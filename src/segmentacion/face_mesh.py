"""Segmentación facial y extracción de la región de interés (ROI) de piel.

Usa la API de MediaPipe Tasks (AI Edge) para detectar el rostro y sus landmarks, y con
ellos arma una máscara de la piel: el óvalo del rostro, recortado en el borde, menos los
ojos, las cejas y los labios, y sin sombras ni píxeles de color atípico (cabello, restos
de labio, brillos). Así la muestra de color queda lo más limpia posible. También delimita
la escena (todo lo que no es la cabeza), que el balance de blancos usa para estimar la
iluminación.
"""

import os
from typing import List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
# Las clases se importan desde el módulo donde se definen. `mediapipe.tasks.python.vision`
# las reexporta con asignaciones (FaceLandmarker = face_landmarker.FaceLandmarker) que el
# análisis estático no puede seguir, y Pylance las ve como variables de tipo desconocido.
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions

# --------------------------------------------------------------------------- #
# Índices de los landmarks de MediaPipe que trazan el contorno del rostro y los
# rasgos que se excluyen de la piel. El modelo devuelve 478 puntos: los 468 de la
# malla facial y 10 del iris. "Izquierdo" y "derecho" se refieren a la imagen, no
# a la persona: OJO_IZQUIERDO es el que aparece a la izquierda de la foto.
# --------------------------------------------------------------------------- #
# Óvalo del rostro (FACEMESH_FACE_OVAL de MediaPipe), desde la parte alta de la frente (10).
CONTORNO_ROSTRO: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
# Contorno completo de cada ojo (16 puntos, como FACEMESH_*_EYE de MediaPipe).
OJO_IZQUIERDO: List[int] = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
OJO_DERECHO: List[int] = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
# Contorno exterior de los dos labios: el superior (61 -> 291) y el inferior de regreso
# (291 -> 61). Cubre también la boca abierta y los dientes.
LABIOS: List[int] = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
CEJA_IZQUIERDA: List[int] = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
CEJA_DERECHA: List[int] = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Márgenes de la máscara, como fracción del ancho del rostro (distancia entre los
# landmarks 234 y 454), para que escalen igual en fotos grandes y pequeñas.
# El borde del óvalo se recorta para quitar mechones de cabello en frente y sienes,
# barba en el contorno de la mandíbula y fondo; ojos, cejas y labios se agrandan
# para quitar pestañas, delineado, vello de las cejas y el borde del labio.
EROSION_ROSTRO: float = 0.06
DILATACION_RASGOS: float = 0.02

# Percentil de brillo (canal V de HSV) que separa las sombras: el 15% más oscuro de la
# máscara (sombras de la nariz, el mentón o el borde del rostro) se excluye de la ROI.
PERCENTIL_SOMBRA: float = 15.0

# Filtro de color robusto: se descartan los píxeles cuyo L*, a* o b* se aleja de la
# mediana de la piel más de Z_MAXIMO_COLOR desviaciones robustas (MAD). Atrapa mechones
# que caen dentro de la cara, restos de labio y brillos. La desviación de cada canal
# nunca se toma menor que DESVIACION_MINIMA_LAB, para no recortar de más una piel muy
# uniforme ni el sombreado natural del rostro.
Z_MAXIMO_COLOR: float = 3.0
DESVIACION_MINIMA_LAB = (8.0, 2.0, 2.0)

# Factor con el que se agranda el contorno del rostro para aproximar la cabeza
# completa (cabello, orejas y parte del cuello). Lo que queda por fuera es la
# escena con la que Gray World estima el iluminante (ver `construir_mascara_escena`).
ESCALA_CABEZA: float = 1.6

# Ruta del modelo de MediaPipe, calculada desde la ubicación de este archivo para que
# funcione sin importar desde qué carpeta se ejecute el programa.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "data", "reference", "face_landmarker.task")

# --------------------------------------------------------------------------- #
# Instancia global del motor de inferencia. Se crea una sola vez (patrón
# Singleton) para no volver a cargar el modelo desde el disco en cada imagen.
# --------------------------------------------------------------------------- #
_landmarker: Optional[FaceLandmarker] = None


def _obtener_landmarker() -> FaceLandmarker:
    """Devuelve la instancia compartida del modelo FaceLandmarker; la crea la primera vez.

    Returns:
        FaceLandmarker: Motor de inferencia cargado en memoria.

    Raises:
        FileNotFoundError: Si el archivo `.task` del modelo no se encuentra en el sistema.
    """
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el modelo Face Landmarker en: {MODEL_PATH}")
        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def _puntos_a_pixeles(landmarks: Sequence, indices: Sequence[int], ancho: int, alto: int) -> np.ndarray:
    """Convierte landmarks normalizados (0-1) de MediaPipe a coordenadas de píxel.

    Args:
        landmarks (Sequence): Landmarks del rostro detectado.
        indices (Sequence[int]): Índices de los landmarks que forman el polígono.
        ancho (int): Ancho de la imagen en píxeles.
        alto (int): Alto de la imagen en píxeles.

    Returns:
        np.ndarray: Matriz N x 2 de enteros con las coordenadas (x, y) de cada punto,
        lista para `cv2.fillPoly`.
    """
    return np.array(
        [[int(landmarks[idx].x * ancho), int(landmarks[idx].y * alto)] for idx in indices],
        dtype=np.int32,
    )


def _excluir_sombras(imagen: np.ndarray, mascara: np.ndarray, percentil: float = PERCENTIL_SOMBRA) -> np.ndarray:
    """Quita de la máscara de piel los píxeles más oscuros (sombras).

    Calcula el brillo (canal V de HSV) solo sobre los píxeles de `mascara` y descarta
    los que quedan por debajo del percentil indicado: suelen ser sombras de la nariz,
    el mentón o el borde del rostro, no piel bien iluminada.

    Args:
        imagen (np.ndarray): Imagen BGR de la que se toma el brillo.
        mascara (np.ndarray): Matriz binaria (0/255) de piel candidata, construida con
            los polígonos de landmarks de MediaPipe (ver `extraer_piel_limpia`).
        percentil (float): Percentil de brillo por debajo del cual un píxel se considera sombra.

    Returns:
        np.ndarray: Nueva máscara binaria (0/255) sin las sombras.
    """
    indices_piel = mascara > 0

    # Si la máscara está vacía, no hay nada que filtrar
    if not np.any(indices_piel):
        return mascara

    # Brillo de cada píxel: canal V del espacio HSV
    valor_v = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)[:, :, 2]

    # Umbral de sombra calculado solo con los píxeles de la máscara
    umbral = np.percentile(valor_v[indices_piel], percentil)

    # Quitar de la máscara los píxeles más oscuros que el umbral
    mascara_sin_sombras = mascara.copy()
    mascara_sin_sombras[(valor_v < umbral) & indices_piel] = 0

    return mascara_sin_sombras


def _excluir_atipicos_de_color(imagen: np.ndarray, mascara: np.ndarray, z_maximo: float = Z_MAXIMO_COLOR) -> np.ndarray:
    """Quita de la máscara los píxeles cuyo color no se parece al de la piel.

    Calcula la mediana y la desviación robusta (MAD) de L*, a* y b* sobre los
    píxeles de la máscara, y descarta los que se alejan más de `z_maximo`
    desviaciones en cualquiera de los tres canales: mechones de cabello que
    caen sobre la cara, restos del labio o de las cejas y brillos especulares.

    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR.
        mascara (np.ndarray): Matriz binaria (0/255) de piel candidata.
        z_maximo (float): Número de desviaciones robustas a partir del cual un píxel se descarta.

    Returns:
        np.ndarray: Nueva máscara binaria (0/255) sin los píxeles atípicos.
    """
    indices_piel = mascara > 0
    if np.count_nonzero(indices_piel) < 50:
        return mascara

    lab = cv2.cvtColor(imagen.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[indices_piel]
    mediana = np.median(lab, axis=0)
    desviacion = np.median(np.abs(lab - mediana), axis=0) * 1.4826  # MAD escalada a desviación estándar
    desviacion = np.maximum(desviacion, DESVIACION_MINIMA_LAB)
    atipicos = (np.abs(lab - mediana) / desviacion > z_maximo).any(axis=1)

    mascara_filtrada = mascara.copy()
    filas, columnas = np.nonzero(indices_piel)
    mascara_filtrada[filas[atipicos], columnas[atipicos]] = 0
    return mascara_filtrada


def _ancho_rostro(landmarks: Sequence, ancho: int, alto: int) -> float:
    """Distancia en píxeles entre los extremos izquierdo (234) y derecho (454) del rostro."""
    return float(np.hypot((landmarks[454].x - landmarks[234].x) * ancho, (landmarks[454].y - landmarks[234].y) * alto))


def _disco(radio: float) -> np.ndarray:
    """Elemento estructurante circular para erosionar o dilatar máscaras."""
    radio_px = max(1, int(round(radio)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio_px + 1, 2 * radio_px + 1))


def detectar_landmarks(imagen: np.ndarray) -> Sequence:
    """Detecta el rostro en la imagen y devuelve sus landmarks.

    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR.

    Returns:
        Sequence: Los landmarks del primer rostro detectado, con coordenadas
        normalizadas (0-1) respecto al ancho y alto de la imagen.

    Raises:
        ValueError: Si la imagen de entrada no es válida o está vacía, o si
            MediaPipe no detecta ningún rostro.
        FileNotFoundError: Si el modelo `.task` no existe en `MODEL_PATH`.
    """
    if imagen is None or not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("La matriz de imagen de entrada no es válida o carece de información.")

    # Convertir la matriz de OpenCV al formato nativo de MediaPipe y ejecutar la
    # inferencia reutilizando el motor alojado en memoria
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
    resultado = _obtener_landmarker().detect(mp_image)

    if not resultado.face_landmarks:
        raise ValueError("MediaPipe no detectó ningún rostro en la imagen.")

    return resultado.face_landmarks[0]


def construir_mascara_escena(landmarks: Sequence, ancho: int, alto: int,
                             escala: float = ESCALA_CABEZA) -> np.ndarray:
    """Marca los píxeles de la escena: todo lo que no es la cabeza de la persona.

    La cabeza se aproxima con la envolvente convexa de todos los landmarks del
    rostro, agrandada `escala` veces alrededor de su centro para cubrir también
    el cabello, las orejas y parte del cuello. Gray World estima el iluminante con
    los píxeles más neutros de esta escena (ver
    `preprocesamiento.gray_world.seleccionar_pixeles_neutros`): si incluyera la piel,
    en un primer plano el promedio sería la propia piel y la corrección la volvería gris.

    Args:
        landmarks (Sequence): Landmarks normalizados (0-1) devueltos por `detectar_landmarks`.
        ancho (int): Ancho de la imagen en píxeles.
        alto (int): Alto de la imagen en píxeles.
        escala (float): Factor de agrandamiento del contorno del rostro.

    Returns:
        np.ndarray: Matriz booleana (alto x ancho); True en los píxeles de la escena.
    """
    puntos = np.array([[p.x * ancho, p.y * alto] for p in landmarks], dtype=np.float32)
    envolvente = cv2.convexHull(puntos).reshape(-1, 2)
    centro = puntos.mean(axis=0)
    cabeza = ((envolvente - centro) * escala + centro).astype(np.int32)

    mascara_cabeza = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara_cabeza, [cabeza], 255)
    return mascara_cabeza == 0


def extraer_piel_limpia(imagen: np.ndarray, landmarks: Optional[Sequence] = None) -> np.ndarray:
    """Extrae la región de interés (ROI) de piel limpia del rostro.

    Detecta los landmarks si no se reciben ya calculados, dibuja el óvalo del rostro
    recortado en el borde, le resta los ojos, las cejas y los labios agrandados, y luego
    quita las sombras (ver `_excluir_sombras`) y los píxeles de color atípico (ver
    `_excluir_atipicos_de_color`).

    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR.
        landmarks (Optional[Sequence]): Landmarks del rostro ya detectados en esta
            imagen o en la misma imagen antes del balance de blancos (que no cambia
            la geometría). Si es None, se detectan aquí con `detectar_landmarks`.

    Returns:
        np.ndarray: Imagen BGR del mismo tamaño que `imagen`, con los píxeles
        fuera del ROI de piel puestos en negro ([0, 0, 0]).

    Raises:
        ValueError: Si la imagen de entrada no es válida o está vacía, o si
            MediaPipe no detecta ningún rostro.
        FileNotFoundError: Si el modelo `.task` no existe en `MODEL_PATH`.
    """
    if imagen is None or not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("La matriz de imagen de entrada no es válida o carece de información.")

    alto, ancho = imagen.shape[:2]

    # 1. Landmarks del rostro (se detectan solo si no vienen dados)
    rostro = landmarks if landmarks is not None else detectar_landmarks(imagen)

    ancho_cara = _ancho_rostro(rostro, ancho, alto)

    # 2. Óvalo del rostro (en blanco), recortado en el borde, donde suelen colarse
    # cabello, barba y fondo
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara, [_puntos_a_pixeles(rostro, CONTORNO_ROSTRO, ancho, alto)], 255)
    mascara = cv2.erode(mascara, _disco(EROSION_ROSTRO * ancho_cara))

    # 3. Ojos, cejas y labios, agrandados un margen, se restan del óvalo
    ruidos = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(ruidos, [
        _puntos_a_pixeles(rostro, OJO_IZQUIERDO, ancho, alto),
        _puntos_a_pixeles(rostro, OJO_DERECHO, ancho, alto),
        _puntos_a_pixeles(rostro, LABIOS, ancho, alto),
        _puntos_a_pixeles(rostro, CEJA_IZQUIERDA, ancho, alto),
        _puntos_a_pixeles(rostro, CEJA_DERECHA, ancho, alto),
    ], 255)
    ruidos = cv2.dilate(ruidos, _disco(DILATACION_RASGOS * ancho_cara))
    mascara[ruidos > 0] = 0

    # 4. Excluir sombras faciales y colores que no son piel
    mascara = _excluir_sombras(imagen, mascara)
    mascara = _excluir_atipicos_de_color(imagen, mascara)

    # 5. Dejar en la imagen solo los píxeles de la máscara; el resto queda en negro
    piel_extraida = cv2.bitwise_and(imagen, imagen, mask=mascara)

    return piel_extraida


# Prueba local: python src/segmentacion/face_mesh.py
if __name__ == "__main__":
    print("Ejecutando prueba local del módulo Face Mesh (AI Edge)...")

    # Prueba 1: una imagen negra no tiene rostro y debe rechazarse con el error esperado
    imagen_prueba = np.zeros((500, 500, 3), dtype=np.uint8)
    try:
        extraer_piel_limpia(imagen_prueba)
        raise AssertionError("Una imagen sin rostro debería producir un ValueError.")
    except ValueError as e:
        assert "ningún rostro" in str(e), f"Error inesperado: {e}"
    print("Prueba 1 exitosa: el modelo se cargó y la imagen sin rostro se rechazó.")

    # Prueba 2: las llamadas siguientes deben reutilizar el mismo motor de inferencia
    # en lugar de volver a cargar el modelo desde el disco
    motor = _obtener_landmarker()
    assert _obtener_landmarker() is motor, "El motor de inferencia debería reutilizarse."
    print("Prueba 2 exitosa: el motor de inferencia se reutiliza entre llamadas.")

    # Prueba 3: máscara de escena con landmarks sintéticos, un cuadrado de 20x20 px en
    # el centro de una imagen de 100x100. Agrandado 1.5 veces cubre unos 30x30 px, así
    # que la escena debe ser cerca del 91% de la imagen y no incluir el centro.
    from types import SimpleNamespace

    cuadrado = [SimpleNamespace(x=x, y=y) for x, y in [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]]
    escena = construir_mascara_escena(cuadrado, 100, 100, escala=1.5)
    assert not escena[50, 50] and escena[5, 5], "La cabeza debe quedar fuera de la escena y las esquinas dentro."
    assert abs(escena.mean() - 0.91) < 0.01, f"Fracción de escena inesperada: {escena.mean():.3f}"
    print(f"Prueba 3 exitosa: la escena cubre el {escena.mean():.1%} de la imagen.")

    # Prueba 4: filtro de color sobre una zona de piel con un mechón de cabello rubio
    # y un resto de labio. Ambos deben salir de la máscara y la piel debe quedarse.
    rng = np.random.default_rng(0)
    parche = np.clip(rng.normal((150, 170, 215), 3, (100, 100, 3)), 0, 255).astype(np.uint8)  # piel (BGR)
    parche[:, 10:16] = (80, 175, 220)   # mechón rubio: mucho más amarillo que la piel
    parche[80:86, 40:60] = (110, 95, 200)  # resto de labio: mucho más rojo que la piel
    filtrada = _excluir_atipicos_de_color(parche, np.full((100, 100), 255, dtype=np.uint8))
    assert not filtrada[:, 10:16].any(), "El mechón de cabello debe salir de la máscara."
    assert not filtrada[80:86, 40:60].any(), "El resto de labio debe salir de la máscara."
    assert (filtrada[:, 30:] > 0).mean() > 0.95, "La piel debe quedarse en la máscara."
    print("Prueba 4 exitosa: el filtro de color quitó el mechón y el labio y conservó la piel.")
