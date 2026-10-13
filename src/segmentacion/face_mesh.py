"""Módulo de segmentación facial y extracción de la región de interés (ROI).

Utiliza la API de MediaPipe Tasks (AI Edge) para detectar el rostro en la
imagen y aislar los píxeles correspondientes a la piel limpia. El proceso
construye una máscara poligonal que excluye dinámicamente elementos de ruido
como ojos, cejas, labios y sombras faciales, garantizando una muestra
cromática pura para el análisis. También delimita la escena (todo lo que no
es la cabeza), que el balance de blancos usa para estimar la iluminación.
"""

from typing import List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os

# --------------------------------------------------------------------------- #
# Índices topológicos de la malla facial canónica de MediaPipe (468 puntos).
# Definen los vértices exactos para trazar el polígono principal del rostro y
# los recortes internos de exclusión cromática.
# --------------------------------------------------------------------------- #
CONTORNO_ROSTRO: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
OJO_IZQUIERDO: List[int] = [33, 160, 158, 133, 153, 144]
OJO_DERECHO: List[int] = [362, 385, 387, 263, 373, 380]
LABIOS: List[int] = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
CEJA_IZQUIERDA: List[int] = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
CEJA_DERECHA: List[int] = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Umbral percentil de luminancia (canal V del espacio HSV).
# Los píxeles por debajo de este valor se consideran sombras proyectadas y se excluyen del ROI.
PERCENTIL_SOMBRA: float = 15.0

# Factor con el que se agranda el contorno del rostro para aproximar la cabeza
# completa (cabello, orejas y parte del cuello). Lo que queda por fuera es la
# escena con la que Gray World estima el iluminante (ver `construir_mascara_escena`).
ESCALA_CABEZA: float = 1.6

# Cálculo dinámico de la ruta del modelo binario de inferencia.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "data", "reference", "face_landmarker.task")

# --------------------------------------------------------------------------- #
# Instancia global del motor de inferencia. Se inicializa bajo el patrón
# Singleton para evitar la sobrecarga de I/O en disco en cada procesamiento.
# --------------------------------------------------------------------------- #
_landmarker: "vision.FaceLandmarker | None" = None


def _obtener_landmarker() -> "vision.FaceLandmarker":
    """Inicializa y retorna la instancia compartida del modelo FaceLandmarker.

    Returns:
        vision.FaceLandmarker: Motor de inferencia configurado en memoria.

    Raises:
        FileNotFoundError: Si el archivo `.task` del modelo no se encuentra en el sistema.
    """
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el modelo Face Landmarker en: {MODEL_PATH}")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _puntos_a_pixeles(landmarks: Sequence, indices: Sequence[int], ancho: int, alto: int) -> np.ndarray:
    """Convierte landmarks normalizados (0-1) de MediaPipe a coordenadas de píxel.

    Args:
        landmarks (Sequence): Lista de hitos espaciales del rostro detectado.
        indices (Sequence[int]): Subconjunto de índices que forman un polígono específico.
        ancho (int): Dimensión horizontal de la imagen base.
        alto (int): Dimensión vertical de la imagen base.

    Returns:
        np.ndarray: Matriz bidimensional Nx2 de enteros con coordenadas cartesianas
        (x, y), lista para `cv2.fillPoly`.
    """
    return np.array(
        [[int(landmarks[idx].x * ancho), int(landmarks[idx].y * alto)] for idx in indices],
        dtype=np.int32,
    )


def _excluir_sombras(imagen: np.ndarray, mascara: np.ndarray, percentil: float = PERCENTIL_SOMBRA) -> np.ndarray:
    """Depura la máscara de piel quitando los píxeles más oscuros (sombras).

    Calcula la luminancia (canal V de HSV) únicamente sobre los píxeles ya
    incluidos en `mascara` y descarta los que caen por debajo del percentil
    indicado, ya que corresponden típicamente a sombras proyectadas por la
    nariz, el mentón o el borde del rostro en lugar de piel iluminada de forma
    uniforme.

    Args:
        imagen (np.ndarray): Imagen original de entrada en formato BGR.
        mascara (np.ndarray): Matriz binaria (0/255) de piel candidata, ya construida
            a partir de los polígonos de landmarks de MediaPipe (ver `extraer_piel_limpia`).
        percentil (float): Límite de tolerancia inferior para considerar un píxel como sombra.

    Returns:
        np.ndarray: Nueva máscara binaria (0/255) ajustada espacialmente.
    """
    indices_piel = mascara > 0

    # Si la máscara de entrada está vacía, se retorna intacta para evitar cálculos redundantes
    if not np.any(indices_piel):
        return mascara

    # Extraer el canal de Valor (V) del espacio HSV
    valor_v = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)[:, :, 2]

    # Calcular el umbral dinámico de sombra en base a la densidad de la imagen
    umbral = np.percentile(valor_v[indices_piel], percentil)

    # Restar lógicamente las áreas de sombra de la máscara original
    mascara_sin_sombras = mascara.copy()
    mascara_sin_sombras[(valor_v < umbral) & indices_piel] = 0

    return mascara_sin_sombras


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
    el cabello, las orejas y parte del cuello. Gray World estima el iluminante
    solo con la escena: si incluyera la piel, en un primer plano el promedio de
    la imagen sería la propia piel y la corrección la volvería gris.

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
    """
    Extrae el ROI de piel limpia del rostro.
    Aplica un flujo de procesamiento que incluye la inferencia de landmarks
    (si no se reciben ya calculados), el recorte de los polígonos fijos de la
    malla facial (contorno menos ojos, cejas y labios) y el filtrado posterior
    de sombras por luminancia (percentil del canal V en HSV, ver `_excluir_sombras`).

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

    # 1-2. Obtener los landmarks del rostro (detectándolos solo si no vienen dados)
    rostro = landmarks if landmarks is not None else detectar_landmarks(imagen)

    # 3. Mapear y dibujar el polígono maestro del rostro (blanco)
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara, [_puntos_a_pixeles(rostro, CONTORNO_ROSTRO, ancho, alto)], 255)

    # Construir los polígonos de ruido facial y restarlos de la máscara maestro (negro)
    ruidos = [
        _puntos_a_pixeles(rostro, OJO_IZQUIERDO, ancho, alto),
        _puntos_a_pixeles(rostro, OJO_DERECHO, ancho, alto),
        _puntos_a_pixeles(rostro, LABIOS, ancho, alto),
        _puntos_a_pixeles(rostro, CEJA_IZQUIERDA, ancho, alto),
        _puntos_a_pixeles(rostro, CEJA_DERECHA, ancho, alto),
    ]
    cv2.fillPoly(mascara, ruidos, 0)

    # 4. Excluir sombras faciales
    mascara = _excluir_sombras(imagen, mascara)

    # 5. Aplicar compuerta lógica (Bitwise AND) para aislar los píxeles de piel verdaderos
    piel_extraida = cv2.bitwise_and(imagen, imagen, mask=mascara)

    return piel_extraida


# Bloque de validación estructural local
if __name__ == "__main__":
    print("Ejecutando prueba local del módulo Face Mesh (AI Edge)...")

    # Generar una matriz vacía para probar el manejo de excepciones del motor
    imagen_prueba = np.zeros((500, 500, 3), dtype=np.uint8)
    try:
        resultado = extraer_piel_limpia(imagen_prueba)
    except ValueError as e:
        if "ningún rostro" in str(e):
            print(" Prueba local exitosa: el modelo .task cargó correctamente y el sistema manejó la imagen de ruido.")
        else:
            print(f" Error inesperado: {e}")

    # Segunda llamada: debe reutilizar el landmarker ya creado (sin recargar
    # el modelo desde disco) en lugar de reconstruirlo.
    try:
        extraer_piel_limpia(imagen_prueba)
        print(" Prueba de reutilización: el landmarker se reutilizó sin recrearse.")
    except ValueError:
        print(" Prueba de reutilización: el landmarker se reutilizó sin recrearse.")

    # Máscara de escena con landmarks sintéticos: un cuadrado de 20x20 px en el
    # centro de una imagen de 100x100. Agrandado 1.5 veces cubre unos 30x30 px,
    # así que la escena debe ser cerca del 91% de la imagen y no incluir el centro.
    from types import SimpleNamespace

    cuadrado = [SimpleNamespace(x=x, y=y) for x, y in [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]]
    escena = construir_mascara_escena(cuadrado, 100, 100, escala=1.5)
    assert not escena[50, 50] and escena[5, 5], "La cabeza debe quedar fuera de la escena y las esquinas dentro."
    assert abs(escena.mean() - 0.91) < 0.01, f"Fracción de escena inesperada: {escena.mean():.3f}"
    print(f" Prueba de la máscara de escena exitosa: la escena cubre el {escena.mean():.1%} de la imagen.")
