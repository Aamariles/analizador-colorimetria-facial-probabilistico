"""Módulo de preprocesamiento para la corrección de balance de blancos.

Implementa el algoritmo Gray World para neutralizar los sesgos de color
introducidos por variaciones en la iluminación de la imagen. El algoritmo
se fundamenta en el principio de que, en una escena natural estándar,
el promedio global de las intensidades de color tiende a un gris neutral.

En un retrato ese supuesto deja de cumplirse si el promedio se calcula sobre
toda la imagen: la piel ocupa buena parte de los píxeles, así que el "gris"
estimado es en realidad el tono de la piel y la corrección la neutraliza
(le quita a* y b*). Por eso `aplicar_gray_world` acepta una máscara con los
píxeles de la escena (todo lo que no es la cabeza): el iluminante se estima
solo con ellos y los factores de escala se aplican a toda la imagen.

Aun sin la cabeza, la escena de un retrato no es gris: el cabello largo, la piel
del cuello y los hombros y la ropa de color son cálidos y harían que la corrección
enfriara la piel. `seleccionar_pixeles_neutros` se queda solo con los píxeles menos
saturados de la escena (paredes, ropa blanca o gris), que son los que de verdad
deberían ser grises.
"""

from typing import Optional

import cv2
import numpy as np

# Valor mínimo de tolerancia para evitar inestabilidad numérica en canales sin información.
_EPSILON = 1e-6

# Fracción mínima de la imagen que debe cubrir la escena para estimar el iluminante.
# Si la cabeza llena prácticamente todo el encuadre no hay escena de la cual
# estimar la luz, y la imagen se devuelve sin corregir.
FRACCION_MINIMA_ESCENA = 0.01

# Fracción de los píxeles de la escena, los menos saturados, que se toman como
# referencia de gris.
FRACCION_NEUTRA = 0.25

# Rango de brillo (canal V de HSV, 0-255) de los píxeles candidatos a gris: los casi
# negros no tienen información de color y los muy brillantes pueden estar recortados.
BRILLO_MINIMO = 40
BRILLO_MAXIMO = 250


def seleccionar_pixeles_neutros(imagen: np.ndarray, mascara: np.ndarray,
                                fraccion: float = FRACCION_NEUTRA) -> np.ndarray:
    """Elige, dentro de una máscara, los píxeles que con más probabilidad son grises.

    Descarta los píxeles casi negros o casi blancos y, de los restantes, conserva la
    fracción indicada con la menor saturación (canal S de HSV). En un retrato eso deja
    la pared del fondo o la ropa blanca y descarta el cabello y la piel del cuello o
    los hombros, que son más saturados.

    Args:
        imagen (np.ndarray): Imagen BGR.
        mascara (np.ndarray): Matriz booleana (alto x ancho) con los píxeles candidatos,
            por ejemplo la escena de `segmentacion.face_mesh.construir_mascara_escena`.
        fraccion (float): Fracción de los candidatos, los menos saturados, que se conserva.

    Returns:
        np.ndarray: Matriz booleana (alto x ancho) con los píxeles neutros seleccionados.
    """
    candidatos = np.asarray(mascara, dtype=bool).copy()
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    saturacion, brillo = hsv[:, :, 1], hsv[:, :, 2]
    candidatos &= (brillo > BRILLO_MINIMO) & (brillo < BRILLO_MAXIMO)
    if not candidatos.any():
        return candidatos
    umbral = np.percentile(saturacion[candidatos], fraccion * 100.0)
    return candidatos & (saturacion <= umbral)


def aplicar_gray_world(imagen: np.ndarray, mascara_escena: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Aplica el algoritmo Gray World a una imagen para corregir el balance de blancos.

    Los promedios por canal (la estimación del iluminante) se calculan sobre los
    píxeles indicados por `mascara_escena`; los factores de escala resultantes
    se aplican a toda la imagen.

    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR.
        mascara_escena (Optional[np.ndarray]): Matriz booleana (alto x ancho) con
            los píxeles que se usan para estimar el iluminante. En retratos debe
            excluir la cabeza (ver `segmentacion.face_mesh.construir_mascara_escena`).
            Si es None se usan todos los píxeles (Gray World clásico).

    Returns:
        np.ndarray: Imagen con el balance de blancos corregido en formato BGR. Si la
        escena cubre menos de `FRACCION_MINIMA_ESCENA` de la imagen, se devuelve la
        imagen sin modificar.

    Raises:
        ValueError: Si la imagen proporcionada está vacía o no es un arreglo válido,
            o si la máscara no tiene el mismo alto y ancho que la imagen.
    """
    if imagen is None or not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("La imagen proporcionada no es válida o está vacía.")

    # Separar la imagen en sus tres canales (Azul, Verde, Rojo)
    b, g, r = cv2.split(imagen)

    # 1. Calcular los promedios de cada canal (B_avg, G_avg, R_avg) sobre la escena
    if mascara_escena is None:
        b_avg, g_avg, r_avg = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))
    else:
        mascara = np.asarray(mascara_escena, dtype=bool)
        if mascara.shape != imagen.shape[:2]:
            raise ValueError(
                f"La máscara de escena debe medir {imagen.shape[:2]} (alto, ancho); se recibió {mascara.shape}."
            )
        if mascara.mean() < FRACCION_MINIMA_ESCENA:
            return imagen
        b_avg, g_avg, r_avg = (float(np.mean(canal[mascara])) for canal in (b, g, r))

    # Si los tres canales están agotados (escena completamente negra) no hay
    # información de color que corregir: se devuelve la imagen sin modificar.
    if b_avg < _EPSILON and g_avg < _EPSILON and r_avg < _EPSILON:
        return imagen

    # 2. Calcular el promedio global (Gray_global)
    gray_global = (b_avg + g_avg + r_avg) / 3.0

    # Asignar un valor mínimo a los promedios nulos para garantizar la estabilidad
    # matemática durante el cálculo de los factores de escala
    b_avg_seguro = b_avg if b_avg >= _EPSILON else _EPSILON
    g_avg_seguro = g_avg if g_avg >= _EPSILON else _EPSILON
    r_avg_seguro = r_avg if r_avg >= _EPSILON else _EPSILON

    # 3. Calcular los factores de escala y aplicarlos a los canales de toda la imagen
    # Se convierte explícitamente a float32 antes de escalar para evitar
    # cualquier ambigüedad de tipo, y np.clip asegura que los valores de los
    # píxeles no superen el rango válido [0, 255] antes de volver a uint8.
    b_new = np.clip(b.astype(np.float32) * (gray_global / b_avg_seguro), 0, 255).astype(np.uint8)
    g_new = np.clip(g.astype(np.float32) * (gray_global / g_avg_seguro), 0, 255).astype(np.uint8)
    r_new = np.clip(r.astype(np.float32) * (gray_global / r_avg_seguro), 0, 255).astype(np.uint8)

    # Reconstruir la imagen fusionando los canales ajustados
    imagen_corregida = cv2.merge((b_new, g_new, r_new))

    return imagen_corregida


# Bloque de prueba local
if __name__ == "__main__":
    print("Ejecutando prueba local del módulo Gray World...")

    # Caso 1: imagen con sesgo rojo en los tres canales (caso general)
    imagen_prueba = np.random.randint(0, 150, (100, 100, 3), dtype=np.uint8)
    imagen_prueba[:, :, 2] = 250  # Saturar intencionalmente el canal R (índice 2)

    try:
        resultado = aplicar_gray_world(imagen_prueba)
        print(f"Forma original: {imagen_prueba.shape} | Promedio Rojo original: {np.mean(imagen_prueba[:, :, 2]):.2f}")
        print(f"Forma corregida: {resultado.shape} | Promedio Rojo corregido: {np.mean(resultado[:, :, 2]):.2f}")
        print("Prueba 1 (caso general) exitosa.")
    except Exception as e:
        print(f"Error en la prueba 1: {e}")

    # Caso 2 (regresión): un único canal completamente agotado (B_avg == 0)
    # mientras los otros dos sí tienen señal. Antes de la corrección esto
    # generaba una división por cero silenciosa (NaN -> 0 al castear a uint8).
    imagen_canal_cero = np.zeros((10, 10, 3), dtype=np.uint8)
    imagen_canal_cero[:, :, 1] = 100  # Canal Verde activo
    imagen_canal_cero[:, :, 2] = 150  # Canal Rojo activo
    try:
        with np.errstate(divide="raise", invalid="raise"):
            resultado_cero = aplicar_gray_world(imagen_canal_cero)
        print(f"Prueba 2 (canal B en cero) exitosa, sin warnings de división. Pixel ejemplo: {resultado_cero[0, 0]}")
    except FloatingPointError as e:
        print(f"Prueba 2 FALLÓ: aún ocurre una operación inválida en punto flotante ({e}).")
    except Exception as e:
        print(f"Error inesperado en la prueba 2: {e}")

    # Caso 3 (regresión del sesgo hacia Verano): retrato sintético en el que la
    # "piel" ocupa la mitad de la imagen sobre un fondo gris neutro. Con el
    # promedio de toda la imagen, Gray World neutraliza la piel; estimando el
    # iluminante solo con la escena, la piel conserva su tono.
    retrato = np.full((100, 100, 3), 128, dtype=np.uint8)
    retrato[15:85, 15:85] = (120, 150, 200)  # piel cálida en BGR
    escena = np.ones((100, 100), dtype=bool)
    escena[15:85, 15:85] = False
    piel_original = retrato[50, 50].astype(int)
    piel_total = aplicar_gray_world(retrato)[50, 50].astype(int)
    piel_escena = aplicar_gray_world(retrato, mascara_escena=escena)[50, 50].astype(int)
    print(f"Prueba 3: piel original {piel_original} | Gray World con toda la imagen {piel_total} | "
          f"Gray World con la escena {piel_escena}")
    assert np.array_equal(piel_escena, piel_original), "Con una escena neutra, la piel no debe cambiar."
    assert piel_total[2] - piel_total[0] < piel_original[2] - piel_original[0], (
        "Con toda la imagen, Gray World debería reducir la diferencia R-B de la piel."
    )
    print("Prueba 3 (la piel no se neutraliza cuando se usa la escena) exitosa.")

    # Caso 4: retrato con cabello largo cálido a los lados de la cara sobre una pared
    # gris. Si se usa toda la escena, el cabello hace que la corrección enfríe la
    # piel; con los píxeles neutros solo cuenta la pared y la piel no cambia.
    retrato = np.full((100, 100, 3), 128, dtype=np.uint8)  # pared gris
    retrato[10:100, 10:35] = (50, 90, 150)  # cabello castaño (BGR), mucho más saturado que la pared
    retrato[10:100, 65:90] = (50, 90, 150)
    retrato[20:80, 35:65] = (120, 150, 200)  # piel
    escena = np.ones((100, 100), dtype=bool)
    escena[20:80, 35:65] = False
    neutros = seleccionar_pixeles_neutros(retrato, escena)
    piel_original = retrato[50, 50].astype(int)
    piel_escena = aplicar_gray_world(retrato, mascara_escena=escena)[50, 50].astype(int)
    piel_neutros = aplicar_gray_world(retrato, mascara_escena=neutros)[50, 50].astype(int)
    print(f"Prueba 4: piel original {piel_original} | Gray World con toda la escena {piel_escena} | "
          f"Gray World con los píxeles neutros {piel_neutros}")
    assert not neutros[50, 20], "El cabello no debe tomarse como referencia de gris."
    assert np.array_equal(piel_neutros, piel_original), "Con la pared como referencia, la piel no debe cambiar."
    assert piel_escena[2] - piel_escena[0] < piel_original[2] - piel_original[0], (
        "Con el cabello en la referencia, Gray World debería enfriar la piel."
    )
    print("Prueba 4 (el cabello no enfría la piel cuando se usan los píxeles neutros) exitosa.")
