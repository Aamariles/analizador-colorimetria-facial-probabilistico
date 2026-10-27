"""Transformación al espacio de color CIELAB y extracción del color dominante de cada zona.

Convierte la región de interés (ROI) de piel al espacio L*a*b*, diseñado para que la
distancia entre dos colores se parezca a la diferencia que percibe el ojo, y agrupa sus
píxeles con K-Means para quedarse con el color más representativo: el centro del grupo
más numeroso. A diferencia de un promedio aritmético, ese valor no se deja arrastrar por
los brillos o las sombras que hayan quedado dentro de la máscara.

`color_dominante` hace lo mismo con cualquier máscara (cabello, iris, labios, cejas) y
permite elegir el grupo más oscuro o el más claro, y las funciones `lab_a_hex`,
`hex_a_lab`, `croma` y `matiz` convierten y describen colores L*a*b*.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Máximo de píxeles que entran a K-Means. Una ROI de piel de una foto de 12 MP tiene
# millones de píxeles y K-Means tardaría varios segundos; una muestra aleatoria (con
# semilla fija) de este tamaño da el mismo color dominante en una fracción del tiempo.
MAX_PIXELES_KMEANS = 30000


def _submuestrear(pixeles: np.ndarray, maximo: int = MAX_PIXELES_KMEANS) -> np.ndarray:
    """Devuelve como mucho `maximo` filas de `pixeles`, elegidas al azar con semilla fija."""
    if len(pixeles) <= maximo:
        return pixeles
    indices = np.random.default_rng(0).choice(len(pixeles), size=maximo, replace=False)
    return pixeles[np.sort(indices)]


def bgr_a_lab(imagen: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR de 8 bits a L*a*b* en unidades reales (L* de 0 a 100).

    Con imágenes de 8 bits, OpenCV escala L* a 0-255 y desplaza a* y b* en 128; con
    flotantes en [0.0, 1.0] devuelve las unidades CIELAB reales.
    """
    return cv2.cvtColor(imagen.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def lab_a_hex(color_lab) -> str:
    """Convierte un color [L*, a*, b*] al código hexadecimal sRGB "#rrggbb" (recortado al gamut)."""
    lab = np.array([[color_lab]], dtype=np.float32)
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)[0, 0]
    r, g, b = (int(round(float(np.clip(c, 0.0, 1.0)) * 255)) for c in (bgr[2], bgr[1], bgr[0]))
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_a_lab(codigo: str) -> np.ndarray:
    """Convierte un código "#rrggbb" a [L*, a*, b*]."""
    codigo = codigo.lstrip("#")
    r, g, b = (int(codigo[i:i + 2], 16) for i in (0, 2, 4))
    return bgr_a_lab(np.array([[[b, g, r]]], dtype=np.uint8))[0, 0].astype(np.float64)


def croma(color_lab) -> float:
    """Croma C* = sqrt(a*² + b*²): qué tan saturado es el color."""
    return float(np.hypot(color_lab[1], color_lab[2]))


def matiz(color_lab) -> float:
    """Ángulo de matiz h en grados (0-360): 0° rojo, 90° amarillo, 180° verde, 270° azul."""
    return float(np.degrees(np.arctan2(color_lab[2], color_lab[1])) % 360.0)


def color_iluminado(imagen: np.ndarray, mascara: np.ndarray, desde: float = 50.0, hasta: float = 95.0,
                    minimo_pixeles: int = 20) -> Optional[Tuple[np.ndarray, int]]:
    """Mediana L*a*b* de los píxeles de una franja de claridad: por defecto, la mitad más iluminada.

    Pensada para el cabello: en una foto, buena parte de sus píxeles son sombra entre los
    mechones y el color que uno reconoce como "su cabello" es el de la parte iluminada. Se
    toman los píxeles con L* entre los percentiles `desde` y `hasta` (sin el 5 % más claro,
    que son brillos) y se calcula la mediana de cada canal. En las 100 celebridades con
    etiqueta externa, esta medida del cabello separa las estaciones mejor que la mediana de
    todo el cabello o que las cejas.

    Args:
        imagen (np.ndarray): Imagen BGR.
        mascara (np.ndarray): Matriz booleana (alto x ancho) con los píxeles de la zona.
        desde (float): Percentil de L* desde el que se toman píxeles.
        hasta (float): Percentil de L* hasta el que se toman píxeles.
        minimo_pixeles (int): Si la zona tiene menos píxeles, no se mide.

    Returns:
        Optional[Tuple[np.ndarray, int]]: El color [L*, a*, b*] y el número de píxeles de
        la zona, o None si la zona tiene muy pocos píxeles.
    """
    mascara = np.asarray(mascara, dtype=bool)
    total = int(mascara.sum())
    if total < minimo_pixeles:
        return None
    pixeles = bgr_a_lab(imagen)[mascara].astype(np.float64)
    bajo, alto = np.percentile(pixeles[:, 0], [desde, hasta])
    franja = pixeles[(pixeles[:, 0] >= bajo) & (pixeles[:, 0] <= alto)]
    if len(franja) == 0:
        franja = pixeles
    return np.median(franja, axis=0), total


def color_dominante(imagen: np.ndarray, mascara: np.ndarray, k: int = 2, elegir: str = "mayor",
                    excluir_brillos: float = 0.0, minimo_pixeles: int = 20) -> Optional[Tuple[np.ndarray, int]]:
    """Color representativo de los píxeles de una máscara, con K-Means en CIELAB.

    Args:
        imagen (np.ndarray): Imagen BGR.
        mascara (np.ndarray): Matriz booleana (alto x ancho) con los píxeles de la zona.
        k (int): Número de grupos de color.
        elegir (str): "mayor" (el grupo con más píxeles), "oscuro" (el de menor L*, para
            separar el vello de las cejas de la piel) o "claro" (el de mayor L*).
        excluir_brillos (float): Fracción de los píxeles más claros que se descarta antes
            de agrupar (brillos especulares en el iris, los labios o el cabello).
        minimo_pixeles (int): Si la zona tiene menos píxeles, no se mide.

    Returns:
        Optional[Tuple[np.ndarray, int]]: El color [L*, a*, b*] y el número de píxeles
        de la zona, o None si la zona tiene muy pocos píxeles.
    """
    mascara = np.asarray(mascara, dtype=bool)
    total = int(mascara.sum())
    if total < minimo_pixeles:
        return None
    pixeles = bgr_a_lab(imagen)[mascara].astype(np.float64)
    if excluir_brillos > 0 and len(pixeles) >= minimo_pixeles * 2:
        umbral = np.percentile(pixeles[:, 0], 100.0 * (1.0 - excluir_brillos))
        pixeles = pixeles[pixeles[:, 0] <= umbral]
    pixeles = _submuestrear(pixeles)
    k_efectivo = max(1, min(k, len(np.unique(np.round(pixeles, 2), axis=0))))
    kmeans = KMeans(n_clusters=k_efectivo, random_state=42, n_init=4).fit(pixeles)
    etiquetas, conteos = np.unique(kmeans.labels_, return_counts=True)
    centros = kmeans.cluster_centers_[etiquetas]
    if elegir == "oscuro":
        indice = int(np.argmin(centros[:, 0]))
    elif elegir == "claro":
        indice = int(np.argmax(centros[:, 0]))
    else:
        indice = int(np.argmax(conteos))
    return centros[indice], total


def extraer_color_dominante_lab(imagen_roi: np.ndarray, k: int = 3) -> np.ndarray:
    """Calcula el color dominante de la piel en CIELAB con K-Means.

    Convierte la ROI a L*a*b*, descarta el fondo negro, divide los píxeles de piel en
    `k` grupos de color y devuelve el centro del grupo con más píxeles.

    Args:
        imagen_roi (np.ndarray): Imagen BGR segmentada, con las zonas excluidas
            (fondo y rasgos faciales) en negro absoluto ([0, 0, 0]).
        k (int): Número de grupos de color en que K-Means divide los píxeles de piel.

    Returns:
        np.ndarray: Vector [L*, a*, b*] del color dominante.

    Raises:
        ValueError: Si la imagen no es válida o está vacía, o si no contiene
            píxeles de piel.
    """
    if imagen_roi is None or not isinstance(imagen_roi, np.ndarray) or imagen_roi.size == 0:
        raise ValueError("La imagen de entrada no es válida o está vacía.")

    # 1. Conversión a CIELAB
    # Con imágenes de 8 bits, OpenCV escala L* a 0-255 y desplaza a* y b* en 128; con
    # flotantes en [0.0, 1.0] devuelve las unidades CIELAB reales (L* de 0 a 100).
    imagen_float = imagen_roi.astype(np.float32) / 255.0
    imagen_lab = cv2.cvtColor(imagen_float, cv2.COLOR_BGR2Lab)

    # 2. Descartar el fondo: solo cuentan los píxeles que no están en negro absoluto
    mascara_fondo = np.all(imagen_roi == [0, 0, 0], axis=-1)
    pixeles_piel_lab = imagen_lab[~mascara_fondo]

    if len(pixeles_piel_lab) == 0:
        raise ValueError("La imagen ROI no contiene píxeles de piel válidos para analizar.")

    # En fotos grandes se agrupa una muestra de los píxeles (ver MAX_PIXELES_KMEANS)
    pixeles_piel_lab = _submuestrear(pixeles_piel_lab)

    # K-Means no puede formar más grupos que colores distintos haya (por ejemplo, en
    # una imagen sintética de un solo color); en ese caso se reduce k.
    colores_unicos = np.unique(pixeles_piel_lab, axis=0)
    k_efectivo = max(1, min(k, len(colores_unicos)))

    # 3. Agrupamiento con K-Means
    # Varias inicializaciones (n_init) reducen el riesgo de quedar en un mínimo local,
    # y la semilla fija (random_state) hace que el resultado sea repetible.
    kmeans = KMeans(n_clusters=k_efectivo, random_state=42, n_init=10)
    kmeans.fit(pixeles_piel_lab)

    # 4. Color dominante: el centro del grupo con más píxeles, que aproxima la moda
    etiquetas, conteos = np.unique(kmeans.labels_, return_counts=True)
    indice_dominante = etiquetas[np.argmax(conteos)]
    color_dominante_lab = kmeans.cluster_centers_[indice_dominante]

    return color_dominante_lab


# Prueba local: python src/colorimetria/cielab_math.py
if __name__ == "__main__":
    print("Ejecutando prueba local del módulo CIELAB...")

    # Imagen sintética de 50x50 con un tono cálido (BGR)
    imagen_prueba = np.full((50, 50, 3), (170, 200, 255), dtype=np.uint8)

    # Bloque negro que simula la parte excluida por la máscara
    imagen_prueba[0:20, 0:20] = [0, 0, 0]

    try:
        color_lab = extraer_color_dominante_lab(imagen_prueba)
        print("Prueba local exitosa.")
        print(f"Color dominante en CIELAB: L*={color_lab[0]:.2f}, a*={color_lab[1]:.2f}, b*={color_lab[2]:.2f}")
    except Exception as e:
        print(f"Error inesperado en la prueba: {e}")
