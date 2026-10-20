"""Transformación al espacio de color CIELAB y extracción del color dominante de la piel.

Convierte la región de interés (ROI) de piel al espacio L*a*b*, diseñado para que la
distancia entre dos colores se parezca a la diferencia que percibe el ojo, y agrupa sus
píxeles con K-Means para quedarse con el color más representativo: el centro del grupo
más numeroso. A diferencia de un promedio aritmético, ese valor no se deja arrastrar por
los brillos o las sombras que hayan quedado dentro de la máscara.
"""


import cv2
import numpy as np
from sklearn.cluster import KMeans


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
