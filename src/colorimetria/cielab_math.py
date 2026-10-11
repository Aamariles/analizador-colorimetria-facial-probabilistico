"""Módulo de transformación de color CIELAB y modelado estadístico.

Convierte regiones de interés (ROI) al espacio de color perceptualmente uniforme
L*a*b* y aplica algoritmos de agrupamiento no supervisado (K-Means) para extraer
la moda estadística del tono. Este enfoque probabilístico mitiga el sesgo numérico
introducido por valores atípicos como brillos especulares o sombras residuales,
superando las limitaciones de un promedio aritmético lineal.
"""


import cv2
import numpy as np
from sklearn.cluster import KMeans


def extraer_color_dominante_lab(imagen_roi: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Transforma la imagen ROI (piel limpia) al espacio perceptualmente uniforme CIELAB
    y extrae el centroide estadístico dominante utilizando K-Means.

    Args:
        imagen_roi (np.ndarray): Matriz de imagen BGR segmentada, donde las áreas
            excluidas (fondo o ruido) están definidas en negro absoluto ([0, 0, 0]).
        k (int): Número de clústeres objetivo para la partición espacial de los datos.

    Returns:
        np.ndarray: Vector unidimensional con las coordenadas [L*, a*, b*] del color
        dominante estimado.

    Raises:
        ValueError: Si la matriz de imagen no es válida, está vacía o carece de
            píxeles analizables.
    """
    if imagen_roi is None or not isinstance(imagen_roi, np.ndarray) or imagen_roi.size == 0:
        raise ValueError("La imagen de entrada no es válida o está vacía.")

    # 1. Normalización de la matriz
    # La conversión precisa al espacio CIELAB en OpenCV requiere que la matriz base
    # se proyecte a una escala de punto flotante [0.0, 1.0] para evitar truncamientos.
    imagen_float = imagen_roi.astype(np.float32) / 255.0
    imagen_lab = cv2.cvtColor(imagen_float, cv2.COLOR_BGR2Lab)

    # 2. Filtrado de ruido
    # Extraer exclusivamente los vectores de color correspondientes a la región de interés
    mascara_fondo = np.all(imagen_roi == [0, 0, 0], axis=-1)
    pixeles_piel_lab = imagen_lab[~mascara_fondo]

    if len(pixeles_piel_lab) == 0:
        raise ValueError("La imagen ROI no contiene píxeles de piel válidos para analizar.")

    # Ajuste dinámico de hiperparámetros
    # Previene la degeneración espacial del algoritmo si la varianza del ROI es inusualmente
    # baja (menor cantidad de colores únicos que el número de clústeres objetivo).
    colores_unicos = np.unique(pixeles_piel_lab, axis=0)
    k_efectivo = max(1, min(k, len(colores_unicos)))

    # 3. Modelado estadístico de agrupamiento (K-Means)
    # Se parametriza con múltiples inicializaciones (n_init) para evitar mínimos locales
    # y garantizar la convergencia global de los centroides.
    kmeans = KMeans(n_clusters=k_efectivo, random_state=42, n_init=10)
    kmeans.fit(pixeles_piel_lab)

    # 4. Extracción de la moda espacial
    # Identificar el clúster con la mayor densidad poblacional (moda matemática)
    etiquetas, conteos = np.unique(kmeans.labels_, return_counts=True)
    indice_dominante = etiquetas[np.argmax(conteos)]
    color_dominante_lab = kmeans.cluster_centers_[indice_dominante]

    return color_dominante_lab


# Bloque de validación estructural local
if __name__ == "__main__":
    print("Ejecutando prueba local del módulo matemático CIELAB...")

    # Generación de matriz sintética (50x50) simulando un tono cálido
    imagen_prueba = np.full((50, 50, 3), (170, 200, 255), dtype=np.uint8)

    # Insertar un bloque de fondo negro simulando la máscara
    imagen_prueba[0:20, 0:20] = [0, 0, 0]

    try:
        color_lab = extraer_color_dominante_lab(imagen_prueba)
        print("Prueba local exitosa.")
        print(f"Vector CIELAB dominante calculado: L*={color_lab[0]:.2f}, a*={color_lab[1]:.2f}, b*={color_lab[2]:.2f}")
    except Exception as e:
        print(f"Error inesperado en la prueba: {e}")
