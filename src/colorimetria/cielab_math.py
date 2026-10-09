import cv2
import numpy as np
from sklearn.cluster import KMeans

def extraer_color_dominante_lab(imagen_roi: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Transforma la imagen ROI (piel limpia) al espacio perceptualmente uniforme CIELAB
    y extrae el centroide estadístico dominante utilizando K-Means.

    Args:
        imagen_roi (np.ndarray): Imagen BGR con el fondo en negro ([0, 0, 0]).
        k (int): Número de clústeres para el análisis espacial.

    Returns:
        np.ndarray: Vector 1D con los valores [L*, a*, b*] del color dominante.
        
    Raises:
        ValueError: Si la imagen no tiene píxeles válidos de piel.
    """
    if imagen_roi is None or not isinstance(imagen_roi, np.ndarray):
        raise ValueError("La imagen de entrada no es válida.")

    # 1. Normalización de la matriz
    # OpenCV requiere que la imagen esté en formato de punto flotante [0, 1] 
    # para no truncar los cálculos matemáticos del espacio CIELAB.
    imagen_float = imagen_roi.astype(np.float32) / 255.0
    imagen_lab = cv2.cvtColor(imagen_float, cv2.COLOR_BGR2Lab)

    # 2. Filtrado de ruido
    # Se excluyen los píxeles del fondo negro absoluto originados por la máscara de MediaPipe
    mascara_fondo = np.all(imagen_roi == [0, 0, 0], axis=-1)
    pixeles_piel_lab = imagen_lab[~mascara_fondo]

    if len(pixeles_piel_lab) == 0:
        raise ValueError("La imagen ROI no contiene píxeles de piel válidos para analizar.")

    # 3. Modelado estadístico con K-Means
    # n_init=10 ejecuta el algoritmo 10 veces con diferentes centroides iniciales para garantizar convergencia
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixeles_piel_lab)

    # 4. Extracción de la moda matemática
    # Encontramos el clúster (tono) que concentra la mayor cantidad de píxeles
    etiquetas, conteos = np.unique(kmeans.labels_, return_counts=True)
    indice_dominante = etiquetas[np.argmax(conteos)]
    color_dominante_lab = kmeans.cluster_centers_[indice_dominante]

    return color_dominante_lab

# Bloque de prueba local
if __name__ == "__main__":
    print("Ejecutando prueba local del módulo matemático CIELAB...")
    
    # Crear una imagen simulada (50x50 píxeles) de color durazno en BGR
    imagen_prueba = np.full((50, 50, 3), (170, 200, 255), dtype=np.uint8)
    # Insertar un bloque de fondo negro simulando la máscara
    imagen_prueba[0:20, 0:20] = [0, 0, 0]

    try:
        color_lab = extraer_color_dominante_lab(imagen_prueba)
        print(f"Prueba local exitosa.")
        print(f"Vector CIELAB dominante calculado: L*={color_lab[0]:.2f}, a*={color_lab[1]:.2f}, b*={color_lab[2]:.2f}")
    except Exception as e:
        print(f"Error inesperado en la prueba: {e}")