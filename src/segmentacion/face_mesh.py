import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# Índices de los landmarks de MediaPipe para formar polígonos
CONTORNO_ROSTRO = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
OJO_IZQUIERDO = [33, 160, 158, 133, 153, 144]
OJO_DERECHO = [362, 385, 387, 263, 373, 380]
LABIOS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
CEJA_IZQUIERDA = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
CEJA_DERECHA = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Ruta absoluta dinámica para encontrar el modelo .task
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'reference', 'face_landmarker.task')

def extraer_piel_limpia(imagen: np.ndarray) -> np.ndarray:
    """
    Detecta el rostro usando la API moderna de MediaPipe Tasks (AI Edge) 
    y extrae únicamente la piel limpia (ROI).
    """
    if imagen is None or not isinstance(imagen, np.ndarray):
        raise ValueError("La imagen de entrada no es válida.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo Face Landmarker en: {MODEL_PATH}")

    alto, ancho = imagen.shape[:2]
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    # 1. Convertir la matriz de OpenCV al formato nativo de MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)

    # 2. Configurar las opciones del motor de inferencia
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    # 3. Crear el landmarker y procesar la imagen
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        resultado = landmarker.detect(mp_image)

        if not resultado.face_landmarks:
            raise ValueError("MediaPipe no detectó ningún rostro en la imagen.")

        rostro = resultado.face_landmarks[0]
        
        def obtener_puntos(indices):
            return np.array([
                [int(rostro[idx].x * ancho), int(rostro[idx].y * alto)]
                for idx in indices
            ], dtype=np.int32)

        # 4. Construir la máscara binaria y aplicar Álgebra de Boole
        mascara = np.zeros((alto, ancho), dtype=np.uint8)
        cv2.fillPoly(mascara, [obtener_puntos(CONTORNO_ROSTRO)], 255)
        
        ruidos = [
            obtener_puntos(OJO_IZQUIERDO),
            obtener_puntos(OJO_DERECHO),
            obtener_puntos(LABIOS),
            obtener_puntos(CEJA_IZQUIERDA),
            obtener_puntos(CEJA_DERECHA)
        ]
        cv2.fillPoly(mascara, ruidos, 0)
        
        piel_extraida = cv2.bitwise_and(imagen, imagen, mask=mascara)
        return piel_extraida

if __name__ == "__main__":
    print("Ejecutando prueba local del módulo Face Mesh (AI Edge)...")
    imagen_prueba = np.zeros((500, 500, 3), dtype=np.uint8)
    try:
        resultado = extraer_piel_limpia(imagen_prueba)
    except Exception as e:
        if "ningún rostro" in str(e):
            print(" Prueba local exitosa: el modelo .task cargó correctamente y el sistema manejó la imagen de ruido.")
        else:
            print(f" Error inesperado: {e}")