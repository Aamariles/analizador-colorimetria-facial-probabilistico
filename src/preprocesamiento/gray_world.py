import cv2
import numpy as np
def aplicar_gray_world(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica el algoritmo Gray World a una imagen para corregir el balance de blancos.
    
    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR (estándar de OpenCV).
        
    Returns:
        np.ndarray: Imagen con el balance de blancos corregido en formato BGR.
        
    Raises:
        ValueError: Si la imagen proporcionada está vacía o no es válida.
    """
    if imagen is None or not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("La imagen proporcionada no es válida o está vacía.")

    # OpenCV carga las imágenes en orden BGR (Azul, Verde, Rojo)
    b, g, r = cv2.split(imagen)

    # 1. Calcular los promedios de cada canal (B_avg, G_avg, R_avg)
    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)

    # Validación para evitar división por cero en imágenes completamente negras
    if b_avg == 0 and g_avg == 0 and r_avg == 0:
        return imagen

    # 2. Calcular el promedio global (Gray_global)
    gray_global = (b_avg + g_avg + r_avg) / 3.0

    # 3. Calcular los factores de escala y aplicarlos a los canales
    # Usamos np.clip para asegurar que los valores de los píxeles no superen 255
    b_new = np.clip(b * (gray_global / b_avg), 0, 255).astype(np.uint8)
    g_new = np.clip(g * (gray_global / g_avg), 0, 255).astype(np.uint8)
    r_new = np.clip(r * (gray_global / r_avg), 0, 255).astype(np.uint8)

    # Reconstruir la imagen fusionando los canales ajustados
    imagen_corregida = cv2.merge((b_new, g_new, r_new))

    return imagen_corregida

# Bloque de prueba local
if __name__ == "__main__":
    import os
    
    # Crear una imagen simulada (ruido aleatorio con sesgo rojo) para probar la función
    print("Ejecutando prueba local del módulo Gray World...")
    imagen_prueba = np.random.randint(0, 150, (100, 100, 3), dtype=np.uint8)
    imagen_prueba[:, :, 2] = 250  # Forzar saturación roja en el canal R (índice 2)
    
    try:
        resultado = aplicar_gray_world(imagen_prueba)
        print(f"Forma original: {imagen_prueba.shape} | Promedio Rojo original: {np.mean(imagen_prueba[:,:,2]):.2f}")
        print(f"Forma corregida: {resultado.shape} | Promedio Rojo corregido: {np.mean(resultado[:,:,2]):.2f}")
        print("✅ Prueba local exitosa.")
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")