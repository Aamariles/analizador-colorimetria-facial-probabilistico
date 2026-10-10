"""Módulo de preprocesamiento para la corrección de balance de blancos.

Implementa el algoritmo Gray World para neutralizar los sesgos de color
introducidos por variaciones en la iluminación de la imagen. El algoritmo 
se fundamenta en el principio de que, en una escena natural estándar, 
el promedio global de las intensidades de color tiende a un gris neutral.
"""

import cv2
import numpy as np

# Valor mínimo de tolerancia para evitar inestabilidad numérica en canales sin información.
_EPSILON = 1e-6


def aplicar_gray_world(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica el algoritmo Gray World a una imagen para corregir el balance de blancos.

    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR.
    Returns:
        np.ndarray: Imagen con el balance de blancos corregido en formato BGR.

    Raises:
       ValueError: Si la imagen proporcionada está vacía o no es un arreglo válido.
    """
    if imagen is None or not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("La imagen proporcionada no es válida o está vacía.")

    # Separar la imagen en sus tres canales (Azul, Verde, Rojo)
    b, g, r = cv2.split(imagen)

    # 1. Calcular los promedios de cada canal (B_avg, G_avg, R_avg)
    b_avg = float(np.mean(b))
    g_avg = float(np.mean(g))
    r_avg = float(np.mean(r))

    # Si los tres canales están agotados (imagen completamente negra) no hay
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

    # 3. Calcular los factores de escala y aplicarlos a los canales
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
        print(f"Forma original: {imagen_prueba.shape} | Promedio Rojo original: {np.mean(imagen_prueba[:,:,2]):.2f}")
        print(f"Forma corregida: {resultado.shape} | Promedio Rojo corregido: {np.mean(resultado[:,:,2]):.2f}")
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
