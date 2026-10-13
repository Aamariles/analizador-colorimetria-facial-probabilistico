"""Orquestador principal del pipeline de Colorimetría Facial Probabilística.

Este script integra todos los módulos del sistema (preprocesamiento,
segmentación geométrica, transformación espacial y clasificación bayesiana)
para procesar una imagen fotográfica desde cero y emitir un diagnóstico
estacional probabilístico.
"""

import argparse
import cv2
import os
import sys

# Importaciones de los módulos del core matemático y de visión
from colorimetria.clasificador import clasificar_estacion
from colorimetria.referencias import ESTACIONES
from medicion import medir_color_piel


def procesar_imagen(ruta_imagen: str) -> None:
    """Ejecuta el pipeline secuencial de análisis sobre una imagen dada.

    La medición del color de piel vive en `medicion.medir_color_piel`, la misma
    función que usa la construcción del dataset_ground_truth: MediaPipe detecta
    el rostro, Gray World estima la luz con la escena (sin la cabeza) y corrige
    la imagen, y K-Means en CIELAB extrae el color dominante de la piel. Luego
    el clasificador bayesiano compara ese color con los centroides estacionales.

    Args:
        ruta_imagen (str): Ruta en el disco hacia la imagen a analizar.
    """
    # Validación inicial de la ruta
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró el archivo en la ruta especificada: {ruta_imagen}")
        sys.exit(1)

    print(f"\nIniciando análisis de imagen: {os.path.basename(ruta_imagen)}")

    # Cargar la matriz de imagen en formato BGR
    imagen_original = cv2.imread(ruta_imagen)
    if imagen_original is None:
        print("Error: El archivo no es una imagen válida o está corrupto.")
        sys.exit(1)

    try:
        # Paso 1: Medición del color de piel (MediaPipe -> Gray World sobre la escena -> CIELAB/K-Means)
        print("[1/2] Midiendo el color de la piel (MediaPipe, Gray World sobre la escena y K-Means en CIELAB)...")
        medicion = medir_color_piel(imagen_original)
        color_lab = medicion.color_lab
        print(f"    -> Vector dominante estimado: L*={color_lab[0]:.1f}, a*={color_lab[1]:.1f}, b*={color_lab[2]:.1f}")
        if not medicion.balance_aplicado:
            print("    -> Aviso: el fondo tiene un color intenso; se omitió la corrección de iluminación.")

        # Paso 2: Inferencia Bayesiana
        print("[2/2] Ejecutando clasificador estacional (Métrica CIEDE2000)...")
        resultados = clasificar_estacion(color_lab)

        # Extracción y formateo del resultado final
        estacion_ganadora = next(iter(resultados))

        print("\n" + "=" * 50)
        print("RESULTADO DEL ANÁLISIS COLORIMÉTRICO")
        print("=" * 50)
        print(f" DIAGNÓSTICO PRINCIPAL : {ESTACIONES[estacion_ganadora].nombre.upper()}")
        print("-" * 50)
        print(" Probabilidades Posteriores:")
        for estacion, prob in resultados.items():
            # Nombre ya acentuado correctamente, tomado de referencias.py
            # (evita mantener aquí una segunda copia del nombre mostrado).
            nombre_mostrado = ESTACIONES[estacion].nombre
            print(f"  • {nombre_mostrado:10s}: {prob * 100:>5.1f}%")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\nSe detuvo el análisis por un error en el pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configuración de la interfaz de línea de comandos (CLI)
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de colorimetría facial probabilística sobre una imagen."
    )
    parser.add_argument(
        "imagen",
        type=str,
        help="Ruta relativa o absoluta hacia el archivo de imagen (.jpg, .png) a analizar.",
    )

    args = parser.parse_args()
    procesar_imagen(args.imagen)
