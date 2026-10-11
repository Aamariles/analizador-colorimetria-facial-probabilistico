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
from preprocesamiento.gray_world import aplicar_gray_world
from segmentacion.face_mesh import extraer_piel_limpia
from colorimetria.cielab_math import extraer_color_dominante_lab
from colorimetria.clasificador import clasificar_estacion
from colorimetria.referencias import ESTACIONES


def procesar_imagen(ruta_imagen: str) -> None:
    """Ejecuta el pipeline secuencial de análisis sobre una imagen dada.

    El orden de las etapas es deliberado: Gray World corrige el sesgo de
    iluminación sobre la imagen completa antes de detectar el rostro y
    aislar la piel, para que la extracción del color dominante en CIELAB
    y la clasificación bayesiana final operen ya sobre un color libre de
    la contaminación del iluminante (ver "Visión general del pipeline" en
    el Marco Teórico del proyecto).

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
        # Paso 1: Preprocesamiento Óptico
        print("[1/4] Aplicando corrección de iluminación (Gray World)...")
        imagen_gw = aplicar_gray_world(imagen_original)

        # Paso 2: Segmentación y Geometría
        print("[2/4] Extrayendo región de interés (MediaPipe AI Edge)...")
        roi_piel = extraer_piel_limpia(imagen_gw)

        # Paso 3: Transformación Matemática y Clustering
        print("[3/4] Calculando moda estadística en espacio CIELAB (K-Means)...")
        color_lab = extraer_color_dominante_lab(roi_piel)
        print(f"    ↳ Vector dominante estimado: L*={color_lab[0]:.1f}, a*={color_lab[1]:.1f}, b*={color_lab[2]:.1f}")

        # Paso 4: Inferencia Bayesiana
        print("[4/4] Ejecutando clasificador estacional (Métrica CIEDE2000)...")
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
