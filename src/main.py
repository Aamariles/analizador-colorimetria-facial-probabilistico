"""Orquestador del pipeline de Colorimetría Facial Probabilística.

Integra los módulos del sistema (preprocesamiento, segmentación facial, transformación
de color y clasificación bayesiana) para analizar una fotografía y emitir un
diagnóstico estacional probabilístico.

Uso, desde la raíz del repositorio:

    python src/main.py ruta/a/la_foto.jpg
    python src/main.py ruta/a/la_foto.jpg --vista

Con --vista, además del diagnóstico, se abre una ventana con la vista de control (la
imagen original, la corregida por Gray World y la piel analizada) y se guarda en
data/raw/previews/.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Módulos del proyecto
from colorimetria.clasificador import clasificar_estacion
from colorimetria.referencias import ESTACIONES
from medicion import MedicionPiel, componer_vista, medir_color_piel

CARPETA_VISTAS = Path(__file__).resolve().parent.parent / "data" / "raw" / "previews"
ALTO_VISTA = 480             # alto de cada panel de la vista de control, en píxeles
ANCHO_MAXIMO_VISTA = 1280    # para que la ventana quepa en la pantalla de un portátil


def mostrar_vista(ruta_imagen: str, imagen: np.ndarray, medicion: MedicionPiel) -> None:
    """Guarda la vista de control en data/raw/previews y la muestra en una ventana.

    La vista pone lado a lado la imagen original, la corregida por Gray World y la piel
    analizada (ver `medicion.componer_vista`). La ventana se cierra al presionar
    cualquier tecla o con su botón de cerrar. Si no se puede abrir una ventana (por
    ejemplo, en un equipo sin pantalla), la vista solo se guarda.

    Args:
        ruta_imagen (str): Ruta de la imagen analizada; su nombre se usa para el archivo de la vista.
        imagen (np.ndarray): Imagen BGR original.
        medicion (MedicionPiel): Resultado de `medir_color_piel` para esa imagen.
    """
    vista = componer_vista(imagen, medicion, alto=ALTO_VISTA, ancho_maximo=ANCHO_MAXIMO_VISTA)
    ruta_vista = CARPETA_VISTAS / f"{Path(ruta_imagen).stem}.jpg"
    ok, buffer = cv2.imencode(".jpg", vista)
    if ok:
        ruta_vista.parent.mkdir(parents=True, exist_ok=True)
        buffer.tofile(str(ruta_vista))  # a diferencia de cv2.imwrite, funciona con tildes en la ruta
        print(f"Vista de control guardada en: {ruta_vista}")

    titulo = f"Vista de control - {Path(ruta_imagen).name}"
    try:
        cv2.imshow(titulo, vista)
        print("Presiona cualquier tecla en la ventana, o ciérrala, para terminar.")
        while cv2.waitKey(100) == -1:
            if cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
    except cv2.error:
        print("No se pudo abrir una ventana en este equipo; abre la imagen guardada.")


def procesar_imagen(ruta_imagen: str, vista: bool = False) -> None:
    """Analiza una imagen y muestra en consola la estación más probable y sus probabilidades.

    La medición del color de piel está en `medicion.medir_color_piel`, la misma
    función que usa la construcción del dataset_ground_truth: MediaPipe detecta el
    rostro, Gray World estima la luz con los píxeles más neutros de la escena y
    corrige la imagen, y K-Means en CIELAB extrae el color dominante de la piel.
    Luego el clasificador bayesiano compara ese color con los centroides estacionales.

    Si la ruta no existe, la imagen no se puede leer o el análisis falla, muestra el
    error y termina el programa con código de salida 1.

    Args:
        ruta_imagen (str): Ruta en el disco de la imagen a analizar.
        vista (bool): Si es True, al final muestra y guarda la vista de control
            (ver `mostrar_vista`).
    """
    # Validación inicial de la ruta
    if not os.path.exists(ruta_imagen):
        print(f"Error: no se encontró el archivo: {ruta_imagen}")
        sys.exit(1)

    print(f"\nAnalizando la imagen: {os.path.basename(ruta_imagen)}")

    # Cargar la imagen en formato BGR
    imagen_original = cv2.imread(ruta_imagen)
    if imagen_original is None:
        print("Error: el archivo no es una imagen válida o está dañado.")
        sys.exit(1)

    try:
        # Paso 1: medición del color de piel (ver medicion.medir_color_piel)
        print("[1/2] Midiendo el color de la piel (MediaPipe, Gray World y K-Means en CIELAB)...")
        medicion = medir_color_piel(imagen_original)
        color_lab = medicion.color_lab
        print(f"    -> Color dominante de la piel: L*={color_lab[0]:.1f}, a*={color_lab[1]:.1f}, b*={color_lab[2]:.1f}")
        if not medicion.balance_aplicado:
            print("    -> Aviso: el fondo tiene un color intenso; se omitió la corrección de iluminación.")

        # Paso 2: clasificación bayesiana
        print("[2/2] Clasificando la estación (distancia CIEDE2000)...")
        resultados = clasificar_estacion(color_lab)

        # Presentación del resultado
        estacion_ganadora = next(iter(resultados))

        print("\n" + "=" * 50)
        print("RESULTADO DEL ANÁLISIS COLORIMÉTRICO")
        print("=" * 50)
        print(f" DIAGNÓSTICO PRINCIPAL: {ESTACIONES[estacion_ganadora].nombre.upper()}")
        print("-" * 50)
        print(" Probabilidades posteriores:")
        for estacion, prob in resultados.items():
            # Nombre ya acentuado correctamente, tomado de referencias.py
            # (evita mantener aquí una segunda copia del nombre mostrado).
            nombre_mostrado = ESTACIONES[estacion].nombre
            print(f"  • {nombre_mostrado:10s}: {prob * 100:>5.1f}%")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\nSe detuvo el análisis por un error en el pipeline: {e}")
        sys.exit(1)

    # La vista va al final para que el diagnóstico ya esté en la consola mientras la ventana está abierta
    if vista:
        mostrar_vista(ruta_imagen, imagen_original, medicion)


if __name__ == "__main__":
    # Configuración de la interfaz de línea de comandos (CLI)
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de colorimetría facial probabilística sobre una imagen."
    )
    parser.add_argument(
        "imagen",
        type=str,
        help="Ruta relativa o absoluta de la imagen que se va a analizar (por ejemplo, .jpg o .png).",
    )
    parser.add_argument(
        "--vista",
        action="store_true",
        help="Muestra en una ventana, y guarda en data/raw/previews, la imagen original, la corregida "
             "por Gray World y la piel analizada.",
    )

    args = parser.parse_args()
    procesar_imagen(args.imagen, vista=args.vista)
