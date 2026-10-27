"""Orquestador del pipeline de Colorimetría Facial Probabilística por línea de comandos.

Integra los módulos del sistema (preprocesamiento, segmentación facial, transformación
de color, clasificación bayesiana y recomendación de paletas HEX) para analizar una
fotografía y emitir un diagnóstico estacional probabilístico. Es el mismo análisis que
hace la API (`analisis.analizar`), sin la interfaz web.

Uso, desde la raíz del repositorio:

    python src/main.py ruta/a/la_foto.jpg
    python src/main.py ruta/a/la_foto.jpg --vista
    python src/main.py ruta/a/la_foto.jpg --json

Con --vista, además del diagnóstico, se abre una ventana con la vista de control (la
imagen original, la corregida por Gray World, la piel analizada y las zonas medidas) y se
guarda en data/raw/previews/. Con --json se imprime el resultado completo en JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Módulos del proyecto
from analisis import analizar
from medicion import MedicionRostro, componer_vista

CARPETA_VISTAS = Path(__file__).resolve().parent.parent / "data" / "raw" / "previews"
ALTO_VISTA = 420             # alto de cada panel de la vista de control, en píxeles
ANCHO_MAXIMO_VISTA = 1600    # para que la ventana quepa en la pantalla de un portátil


def mostrar_vista(ruta_imagen: str, imagen: np.ndarray, medicion: MedicionRostro) -> None:
    """Guarda la vista de control en data/raw/previews y la muestra en una ventana.

    La vista pone lado a lado la imagen original, la corregida por Gray World, la piel
    analizada y las zonas medidas (ver `medicion.componer_vista`). La ventana se cierra al
    presionar cualquier tecla o con su botón de cerrar. Si no se puede abrir una ventana
    (por ejemplo, en un equipo sin pantalla), la vista solo se guarda.

    Args:
        ruta_imagen (str): Ruta de la imagen analizada; su nombre se usa para el archivo de la vista.
        imagen (np.ndarray): Imagen BGR original.
        medicion (MedicionRostro): Medición de esa imagen.
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


def imprimir_resultado(resultado: dict) -> None:
    """Muestra en consola el diagnóstico: estación, subtipo, probabilidades, perfil y paleta."""
    linea = "=" * 60
    print("\n" + linea)
    print("RESULTADO DEL ANÁLISIS COLORIMÉTRICO")
    print(linea)
    print(f" ESTACIÓN MÁS PROBABLE: {resultado['estacion']['nombre'].upper()}"
          f"  ({resultado['estacion']['probabilidad'] * 100:.1f} %)")
    print(f" SUBTIPO (12 estaciones): {resultado['subtipo']['nombre']} - {resultado['subtipo']['lema']}")
    print("-" * 60)
    print(" Probabilidades posteriores:")
    for p in resultado["probabilidades"]:
        print(f"  • {p['nombre']:10s}: {p['probabilidad'] * 100:>5.1f} %")
    print(" Perfil (percentil frente al dataset):")
    for eje in resultado["ejes"].values():
        print(f"  • {eje['nombre']:12s}: {eje['valor'] * 100:>5.1f}  ({eje['bajo']} -> {eje['alto']})")
    print(" Colores medidos:")
    for zona in resultado["zonas"].values():
        extra = " (estimado con las cejas)" if zona["fuente"] == "cejas" else ""
        print(f"  • {zona['nombre']:12s}: {zona['hex']}  L*={zona['lab'][0]:.1f} a*={zona['lab'][1]:.1f} "
              f"b*={zona['lab'][2]:.1f}{extra}")
    paleta = resultado["paleta"]
    print(" Paleta favorecedora: " + " ".join(c["hex"] for c in paleta["favorecedores"]))
    print(" Colores a evitar:    " + " ".join(c["hex"] for c in paleta["evitar"]))
    for aviso in resultado["avisos"]:
        print(f" Aviso: {aviso}")
    modelo = resultado.get("modelo") or {}
    if modelo.get("exactitud") is not None:
        print(f" Modelo: acierta la estación en {modelo['exactitud'] * 100:.0f} % de los casos de validación "
              f"(IC 95 %: {modelo['ic95'][0] * 100:.0f}-{modelo['ic95'][1] * 100:.0f} %, etiquetas "
              f"'{modelo['etiquetas']}', n = {modelo['n']}).")
    print(f" Tiempo de análisis: {resultado['tiempo_ms']} ms")
    print(linea + "\n")


def procesar_imagen(ruta_imagen: str, vista: bool = False, como_json: bool = False) -> None:
    """Analiza una imagen y muestra el resultado en consola.

    Si la ruta no existe, la imagen no se puede leer o el análisis falla, muestra el
    error y termina el programa con código de salida 1.

    Args:
        ruta_imagen (str): Ruta en el disco de la imagen a analizar.
        vista (bool): Si es True, al final muestra y guarda la vista de control.
        como_json (bool): Si es True, imprime el resultado completo en JSON.
    """
    if not os.path.exists(ruta_imagen):
        print(f"Error: no se encontró el archivo: {ruta_imagen}")
        sys.exit(1)
    # np.fromfile + imdecode funciona también con tildes en la ruta (cv2.imread falla en Windows)
    imagen_original = cv2.imdecode(np.fromfile(ruta_imagen, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagen_original is None:
        print("Error: el archivo no es una imagen válida o está dañado.")
        sys.exit(1)

    if not como_json:
        print(f"\nAnalizando la imagen: {os.path.basename(ruta_imagen)}")
    try:
        resultado, medicion = analizar(imagen_original)
    except Exception as e:
        print(f"\nSe detuvo el análisis por un error en el pipeline: {e}")
        sys.exit(1)

    if como_json:
        print(json.dumps(resultado, ensure_ascii=False, indent=1))
    else:
        imprimir_resultado(resultado)
    # La vista va al final para que el diagnóstico ya esté en la consola mientras la ventana está abierta
    if vista:
        mostrar_vista(ruta_imagen, imagen_original, medicion)


if __name__ == "__main__":
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
             "por Gray World, la piel analizada y las zonas medidas.",
    )
    parser.add_argument("--json", action="store_true", help="Imprime el resultado completo en JSON.")

    args = parser.parse_args()
    procesar_imagen(args.imagen, vista=args.vista, como_json=args.json)
