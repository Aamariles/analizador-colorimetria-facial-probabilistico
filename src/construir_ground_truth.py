"""Herramienta para construir el dataset_ground_truth (hito 10 del Acta de Constitución).

Procesa las imágenes de referencia (rostros cuya estación cromática ya se conoce) con la
misma medición que usa el orquestador (`medicion.medir_color_piel`: MediaPipe -> Gray
World -> CIELAB/K-Means) y guarda los resultados en archivos CSV, listos para calibrar
los centroides con `colorimetria.referencias.calibrar_desde_ground_truth` y para medir
la precisión.

Subcomandos (se ejecutan desde la raíz del repositorio):

    python src/construir_ground_truth.py preparar   # crea carpetas y sincroniza el registro
    python src/construir_ground_truth.py extraer    # calcula el L*a*b* de cada imagen
    python src/construir_ground_truth.py dividir    # asigna calibración o validación
    python src/construir_ground_truth.py resumen    # control de calidad antes de entregar

Las imágenes van en data/raw/imagenes/ (carpeta que git ignora) y se nombran
<estacion>_<numero>.<extension>, por ejemplo invierno_1.jpg; las extensiones válidas son
.jpg, .jpeg, .png y .webp. También se aceptan nombres como "invierno 1.jpg" y "otoño_1.jpg".

Todas las imágenes son fotos de internet (ninguna está generada con IA). El registro
(registro_imagenes.csv) anota, para cada una, las dos fuentes que respaldan su estación y
la iluminación de la foto; el subtipo de 12 estaciones y las notas son opcionales.
"""

import argparse
import csv
import hashlib
import io
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_IMAGENES = RAIZ / "data" / "raw" / "imagenes"
CARPETA_PREVIEWS = RAIZ / "data" / "raw" / "previews"
CARPETA_GT = RAIZ / "data" / "ground_truth"
ARCHIVO_REGISTRO = CARPETA_GT / "registro_imagenes.csv"
ARCHIVO_MUESTRAS = CARPETA_GT / "muestras_lab.csv"

ESTACIONES_VALIDAS = ("primavera", "verano", "otono", "invierno")
ILUMINACIONES = ("natural", "calida", "fria", "estudio", "desconocida")
MINIMO_POR_ESTACION = 25
MINIMO_VALIDACION = 5
MIN_LADO_PX = 400            # imágenes con un lado menor se marcan como baja resolución
MIN_PIXELES_PIEL = 3500      # una ROI de piel con menos píxeles se marca para revisar:
#                              con tan poca piel, el color medido es poco confiable

PATRON_NOMBRE = re.compile(r"^(primavera|verano|otono|otoño|invierno)[ _-]?(\d+)\.(jpe?g|png|webp)$",
                           re.IGNORECASE)
EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp")
ARCHIVOS_DEL_SISTEMA = {"desktop.ini", "thumbs.db", ".ds_store"}

COLUMNAS_REGISTRO = ["archivo", "estacion", "fuente_estacion_1", "fuente_estacion_2", "subtipo_12",
                     "iluminacion", "notas"]
# Columnas que toda fila del registro debe tener llenas, además de la iluminación
CAMPOS_OBLIGATORIOS = ("fuente_estacion_1", "fuente_estacion_2")
# Columnas de versiones anteriores del registro, que `preparar` quita si las encuentra
COLUMNAS_RETIRADAS = ("origen", "persona", "url_imagen", "herramienta_ia", "prompt_ia")
COLUMNAS_MUESTRAS = ["archivo", "estacion", "L", "a", "b", "pixeles_piel", "ancho", "alto",
                     "estado", "detalle", "subconjunto"]


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def sin_tildes(texto: str) -> str:
    """Pasa a minúsculas y quita tildes ('Cálida' -> 'calida', 'Otoño' -> 'otono')."""
    normal = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(c for c in normal if not unicodedata.combining(c))


def interpretar_nombre(nombre: str) -> Optional[Tuple[str, int]]:
    """Devuelve (estación, número), por ejemplo ('otono', 3) para 'Otoño 3.png'; None si el nombre no es válido."""
    coincidencia = PATRON_NOMBRE.match(unicodedata.normalize("NFC", nombre.strip()))
    if coincidencia is None:
        return None
    return sin_tildes(coincidencia.group(1)), int(coincidencia.group(2))


def orden_estacion(estacion: str) -> int:
    """Posición de la estación en ESTACIONES_VALIDAS, para ordenar; las desconocidas van al final."""
    return ESTACIONES_VALIDAS.index(estacion) if estacion in ESTACIONES_VALIDAS else len(ESTACIONES_VALIDAS)


def leer_csv(ruta: Path) -> Tuple[List[Dict[str, str]], str]:
    """Lee un CSV aceptando el formato de Excel en español (separador ';', BOM, cp1252).

    Returns:
        Tuple[List[Dict[str, str]], str]: Filas como diccionarios y el separador detectado.
    """
    datos = ruta.read_bytes()
    texto = ""
    for codificacion in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            texto = datos.decode(codificacion)
            break
        except UnicodeDecodeError:
            continue
    lineas = texto.splitlines()
    primera = lineas[0] if lineas else ""
    separador = ";" if primera.count(";") > primera.count(",") else ","
    lector = csv.DictReader(io.StringIO(texto), delimiter=separador, restkey="_sobrante")
    filas = []
    for fila in lector:
        filas.append({(clave or "").strip(): (valor.strip() if isinstance(valor, str) else "")
                      for clave, valor in fila.items() if clave != "_sobrante"})
    return filas, separador


def escribir_csv(ruta: Path, filas: List[Dict[str, str]], columnas: List[str],
                 separador: str = ",", para_excel: bool = False) -> None:
    """Escribe un CSV. Con `para_excel=True` agrega BOM para que Excel muestre bien las tildes."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    codificacion = "utf-8-sig" if para_excel else "utf-8"
    with open(ruta, "w", newline="", encoding=codificacion) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, delimiter=separador, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: fila.get(c, "") for c in columnas})


def listar_imagenes(avisar: bool = True) -> List[Tuple[Path, str, int]]:
    """Lista las imágenes válidas de la carpeta, ordenadas por estación y número.

    Omite los archivos del sistema, los de nombre o extensión no válidos y los que
    repiten el número de otra imagen de la misma estación.

    Args:
        avisar (bool): Si es True, muestra en consola por qué se omite cada archivo.

    Returns:
        List[Tuple[Path, str, int]]: Para cada imagen, su ruta, su estación y su número.
    """
    if not CARPETA_IMAGENES.exists():
        return []
    validas, vistas = [], {}
    for ruta in sorted(CARPETA_IMAGENES.iterdir()):
        if not ruta.is_file() or ruta.name.lower() in ARCHIVOS_DEL_SISTEMA or ruta.name.startswith("."):
            continue
        datos = interpretar_nombre(ruta.name)
        if datos is None:
            if avisar:
                motivo = ("extensión no soportada (usa .jpg, .png o .webp)" if ruta.suffix.lower() not in EXTENSIONES
                          else "nombre no válido (usa <estacion>_<numero>, por ejemplo invierno_1.jpg)")
                print(f"  [AVISO] {ruta.name}: {motivo}; se omite.")
            continue
        if datos in vistas:
            if avisar:
                print(f"  [AVISO] {ruta.name} repite el número de {vistas[datos]}; se omite.")
            continue
        vistas[datos] = ruta.name
        validas.append((ruta, datos[0], datos[1]))
    return sorted(validas, key=lambda x: (orden_estacion(x[1]), x[2]))


def cargar_registro() -> Tuple[List[Dict[str, str]], str]:
    """Lee el registro de imágenes; si todavía no existe, devuelve una lista vacía y ';' como separador."""
    if not ARCHIVO_REGISTRO.exists():
        return [], ";"
    return leer_csv(ARCHIVO_REGISTRO)


def problemas_registro(fila: Dict[str, str]) -> List[str]:
    """Campos faltantes o inválidos de una fila del registro."""
    faltan = []
    datos = interpretar_nombre(fila.get("archivo", ""))
    if datos is None:
        faltan.append("archivo (nombre no válido)")
    elif sin_tildes(fila.get("estacion", "")) != datos[0]:
        faltan.append(f"estacion (el nombre del archivo dice '{datos[0]}')")
    faltan += [campo for campo in CAMPOS_OBLIGATORIOS if not fila.get(campo, "").strip()]
    iluminacion = sin_tildes(fila.get("iluminacion", ""))
    if not iluminacion:
        faltan.append("iluminacion")
    elif iluminacion not in ILUMINACIONES:
        faltan.append(f"iluminacion ('{fila.get('iluminacion')}' no es una de: {', '.join(ILUMINACIONES)})")
    return faltan


def leer_imagen(ruta: Path) -> Optional[np.ndarray]:
    """Lee una imagen BGR aunque la ruta tenga tildes o eñes (cv2.imread falla con ellas en Windows).

    Returns:
        Optional[np.ndarray]: La imagen, o None si el archivo está vacío o no es una imagen válida.
    """
    import cv2
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def guardar_preview(ruta: Path, original: np.ndarray, corregida: np.ndarray, roi: np.ndarray) -> None:
    """Guarda, lado a lado, la imagen original, la corregida por Gray World y la piel extraída.

    Si la corrección de iluminación se descartó, el panel del medio es igual al original.
    """
    import cv2
    alto = 360

    def escalar(imagen: np.ndarray) -> np.ndarray:
        h, w = imagen.shape[:2]
        return cv2.resize(imagen, (max(1, round(w * alto / h)), alto))

    lado_a_lado = np.hstack([escalar(original), escalar(corregida), escalar(roi)])
    ok, buffer = cv2.imencode(".jpg", lado_a_lado)
    if ok:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        buffer.tofile(str(ruta))


def cargar_muestras(subconjunto: str = "calibracion",
                    ruta: Path = ARCHIVO_MUESTRAS) -> Dict[str, List[Tuple[float, float, float]]]:
    """Agrupa por estación los L*a*b* de un subconjunto, en el formato de `calibrar_desde_ground_truth`.

    Args:
        subconjunto (str): "calibracion" o "validacion".
        ruta (Path): CSV generado por los subcomandos `extraer` y `dividir`.

    Returns:
        Dict[str, List[Tuple[float, float, float]]]: Para cada estación, la lista de (L*, a*, b*).
    """
    filas, _ = leer_csv(ruta)
    muestras: Dict[str, List[Tuple[float, float, float]]] = {e: [] for e in ESTACIONES_VALIDAS}
    for fila in filas:
        if fila.get("estado") == "ok" and fila.get("subconjunto") == subconjunto:
            muestras[fila["estacion"]].append((float(fila["L"]), float(fila["a"]), float(fila["b"])))
    return muestras


# --------------------------------------------------------------------------- #
# Subcomandos
# --------------------------------------------------------------------------- #
def cmd_preparar(args: argparse.Namespace) -> int:
    """Crea las carpetas y agrega al registro una fila por cada imagen nueva."""
    CARPETA_IMAGENES.mkdir(parents=True, exist_ok=True)
    CARPETA_GT.mkdir(parents=True, exist_ok=True)
    filas, separador = cargar_registro()
    # Se conservan las columnas extra que se hayan agregado a mano, salvo las que ya no se usan
    anteriores = list(filas[0].keys()) if filas else []
    retiradas = [c for c in anteriores if c in COLUMNAS_RETIRADAS]
    columnas = COLUMNAS_REGISTRO + [c for c in anteriores
                                    if c not in COLUMNAS_REGISTRO and c not in COLUMNAS_RETIRADAS]
    existentes = {f.get("archivo", "") for f in filas}
    imagenes = listar_imagenes()
    nuevas = 0
    for ruta, estacion, _ in imagenes:
        if ruta.name not in existentes:
            filas.append({"archivo": ruta.name, "estacion": estacion})
            nuevas += 1

    def clave(fila: Dict[str, str]) -> Tuple[int, int]:
        datos = interpretar_nombre(fila.get("archivo", ""))
        return (orden_estacion(datos[0]), datos[1]) if datos else (len(ESTACIONES_VALIDAS) + 1, 0)

    filas.sort(key=clave)
    escribir_csv(ARCHIVO_REGISTRO, filas, columnas, separador=separador, para_excel=True)
    en_carpeta = {r.name for r, _, _ in imagenes}
    huerfanas = [f["archivo"] for f in filas if f.get("archivo") and f["archivo"] not in en_carpeta]
    print(f"Carpeta de imágenes: {CARPETA_IMAGENES}")
    print(f"Registro: {ARCHIVO_REGISTRO}")
    print(f"Imágenes válidas en la carpeta: {len(imagenes)} | filas nuevas en el registro: {nuevas}")
    if retiradas:
        print(f"  [AVISO] se quitaron del registro las columnas que ya no se usan: {', '.join(retiradas)}")
    if huerfanas:
        print(f"  [AVISO] {len(huerfanas)} fila(s) del registro no tienen imagen: {', '.join(huerfanas[:10])}")
    return 0


def cmd_extraer(args: argparse.Namespace) -> int:
    """Calcula el L*a*b* dominante de cada imagen con el pipeline del proyecto."""
    imagenes = listar_imagenes()
    if not imagenes:
        print(f"No hay imágenes válidas en {CARPETA_IMAGENES}. Corre primero 'preparar' y agrega imágenes.")
        return 1

    from medicion import medir_color_piel

    filas = []
    total = len(imagenes)
    for i, (ruta, estacion, _) in enumerate(imagenes, start=1):
        fila = {"archivo": ruta.name, "estacion": estacion}
        imagen = leer_imagen(ruta)
        if imagen is None:
            fila.update(estado="error", detalle="no se pudo leer la imagen (formato no soportado o archivo dañado)")
        else:
            alto, ancho = imagen.shape[:2]
            fila.update(ancho=str(ancho), alto=str(alto))
            try:
                medicion = medir_color_piel(imagen)
                corregida, roi, lab = medicion.imagen_corregida, medicion.roi_piel, medicion.color_lab
            except FileNotFoundError as error:
                print(f"ERROR: {error}")
                return 1
            except ValueError as error:
                estado = "sin_rostro" if "ningún rostro" in str(error) else "error"
                fila.update(estado=estado, detalle=str(error))
            except Exception as error:  # cualquier otro fallo de una imagen no debe detener el lote
                fila.update(estado="error", detalle=f"{type(error).__name__}: {error}")
            else:
                pixeles = int(np.count_nonzero(np.any(roi != 0, axis=-1)))
                avisos = []
                if min(ancho, alto) < MIN_LADO_PX:
                    avisos.append(f"baja resolución ({ancho}x{alto})")
                if pixeles < MIN_PIXELES_PIEL:
                    avisos.append(f"poca piel detectada ({pixeles} px)")
                if not medicion.balance_aplicado:
                    avisos.append("fondo de color: se omitió la corrección de iluminación")
                fila.update(L=f"{lab[0]:.2f}", a=f"{lab[1]:.2f}", b=f"{lab[2]:.2f}", pixeles_piel=str(pixeles),
                            estado="ok", detalle="; ".join(avisos))
                if args.previews:
                    guardar_preview(CARPETA_PREVIEWS / f"{ruta.stem}.jpg", imagen, corregida, roi)
        filas.append(fila)
        valores = f"L={fila['L']} a={fila['a']} b={fila['b']}" if fila["estado"] == "ok" else fila["detalle"]
        print(f"[{i:3d}/{total}] {ruta.name:28s} {fila['estado']:10s} {valores}")

    escribir_csv(ARCHIVO_MUESTRAS, filas, COLUMNAS_MUESTRAS)
    correctas = sum(1 for f in filas if f["estado"] == "ok")
    print(f"\nListo: {correctas}/{total} imágenes procesadas sin error -> {ARCHIVO_MUESTRAS}")
    if args.previews:
        print(f"Vistas previas en: {CARPETA_PREVIEWS}")
    print("Siguiente paso: python src/construir_ground_truth.py dividir")
    return 0


def cmd_dividir(args: argparse.Namespace) -> int:
    """Separa, por estación, las imágenes válidas en calibración y validación."""
    if not ARCHIVO_MUESTRAS.exists():
        print("No existe muestras_lab.csv. Corre primero: python src/construir_ground_truth.py extraer")
        return 1
    filas, _ = leer_csv(ARCHIVO_MUESTRAS)
    for fila in filas:
        fila["subconjunto"] = ""

    generador = random.Random(args.semilla)
    print(f"Proporción para validación: {args.proporcion:.0%} | semilla: {args.semilla}")
    for estacion in ESTACIONES_VALIDAS:
        validas = sorted((f for f in filas if f["estacion"] == estacion and f["estado"] == "ok"),
                         key=lambda f: f["archivo"])
        objetivo = int(len(validas) * args.proporcion + 0.5)
        elegidas = {f["archivo"] for f in generador.sample(validas, min(objetivo, len(validas)))}
        for fila in validas:
            fila["subconjunto"] = "validacion" if fila["archivo"] in elegidas else "calibracion"
        print(f"  {estacion:10s} válidas={len(validas):3d}  calibración={len(validas) - len(elegidas):3d}  "
              f"validación={len(elegidas):3d}")

    escribir_csv(ARCHIVO_MUESTRAS, filas, COLUMNAS_MUESTRAS)
    print(f"\nColumna 'subconjunto' actualizada en {ARCHIVO_MUESTRAS}")
    print("Siguiente paso: python src/construir_ground_truth.py resumen")
    return 0


def cmd_resumen(args: argparse.Namespace) -> int:
    """Muestra conteos, promedios L*a*b* y la lista de pendientes antes de entregar."""
    from colorimetria.referencias import ESTACIONES

    imagenes = listar_imagenes(avisar=False)
    registro, _ = cargar_registro()
    muestras, _ = leer_csv(ARCHIVO_MUESTRAS) if ARCHIVO_MUESTRAS.exists() else ([], ",")
    pendientes: List[str] = []

    print("\nIMÁGENES POR ESTACIÓN")
    print(f"  {'estación':10s} {'carpeta':>7s} {'ok':>4s} {'fallas':>6s} {'calib.':>6s} {'valid.':>6s}")
    for estacion in ESTACIONES_VALIDAS:
        en_carpeta = sum(1 for _, e, _ in imagenes if e == estacion)
        filas = [f for f in muestras if f.get("estacion") == estacion]
        ok = [f for f in filas if f.get("estado") == "ok"]
        conteo = {
            "fallas": len(filas) - len(ok),
            "calibracion": sum(1 for f in ok if f.get("subconjunto") == "calibracion"),
            "validacion": sum(1 for f in ok if f.get("subconjunto") == "validacion"),
        }
        print(f"  {estacion:10s} {en_carpeta:7d} {len(ok):4d} {conteo['fallas']:6d} "
              f"{conteo['calibracion']:6d} {conteo['validacion']:6d}")
        if len(ok) < MINIMO_POR_ESTACION:
            pendientes.append(f"{estacion}: {len(ok)} imagen(es) válida(s) de {MINIMO_POR_ESTACION} requeridas")
        if muestras and conteo["validacion"] < MINIMO_VALIDACION:
            pendientes.append(f"{estacion}: {conteo['validacion']} imagen(es) en validación "
                              f"(mínimo {MINIMO_VALIDACION}; corre 'dividir' o agrega más imágenes)")

    print("\nPROMEDIO L*a*b* POR ESTACIÓN (imágenes ok) frente al centroide actual de referencias.py")
    for estacion in ESTACIONES_VALIDAS:
        valores = np.array([[float(f["L"]), float(f["a"]), float(f["b"])] for f in muestras
                            if f.get("estacion") == estacion and f.get("estado") == "ok"])
        referencia = ESTACIONES[estacion].lab
        if len(valores):
            m = valores.mean(axis=0)
            print(f"  {estacion:10s} L={m[0]:6.2f} a={m[1]:6.2f} b={m[2]:6.2f}   "
                  f"(referencia: L={referencia[0]:.0f} a={referencia[1]:.0f} b={referencia[2]:.0f})")
        else:
            print(f"  {estacion:10s} sin datos")

    # ---- registro -------------------------------------------------------- #
    nombres_carpeta = {r.name for r, _, _ in imagenes}
    nombres_registro = {f.get("archivo", "") for f in registro}
    sin_registro = sorted(nombres_carpeta - nombres_registro)
    sin_imagen = sorted(n for n in nombres_registro - nombres_carpeta if n)
    if sin_registro:
        pendientes.append(f"{len(sin_registro)} imagen(es) sin fila en el registro (corre 'preparar'): "
                          f"{', '.join(sin_registro[:8])}")
    if sin_imagen:
        pendientes.append(f"{len(sin_imagen)} fila(s) del registro sin imagen: {', '.join(sin_imagen[:8])}")
    sobrantes = [c for c in (registro[0].keys() if registro else []) if c in COLUMNAS_RETIRADAS]
    if sobrantes:
        pendientes.append(f"el registro todavía tiene columnas que ya no se usan ({', '.join(sobrantes)}); "
                          "corre 'preparar' para quitarlas")
    for fila in registro:
        faltan = problemas_registro(fila)
        if faltan and fila.get("archivo") in nombres_carpeta:
            pendientes.append(f"registro {fila.get('archivo')}: falta o está mal {', '.join(faltan)}")

    huellas: Dict[str, List[str]] = {}
    for ruta, _, _ in imagenes:
        huellas.setdefault(hashlib.md5(ruta.read_bytes()).hexdigest(), []).append(ruta.name)
    for archivos in huellas.values():
        if len(archivos) > 1:
            pendientes.append(f"imágenes idénticas: {', '.join(archivos)}")

    # ---- muestras ---------------------------------------------------------- #
    if not muestras:
        pendientes.append("aún no existe muestras_lab.csv (corre 'extraer')")
    else:
        procesadas = {f.get("archivo", "") for f in muestras}
        if procesadas != nombres_carpeta:
            pendientes.append("muestras_lab.csv no coincide con la carpeta de imágenes (corre 'extraer' y "
                              "luego 'dividir' de nuevo)")
        for fila in muestras:
            if fila.get("estado") != "ok":
                pendientes.append(f"{fila.get('archivo')}: {fila.get('estado')} ({fila.get('detalle')}); "
                                  "reemplázala o retírala")
            elif fila.get("detalle"):
                pendientes.append(f"{fila.get('archivo')}: revisar a ojo -> {fila.get('detalle')}")

    print("\nPENDIENTES")
    if pendientes:
        for linea in pendientes:
            print(f"  - {linea}")
        print(f"\n{len(pendientes)} pendiente(s). Corrígelos y vuelve a correr el resumen.")
    else:
        print("  Ninguno. El dataset_ground_truth está listo para entregar.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Interpreta el subcomando de la línea de comandos, lo ejecuta y devuelve el código de salida."""
    parser = argparse.ArgumentParser(description="Construcción del dataset_ground_truth (hito 10).")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("preparar", help="Crea las carpetas y sincroniza el registro con las imágenes.")
    p_extraer = sub.add_parser("extraer", help="Calcula el L*a*b* de cada imagen con el pipeline.")
    p_extraer.add_argument("--previews", action="store_true",
                           help="Guarda en data/raw/previews la original, la corregida y la piel extraída.")
    p_dividir = sub.add_parser("dividir", help="Asigna cada imagen a calibración o validación.")
    p_dividir.add_argument("--proporcion", type=float, default=0.30,
                           help="Fracción de cada estación que va a validación (por defecto 0.30).")
    p_dividir.add_argument("--semilla", type=int, default=42, help="Semilla para que la división sea repetible.")
    sub.add_parser("resumen", help="Control de calidad antes de entregar.")
    args = parser.parse_args(argv)
    comandos = {"preparar": cmd_preparar, "extraer": cmd_extraer, "dividir": cmd_dividir, "resumen": cmd_resumen}
    return comandos[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
