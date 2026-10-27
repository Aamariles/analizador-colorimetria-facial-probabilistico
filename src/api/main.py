"""API REST local de Prisma (FastAPI): análisis, exploración, paletas, enlace PC-celular e informe.

Corre en el PC principal y la abren, por la red local, el navegador del PC y el del celular
(arquitectura handoff del Acta). Sirve también la interfaz web (carpeta `frontend/`), de modo
que ambos dispositivos entran por la misma dirección: https://<IP del PC>:8443.

Para levantarla con HTTPS (necesario para que el celular pueda abrir la cámara), usar
`python src/servidor.py`. La documentación interactiva queda en /docs.

Rutas:
    GET  /api/salud                       estado del servidor, del modelo y direcciones de red
    GET  /api/paletas                     los 12 subtipos con sus colores (portada)
    POST /api/analizar                    foto -> estación, probabilidades, subtipo, zonas y paleta
    POST /api/explorar                    rasgos + ejes deseados -> probabilidades de exploración
    POST /api/enlace                      crea un código de enlace PC-celular
    GET  /api/enlace/{codigo}             ¿el código sigue vigente?
    PUT  /api/enlace/{codigo}/foto        el celular deja la foto
    GET  /api/enlace/{codigo}/foto        el PC la recoge (y se borra del servidor)
    DELETE /api/enlace/{codigo}           cierra el enlace
    POST /api/informe                     PDF del resultado: descargarlo o enviarlo por correo

Política de cero almacenamiento: las fotos solo viven en memoria durante la petición (o
mientras esperan a que el PC las recoja, como máximo dos minutos) y nunca se escriben en disco.
"""

import json
import smtplib
import threading
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from analisis import analizar, describir_modelo, explorar, modelo_por_defecto
from api.enlace import VIGENCIA_CODIGO, AlmacenEnlaces
from api.red import ips_locales
from colorimetria.modelo import NOMBRES_EJES
from informe.correo import CorreoNoConfigurado, configuracion_correo, enviar_informe, es_correo_valido
from informe.pdf import generar_pdf, texto_resumen
from medicion import ZONAS
from recomendador.paletas import vista_previa
from segmentacion.segmentador import modelo_disponible

VERSION = "1.0.0"
TAMANO_MAXIMO_FOTO = 12 * 1024 * 1024  # bytes
RAIZ = Path(__file__).resolve().parents[2]
CARPETA_FRONTEND = RAIZ / "frontend"

app = FastAPI(
    title="Prisma · Analizador de Colorimetría Facial Probabilístico",
    version=VERSION,
    description="API local del proyecto (Programación 2 y Probabilidad, Universidad El Bosque). "
                "Las fotos se procesan en memoria y no se guardan.",
)
enlaces = AlmacenEnlaces()
# MediaPipe usa un solo motor compartido: los análisis se atienden de a uno.
_candado_analisis = threading.Lock()


@app.middleware("http")
async def sin_cache(request: Request, siguiente):
    """Las respuestas de la API (fotos y resultados) no deben quedar en la caché del navegador."""
    respuesta = await siguiente(request)
    if request.url.path.startswith("/api/"):
        respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
async def _leer_subida(archivo: UploadFile) -> bytes:
    """Lee un archivo subido con un tope de tamaño (para no llenar la memoria)."""
    datos = await archivo.read(TAMANO_MAXIMO_FOTO + 1)
    if not datos:
        raise HTTPException(400, "No llegó ninguna foto.")
    if len(datos) > TAMANO_MAXIMO_FOTO:
        raise HTTPException(413, "La foto pesa más de 12 MB.")
    return datos


def _decodificar(datos: bytes) -> np.ndarray:
    """Convierte los bytes de una foto (JPG, PNG o WebP) en una imagen BGR."""
    imagen = cv2.imdecode(np.frombuffer(datos, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        raise HTTPException(400, "El archivo no es una imagen válida (usa JPG o PNG).")
    return imagen


def _correcciones(texto: Optional[str]) -> Dict[str, Tuple[float, float]]:
    """Interpreta las zonas corregidas por el usuario: {"cabello": [x, y], ...} con x, y entre 0 y 1."""
    if not texto:
        return {}
    try:
        datos = json.loads(texto)
        correcciones = {}
        for zona, punto in datos.items():
            x, y = float(punto[0]), float(punto[1])
            if zona not in ZONAS or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(zona)
            correcciones[zona] = (x, y)
        return correcciones
    except (ValueError, TypeError, IndexError, AttributeError, json.JSONDecodeError):
        raise HTTPException(400, "Las correcciones de zonas no son válidas.")


def _nombre_archivo(texto: str) -> str:
    """Texto sin tildes ni espacios, para nombres de archivo ("Otoño Profundo" -> "otono-profundo")."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return "-".join(sin_tildes.lower().split())


# --------------------------------------------------------------------------- #
# Estado y paletas
# --------------------------------------------------------------------------- #
@app.get("/api/salud")
def salud(request: Request) -> Dict:
    """Estado del servidor: versión, modelo calibrado, segmentador, correo y direcciones de red.

    `enlaces_red` son las direcciones con que el celular puede abrir esta misma página (la
    interfaz las usa para el código QR cuando en el PC se abrió como "localhost").
    """
    try:
        modelo = describir_modelo(modelo_por_defecto())
    except FileNotFoundError:
        modelo = None
    puerto = f":{request.url.port}" if request.url.port else ""
    enlaces = [f"{request.url.scheme}://{ip}{puerto}/" for ip in ips_locales()]
    return {"estado": "ok", "version": VERSION, "modelo": modelo, "segmentador": modelo_disponible(),
            "correo": configuracion_correo() is not None, "enlaces_red": enlaces}


@app.get("/api/paletas")
def paletas() -> Dict:
    """Los 12 subtipos con su lema y sus colores (vista previa de la portada)."""
    return {"subtipos": vista_previa()}


# --------------------------------------------------------------------------- #
# Análisis y exploración
# --------------------------------------------------------------------------- #
def _analizar_en_serie(imagen: np.ndarray, puntos: Dict[str, Tuple[float, float]]) -> Dict:
    """Analiza con el candado tomado (un análisis a la vez)."""
    with _candado_analisis:
        resultado, _ = analizar(imagen, puntos)
    return resultado


@app.post("/api/analizar")
async def api_analizar(foto: UploadFile = File(...), correcciones: Optional[str] = Form(None)) -> Dict:
    """Analiza una foto de rostro y devuelve el resultado completo (ver `analisis.analizar`).

    Las zonas del rostro se detectan solas. `correcciones` es opcional y la interfaz no lo
    usa: fija el punto de alguna zona ({"cabello": [x, y]}, con x e y entre 0 y 1), por
    ejemplo para estudiar un caso en que la detección automática falla.
    """
    imagen = _decodificar(await _leer_subida(foto))
    puntos = _correcciones(correcciones)
    try:
        # El cálculo corre en un hilo aparte para no frenar las demás peticiones (por
        # ejemplo, el PC que pregunta si ya llegó la foto del celular).
        resultado = await run_in_threadpool(_analizar_en_serie, imagen, puntos)
    except ValueError as error:
        if "rostro" in str(error):
            raise HTTPException(422, "No se detectó un rostro. Encuadra la cara de frente, dentro del óvalo y "
                                     "con buena luz.")
        raise HTTPException(422, f"No se pudo analizar la foto: {error}")
    except FileNotFoundError as error:
        raise HTTPException(500, f"Falta un archivo del modelo en el servidor: {error}")
    finally:
        del imagen  # cero almacenamiento: la foto no sobrevive a la petición
    return resultado


class PeticionExplorar(BaseModel):
    """Rasgos de un análisis y el percentil (0-1) deseado para cada eje que se mueve."""

    rasgos: Dict[str, Optional[float]]
    ejes: Dict[str, float] = Field(default_factory=dict)


@app.post("/api/explorar")
def api_explorar(peticion: PeticionExplorar) -> Dict:
    """Probabilidades, subtipo y paleta si los ejes fueran otros (para estudiar el modelo; no
    cambia el resultado oficial y la interfaz no lo usa)."""
    for eje, valor in peticion.ejes.items():
        if eje not in NOMBRES_EJES or not 0.0 <= valor <= 1.0:
            raise HTTPException(400, f"Eje no válido: {eje}")
    return explorar(peticion.rasgos, peticion.ejes)


# --------------------------------------------------------------------------- #
# Enlace PC-celular
# --------------------------------------------------------------------------- #
@app.post("/api/enlace")
def crear_enlace() -> Dict:
    """Crea un código para que el celular le envíe la foto a este PC."""
    try:
        return {"codigo": enlaces.crear(), "vigencia_segundos": VIGENCIA_CODIGO}
    except RuntimeError as error:
        raise HTTPException(503, str(error))


@app.get("/api/enlace/{codigo}")
def estado_enlace(codigo: str) -> Dict:
    """Indica si un código sigue vigente (el celular lo revisa antes de tomar la foto)."""
    return {"vigente": enlaces.existe(codigo)}


@app.put("/api/enlace/{codigo}/foto")
async def dejar_foto(codigo: str, foto: UploadFile = File(...)) -> Dict:
    """El celular entrega la foto; queda en memoria hasta que el PC la recoja."""
    datos = await _leer_subida(foto)
    _decodificar(datos)  # rechaza lo que no sea una imagen
    if not enlaces.dejar_foto(codigo, datos):
        raise HTTPException(404, "El código no existe o venció. Pide uno nuevo en la pantalla del PC.")
    return {"recibida": True}


@app.get("/api/enlace/{codigo}/foto")
def recoger_foto(codigo: str) -> Response:
    """El PC recoge la foto (204 si todavía no llegó); al entregarla se borra del servidor."""
    try:
        foto = enlaces.recoger_foto(codigo)
    except KeyError:
        raise HTTPException(404, "El código no existe o venció.")
    if foto is None:
        return Response(status_code=204)
    tipo = "image/png" if foto[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return Response(content=foto, media_type=tipo)


@app.delete("/api/enlace/{codigo}")
def cerrar_enlace(codigo: str) -> Dict:
    """Cierra el enlace y descarta cualquier foto pendiente."""
    enlaces.cerrar(codigo)
    return {"cerrado": True}


# --------------------------------------------------------------------------- #
# Informe PDF y correo
# --------------------------------------------------------------------------- #
@app.post("/api/informe")
async def informe(resultado: str = Form(...), accion: str = Form("descargar"), correo: str = Form(""),
                  acepta_envio: bool = Form(False), incluir_foto: bool = Form(False),
                  foto: Optional[UploadFile] = File(None)) -> Response:
    """Genera el informe en PDF y lo devuelve para descargar o lo envía por correo.

    Args:
        resultado: El JSON que devolvió /api/analizar.
        accion: "descargar" o "enviar".
        correo: Dirección de destino (solo para "enviar").
        acepta_envio: Consentimiento para enviar los resultados al correo indicado.
        incluir_foto: Consentimiento para incluir la foto en el informe.
        foto: La foto, que solo se usa si `incluir_foto` es verdadero.
    """
    try:
        datos = json.loads(resultado)
        nombre = _nombre_archivo(datos["subtipo"]["nombre"])
    except (json.JSONDecodeError, KeyError, TypeError):
        raise HTTPException(400, "El resultado del análisis no es válido.")
    foto_jpeg = None
    if incluir_foto:
        if foto is None:
            raise HTTPException(400, "Marcaste incluir la foto, pero no llegó ninguna foto.")
        imagen = _decodificar(await _leer_subida(foto))
        ok, codificada = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 88])
        foto_jpeg = codificada.tobytes() if ok else None
        del imagen
    try:
        pdf = await run_in_threadpool(generar_pdf, datos, foto_jpeg)
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "El resultado del análisis está incompleto.")
    archivo = f"prisma-informe-{nombre}.pdf"

    if accion == "descargar":
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{archivo}"'})
    if accion != "enviar":
        raise HTTPException(400, "Acción no válida (usa 'descargar' o 'enviar').")
    if not acepta_envio:
        raise HTTPException(400, "Para enviar el informe tienes que autorizar el envío a tu correo.")
    if not es_correo_valido(correo):
        raise HTTPException(400, "Escribe un correo válido.")
    try:
        await run_in_threadpool(enviar_informe, correo, f"Tu colorimetría Prisma: {datos['subtipo']['nombre']}",
                                texto_resumen(datos), pdf, archivo)
    except CorreoNoConfigurado as error:
        raise HTTPException(503, str(error))
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(502, "El servidor de correo rechazó la cuenta remitente (revisa la contraseña de "
                                 "aplicación en .env).")
    except (smtplib.SMTPException, OSError) as error:
        raise HTTPException(502, f"No se pudo enviar el correo: {error}")
    return Response(content=json.dumps({"enviado": True, "destinatario": correo.strip()}),
                    media_type="application/json")


# La interfaz web se sirve desde la raíz, después de las rutas de la API.
if CARPETA_FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(CARPETA_FRONTEND), html=True), name="interfaz")
