"""Análisis completo de una foto: las seis fases del pipeline en una sola llamada.

`analizar` encadena la medición del rostro (Gray World, MediaPipe y K-Means en CIELAB por
zona), el clasificador bayesiano (probabilidad de cada estación y subtipo de 12) y el
recomendador de paletas HEX, y devuelve un diccionario listo para convertir a JSON. Lo usan
la API (`api/main.py`), la línea de comandos (`main.py`) y el informe PDF, para que todos
muestren exactamente el mismo resultado.

La imagen solo vive en memoria durante la llamada (política de cero almacenamiento del
Acta): no se escribe en disco ni queda guardada en el resultado.
"""

import math
import time
from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from colorimetria.modelo import (ESTACIONES, NOMBRES_EJES, NOMBRES_ESTACIONES, RASGOS, ModeloEstacional,
                                 percentil_a_z, calcular_rasgos)
from medicion import NOMBRES_ZONAS, ZONAS, MedicionRostro, medir_rostro
from recomendador.paletas import recomendar


@lru_cache(maxsize=1)
def modelo_por_defecto() -> ModeloEstacional:
    """El modelo calibrado de data/reference (se lee una sola vez).

    Raises:
        FileNotFoundError: Si no existe; hay que correr `construir_ground_truth.py calibrar`.
    """
    return ModeloEstacional.cargar()


def _numero(valor: float) -> Optional[float]:
    """Número apto para JSON (NaN no existe en JSON y se convierte en None)."""
    return None if valor is None or (isinstance(valor, float) and math.isnan(valor)) else round(float(valor), 4)


def zonas_para_modelo(zonas: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Colores de las zonas que entran al modelo; el cabello cae en las cejas si no se midió."""
    resultado = {nombre: np.asarray(lab) for nombre, lab in zonas.items()}
    if "cabello" not in resultado and "cejas" in resultado:
        resultado["cabello"] = resultado["cejas"]
    return resultado


def _probabilidades(probabilidades: Mapping[str, float]):
    return [{"clave": e, "nombre": NOMBRES_ESTACIONES[e], "probabilidad": round(float(p), 4)}
            for e, p in probabilidades.items()]


def describir_modelo(modelo: ModeloEstacional) -> Dict:
    """Cómo se calibró el modelo y qué tan bien acierta (para mostrarlo al usuario).

    Si ya se midió en el subconjunto de validación (`construir_ground_truth.py evaluar
    --guardar`), se informa esa precisión, que es la que pide el Acta; si no, la de la
    validación cruzada de la calibración.
    """
    evaluacion = modelo.parametros.get("evaluacion") or {}
    validacion = evaluacion.get("validacion")
    if validacion:
        return {"tipo": modelo.parametros.get("tipo"), "etiquetas": validacion.get("etiquetas"),
                "n": validacion.get("n"), "exactitud": validacion.get("exactitud"), "ic95": validacion.get("ic95"),
                "top2": validacion.get("top2"), "fecha": validacion.get("fecha"), "metodo": "validacion"}
    metricas = evaluacion.get("metricas") or {}
    return {"tipo": modelo.parametros.get("tipo"), "etiquetas": evaluacion.get("etiquetas"),
            "n": evaluacion.get("n"), "exactitud": metricas.get("exactitud"), "ic95": metricas.get("ic95"),
            "top2": metricas.get("top2"), "fecha": evaluacion.get("fecha"), "metodo": "validacion_cruzada"}


def resultado_desde_rasgos(rasgos: Mapping[str, float], modelo: ModeloEstacional,
                           piel_lab: Optional[np.ndarray] = None) -> Dict:
    """Clasificación, ejes, subtipo y paleta a partir de los rasgos (sin imagen)."""
    probabilidades = modelo.probabilidades(rasgos)
    estacion = next(iter(probabilidades))
    ejes_z = modelo.ejes_z(rasgos)
    percentiles = modelo.ejes_percentil(rasgos)
    clave_subtipo, nombre_subtipo = modelo.subtipo(estacion, ejes_z)
    paleta = recomendar(clave_subtipo, piel_lab)
    return {
        "estacion": {"clave": estacion, "nombre": NOMBRES_ESTACIONES[estacion],
                     "probabilidad": round(float(probabilidades[estacion]), 4)},
        "probabilidades": _probabilidades(probabilidades),
        "subtipo": {"clave": clave_subtipo, "nombre": nombre_subtipo, "lema": paleta["lema"]},
        "ejes": {eje: {"nombre": NOMBRES_EJES[eje][0], "bajo": NOMBRES_EJES[eje][1], "alto": NOMBRES_EJES[eje][2],
                       "valor": round(percentiles[eje], 4), "z": round(ejes_z[eje], 4)} for eje in NOMBRES_EJES},
        "rasgos": {nombre: _numero(rasgos.get(nombre, math.nan)) for nombre in RASGOS},
        "paleta": paleta,
    }


def analizar(imagen: np.ndarray, correcciones: Optional[Dict[str, Tuple[float, float]]] = None,
             modelo: Optional[ModeloEstacional] = None) -> Tuple[Dict, MedicionRostro]:
    """Analiza una foto de rostro y devuelve el resultado completo.

    Args:
        imagen (np.ndarray): Imagen BGR.
        correcciones (Optional[Dict[str, Tuple[float, float]]]): Zonas que el usuario
            fijó a mano (opcional; la interfaz no lo usa, ver `medicion.medir_rostro`).
        modelo (Optional[ModeloEstacional]): Modelo a usar; por defecto el calibrado.

    Returns:
        Tuple[Dict, MedicionRostro]: El resultado (apto para JSON) y la medición, por si se
        quiere componer la vista de control.

    Raises:
        ValueError: Si la imagen no es válida o no tiene un rostro.
        FileNotFoundError: Si falta el modelo de MediaPipe o el modelo calibrado.
    """
    inicio = time.perf_counter()
    modelo = modelo or modelo_por_defecto()
    medicion = medir_rostro(imagen, correcciones)
    lab = {nombre: zona.lab for nombre, zona in medicion.zonas.items()}
    rasgos = calcular_rasgos(zonas_para_modelo(lab))
    resultado = resultado_desde_rasgos(rasgos, modelo, piel_lab=lab.get("piel"))
    resultado["zonas"] = {
        nombre: {"nombre": NOMBRES_ZONAS[nombre], "hex": zona.hex, "lab": [round(float(v), 2) for v in zona.lab],
                 "punto": [round(zona.punto[0], 4), round(zona.punto[1], 4)], "fuente": zona.fuente,
                 "pixeles": zona.pixeles}
        for nombre, zona in medicion.zonas.items() if nombre in ZONAS
    }
    resultado["avisos"] = list(medicion.avisos)
    resultado["balance_aplicado"] = bool(medicion.balance_aplicado)
    resultado["modelo"] = describir_modelo(modelo)
    resultado["tiempo_ms"] = int(round((time.perf_counter() - inicio) * 1000))
    return resultado, medicion


def explorar(rasgos: Mapping[str, Optional[float]], ejes_objetivo: Mapping[str, float],
             modelo: Optional[ModeloEstacional] = None) -> Dict:
    """Probabilidades y paleta si los ejes fueran otros (para estudiar cómo responde el modelo).

    Args:
        rasgos (Mapping[str, Optional[float]]): Los rasgos de un análisis (`resultado["rasgos"]`).
        ejes_objetivo (Mapping[str, float]): Percentil (0-1) deseado para cada eje que se mueve.
        modelo (Optional[ModeloEstacional]): Modelo a usar; por defecto el calibrado.

    Returns:
        Dict: Estación, probabilidades, subtipo y paleta de la exploración (no es el resultado oficial).
    """
    modelo = modelo or modelo_por_defecto()
    limpios = {n: (math.nan if v is None else float(v)) for n, v in rasgos.items()}
    probabilidades = modelo.explorar(limpios, ejes_objetivo)
    estacion = next(iter(probabilidades))
    ejes_z = modelo.ejes_z(limpios)
    for eje, percentil in ejes_objetivo.items():
        if eje in ejes_z:
            ejes_z[eje] = percentil_a_z(float(percentil))
    clave_subtipo, nombre_subtipo = modelo.subtipo(estacion, ejes_z)
    paleta = recomendar(clave_subtipo)
    return {"estacion": {"clave": estacion, "nombre": NOMBRES_ESTACIONES[estacion],
                         "probabilidad": round(float(probabilidades[estacion]), 4)},
            "probabilidades": _probabilidades(probabilidades),
            "subtipo": {"clave": clave_subtipo, "nombre": nombre_subtipo, "lema": paleta["lema"]},
            "paleta": paleta, "exploracion": True}


__all__ = ["analizar", "explorar", "resultado_desde_rasgos", "modelo_por_defecto", "describir_modelo",
           "zonas_para_modelo", "ESTACIONES"]
