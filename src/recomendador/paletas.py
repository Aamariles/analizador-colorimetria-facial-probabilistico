"""Recomendador de paletas HEX (sexta fase del pipeline y hito 8 del Acta).

Lee el diccionario de paletas de `data/reference/paletas_hex.json` (lo genera
`recomendador/generar_paletas.py` con reglas en CIELAB) y arma la recomendación de una
persona: la paleta de su subtipo (colores favorecedores, neutros, colores a evitar,
maquillaje, cabello y metales) más un tono de base de maquillaje calculado con el color
real de su piel.
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from colorimetria.cielab_math import lab_a_hex

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_PALETAS = os.path.join(BASE_DIR, "data", "reference", "paletas_hex.json")

# Diferencia de claridad (L*) entre los tres tonos de base sugeridos.
PASO_BASE = 4.0


@lru_cache(maxsize=1)
def cargar(ruta: str = RUTA_PALETAS) -> Dict:
    """Lee el diccionario de paletas (una sola vez; queda en memoria).

    Raises:
        FileNotFoundError: Si el archivo no existe (correr `generar_paletas.py`).
    """
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def subtipos() -> List[str]:
    """Claves de los 12 subtipos, en el orden del diccionario."""
    return list(cargar()["subtipos"].keys())


def paleta(subtipo: str) -> Dict:
    """Paleta completa de un subtipo (por ejemplo "otono_profundo").

    Raises:
        KeyError: Si el subtipo no existe.
    """
    datos = cargar()["subtipos"]
    if subtipo not in datos:
        raise KeyError(f"Subtipo desconocido: '{subtipo}'. Opciones: {', '.join(datos)}")
    return datos[subtipo]


def bases_maquillaje(piel_lab: Optional[Sequence[float]], paleta_subtipo: Dict) -> List[Dict]:
    """Tres tonos de base de maquillaje: uno más claro, el ideal y uno más profundo.

    Se calculan con el color medido de la piel (corregido por Gray World), así que son
    personales y no de la estación. Si no hay piel medida, se usa un tono neutro según la
    profundidad del subtipo.

    Args:
        piel_lab (Optional[Sequence[float]]): Color [L*, a*, b*] de la piel.
        paleta_subtipo (Dict): Paleta del subtipo (para el caso sin piel medida).

    Returns:
        List[Dict]: Tres colores {"hex", "nombre", "lab"}.
    """
    if piel_lab is None:
        profundidad = paleta_subtipo["parametros"]["profundidad"]
        calido = paleta_subtipo["parametros"]["calidez"] > 0
        piel_lab = [78 - 38 * profundidad, 12.0, 22.0 if calido else 14.0]
    base = np.asarray(piel_lab, dtype=np.float64)
    tonos = []
    for desplazamiento, nombre in ((PASO_BASE, "Un tono más claro"), (0.0, "Tono base ideal"),
                                   (-PASO_BASE, "Un tono más profundo")):
        lab = base + np.array([desplazamiento, 0.0, 0.0])
        lab[0] = min(max(lab[0], 5.0), 97.0)
        tonos.append({"hex": lab_a_hex(lab), "nombre": nombre, "lab": [round(float(v), 2) for v in lab]})
    return tonos


def recomendar(subtipo: str, piel_lab: Optional[Sequence[float]] = None) -> Dict:
    """Recomendación completa para una persona: la paleta de su subtipo más su base de maquillaje.

    Args:
        subtipo (str): Clave del subtipo (ver `colorimetria.modelo.SUBTIPOS`).
        piel_lab (Optional[Sequence[float]]): Color medido de la piel, para la base.

    Returns:
        Dict: Copia de la paleta con "maquillaje"["base"] personalizada.
    """
    datos = json.loads(json.dumps(paleta(subtipo)))  # copia profunda: el diccionario en memoria no cambia
    datos["maquillaje"]["base"] = bases_maquillaje(piel_lab, datos)
    return datos


def vista_previa() -> List[Dict]:
    """Nombre, lema y colores favorecedores de los 12 subtipos (para la portada de la interfaz)."""
    return [{"clave": clave, "nombre": p["nombre"], "lema": p["lema"],
             "favorecedores": p["favorecedores"], "labial": p["maquillaje"]["labial"],
             "metales": [m["nombre"] for m in p["metales"]], "cabello": p["cabello"]["consejos"][0]}
            for clave, p in cargar()["subtipos"].items()]
