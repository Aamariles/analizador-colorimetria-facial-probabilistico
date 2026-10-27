"""Genera el diccionario de paletas HEX de las 12 estaciones (data/reference/paletas_hex.json).

Las paletas se construyen con reglas en el espacio CIELAB, en coordenadas cilíndricas LCh
(L* claridad, C* croma, h matiz), a partir de cuatro parámetros de cada subtipo, los mismos
del prototipo de la interfaz:

- calidez (-1 frío a 1 cálido): elige las familias de color cuyo matiz combina con esa
  temperatura y desplaza levemente su matiz hacia el amarillo (cálido) o el azul (frío);
- profundidad (0 claro a 1 profundo): fija la claridad L* de los colores;
- claridad (0 suave a 1 brillante): fija el croma C* (qué tan limpio o empolvado);
- contraste (0 bajo a 1 alto): se usa en los consejos.

Los colores a evitar son los opuestos: las familias de temperatura contraria, con el croma y
la claridad invertidos. Cada color se lleva al gamut sRGB reduciendo su croma (sin cambiar
L* ni h) y se nombra por su familia y un modificador (pastel, apagado, profundo, intenso).

Para regenerar el archivo (por ejemplo, después de ajustar una regla):

    python src/recomendador/generar_paletas.py

Si el equipo tiene paletas propias, puede editar el JSON directamente: el módulo
`recomendador.paletas` solo lee ese archivo.
"""

import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colorimetria.cielab_math import croma, lab_a_hex  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_PALETAS = os.path.join(BASE_DIR, "data", "reference", "paletas_hex.json")

# Subtipo: (clave, nombre, estación, calidez, profundidad, claridad, contraste, lema)
SUBTIPOS: List[Tuple[str, str, str, float, float, float, float, str]] = [
    ("primavera_clara", "Primavera Clara", "primavera", .55, .12, .55, .30, "Luz dorada y pasteles cálidos"),
    ("primavera_calida", "Primavera Cálida", "primavera", .95, .40, .65, .45, "Oro, coral y verdes de sol"),
    ("primavera_brillante", "Primavera Brillante", "primavera", .50, .40, .95, .75, "Color limpio y vivo, con calidez"),
    ("verano_claro", "Verano Claro", "verano", -.40, .12, .40, .25, "Acuarela fría y plata delicada"),
    ("verano_frio", "Verano Frío", "verano", -.90, .45, .40, .40, "Rosa empolvado y azul grisáceo"),
    ("verano_suave", "Verano Suave", "verano", -.30, .50, .08, .20, "Bruma elegante en tonos ahumados"),
    ("otono_suave", "Otoño Suave", "otono", .35, .50, .08, .25, "Tierra apagada y salvia"),
    ("otono_calido", "Otoño Cálido", "otono", .95, .60, .35, .50, "Especias, óxido y oliva"),
    ("otono_profundo", "Otoño Profundo", "otono", .60, .90, .50, .80, "Burdeos, bosque y caoba"),
    ("invierno_profundo", "Invierno Profundo", "invierno", -.50, .92, .65, .92, "Joya oscura y contraste total"),
    ("invierno_frio", "Invierno Frío", "invierno", -.95, .60, .70, .75, "Zafiro, fucsia y plata"),
    ("invierno_brillante", "Invierno Brillante", "invierno", -.55, .50, .97, .95, "Cristalino, nítido y eléctrico"),
]

# Familias de color: (matiz CIELAB en grados, nombre, afinidad de temperatura de -1 a 1).
# Los matices salen de colores de referencia (por ejemplo, coral #FF7F50 tiene h = 46°,
# mostaza #E1AD01 h = 85°, azul real #4169E1 h = 292°).
FAMILIAS: List[Tuple[float, str, float]] = [
    (5, "Rosa", -.5), (25, "Rojo", .1), (40, "Coral", .8), (48, "Terracota", .9), (60, "Naranja", .8),
    (80, "Mostaza", .9), (100, "Oliva", .7), (125, "Lima", .4), (145, "Verde", 0.0), (175, "Verde azulado", -.2),
    (195, "Turquesa", -.1), (235, "Azul cielo", -.5), (275, "Azul", -.7), (295, "Índigo", -.8),
    (312, "Violeta", -.8), (328, "Malva", -.7), (345, "Fucsia", -.8),
]
# Colores de tema de cada subtipo (degradado del título y del anillo de probabilidad), los
# mismos del prototipo de la interfaz.
TEMAS: Dict[str, Tuple[str, str]] = {
    "primavera_clara": ("#f0a64a", "#ee8a7c"), "primavera_calida": ("#e9a23b", "#e0663f"),
    "primavera_brillante": ("#f4b400", "#ee5a8a"), "verano_claro": ("#9db4d9", "#c9a7d8"),
    "verano_frio": ("#7f9fcf", "#d79bb3"), "verano_suave": ("#8fa3b3", "#b9a2b8"),
    "otono_suave": ("#c9a24a", "#8c9a5b"), "otono_calido": ("#c8872e", "#a8502d"),
    "otono_profundo": ("#c4503a", "#8a9a45"), "invierno_profundo": ("#3f63d6", "#aab4cc"),
    "invierno_frio": ("#2563c9", "#b8c2d6"), "invierno_brillante": ("#2f5de0", "#d63384"),
}
VARIACION = (0.0, .08, -.06, .04, -.1, .1)  # alterna claridad y croma para que la paleta no sea plana
CROMA_MAXIMA_PASTEL = 38.0  # croma máximo de un color muy claro (L* > 78)

NEUTROS_CALIDOS = ("Marfil", "Arena", "Camel", "Chocolate", "Espresso")
NEUTROS_FRIOS = ("Blanco roto", "Gris perla", "Gris piedra", "Azul marino", "Carbón")

# Metales (nombre: degradado de dos colores para dibujarlos) y su uso según la temperatura.
METALES: Dict[str, Tuple[str, str]] = {
    "Oro amarillo": ("#f6dc8a", "#c9972b"), "Cobre": ("#f0b08a", "#b8602f"), "Bronce": ("#d9a86c", "#8f5f2a"),
    "Plata": ("#f4f5f7", "#a3a9b3"), "Oro blanco": ("#f2eee6", "#bdb3a0"), "Platino": ("#e9edf1", "#8e98a6"),
    "Oro rosa": ("#f6cdbd", "#c98a78"), "Oro claro": ("#f8e7b0", "#d3b25a"),
}


def _acotar(x: float, a: float, b: float) -> float:
    return min(b, max(a, x))


def _en_gamut(L: float, C: float, h: float) -> np.ndarray:
    """Color L*a*b* con el croma reducido lo justo para que exista en sRGB (búsqueda binaria)."""
    import cv2

    def lab(c: float) -> np.ndarray:
        return np.array([L, c * math.cos(math.radians(h)), c * math.sin(math.radians(h))])

    def dentro(c: float) -> bool:
        bgr = cv2.cvtColor(np.array([[lab(c)]], dtype=np.float32), cv2.COLOR_Lab2BGR)[0, 0]
        return bool(np.all(bgr >= -1e-3) and np.all(bgr <= 1 + 1e-3))

    if dentro(C):
        return lab(C)
    bajo, alto = 0.0, C
    for _ in range(25):
        medio = (bajo + alto) / 2
        bajo, alto = (medio, alto) if dentro(medio) else (bajo, medio)
    return lab(bajo)


def _hacia(h: float, objetivo: float, grados: float) -> float:
    """Gira el matiz h hacia `objetivo` como mucho `grados` (sin pasarse)."""
    diferencia = (objetivo - h + 180.0) % 360.0 - 180.0
    paso = math.copysign(min(abs(grados), abs(diferencia)), diferencia)
    return (h + paso) % 360.0


def _modificador(L: float, C: float) -> str:
    """Adjetivo del nombre de un color según su claridad y croma (como en el prototipo)."""
    if L > 78:
        return " pastel"
    if C < 22:
        return " apagado"
    if L < 32:
        return " profundo"
    if C > 62:
        return " intenso"
    return ""


def _color(lab: np.ndarray, nombre: str) -> Dict:
    return {"hex": lab_a_hex(lab), "nombre": nombre, "lab": [round(float(v), 2) for v in lab]}


def _color_familia(h: float, nombre: str, L: float, C: float, calidez: float) -> Dict:
    """Color de una familia, con el matiz llevado hacia el amarillo (cálido) o el azul (frío)."""
    h = _hacia(h, 90.0, 8 * calidez) if calidez > 0 else _hacia(h, 280.0, 8 * -calidez)
    lab = _en_gamut(L, C, h)
    return _color(lab, nombre + _modificador(lab[0], croma(lab)))


def paleta_de(clave: str, nombre: str, estacion: str, calidez: float, profundidad: float, claridad: float,
              contraste: float, lema: str) -> Dict:
    """Construye la paleta completa de un subtipo (ver el docstring del módulo)."""
    cercanas = sorted(range(len(FAMILIAS)), key=lambda j: abs(FAMILIAS[j][2] - calidez))[:12]
    favorecedores = []
    for k, j in enumerate(sorted(cercanas)):
        v = VARIACION[k % len(VARIACION)]
        L = _acotar(88 - 52 * profundidad + 45 * v, 18, 93)
        C = _acotar(18 + 62 * claridad + 40 * v, 8, 85)
        if L > 78:  # un color muy claro y saturado se ve fluorescente, no pastel
            C = min(C, CROMA_MAXIMA_PASTEL)
        favorecedores.append(_color_familia(FAMILIAS[j][0], FAMILIAS[j][1], L, C, calidez))
    for color in sorted(favorecedores, key=lambda c: -croma(c["lab"]))[:3]:
        color["estrella"] = True

    calido = calidez > 0
    nombres_neutros = NEUTROS_CALIDOS if calido else NEUTROS_FRIOS
    neutros = []
    for i, L in enumerate((94, 80, 62, 38, 20)):
        L = _acotar(L - 12 * (profundidad - 0.5), 12, 96)
        if calido:
            lab = _en_gamut(L, 6 + 18 * abs(calidez) * (0.4 + 0.6 * (i in (2, 3))), 70)
        elif nombres_neutros[i] == "Azul marino":
            lab = _en_gamut(min(L, 28), 28, 285)
        else:
            lab = _en_gamut(L, 3, 260)
        neutros.append(_color(lab, nombres_neutros[i]))

    lejanas = sorted(range(len(FAMILIAS)), key=lambda j: -abs(FAMILIAS[j][2] - calidez))[:6]
    evitar = []
    for k, j in enumerate(lejanas):
        L = 85.0 if profundidad > .55 else 22.0 + 4 * k
        C = 70.0 if claridad < .5 else 15.0
        evitar.append(_color_familia(FAMILIAS[j][0], FAMILIAS[j][1], L, C, -calidez))

    matices_labios = (40, 48, 32) if calido else (5, 355, 335)
    nombres_labios = ("Coral", "Terracota", "Rojo tomate") if calido else ("Rosa", "Frambuesa", "Ciruela")
    L_labios = _acotar(62 - 24 * profundidad, 30, 70)
    C_labios = _acotar(30 + 40 * claridad, 20, 75)
    labial = [_color(_en_gamut(L_labios, C_labios, h), n) for h, n in zip(matices_labios, nombres_labios)]
    rubor = [_color(_en_gamut(min(L_labios + 14, 82), max(C_labios - 12, 15), h), n + " suave")
             for h, n in zip(matices_labios[:2], nombres_labios[:2])]
    sombras = [neutros[1], neutros[2], neutros[3], favorecedores[6]]

    niveles = (20, 25, 32, 40) if profundidad > .6 else (28, 35, 43, 52) if profundidad > .3 else (45, 55, 65, 75)
    if calido:
        tonos_cabello = [_color(_en_gamut(L, 26, h), n) for L, h, n in
                         zip(niveles, (60, 45, 70, 55), ("Castaño chocolate", "Castaño caoba", "Castaño miel",
                                                         "Cobrizo suave"))]
        consejo_cabello = ("castaño chocolate cálido o caoba" if profundidad > .6 else
                           "castaño miel o dorado suave" if profundidad > .3 else "rubio dorado o miel claro")
    else:
        tonos_cabello = [_color(_en_gamut(L, 8, h), n) for L, h, n in
                         zip(niveles, (60, 50, 80, 70), ("Castaño ceniza", "Cacao frío", "Rubio ceniza", "Gris perla"))]
        consejo_cabello = ("negro azulado o castaño oscuro frío" if profundidad > .6 else
                           "castaño ceniza o cacao frío" if profundidad > .3 else "rubio ceniza o platino suave")
    consejo_contraste = ("tonos mezclados y suaves, sin contrastes duros" if claridad < .3 else
                         "admite reflejos nítidos y contraste marcado" if claridad > .7
                         else "reflejos sutiles y naturales")

    if calidez > .25:
        metales = ["Oro amarillo", "Cobre", "Bronce"]
    elif calidez < -.25:
        metales = ["Plata", "Oro blanco", "Platino"]
    else:
        metales = ["Oro rosa", "Oro claro"]
    consejo = ("En vez de negro puro, usa espresso o chocolate intenso." if calido else
               "El negro puro y el blanco óptico sí te funcionan." if profundidad > .7 else
               "En vez de negro puro, usa carbón o azul marino.")

    return {
        "clave": clave, "nombre": nombre, "estacion": estacion, "lema": lema, "tema": list(TEMAS[clave]),
        "parametros": {"calidez": calidez, "profundidad": profundidad, "claridad": claridad, "contraste": contraste},
        "favorecedores": favorecedores, "neutros": neutros, "evitar": evitar,
        "maquillaje": {"labial": labial, "rubor": rubor, "sombras": sombras},
        "cabello": {"tonos": tonos_cabello, "consejos": [consejo_cabello, consejo_contraste]},
        "metales": [{"nombre": m, "degradado": list(METALES[m])} for m in metales],
        "consejo": consejo,
    }


def generar() -> Dict:
    """Paletas de los 12 subtipos, indexadas por su clave."""
    return {"version": 1, "metodo": "reglas en CIELAB/LCh (ver src/recomendador/generar_paletas.py)",
            "subtipos": {s[0]: paleta_de(*s) for s in SUBTIPOS}}


if __name__ == "__main__":
    datos = generar()
    with open(RUTA_PALETAS, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=1)
        archivo.write("\n")
    for clave, p in datos["subtipos"].items():
        print(f"{p['nombre']:20s} " + " ".join(c["hex"] for c in p["favorecedores"]))
    print(f"\nPaletas guardadas en {RUTA_PALETAS}")
