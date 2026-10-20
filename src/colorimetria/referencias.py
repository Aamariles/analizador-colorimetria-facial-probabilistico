"""Módulo de referencias cromáticas estacionales en el espacio CIELAB.

Define los centroides matemáticos (L*a*b*) y la dispersión base para cada una
de las cuatro estaciones de la teoría del color (Primavera, Verano, Otoño, Invierno).
Incluye herramientas de calibración estadística para ajustar iterativamente
estos centroides a partir de conjuntos de datos empíricos supervisados.

Estado de los valores: los centroides de `ESTACIONES` son una primera aproximación
razonada (a partir de los ejes cualitativos de temperatura y claridad del análisis
de color por estaciones), no un dataset calibrado. El Acta de Constitución (sección
XII) exige construir un Ground Truth etiquetado manualmente; `calibrar_desde_ground_truth`
existe para reemplazar estos valores por ese dataset en cuanto esté disponible.
"""

from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from skimage.color import deltaE_ciede2000


class EstacionReferencia(NamedTuple):
    """Estructura de datos para los parámetros estadísticos de una estación cromática.

    Attributes:
        nombre (str): Identificador formal de la estación.
        lab (Tuple[float, float, float]): Coordenadas del centroide en el espacio L*a*b*.
        sigma (float): Dispersión esperada (tolerancia) alrededor del centroide,
            expresada en unidades ΔE00. Define la amplitud del kernel gaussiano
            en la clasificación bayesiana; un valor mayor admite más variación
            intra-clase.
    """

    nombre: str
    lab: Tuple[float, float, float]
    sigma: float


# Definición heurística de los clústeres estacionales basada en los ejes de
# temperatura (b* alto para cálidos, b* bajo para fríos) y luminosidad (L*).
# Los valores de dispersión (sigma) se inicializan simétricamente en un umbral
# perceptual estándar (12.0) y están diseñados para asimetrizarse de forma natural
# tras la calibración con datos reales.
ESTACIONES: Dict[str, EstacionReferencia] = {
    "primavera": EstacionReferencia(nombre="Primavera", lab=(72.0, 18.0, 28.0), sigma=12.0),
    "verano": EstacionReferencia(nombre="Verano", lab=(68.0, 14.0, 16.0), sigma=12.0),
    "otono": EstacionReferencia(nombre="Otoño", lab=(58.0, 20.0, 30.0), sigma=12.0),
    "invierno": EstacionReferencia(nombre="Invierno", lab=(55.0, 16.0, 14.0), sigma=12.0),
}


def calibrar_desde_ground_truth(
    muestras_por_estacion: Dict[str, List[Tuple[float, float, float]]],
    sigma_minimo: float = 2.0,
) -> Dict[str, EstacionReferencia]:
    """Recalibra los centroides estacionales y su dispersión a partir de muestras reales ya etiquetadas.

    Sustituye los valores heurísticos iniciales calculando el promedio espacial
    y la dispersión (en unidades ΔE00) a partir de un conjunto de muestras reales
    (Ground Truth) previamente clasificadas y validadas. El resultado se pasa al
    clasificador con el parámetro `referencias` de `clasificar_estacion`.

    Args:
        muestras_por_estacion: Para cada clave de estación ("primavera",
            "verano", "otono", "invierno"), una lista de tuplas (L*, a*, b*)
            correspondientes a muestras reales etiquetadas con esa estación.
        sigma_minimo (float): Límite inferior de tolerancia para la dispersión,
            en unidades ΔE00. Previene el colapso del modelo de verosimilitud si
            la varianza de una muestra empírica es excesivamente baja. El valor
            por defecto (2.0) equivale al límite anterior de 4 unidades de
            desviación estándar en L*, expresado en la nueva escala.

    Returns:
        Dict[str, EstacionReferencia]: Diccionario con las referencias estacionales
        actualizadas según la distribución probabilística real de los datos.

    Raises:
        ValueError: Si una o más estaciones carecen de muestras en el dataset de entrada.
    """
    nuevas_referencias: Dict[str, EstacionReferencia] = {}
    for clave, referencia_actual in ESTACIONES.items():
        muestras = muestras_por_estacion.get(clave, [])
        if not muestras:
            raise ValueError(f"No hay muestras de calibración (Ground Truth) para la estación '{clave}'.")

        datos = np.array(muestras, dtype=np.float64)
        promedio = datos.mean(axis=0)
        centroide = (float(promedio[0]), float(promedio[1]), float(promedio[2]))

        # La dispersión se mide en las mismas unidades que usa el clasificador: la
        # distancia ΔE00 de cada muestra a su centroide. Para una gaussiana isotrópica
        # en 3 dimensiones E[d²] = 3σ², de donde σ = sqrt(media(d²) / 3). Así recoge la
        # variación en L*, a* y b*, no solo la de la luminosidad.
        distancias = deltaE_ciede2000(datos, np.tile(centroide, (len(datos), 1)))
        sigma_observado = float(max(np.sqrt(np.mean(distancias ** 2) / 3.0), sigma_minimo))

        nuevas_referencias[clave] = EstacionReferencia(
            nombre=referencia_actual.nombre,
            lab=centroide,
            sigma=sigma_observado,
        )

    return nuevas_referencias
