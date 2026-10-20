"""Referencias de color de las cuatro estaciones en el espacio CIELAB.

Define, para Primavera, Verano, Otoño e Invierno, el centroide L*a*b* y la dispersión
(sigma) que usa el clasificador, y la función que recalcula ambos valores a partir de
muestras etiquetadas (el dataset_ground_truth).

Estado de los valores: los centroides de `ESTACIONES` son una primera aproximación
razonada a partir de los ejes de temperatura (cálido/frío) y claridad del análisis de
color por estaciones; no salen de datos medidos. El Acta de Constitución (sección XII)
exige construir un Ground Truth etiquetado, y `calibrar_desde_ground_truth` reemplaza
estos valores con ese dataset (hito 11).
"""

from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from skimage.color import deltaE_ciede2000


class EstacionReferencia(NamedTuple):
    """Parámetros estadísticos de una estación cromática.

    Attributes:
        nombre (str): Nombre de la estación para mostrar, con tildes (por ejemplo, "Otoño").
        lab (Tuple[float, float, float]): Centroide de la estación en el espacio L*a*b*.
        sigma (float): Dispersión esperada alrededor del centroide, en unidades ΔE00.
            Es el ancho de la campana gaussiana del clasificador bayesiano: un valor
            mayor admite más variación dentro de la estación.
    """

    nombre: str
    lab: Tuple[float, float, float]
    sigma: float


# Centroides definidos a partir de los ejes de temperatura (b* alto en las estaciones
# cálidas, bajo en las frías) y claridad (L*). Todas las estaciones empiezan con la
# misma dispersión, amplia (12.0 ΔE00), para que ninguna quede favorecida; la
# calibración le asigna a cada una la dispersión que muestren sus datos.
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
    """Recalcula el centroide y la dispersión de cada estación a partir de muestras etiquetadas.

    El centroide es el promedio L*a*b* de las muestras de la estación, y la dispersión
    se mide en unidades ΔE00, las mismas que usa el clasificador. El resultado se pasa
    a `clasificar_estacion` con su parámetro `referencias`.

    Args:
        muestras_por_estacion (Dict[str, List[Tuple[float, float, float]]]): Para cada
            clave de estación ("primavera", "verano", "otono", "invierno"), la lista de
            colores (L*, a*, b*) de las muestras etiquetadas con esa estación.
        sigma_minimo (float): Dispersión mínima, en unidades ΔE00. Evita que una
            estación con pocas muestras muy parecidas quede con una campana tan
            estrecha que el clasificador se vuelva demasiado seguro.

    Returns:
        Dict[str, EstacionReferencia]: Las referencias recalculadas, con las mismas
        claves y nombres que `ESTACIONES`.

    Raises:
        ValueError: Si alguna estación no tiene muestras.
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
