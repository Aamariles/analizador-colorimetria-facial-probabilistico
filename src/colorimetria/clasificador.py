"""Clasificación probabilística de la estación cromática a partir de la distancia perceptual.

A partir del color dominante de la piel en CIELAB, estima la probabilidad de que el
tono de piel pertenezca a cada una de las cuatro estaciones cromáticas (Primavera,
Verano, Otoño, Invierno) con un modelo bayesiano: distancia ΔE00 → verosimilitud
gaussiana → probabilidad posterior.

Nota de alcance: los centroides de referencia (`colorimetria.referencias.ESTACIONES`)
son una primera aproximación razonada, no un dataset calibrado (ver el docstring de
ese módulo). Las probabilidades que produce este clasificador deben leerse como
ilustrativas hasta que se calibren con el Ground Truth que exige el Acta (sección XII).
Las referencias calibradas (`calibrar_desde_ground_truth`) se pasan con el parámetro
`referencias` de las funciones de este módulo.
"""

from typing import Dict, Optional

import numpy as np
from skimage.color import deltaE_ciede2000

# Import relativo dentro del paquete `colorimetria`. Si el archivo se ejecuta
# directamente (python src/colorimetria/clasificador.py) no hay paquete, así que
# `referencias` se importa como módulo suelto.
try:
    from .referencias import ESTACIONES, EstacionReferencia
except ImportError:
    from referencias import ESTACIONES, EstacionReferencia


def calcular_distancias_cie00(
    color_lab: np.ndarray,
    referencias: Optional[Dict[str, EstacionReferencia]] = None,
) -> Dict[str, float]:
    """Calcula la distancia perceptual CIEDE2000 entre un color y cada centroide estacional.

    Args:
        color_lab (np.ndarray): Vector [L*, a*, b*] del color dominante de la piel.
        referencias (Optional[Dict[str, EstacionReferencia]]): Centroides con los que
            se compara. Si es None, se usan los de `referencias.ESTACIONES`.

    Returns:
        Dict[str, float]: Distancia ΔE00 a cada estación ("primavera", "verano",
        "otono", "invierno"). Un valor menor indica un color más parecido.

    Raises:
        ValueError: Si el vector de color no tiene exactamente 3 componentes.
    """
    referencias = ESTACIONES if referencias is None else referencias
    color_lab = np.asarray(color_lab, dtype=np.float64)
    if color_lab.shape != (3,):
        raise ValueError(
            f"Se esperaba un vector [L*, a*, b*] de 3 componentes; se recibió uno de forma {color_lab.shape}."
        )

    distancias: Dict[str, float] = {}
    for clave, referencia in referencias.items():
        centroide = np.array(referencia.lab, dtype=np.float64)
        distancias[clave] = float(deltaE_ciede2000(color_lab, centroide))

    return distancias


def clasificar_estacion(
    color_lab: np.ndarray,
    prior: Optional[Dict[str, float]] = None,
    referencias: Optional[Dict[str, EstacionReferencia]] = None,
) -> Dict[str, float]:
    """Calcula la probabilidad posterior de cada estación para un color de piel.

    Args:
        color_lab (np.ndarray): Vector [L*, a*, b*] del color dominante de la piel.
        prior (Optional[Dict[str, float]]): Distribución de probabilidad a priori
            (debe sumar 1.0). Si es None, se usa una distribución uniforme (25% por
            estación): a priori ninguna estación es más probable que otra.
        referencias (Optional[Dict[str, EstacionReferencia]]): Centroides y
            dispersiones de cada estación, por ejemplo los que devuelve
            `calibrar_desde_ground_truth`. Si es None, se usan los de `ESTACIONES`.

    Returns:
        Dict[str, float]: Probabilidad posterior (de 0.0 a 1.0) de cada estación; las
        probabilidades suman 1.0 y están ordenadas de mayor a menor, de modo que la
        estación más probable queda primero.

    Raises:
        ValueError: Si `prior` no cubre las estaciones de las referencias o no suma 1.0.
    """
    referencias = ESTACIONES if referencias is None else referencias
    claves = list(referencias.keys())

    # Distribución a priori: uniforme si no se indica otra
    if prior is None:
        prior = {clave: 1.0 / len(claves) for clave in claves}
    else:
        if set(prior.keys()) != set(claves):
            raise ValueError(f"El prior debe tener exactamente las claves {claves}.")
        if not np.isclose(sum(prior.values()), 1.0, atol=1e-6):
            raise ValueError(f"El prior debe sumar 1.0 (suma actual: {sum(prior.values())}).")

    distancias = calcular_distancias_cie00(color_lab, referencias)

    # Log-verosimilitud de una gaussiana isotrópica en el espacio L*a*b* (3 dimensiones),
    # medida con la distancia ΔE00: log p = -d² / (2σ²) - 3·ln(σ) + constante.
    # El término -3·ln(σ) es la normalización de la densidad: no cambia nada mientras
    # todas las estaciones tengan la misma σ, pero tras la calibración cada estación
    # tiene la suya y, sin él, la estación más dispersa absorbería los casos dudosos.
    # Se trabaja en logaritmos y se normaliza con la técnica log-sum-exp: con
    # distancias muy grandes, exponenciar directamente daría 0.0 para todas las
    # estaciones y la división posterior no tendría sentido.
    log_pesos = {}
    for clave in claves:
        sigma = referencias[clave].sigma
        log_verosimilitud = -(distancias[clave] ** 2) / (2.0 * sigma ** 2) - 3.0 * np.log(sigma)
        log_pesos[clave] = log_verosimilitud + np.log(prior[clave])

    # Normalización log-sum-exp
    log_maximo = max(log_pesos.values())
    pesos_normalizados = {clave: np.exp(valor - log_maximo) for clave, valor in log_pesos.items()}
    suma_pesos = sum(pesos_normalizados.values())

    posterior = {clave: float(peso / suma_pesos) for clave, peso in pesos_normalizados.items()}

    # Ordenar de mayor a menor probabilidad
    return dict(sorted(posterior.items(), key=lambda item: item[1], reverse=True))


# Prueba local: python src/colorimetria/clasificador.py
if __name__ == "__main__":
    print("Ejecutando prueba local del clasificador bayesiano de estación cromática...")

    # Caso 1: un color cercano al centroide de Otoño (L*=58, a*=20, b*=30)
    color_prueba = np.array([59.0, 19.0, 29.0])
    posterior = clasificar_estacion(color_prueba)
    distancias = calcular_distancias_cie00(color_prueba)

    print(f"\nColor de prueba (CIELAB): {color_prueba}")
    print("Distancias ΔE00 a cada centroide:")
    for clave, distancia in sorted(distancias.items(), key=lambda item: item[1]):
        print(f"  {ESTACIONES[clave].nombre:12s}: ΔE00 = {distancia:6.2f}")

    print("\nProbabilidad posterior por estación:")
    suma = 0.0
    for clave, probabilidad in posterior.items():
        print(f"  {ESTACIONES[clave].nombre:12s}: {probabilidad * 100:5.1f}%")
        suma += probabilidad

    estacion_ganadora = next(iter(posterior))
    assert abs(suma - 1.0) < 1e-9, "Las probabilidades deben sumar 1.0"
    assert estacion_ganadora == "otono", (
        "El color de prueba debería clasificar como 'otono' (es el centroide más cercano)"
    )
    print(f"\nPrueba 1 exitosa: suma = {suma:.6f}, estación más probable = '{estacion_ganadora}' (esperada: 'otono').")

    # Caso 2 (regresión del prior): con el mismo color, un prior que favorece
    # fuertemente a "invierno" debe poder cambiar la estación ganadora.
    prior_sesgado = {"primavera": 0.05, "verano": 0.05, "otono": 0.05, "invierno": 0.85}
    posterior_sesgado = clasificar_estacion(color_prueba, prior=prior_sesgado)
    ganadora_sesgada = next(iter(posterior_sesgado))
    assert ganadora_sesgada == "invierno", "Un prior fuertemente sesgado debe poder cambiar la estación ganadora."
    print(f"\nPrueba 2 exitosa (prior sesgado hacia invierno): estación más probable = '{ganadora_sesgada}'.")

    # Caso 3 (regresión de la normalización): dos estaciones con dispersiones distintas.
    # El color queda a ΔE00 ≈ 2.2 de "compacta" (σ=4) y a ΔE00 ≈ 1.3 de "dispersa" (σ=12).
    # La densidad de la estación compacta en ese punto es unas 23 veces mayor, así
    # que debe ganar; sin el término -3·ln(σ) ganaba la estación dispersa.
    referencias_prueba = {
        "compacta": EstacionReferencia(nombre="Compacta", lab=(60.0, 15.0, 20.0), sigma=4.0),
        "dispersa": EstacionReferencia(nombre="Dispersa", lab=(64.0, 15.0, 20.0), sigma=12.0),
    }
    color_intermedio = np.array([62.5, 15.0, 20.0])
    d = calcular_distancias_cie00(color_intermedio, referencias_prueba)
    posterior_3 = clasificar_estacion(color_intermedio, referencias=referencias_prueba)
    print(f"\nPrueba 3: ΔE00 compacta = {d['compacta']:.2f}, ΔE00 dispersa = {d['dispersa']:.2f} -> "
          + ", ".join(f"{k} {v * 100:.1f}%" for k, v in posterior_3.items()))
    assert next(iter(posterior_3)) == "compacta", "La estación compacta debe ganar en un punto donde es más densa."
    print("Prueba 3 (normalización por dispersión) exitosa.")
