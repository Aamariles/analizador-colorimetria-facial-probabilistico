"""Módulo de clasificación probabilística estacional basada en distancia perceptual.

Utiliza el vector CIELAB dominante para estimar la probabilidad de que
el tono de piel analizado pertenezca a cada una de las cuatro estaciones
cromáticas (Primavera, Verano, Otoño, Invierno).
"""

from typing import Dict, Optional

import numpy as np
from skimage.color import deltaE_ciede2000

try:
# Manejo de importaciones para soportar tanto la ejecución como script independiente
# (pruebas locales) como la importación relativa dentro del paquete principal.
    from .referencias import ESTACIONES
except ImportError:
    from referencias import ESTACIONES


def calcular_distancias_cie00(color_lab: np.ndarray) -> Dict[str, float]:
    """
    Calcula la distancia perceptual CIEDE2000 entre el color de entrada 
    y los centroides estacionales.

    Args:
        color_lab (np.ndarray): Vector [L*, a*, b*] del color dominante extraído.

    Returns:
        Dict[str, float]: Diccionario con la distancia ΔE00 calculada para cada 
        estación ("primavera", "verano", "otono", "invierno"). Un valor menor 
        indica mayor similitud cromática perceptual.

    Raises:
        ValueError: Si el vector de color no tiene exactamente 3 componentes espaciales.
    """
    color_lab = np.asarray(color_lab, dtype=np.float64)
    if color_lab.shape != (3,):
        raise ValueError(f"Se esperaba un vector [L*, a*, b*] de 3 componentes; se recibió forma {color_lab.shape}.")

    distancias: Dict[str, float] = {}
    for clave, referencia in ESTACIONES.items():
        centroide = np.array(referencia.lab, dtype=np.float64)
        distancias[clave] = float(deltaE_ciede2000(color_lab, centroide))

    return distancias


def clasificar_estacion(
    color_lab: np.ndarray,
    prior: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    
    """Clasifica el color de piel en una distribución de probabilidad bayesiana
    sobre las cuatro estaciones cromáticas.

    Args:
        color_lab (np.ndarray): Vector [L*, a*, b*] del color dominante.
        prior (Optional[Dict[str, float]]): Distribución de probabilidad a priori 
            (debe sumar 1.0). Si es None, se asume una distribución uniforme (25% 
            por estación), indicando que a priori ninguna estación es más probable.

    Returns:
        Dict[str, float]: Probabilidad posterior (0.0 a 1.0) de pertenecer a
        cada estación; las cuatro suman 1.0. Ordenado de mayor a menor
        probabilidad para que la estación más probable quede primero.

    Raises:
        ValueError: Si `prior` no cubre las cuatro estaciones o no suma 1.0.
    """
    claves = list(ESTACIONES.keys())

    # Configuración de la distribución a priori por defecto (Uniforme)
    if prior is None:
        prior = {clave: 1.0 / len(claves) for clave in claves}
    else:
        if set(prior.keys()) != set(claves):
            raise ValueError(f"El prior debe tener exactamente las claves {claves}.")
        if not np.isclose(sum(prior.values()), 1.0, atol=1e-6):
            raise ValueError(f"El prior debe sumar 1.0 (suma actual: {sum(prior.values())}).")

    distancias = calcular_distancias_cie00(color_lab)

    # Cálculo de la log-verosimilitud usando un kernel gaussiano sobre la distancia ΔE00.
    # Se aplica la técnica log-sum-exp antes de normalizar para garantizar la
    # estabilidad numérica frente a distancias atípicas muy grandes que podrían
    # generar un colapso a 0.0 al exponenciar prematuramente.
    log_pesos = {}
    for clave in claves:
        sigma = ESTACIONES[clave].sigma
        log_verosimilitud = -(distancias[clave] ** 2) / (2.0 * sigma ** 2)
        log_pesos[clave] = log_verosimilitud + np.log(prior[clave])

    # Normalización log-sum-exp
    log_maximo = max(log_pesos.values())
    pesos_normalizados = {clave: np.exp(valor - log_maximo) for clave, valor in log_pesos.items()}
    suma_pesos = sum(pesos_normalizados.values())

    posterior = {clave: float(peso / suma_pesos) for clave, peso in pesos_normalizados.items()}

    # Ordenar el resultado para posicionar la estación con mayor probabilidad al inicio
    return dict(sorted(posterior.items(), key=lambda item: item[1], reverse=True))


# Bloque de validación estructural local
if __name__ == "__main__":
    print("Ejecutando prueba local del clasificador bayesiano de estación cromática...")

    # Caso 1: un color cercano al centroide de "Otono" (L*=58, a*=20, b*=30)
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
    assert estacion_ganadora == "otono", "El color de prueba debería clasificar como 'otono' (es el centroide más cercano)"
    print(f"\nPrueba 1 exitosa: suma = {suma:.6f}, estación más probable = '{estacion_ganadora}' (esperada: 'otono').")

    # Caso 2 (regresión del prior): mismo color, pero con un prior que favorece
    # fuertemente a "invierno" debe poder cambiar la estación ganadora.
    prior_sesgado = {"primavera": 0.05, "verano": 0.05, "otono": 0.05, "invierno": 0.85}
    posterior_sesgado = clasificar_estacion(color_prueba, prior=prior_sesgado)
    ganadora_sesgada = next(iter(posterior_sesgado))
    print(f"\nPrueba 2 (prior sesgado hacia invierno): estación más probable = '{ganadora_sesgada}'.")
