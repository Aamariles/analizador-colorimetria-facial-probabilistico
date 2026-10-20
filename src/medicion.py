"""Medición del color de piel: la secuencia de etapas que comparten la aplicación y el dataset.

El orden de las etapas es deliberado:

1. MediaPipe detecta el rostro sobre la imagen original (la geometría no depende del color).
2. Gray World estima el iluminante solo con los píxeles más neutros de la escena (lo
   que no es la cabeza): la pared o la ropa blanca, no el cabello ni la piel del cuello
   o los hombros. Luego corrige la imagen completa. Si lo estimara con toda la imagen,
   en un primer plano el promedio sería la propia piel y la corrección la volvería gris;
   si usara toda la escena, el cabello largo y los hombros la enfriarían. En ambos casos
   la piel se desplazaría hacia Verano (ver `preprocesamiento.gray_world`).
3. Se aísla la piel limpia con los mismos landmarks.
4. K-Means en CIELAB devuelve el color dominante de la piel.
5. Control de plausibilidad: la piel humana siempre tiene a* y b* positivos (su matiz
   está entre el rojo y el amarillo). Si la corrección la deja verdosa (a* < 0) o
   azulada (b* < 0), la escena no era gris (por ejemplo, un fondo de color intenso) y
   la corrección se descarta, siempre que la piel sin corregir sí sea plausible.

`main.py` y `construir_ground_truth.py` usan `medir_color_piel` para que la aplicación y
el dataset_ground_truth se midan exactamente igual: unos centroides calibrados con una
medición distinta de la que usa la aplicación no servirían para clasificar.
"""

from typing import NamedTuple

import numpy as np

from colorimetria.cielab_math import extraer_color_dominante_lab
from preprocesamiento.gray_world import aplicar_gray_world, seleccionar_pixeles_neutros
from segmentacion.face_mesh import construir_mascara_escena, detectar_landmarks, extraer_piel_limpia


class MedicionPiel(NamedTuple):
    """Resultado de medir el color de piel de una imagen.

    Attributes:
        imagen_corregida (np.ndarray): Imagen BGR completa usada para medir (la
            original si se descartó el balance de blancos).
        roi_piel (np.ndarray): Imagen BGR con solo la piel analizada (el resto en negro).
        color_lab (np.ndarray): Vector [L*, a*, b*] del color dominante de la piel.
        balance_aplicado (bool): False si la corrección de Gray World se descartó
            porque dejaba la piel con un color imposible.
    """

    imagen_corregida: np.ndarray
    roi_piel: np.ndarray
    color_lab: np.ndarray
    balance_aplicado: bool


def es_piel_plausible(color_lab: np.ndarray) -> bool:
    """Indica si un color L*a*b* puede ser piel humana: a* y b* no negativos.

    Args:
        color_lab (np.ndarray): Vector [L*, a*, b*].

    Returns:
        bool: False si el color es verdoso (a* < 0) o azulado (b* < 0).
    """
    return bool(color_lab[1] >= 0.0 and color_lab[2] >= 0.0)


def medir_color_piel(imagen: np.ndarray) -> MedicionPiel:
    """Mide el color dominante de la piel de una fotografía.

    Sigue las etapas descritas en el docstring de este módulo.

    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR.

    Returns:
        MedicionPiel: La imagen usada, la piel aislada, su color dominante en CIELAB
        y si se aplicó el balance de blancos.

    Raises:
        ValueError: Si la imagen no es válida, si MediaPipe no detecta ningún rostro
            o si no quedan píxeles de piel para analizar.
        FileNotFoundError: Si el modelo `.task` de MediaPipe no existe.
    """
    landmarks = detectar_landmarks(imagen)
    alto, ancho = imagen.shape[:2]
    escena = construir_mascara_escena(landmarks, ancho, alto)
    referencia_gris = seleccionar_pixeles_neutros(imagen, escena)
    imagen_corregida = aplicar_gray_world(imagen, mascara_escena=referencia_gris)
    roi_piel = extraer_piel_limpia(imagen_corregida, landmarks=landmarks)
    color_lab = extraer_color_dominante_lab(roi_piel)

    if not es_piel_plausible(color_lab):
        roi_original = extraer_piel_limpia(imagen, landmarks=landmarks)
        color_original = extraer_color_dominante_lab(roi_original)
        if es_piel_plausible(color_original):
            return MedicionPiel(imagen, roi_original, color_original, balance_aplicado=False)

    return MedicionPiel(imagen_corregida, roi_piel, color_lab, balance_aplicado=True)
