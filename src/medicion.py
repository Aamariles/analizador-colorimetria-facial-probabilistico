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

`componer_vista` arma la vista de control de una medición (la original, la corregida por
Gray World y la piel analizada), la misma que muestra `main.py --vista` y que guarda
`construir_ground_truth.py extraer --previews`.
"""

from typing import NamedTuple, Optional

import cv2
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


# --------------------------------------------------------------------------- #
# Vista de control
# --------------------------------------------------------------------------- #
ALTO_TITULO = 32               # alto en píxeles de la barra con el título de cada panel
SEPARACION = 6                 # ancho de la franja que separa los paneles
COLOR_FONDO = (45, 45, 45)     # gris oscuro (BGR) de las barras y las franjas


def _con_titulo(panel: np.ndarray, titulo: str) -> np.ndarray:
    """Agrega encima del panel una barra con su título centrado.

    La fuente de OpenCV no tiene tildes ni eñes, así que los títulos se escriben sin ellas.
    """
    ancho = panel.shape[1]
    barra = np.full((ALTO_TITULO, ancho, 3), COLOR_FONDO, dtype=np.uint8)
    fuente, escala, grosor = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (ancho_texto, alto_texto), _ = cv2.getTextSize(titulo, fuente, escala, grosor)
    if ancho_texto > ancho - 12:  # en un panel angosto el título se achica para caber
        escala = max(0.3, escala * (ancho - 12) / ancho_texto)
        (ancho_texto, alto_texto), _ = cv2.getTextSize(titulo, fuente, escala, grosor)
    origen = (max(0, (ancho - ancho_texto) // 2), (ALTO_TITULO + alto_texto) // 2)
    cv2.putText(barra, titulo, origen, fuente, escala, (255, 255, 255), grosor, cv2.LINE_AA)
    return np.vstack([barra, panel])


def componer_vista(imagen: np.ndarray, medicion: MedicionPiel, alto: int = 360,
                   ancho_maximo: Optional[int] = None) -> np.ndarray:
    """Arma la vista de control de una medición: tres paneles lado a lado, cada uno con su título.

    1. La imagen original.
    2. La imagen con el balance de blancos de Gray World. Si la corrección se descartó
       por un fondo de color, es igual a la original y el título lo indica.
    3. La piel analizada (lo demás en negro), con un cuadro en la esquina inferior
       derecha del color dominante que se usa para clasificar.

    Args:
        imagen (np.ndarray): Imagen BGR original, la misma que se le pasó a `medir_color_piel`.
        medicion (MedicionPiel): Resultado de `medir_color_piel` para esa imagen.
        alto (int): Alto de cada panel en píxeles, sin contar el título.
        ancho_maximo (Optional[int]): Si se indica, los paneles se achican para que la
            vista no sea más ancha que esto (por ejemplo, para que quepa en la pantalla).

    Returns:
        np.ndarray: Imagen BGR con la vista de control.
    """
    alto_original, ancho_original = imagen.shape[:2]
    alto = max(1, alto)
    ancho = max(1, round(ancho_original * alto / alto_original))
    if ancho_maximo and 3 * ancho + 2 * SEPARACION > ancho_maximo:
        ancho = max(1, (ancho_maximo - 2 * SEPARACION) // 3)
        alto = max(1, round(alto_original * ancho / ancho_original))
    interpolacion = cv2.INTER_AREA if alto < alto_original else cv2.INTER_LINEAR

    def escalar(panel: np.ndarray) -> np.ndarray:
        return cv2.resize(panel, (ancho, alto), interpolation=interpolacion)

    piel = escalar(medicion.roi_piel)
    lado = max(16, min(ancho, alto) // 5)
    margen = max(4, lado // 6)
    x, y = ancho - margen - lado, alto - margen - lado
    # El color dominante está en L*a*b* real (L* de 0 a 100): se pasa a BGR para dibujarlo
    color = cv2.cvtColor(np.array([[medicion.color_lab]], dtype=np.float32), cv2.COLOR_Lab2BGR)[0, 0]
    color_bgr = tuple(int(c) for c in np.clip(np.round(color * 255), 0, 255))
    cv2.rectangle(piel, (x, y), (x + lado, y + lado), color_bgr, cv2.FILLED)
    cv2.rectangle(piel, (x, y), (x + lado, y + lado), (255, 255, 255), 1)

    titulo_balance = "2. Balance de blancos" if medicion.balance_aplicado else "2. Sin balance (fondo de color)"
    separador = np.full((ALTO_TITULO + alto, SEPARACION, 3), COLOR_FONDO, dtype=np.uint8)
    return np.hstack([_con_titulo(escalar(imagen), "1. Original"), separador,
                      _con_titulo(escalar(medicion.imagen_corregida), titulo_balance), separador,
                      _con_titulo(piel, "3. Piel analizada")])
