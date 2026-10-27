"""Enlace PC-celular (handoff): el celular le entrega la foto al PC a través de la API local.

El PC pide un código de 6 caracteres y lo muestra (con un QR). El celular abre la misma
página con ese código, toma la foto y la sube; el PC pregunta cada segundo si ya llegó y,
cuando la recoge, la foto se borra del servidor. Todo vive en memoria (RAM) y caduca solo:
el código a los `VIGENCIA_CODIGO` segundos y la foto no recogida a los `VIGENCIA_FOTO`,
de acuerdo con la política de cero almacenamiento del Acta.
"""

import secrets
import threading
import time
from typing import Dict, Optional

# Alfabeto sin caracteres que se confunden al leerlos (0/O, 1/I/L).
ALFABETO = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
LONGITUD_CODIGO = 6
VIGENCIA_CODIGO = 15 * 60   # segundos que dura un código sin usarse
VIGENCIA_FOTO = 2 * 60      # segundos que una foto espera a que el PC la recoja
MAXIMO_CODIGOS = 50         # tope de enlaces abiertos a la vez (protege la memoria)


class _Enlace:
    __slots__ = ("creado", "foto", "foto_recibida")

    def __init__(self) -> None:
        self.creado = time.monotonic()
        self.foto: Optional[bytes] = None
        self.foto_recibida = 0.0


class AlmacenEnlaces:
    """Códigos de enlace y fotos en espera, en memoria y con caducidad.

    Es seguro usarlo desde varios hilos (FastAPI atiende las peticiones en paralelo).
    """

    def __init__(self) -> None:
        self._enlaces: Dict[str, _Enlace] = {}
        self._candado = threading.Lock()

    def _limpiar(self) -> None:
        """Borra los códigos vencidos y las fotos que nadie recogió a tiempo (con el candado tomado)."""
        ahora = time.monotonic()
        for codigo in [c for c, e in self._enlaces.items() if ahora - e.creado > VIGENCIA_CODIGO]:
            del self._enlaces[codigo]
        for enlace in self._enlaces.values():
            if enlace.foto is not None and ahora - enlace.foto_recibida > VIGENCIA_FOTO:
                enlace.foto = None

    def crear(self) -> str:
        """Crea un código nuevo.

        Raises:
            RuntimeError: Si ya hay `MAXIMO_CODIGOS` enlaces abiertos.
        """
        with self._candado:
            self._limpiar()
            if len(self._enlaces) >= MAXIMO_CODIGOS:
                raise RuntimeError("Hay demasiados enlaces abiertos; espera unos minutos.")
            while True:
                codigo = "".join(secrets.choice(ALFABETO) for _ in range(LONGITUD_CODIGO))
                if codigo not in self._enlaces:
                    self._enlaces[codigo] = _Enlace()
                    return codigo

    def existe(self, codigo: str) -> bool:
        """Indica si el código está vigente."""
        with self._candado:
            self._limpiar()
            return codigo.upper() in self._enlaces

    def dejar_foto(self, codigo: str, foto: bytes) -> bool:
        """Guarda en memoria la foto que manda el celular (reemplaza una anterior sin recoger).

        Returns:
            bool: False si el código no existe o venció.
        """
        with self._candado:
            self._limpiar()
            enlace = self._enlaces.get(codigo.upper())
            if enlace is None:
                return False
            enlace.foto = foto
            enlace.foto_recibida = time.monotonic()
            return True

    def recoger_foto(self, codigo: str) -> Optional[bytes]:
        """Entrega la foto al PC y la borra del servidor; None si todavía no llegó.

        Raises:
            KeyError: Si el código no existe o venció.
        """
        with self._candado:
            self._limpiar()
            enlace = self._enlaces.get(codigo.upper())
            if enlace is None:
                raise KeyError(codigo)
            foto, enlace.foto = enlace.foto, None
            return foto

    def cerrar(self, codigo: str) -> None:
        """Elimina un código y su foto, si la hay."""
        with self._candado:
            self._enlaces.pop(codigo.upper(), None)

    def __len__(self) -> int:
        with self._candado:
            self._limpiar()
            return len(self._enlaces)
