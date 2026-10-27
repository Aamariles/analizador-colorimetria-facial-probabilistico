"""Direcciones del PC en la red local (Wi-Fi o hotspot).

Las usan `servidor.py`, para crear el certificado HTTPS e imprimir las direcciones al
arrancar, y la API, para que la interfaz arme el código QR con la dirección de red del PC
aunque en el PC se haya abierto la página como "localhost" (el celular no puede abrir
"localhost": esa dirección apunta al propio celular).
"""

import socket
from typing import List


def ips_locales() -> List[str]:
    """Direcciones IPv4 de este PC en la red local (sin 127.x ni 169.254.x).

    La primera es la de la interfaz que sale a la red (la del Wi-Fi o del hotspot).
    """
    ips: List[str] = []
    try:  # conectar un socket UDP no envía nada, pero revela la IP de la interfaz de salida
        sonda = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sonda.connect(("10.255.255.255", 1))
        ips.append(sonda.getsockname()[0])
        sonda.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return [ip for ip in ips if not ip.startswith(("127.", "169.254.", "0."))]
