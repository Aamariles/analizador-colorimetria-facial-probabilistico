"""Arranca la API y la interfaz web en la red local, con HTTPS.

El navegador del celular solo abre la cámara en un "contexto seguro": HTTPS o localhost.
Como el celular entra por la IP del PC (https://192.168.x.x:8443), el servidor necesita un
certificado. Este script crea uno autofirmado para las IP actuales del PC (en la carpeta
`certs/`, que git ignora) y lo renueva solo si cambian las IP o si vence.

Uso, desde la raíz del repositorio:

    python src/servidor.py                 # HTTPS en el puerto 8443
    python src/servidor.py --puerto 9000
    python src/servidor.py --sin-https     # HTTP (solo para probar en el mismo PC: http://localhost:8000)

Al abrir la dirección por primera vez, el navegador avisa que el certificado no es de una
autoridad conocida (es autofirmado). Hay que aceptarlo una vez:
- Android (Chrome): "Configuración avanzada" > "Acceder a 192.168.x.x (sitio no seguro)".
- iPhone (Safari): "Mostrar detalles" > "visitar este sitio web" > "Visitar sitio web".
- PC (Chrome o Edge): "Configuración avanzada" > "Continuar".
"""

import argparse
import datetime
import ipaddress
import sys
from pathlib import Path
from typing import List, Tuple

from api.red import ips_locales

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_CERTIFICADOS = RAIZ / "certs"
ARCHIVO_CLAVE = CARPETA_CERTIFICADOS / "prisma.key"
ARCHIVO_CERTIFICADO = CARPETA_CERTIFICADOS / "prisma.crt"
DIAS_VIGENCIA = 365


def _certificado_sirve(ips: List[str]) -> bool:
    """Indica si el certificado existente cubre todas las IP actuales y no está por vencer."""
    if not (ARCHIVO_CLAVE.exists() and ARCHIVO_CERTIFICADO.exists()):
        return False
    from cryptography import x509

    certificado = x509.load_pem_x509_certificate(ARCHIVO_CERTIFICADO.read_bytes())
    try:
        nombres = certificado.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return False
    cubiertas = {str(ip) for ip in nombres.get_values_for_type(x509.IPAddress)}
    vence = certificado.not_valid_after_utc if hasattr(certificado, "not_valid_after_utc") \
        else certificado.not_valid_after.replace(tzinfo=datetime.timezone.utc)
    return set(ips) <= cubiertas and vence - datetime.datetime.now(datetime.timezone.utc) > datetime.timedelta(days=7)


def generar_certificado(ips: List[str]) -> Tuple[Path, Path]:
    """Crea (o reutiliza) un certificado autofirmado para localhost y las IP dadas.

    Returns:
        Tuple[Path, Path]: Rutas de la clave privada y del certificado.
    """
    if _certificado_sirve(ips):
        return ARCHIVO_CLAVE, ARCHIVO_CERTIFICADO
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    clave = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nombre = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Prisma (red local)"),
                        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Analizador de Colorimetría Facial")])
    ahora = datetime.datetime.now(datetime.timezone.utc)
    alternativos = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    alternativos += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    certificado = (
        x509.CertificateBuilder()
        .subject_name(nombre).issuer_name(nombre)
        .public_key(clave.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(ahora - datetime.timedelta(minutes=5))
        .not_valid_after(ahora + datetime.timedelta(days=DIAS_VIGENCIA))
        .add_extension(x509.SubjectAlternativeName(alternativos), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(clave, hashes.SHA256())
    )
    CARPETA_CERTIFICADOS.mkdir(exist_ok=True)
    ARCHIVO_CLAVE.write_bytes(clave.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
    ARCHIVO_CERTIFICADO.write_bytes(certificado.public_bytes(serialization.Encoding.PEM))
    return ARCHIVO_CLAVE, ARCHIVO_CERTIFICADO


def main(argv=None) -> int:
    """Interpreta los argumentos, prepara el certificado y arranca uvicorn."""
    parser = argparse.ArgumentParser(description="Servidor local de Prisma (API + interfaz web).")
    parser.add_argument("--puerto", type=int, default=None, help="Puerto (por defecto 8443 con HTTPS, 8000 sin él).")
    parser.add_argument("--host", default="0.0.0.0", help="Interfaz donde escuchar (0.0.0.0 = toda la red local).")
    parser.add_argument("--sin-https", action="store_true",
                        help="Servir por HTTP (el celular no podrá abrir la cámara).")
    args = parser.parse_args(argv)

    import uvicorn
    from api.main import app

    ips = ips_locales()
    puerto = args.puerto or (8000 if args.sin_https else 8443)
    esquema = "http" if args.sin_https else "https"
    opciones = {}
    if not args.sin_https:
        clave, certificado = generar_certificado(ips)
        opciones = {"ssl_keyfile": str(clave), "ssl_certfile": str(certificado)}

    print("\n  Prisma · Analizador de Colorimetría Facial Probabilístico")
    print(f"  En este PC:        {esquema}://localhost:{puerto}")
    for ip in ips:
        print(f"  En la red local:   {esquema}://{ip}:{puerto}   (ábrela en el celular)")
    if not ips:
        print("  No se encontró una IP de red local: conecta el PC al Wi-Fi o al hotspot y vuelve a correrlo.")
    if not args.sin_https:
        print("  La primera vez, acepta el aviso del certificado autofirmado (ver el docstring de src/servidor.py).")
    print("  Documentación de la API: /docs    ·    Ctrl+C para detener\n")
    uvicorn.run(app, host=args.host, port=puerto, log_level="warning", **opciones)
    return 0


if __name__ == "__main__":
    sys.exit(main())
