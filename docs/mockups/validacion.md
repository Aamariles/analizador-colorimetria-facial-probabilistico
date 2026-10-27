# Hito 7 · Mockups y validación del diseño

## Prototipo

`prisma-prototipo.html` es el prototipo navegable de la interfaz (diseño de Andrés). Se
abre con doble clic en cualquier navegador. Su análisis de color es simulado: sirve para
validar el flujo, el diseño y la forma del informe. La interfaz final (`frontend/`) conserva
su diseño y su CSS, y reemplaza esa lógica por la API local.

Flujo (cuatro ventanas, igual en el prototipo y en la interfaz final): Inicio → Enlace (QR
con el celular) → Cámara (guía de encuadre y control de luz) → Resultado (colores medidos,
estación, probabilidades, paletas, maquillaje, joyería e informe PDF por descarga o
correo). No hay ventana de muestras: apenas llega la foto, el sistema ubica el rostro, mide
solo sus zonas y pasa directo al resultado. Si la foto no tiene un rostro, no pasa al
resultado: la descarta y pide, en la misma ventana, subir o tomar otra.

En el prototipo, el rostro se ubica en el navegador con MediaPipe (el mismo modelo de
puntos del rostro que usa el backend), que se descarga de internet la primera vez. Sin
conexión no se puede revisar la foto, así que tampoco avanza: pide revisar la conexión e
intentarlo de nuevo.

## Validación

| Quién | Fecha | Resultado | Notas |
|---|---|---|---|
| Gerente de Proyecto (Andrés) | 25/10/2026 | Aprobado | Pidió quitar el análisis simulado del prototipo, conservar las 12 estaciones como subtipo y agregar el consentimiento para incluir la foto en el correo. |
| Gerente de Proyecto (Andrés) | 26/10/2026 | Cambio | Quitar la ventana Muestras: la medición de las zonas la debe hacer el sistema de manera automática, no el usuario. |
| Gerente de Proyecto (Andrés) | 26/10/2026 | Cambio | Si al tomar o subir la foto no se encuentra un rostro, no pasar al resultado: pedir que se tome o se suba otra. |
| Patrocinador (docente titular) | 25/10/2026 | Aprobado | Aprobación comunicada por el Gerente de Proyecto. |
| Usuarios de prueba | Pendiente | | Ver el protocolo de abajo. |

Cambios de diseño que salieron de la validación:

- Sin ventana de muestras ni controles de ajuste (prototipo y `frontend/`): el sistema mide
  solo las zonas del rostro y apenas termina muestra el resultado. Los colores medidos y
  los avisos de la medición (por ejemplo, de la luz o del cabello estimado con las cejas)
  se ven en el resultado, sin que el usuario tenga que hacer nada. El prototipo mide la
  piel, el cabello, el iris y los labios; `frontend/` agrega las cejas y la esclerótica.
- Sin rostro no hay resultado (prototipo y `frontend/`): si la foto no muestra un rostro,
  se descarta y la misma ventana pide otra. En la cámara se vuelve a abrir la vista en
  vivo; con el celular, el mismo código sigue sirviendo para mandar otra foto.
- El envío por correo pide dos autorizaciones separadas: enviar los resultados y, aparte,
  incluir la foto (`frontend/`).
- El resultado muestra la precisión del modelo medida en validación, con su intervalo de
  confianza (`frontend/`).

## Protocolo con usuarios de prueba

Cinco personas que no hayan participado en el proyecto, con celulares Android y iPhone, en
la red local (hotspot) que se usará en la sustentación.

Tareas (sin ayuda del equipo; se anota si la persona la completa y cuánto tarda):

1. Desde la portada, conectar el celular con el código QR.
2. Tomar la foto con la guía en vivo y enviarla a la pantalla principal.
3. En el resultado, encontrar los colores que midió el sistema y la probabilidad de cada
   estación.
4. Encontrar los colores a evitar y la prueba A/B.
5. Descargar el informe en PDF y enviarlo a su correo sin incluir la foto.

Preguntas al final (de 1 = muy en desacuerdo a 5 = muy de acuerdo):

1. Entendí qué se hace con mi foto y que no se guarda.
2. Supe en todo momento qué hacer en cada paso.
3. El resultado y las probabilidades se entienden.
4. El informe en PDF es claro y útil.
5. Usaría Prisma otra vez o se lo recomendaría a alguien.

Y una pregunta abierta: ¿qué cambiarías?

| Usuario | Celular | Tareas completadas (1-5) | Tiempo total | Preguntas (1-5) | Comentarios |
|---|---|---|---|---|---|
| U1 | | | | | |
| U2 | | | | | |
| U3 | | | | | |
| U4 | | | | | |
| U5 | | | | | |
