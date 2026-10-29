/* ==========================================================================
   PRISMA · lógica de la interfaz web (hito 9)
   --------------------------------------------------------------------------
   Esta página no analiza colores: captura la foto, la envía a la API local
   (src/api/main.py) y muestra lo que el computador calculó. Las zonas del rostro
   las mide el computador solo: apenas llega la foto se analiza y se muestra el
   resultado, sin pasos intermedios para el usuario. Lo único que mide el
   navegador es la luz y el encuadre de la cámara en vivo, para guiar la captura
   (sección 6).
   Política de cero almacenamiento: la foto vive solo en memoria (st.foto y
   st.url) y wipe() la descarta.
   ========================================================================== */
'use strict';

/* 1. UTILIDADES ============================================================ */
/** Atajos de DOM: `$` = primer elemento, `$$` = arreglo de todos. */
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
/** Escapa texto para insertarlo en HTML. */
const esc = t => String(t ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
/** Primera letra en mayúscula. */
const mayuscula = t => t.charAt(0).toUpperCase() + t.slice(1);
/** Porcentaje entero de una probabilidad 0-1. */
const pct = p => Math.round(p * 100);
/** Aviso flotante breve. @param {string} t */
const toast = t => {const e = $('#toast'); e.textContent = t; e.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => e.hidden = true, 4500)};

/**
 * Llama a la API local con tiempo límite y convierte los errores en mensajes claros.
 * @param {string} ruta Ruta de la API (por ejemplo "/api/analizar").
 * @param {RequestInit} opciones Opciones de fetch.
 * @param {number} segundos Tiempo máximo de espera.
 * @returns {Promise<Response>} La respuesta (siempre con estado 2xx).
 */
async function api(ruta, opciones = {}, segundos = 30) {
  const control = new AbortController(), reloj = setTimeout(() => control.abort(), segundos * 1000);
  try {
    const r = await fetch(ruta, {...opciones, signal: control.signal, cache: 'no-store'});
    if (!r.ok) {
      let mensaje = `El computador respondió con un error (${r.status}).`;
      try {const d = await r.json(); if (d.detail) mensaje = typeof d.detail === 'string' ? d.detail : 'Los datos enviados no son válidos.'} catch (e) { /* respuesta sin JSON */ }
      const error = new Error(mensaje); error.estado = r.status; throw error;
    }
    return r;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('El computador tardó demasiado en responder. Revisa que el celular y el computador estén en la misma red.');
    if (e instanceof TypeError) throw new Error('No hay conexión con el computador. Revisa que estén en la misma red (Wi-Fi o hotspot) y que el servidor esté encendido.');
    throw e;
  } finally {clearTimeout(reloj)}
}

/** Carga un Blob/File en un <img> (los navegadores actuales ya respetan la orientación EXIF). */
const leerImagen = fuente => new Promise((ok, mal) => {const im = new Image(), u = URL.createObjectURL(fuente); im.onload = () => {URL.revokeObjectURL(u); ok(im)}; im.onerror = () => {URL.revokeObjectURL(u); mal(new Error('El archivo no es una imagen válida.'))}; im.src = u});
/** Dibuja una imagen o un video en un canvas de como máximo `lado` píxeles en su lado mayor. */
function reducir(fuente, ancho, alto, lado) {const k = Math.min(1, lado / Math.max(ancho, alto)), c = document.createElement('canvas'); c.width = Math.round(ancho * k); c.height = Math.round(alto * k); c.getContext('2d').drawImage(fuente, 0, 0, c.width, c.height); return c}
/** Canvas → Blob JPEG. */
const aJpeg = (c, calidad = .9) => new Promise(ok => c.toBlob(ok, 'image/jpeg', calidad));

/* 2. ESTADO Y NAVEGACIÓN =================================================== */
/**
 * Estado global (único objeto mutable de la app).
 *  foto   Blob JPEG de la foto (≤1280 px)      url   URL de objeto de la foto      res  resultado de /api/analizar
 *  dr/di/dj  modo y colores de la prueba en rostro   pair/local  banderas del flujo   codigo/sondeo  enlace con el celular
 */
const st = {foto: null, url: '', res: null, dr: 'best', di: 0, dj: 0, pair: 0, local: 0, codigo: '', sondeo: null};
const V = ['home', 'pair', 'phone', 'result'];
const ZONAS = ['piel', 'cabello', 'cejas', 'iris', 'esclerotica', 'labios'];
const EJES = ['temperatura', 'profundidad', 'croma', 'contraste'];

/** Muestra una ventana, oculta las demás y actualiza la barra de pasos. @param {string} v */
function show(v) {
  limpiarAvisos();
  V.forEach(x => $('#v-' + x).hidden = x !== v);
  const k = V.indexOf(v);
  $$('#steps li').forEach((l, i) => {l.className = (i === k ? 'on' : i < k ? 'ok' : '') + (i !== k && !avail(i + 1) ? ' lk' : ''); l.hidden = i === 1 && !st.pair});
  $('#steps').hidden = v === 'phone' && !st.local;
  scrollTo(0, 0);
}
/** ¿Está disponible el paso n (1-4)? */
const avail = n => n === 1 || (n === 2 && !!st.pair) || (n === 3 && !!st.foto) || (n === 4 && !!st.res);
/** Etapa de la guía rápida que explica cada paso de la barra. */
const hm = [0, 1, 2, 2, 3];
/** Guía rápida: completa (`all`) o solo la etapa del paso `n`. */
function help(n, all) {$$('#hw .hws li').forEach(l => l.hidden = !all && +l.dataset.n !== hm[n]); $('#hw .hw-i').hidden = !all; $('#hw').classList.toggle('full', !!all); $('#hw h2').textContent = all ? 'Conoce cómo funciona Prisma paso a paso' : '¿Qué hace esta ventana?'; $('#hw').hidden = false; $('#hw .hw').scrollTop = 0}
/** Navega al paso n de la barra; si está bloqueado, explica esa etapa. */
function nav(n) {
  if (V.findIndex(x => !$('#v-' + x).hidden) === n - 1) return;
  if (!avail(n)) {help(n); return}
  detenerCamara(); detenerSondeo();
  if (n === 1) show('home'); else if (n === 2) $('#phone').click(); else if (n === 3) {st.pair = 0; capUI(1); show('phone')} else {show('result'); render()}
}
$('#steps').onclick = e => {const l = e.target.closest('li'); if (l) nav(+l.dataset.n)};
$('#qm').onclick = () => help(0, true); $('#hwx').onclick = () => $('#hw').hidden = true; $('#hwgo').onclick = () => {$('#hw').hidden = true; nav(1)};
$('#hw').onclick = e => {if (e.target.id === 'hw') $('#hw').hidden = true};
addEventListener('keydown', e => {if (e.key === 'Escape') {$('#hw').hidden = true; $('#rv').hidden = true}});

/** Cero almacenamiento: descarta la foto y el resultado. */
function wipe() {
  if (st.url) URL.revokeObjectURL(st.url);
  Object.assign(st, {foto: null, url: '', res: null});
  $('#toast').hidden = true; avisoInicio('');
  ['#dimg', '#avt', '#pp'].forEach(s => $(s).removeAttribute('src')); pimg = null;
}

/* 3. CONSENTIMIENTO (Ley 1581 de 2012) ===================================== */
/** Pide la autorización una vez por sesión antes de usar la cámara o subir una foto; luego ejecuta `cb`. */
let okc = 0, pend = null;
const ask = cb => {if (okc) return cb(); pend = cb; $('#hdb').innerHTML = $('#hdt').innerHTML; $('#hd').hidden = false};
$('#hy').onclick = () => {okc = 1; $('#hd').hidden = true; if (pend) pend(); pend = null};
$('#hn').onclick = () => {$('#hd').hidden = true; pend = null};
// La selección de archivo también espera el consentimiento.
$$('.cta label,#pfl').forEach(l => l.addEventListener('click', e => {if (!okc) {e.preventDefault(); ask(() => l.querySelector('input').click())}}));

/* 4. PORTADA: vista previa de las 12 paletas (datos de /api/paletas) ======== */
let ti = 0, hold = 0, vistas = [];
const swp = (e, f, d = 0) => {setTimeout(() => e.classList.add('f'), d); setTimeout(() => {f(); e.classList.remove('f')}, d + 450)}, cell = '<div class="sw"><i></i><span><b></b><small></small></span></div>';
function teaser() {
  if (!vistas.length) return;
  const g = vistas[ti], c = g.favorecedores, z = $('#tz');
  if (!z.firstChild) z.innerHTML = `<p class="eyebrow" id="tn"></p><div class="strip">${'<i></i>'.repeat(12)}</div><div class="sws">${cell.repeat(4)}</div><div class="blur"><div class="sws">${cell.repeat(4)}</div><p id="tl"></p><p id="tm"></p></div>`;
  swp($('#tn'), () => $('#tn').textContent = `Vista previa · ejemplo ${g.nombre}`);
  $$('#tz .strip i').forEach((e, k) => {const q = c[k % c.length]; e.style.transitionDelay = '0s,' + k * 45 + 'ms'; e.style.background = q.hex; e.title = q.nombre});
  $$('#tz .sw').forEach((e, k) => {const q = c[k % c.length], i = e.firstChild, b = e.querySelector('b'), s = e.querySelector('small'); i.style.transitionDelay = '0s,' + k * 70 + 'ms'; i.style.background = q.hex; swp(b, () => b.textContent = q.nombre, k * 70); swp(s, () => s.textContent = q.hex, k * 70)});
  swp($('#tl'), () => $('#tl').innerHTML = '<b>Labios:</b> ' + g.labial.map(x => esc(x.nombre + ' ' + x.hex)).join(', '));
  swp($('#tm'), () => $('#tm').innerHTML = '<b>Metales:</b> ' + esc(g.metales.join(', ')) + '. <b>Cabello:</b> ' + esc(g.cabello) + '.', 120);
}
$('#unlock').onclick = () => $('#take').click();
$('#teaser').onmouseenter = () => hold = 1; $('#teaser').onmouseleave = () => hold = 0;
setInterval(() => {if (!hold && vistas.length && !$('#v-home').hidden) {ti = (ti + 1) % vistas.length; teaser()}}, 5000);
api('/api/paletas', {}, 10).then(r => r.json()).then(d => {vistas = d.subtipos; teaser()}).catch(() => {$('#teaser').hidden = true});

/* 5. ENTRADA DE FOTO: archivo, cámara del PC y enlace con el celular ======= */
/** Lo que devuelve analizarFoto cuando la API no encuentra un rostro en la foto (422). */
const SIN_ROSTRO = 'sin rostro';
/** Qué se le pide al usuario cuando no hay un rostro, según la ventana: la foto se descarta y no se pasa al resultado. */
const PIDE_OTRA = {
  home: 'No encontramos un rostro en la imagen. Sube otra foto de frente, con el rostro despejado y buena luz.',
  phone: 'No encontramos un rostro en la foto. Vuelve a tomarla: centra tu rostro en el óvalo, de frente y con buena luz.',
  pair: 'La foto que llegó del celular no muestra un rostro.'
};
/** Muestra un aviso (portada #hst, cámara #cerr o enlace #perr) y lo deja a la vista; con texto vacío lo oculta. */
const aviso = (sel, t) => {const e = $(sel); e.innerHTML = t ? '<svg class="ic"><use href="#i-alert"/></svg>' + esc(t) : ''; e.hidden = !t; if (t) e.scrollIntoView({block: 'center', behavior: 'smooth'})};
const avisoInicio = t => aviso('#hst', t);
const limpiarAvisos = () => ['#hst', '#cerr', '#perr'].forEach(s => aviso(s, ''));
$('#f2').onchange = async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  st.pair = st.local = 0;
  const error = await analizarFoto(f);
  if (error) avisoInicio(error === SIN_ROSTRO ? PIDE_OTRA.home : error);
};
$('#again').onclick = () => {wipe(); st.pair = st.local = 0; show('home')};

/** Enlace (PC): crea el código, dibuja el QR y pregunta cada segundo si llegó la foto del celular. */
async function abrirEnlace() {
  st.pair = 1; st.local = 0; $('#v-pair').insertBefore($('#hp'), $('#v-pair .row')); $('#hp').hidden = false; show('pair');
  $('#code').textContent = '······'; $('#qr').innerHTML = ''; estadoEnlace('Creando el código…', 0);
  try {
    await enlaceDeRed();
    const d = await (await api('/api/enlace', {method: 'POST'}, 10)).json();
    st.codigo = d.codigo; $('#code').textContent = d.codigo; qr(); sondear();
  } catch (e) {estadoEnlace(e.message, 0)}
}
const estadoEnlace = (t, vivo) => {$('#hs').textContent = t; $('.live').classList.toggle('off', !vivo)};
/** QR con el enlace de esta página + #tel-CÓDIGO, que abre la vista del celular. */
function qr() {$('#qr').innerHTML = ''; if (window.QRCode && st.codigo) new QRCode($('#qr'), {text: $('#lnk').value.trim() + '#tel-' + st.codigo, width: 220, height: 220})}
let lnkEditado = false;
$('#lnk').value = location.href.split('#')[0]; $('#lnk').oninput = () => {lnkEditado = true; qr()};
/** Si en el PC se abrió la página como "localhost", el QR usa la dirección de red del PC
    (en el celular, "localhost" apunta al propio celular). */
async function enlaceDeRed() {
  if (lnkEditado || !/^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)) return;
  try {
    const d = await (await api('/api/salud', {}, 8)).json();
    if (d.enlaces_red && d.enlaces_red.length) $('#lnk').value = d.enlaces_red[0];
    else toast('Este PC no tiene dirección en la red local: conéctalo al Wi-Fi o al hotspot y vuelve a intentarlo.');
  } catch (e) { /* se deja el enlace actual; se puede corregir a mano */ }
}
function detenerSondeo() {clearTimeout(st.sondeo); st.sondeo = null}
/** Pregunta por la foto cada segundo mientras la ventana de enlace esté abierta. */
function sondear() {
  detenerSondeo(); estadoEnlace('Esperando la foto del celular…', 1);
  const paso = async () => {
    if ($('#v-pair').hidden || !st.codigo) return;
    try {
      const r = await api(`/api/enlace/${st.codigo}/foto`, {}, 8);
      if (r.status === 200) {
        estadoEnlace('Foto recibida: analizando…', 1);
        const error = await analizarFoto(await r.blob());
        if (!error) { // el código ya cumplió: se cierra
          api(`/api/enlace/${st.codigo}`, {method: 'DELETE'}).catch(() => {}); st.codigo = ''; return;
        }
        aviso('#perr', (error === SIN_ROSTRO ? PIDE_OTRA.pair : error) + ' Toma otra foto con el celular: el mismo código sigue sirviendo.');
        estadoEnlace('Esperando la foto del celular…', 1);
      }
    } catch (e) {
      if (e.estado === 404) {estadoEnlace('El código venció; creando uno nuevo…', 0); abrirEnlace(); return}
      estadoEnlace(e.message, 0);
    }
    st.sondeo = setTimeout(paso, 1000);
  };
  st.sondeo = setTimeout(paso, 1000);
}
$('#phone').onclick = () => ask(abrirEnlace);
$('#pback').onclick = () => {detenerSondeo(); if (st.codigo) api(`/api/enlace/${st.codigo}`, {method: 'DELETE'}).catch(() => {}); st.codigo = ''; st.pair = 0; $('#v-home .card').after($('#hp')); $('#hp').hidden = true; show('home')};
$('#hascode').onclick = e => {e.preventDefault(); capUI(0); show('phone')};
// "Probar conexión": mide la latencia hasta la API.
$('#tc').onclick = async () => {const t0 = performance.now(); estadoEnlace('Probando conexión…', 1); try {const d = await (await api('/api/salud', {}, 8)).json(); estadoEnlace(`Conexión OK · ${Math.round(performance.now() - t0)} ms · servidor ${d.version}`, 1); if (st.codigo) sondear()} catch (e) {estadoEnlace(e.message, 0)}};

/* 6. CÁMARA EN VIVO Y GUÍA DE CAPTURA ====================================== */
/* Lo único que el navegador mide de la imagen es la luz y el encuadre, para habilitar
   el botón de captura. No interviene en el análisis de color. */
let pimg = null, stream = null, tm = null, turnoCamara = 0;
/** sRGB → L*a*b* aproximado (D65), solo para la guía de luz. */
const labGuia = (r, g, b) => {const f = v => (v /= 255) <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4, R = f(r), G = f(g), B = f(b), q = t => t > .008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116, X = q((R * .4124 + G * .3576 + B * .1805) / .95047), Y = q(R * .2126 + G * .7152 + B * .0722), Z = q((R * .0193 + G * .1192 + B * .9505) / 1.08883); return [116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)]};
/** Control de luz de un fotograma (región central): luminosidad, luz lateral y dominante cálida/fría. */
const chk0 = s => {const c = document.createElement('canvas'); c.width = 48; c.height = 64; const x = c.getContext('2d'); x.drawImage(s, 0, 0, 48, 64); const d = x.getImageData(12, 12, 24, 40).data; let R = 0, G = 0, B = 0, A = 0, Z = 0; const n = d.length / 4; for (let i = 0; i < d.length; i += 4) {R += d[i]; G += d[i + 1]; B += d[i + 2]; const q = labGuia(d[i], d[i + 1], d[i + 2])[0]; (i / 4 % 24 < 12 ? A += q : Z += q)} const l = labGuia(R / n, G / n, B / n), df = Math.abs(A - Z) / (n / 2); return l[0] < 35 ? {ok: 0, msg: 'Luz insuficiente: busca una luz más intensa y neutra.'} : l[0] > 88 ? {ok: 0, msg: 'Sobreexpuesta: evita el flash y la luz directa.'} : df > 16 ? {ok: 0, msg: 'Luz lateral o sombras marcadas: gira hacia la luz.'} : l[2] > 30 ? {ok: 0, msg: 'Luz muy cálida: busca una luz más neutra.'} : l[2] < 7 ? {ok: 0, msg: 'Luz muy fría: busca una luz más neutra.'} : {ok: 1, msg: 'Iluminación óptima'}};
/** Heurística de encuadre: ¿más de la mitad de la región central tiene tono de piel (YCbCr)? */
const skin = s => {const c = document.createElement('canvas'); c.width = 48; c.height = 64; const x = c.getContext('2d'); x.drawImage(s, 0, 0, 48, 64); const d = x.getImageData(12, 12, 24, 40).data; let K = 0; for (let i = 0; i < d.length; i += 4) {const b = 128 - .1687 * d[i] - .3313 * d[i + 1] + .5 * d[i + 2], r = 128 + .5 * d[i] - .4187 * d[i + 1] - .0813 * d[i + 2]; if (b > 77 && b < 127 && r > 133 && r < 173) K++} return K / (d.length / 4) > .5};
const chk = s => {const r = chk0(s); r.fr = skin(s); return r};
/** Pinta el aviso de luz, el tinte parpadeante, la línea de escaneo y el color de la guía. */
const ban = r => {const e = $('#plt'), cm = $('.cam'), live = !$('#vid').hidden, k = r.ok ? '' : /cálida/.test(r.msg) ? 'w' : /fría/.test(r.msg) ? 'c' : 'd'; e.hidden = false; e.className = 'lt ' + (r.ok ? 'ok' : 'warn'); e.innerHTML = '<svg class="ic"><use href="#i-' + (r.ok ? 'check' : 'alert') + '"/></svg>' + (r.ok ? (r.fr ? 'Iluminación óptima · rostro encuadrado' : 'Iluminación óptima · encuadra tu rostro en el óvalo') : esc(r.msg)); cm.className = 'cam ' + (live ? 'live ' + k : ''); $('.gd').classList.toggle('go', !!(r.ok && r.fr))};
/** Guarda la captura en memoria (JPEG ≤1280 px), la muestra y prepara el botón de envío. @param {HTMLCanvasElement} c */
async function setPhoto(c) {pimg = await aJpeg(c, .9); $('#pp').src = URL.createObjectURL(pimg); $('#pp').hidden = false; const r = chk(c); ban(r); $('#send').hidden = false; $('#send').textContent = (st.local ? 'Analizar esta foto' : 'Enviar a la pantalla') + (r.ok ? '' : ' de todos modos'); $('#pst').textContent = ''}
/** Apaga la cámara en vivo (y cancela una que se esté abriendo). */
function detenerCamara() {turnoCamara++; clearInterval(tm); stream && stream.getTracks().forEach(t => t.stop()); stream = null; $('#vid').hidden = true; $('#snap').hidden = true; $('#live').hidden = false}
/** Inicia la cámara frontal y evalúa un fotograma cada 500 ms; "Capturar" solo se habilita con buena luz. */
async function enVivo() {
  if (!window.isSecureContext || !navigator.mediaDevices) {$('#pst').textContent = 'La cámara en vivo necesita abrir la página con https. Usa la cámara del sistema (enlace de abajo).'; return}
  const turno = ++turnoCamara;
  try {
    const s = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user', width: {ideal: 1280}, height: {ideal: 1280}}});
    // Si mientras se abría la cámara el usuario eligió una foto (o salió), no se toma la pantalla.
    if (turno !== turnoCamara) {s.getTracks().forEach(t => t.stop()); return}
    stream = s;
    const v = $('#vid'); v.srcObject = stream; await v.play();
    if (turno !== turnoCamara) return;
    v.hidden = false; $('#live').hidden = true; $('#snap').hidden = false; $('#pp').hidden = $('#send').hidden = true;
    tm = setInterval(() => {if (!v.videoWidth) return; const r = chk(v); ban(r); $('#snap').disabled = !r.ok; $('#snap').classList.toggle('ready', !!(r.ok && r.fr))}, 500);
  } catch (e) {if (turno !== turnoCamara) return; detenerCamara(); $('#pst').textContent = 'El navegador no permitió abrir la cámara. Revisa el permiso o usa la cámara del sistema (enlace de abajo).'}
}
$('#live').onclick = () => ask(enVivo);
$('#snap').onclick = () => {const v = $('#vid'), c = reducir(v, v.videoWidth, v.videoHeight, 1280); detenerCamara(); setPhoto(c)};
// Foto desde la cámara del sistema (sobre todo en el celular).
$('#pf').onchange = async e => {const f = e.target.files[0]; e.target.value = ''; if (!f) return; detenerCamara(); pimg = null; $('#send').hidden = true; try {const im = await leerImagen(f); setPhoto(reducir(im, im.naturalWidth, im.naturalHeight, 1280))} catch (err) {toast(err.message)}};
/** Envío: en modo local se analiza aquí; en modo celular la foto viaja al PC por la API. */
$('#send').onclick = async () => {
  if (!pimg) return;
  if (st.local) {
    detenerCamara(); $('#pst').textContent = '';
    const error = await analizarFoto(pimg);
    if (error) { // la foto no sirve: se descarta, se pide otra y se reabre la cámara en vivo
      if ($('#pp').src.startsWith('blob:')) URL.revokeObjectURL($('#pp').src);
      pimg = null; $('#pp').removeAttribute('src'); $('#pp').hidden = $('#send').hidden = true;
      aviso('#cerr', error === SIN_ROSTRO ? PIDE_OTRA.phone : error); enVivo();
    }
    return;
  }
  const codigo = $('#pc').value.trim().toUpperCase();
  if (!codigo) {$('#pst').textContent = 'Escribe el código que aparece en la pantalla principal.'; return}
  $('#pst').innerHTML = '<span class="spin"></span> Enviando…'; $('#send').disabled = true;
  const fd = new FormData(); fd.append('foto', pimg, 'foto.jpg');
  try {await api(`/api/enlace/${codigo}/foto`, {method: 'PUT', body: fd}, 20); $('#pst').textContent = '¡Listo! Mira la pantalla principal.'; pimg = null; $('#pp').hidden = true; $('#send').hidden = true}
  catch (e) {$('#pst').textContent = e.estado === 404 ? 'El código no existe o venció. Revisa la pantalla principal.' : e.message}
  finally {$('#send').disabled = false}
};
/** Adapta la ventana de captura: modo local (l=1, cámara del PC) o modo celular (l=0, con código). */
const capUI = l => {st.local = l; $('#v-phone h1').textContent = l ? 'Captura inteligente' : 'Tu foto para Prisma'; $('#v-phone .lead').textContent = l ? 'Alinea tus ojos y labios con las guías. Cuando la luz sea óptima, podrás capturar.' : 'Toma una selfie con luz lo más neutra posible y envíala a la pantalla principal.'; [$('#pc'), $('#pc').previousElementSibling].forEach(e => e.hidden = !!l); $('#send').hidden = true; $('#pp').hidden = true; $('#pst').textContent = ''};
$('#take').onclick = () => ask(() => {st.pair = 0; capUI(1); show('phone'); enVivo()});
$('#cback').onclick = e => {e.preventDefault(); detenerCamara(); wipe(); st.pair = st.local = 0; show('home')};

/* 7. ANÁLISIS AUTOMÁTICO ================================================= */
/** Pantalla de carga con mensajes mientras el computador trabaja. @param {boolean} on @param {HTMLCanvasElement} [lienzo] la foto */
function cargando(on, lienzo) {
  const o = $('#ld'), b = $('#lb'), t = $('#lt2'), M = ['Corrigiendo la iluminación…', 'Ubicando tu rostro y tu cabello…', 'Midiendo tus colores en CIELAB…', 'Calculando tu afinidad estacional…', 'Generando tus combinaciones ideales…'];
  clearInterval(cargando.t);
  // Mientras el PC analiza, la página de fondo (con sus animaciones bajo el desenfoque) se
  // oculta: dibujarla le quita procesador al análisis cuando el PC es a la vez servidor y pantalla.
  document.body.classList.toggle('ocupado', on);
  if (!on) {b.style.width = '100%'; setTimeout(() => o.hidden = true, 200); return}
  const r = $('#roi'), k = Math.min(1, 240 / lienzo.width); r.width = lienzo.width * k; r.height = lienzo.height * k; r.getContext('2d').drawImage(lienzo, 0, 0, r.width, r.height);
  o.hidden = false; let p = 0, i = 0; b.style.width = '4%'; t.textContent = M[0];
  cargando.t = setInterval(() => {p += (92 - p) * .12; b.style.width = p + '%'; if (++i % 6 === 0) t.textContent = M[Math.min(M.length - 1, i / 6 | 0)]}, 150);
}
/**
 * Analiza una foto (archivo o blob): la reduce a ≤1280 px, la envía a /api/analizar y, si todo
 * sale bien, muestra el resultado. El computador ubica y mide solo las zonas del rostro.
 * @param {Blob} fuente La foto.
 * @returns {Promise<string>} Vacío si se mostró el resultado; SIN_ROSTRO si la foto no tiene un
 *   rostro (no se pasa al resultado y se pide otra); si no, el mensaje de error para mostrarlo
 *   donde está el usuario (portada, cámara o enlace con el celular).
 */
async function analizarFoto(fuente) {
  limpiarAvisos();
  let im;
  try {im = await leerImagen(fuente)} catch (e) {return e.message}
  const lienzo = reducir(im, im.naturalWidth, im.naturalHeight, 1280), foto = await aJpeg(lienzo, .9);
  const fd = new FormData(); fd.append('foto', foto, 'foto.jpg');
  cargando(true, lienzo);
  let res;
  try {res = await (await api('/api/analizar', {method: 'POST', body: fd}, 45)).json()}
  catch (e) {cargando(false); return e.estado === 422 && /rostro/.test(e.message) ? SIN_ROSTRO : e.message}
  cargando(false); wipe();
  Object.assign(st, {foto, url: URL.createObjectURL(foto), res, dr: 'best', di: 0, dj: 0});
  show('result'); render();
  return '';
}

/* 9. RESULTADOS ============================================================= */
/** Muestra de color; `x` agrega un prefijo (la estrella de los colores estrella). */
const sw = (c, x = '') => `<div class="sw"><i style="background:${c.hex}"></i><span>${x}${esc(c.nombre)}<small>${c.hex}</small></span></div>`, grid = (a, x) => `<div class="sws">${a.map(c => sw(c, x && x(c))).join('')}</div>`;
/** Cuatro palabras que describen el subtipo (de los parámetros de su paleta, como el informe),
    en el género de la estación: Primavera Cálida, Verano Frío. */
const palabras = (p, estacion) => {
  const g = (m, f) => estacion === 'primavera' ? f : m;
  return [p.calidez > .25 ? g('Cálido', 'Cálida') : p.calidez < -.25 ? g('Frío', 'Fría') : g('Neutro', 'Neutra'),
    p.profundidad < .3 ? g('Claro', 'Clara') : p.profundidad > .7 ? g('Profundo', 'Profunda') : g('Medio', 'Media'),
    p.claridad < .3 ? 'Suave' : p.claridad > .7 ? 'Brillante' : g('Equilibrado', 'Equilibrada'),
    p.contraste < .34 ? 'Contraste bajo' : p.contraste < .67 ? 'Contraste medio' : 'Contraste alto'];
};
/** Anillo de probabilidad animado hasta p %. */
function gauge(p) {const f = $('#gfg'), v = $('#gv'), a = +v.dataset.p || 0; v.dataset.p = p; setTimeout(() => {f.style.strokeDashoffset = 326.7 * (1 - p / 100); const t0 = performance.now(); (function s(t) {const k = Math.min(1, (t - t0) / 1400); v.textContent = Math.round(a + (p - a) * k) + '%'; if (k < 1) requestAnimationFrame(s)})(t0)}, 80)}
/** Dibuja toda la ventana de resultados a partir del resultado oficial. */
function render() {
  const r = st.res, g = r.paleta, pr = pct(r.estacion.probabilidad), h = $('.hero');
  $('#af').textContent = `${pr} % de probabilidad de ${r.estacion.nombre}`; gauge(pr);
  h.style.setProperty('--g1', g.tema[0]); h.style.setProperty('--g2', g.tema[1]);
  h.style.background = `linear-gradient(135deg,${g.favorecedores[0].hex}44,${g.favorecedores[5].hex}33 55%,${g.favorecedores[10].hex}44)`;
  $('#sn').textContent = r.subtipo.nombre; $('#tg').textContent = r.subtipo.lema;
  $('#strip').innerHTML = g.favorecedores.map(c => `<i style="background:${c.hex}" title="${esc(c.nombre)}"></i>`).join('');
  $('#chp').innerHTML = palabras(g.parametros, g.estacion).map(t => `<span>${t}</span>`).join('');
  $('#avt').src = $('#dimg').src = st.url;
  // Colores que midió el computador en cada zona, y avisos de la medición (luz, cabello...).
  $('#med').innerHTML = ZONAS.filter(n => r.zonas[n]).map(n => {const q = r.zonas[n]; return `<div class="chip"><i style="background:${q.hex}"></i><span>${esc(q.nombre)}<small>${q.hex}${q.fuente === 'cejas' ? ' · estimado con las cejas' : ''}</small></span></div>`}).join('');
  $('#avs').innerHTML = r.avisos.map(a => `<li>${esc(a)}</li>`).join('');
  // Perfil por ejes: el marcador parte del centro y se anima hasta su percentil.
  const tr = {temperatura: 't1', profundidad: 't2', croma: 't3', contraste: 't2'};
  $('#az').innerHTML = EJES.map(k => {const e = r.ejes[k]; return `<div class="trk"><div class="th"><b>${esc(e.nombre)}</b><small>percentil ${pct(e.valor)}</small></div><div class="tr ${tr[k]}"><i data-v="${pct(e.valor)}" style="left:50%"></i></div><div class="te"><span>${esc(e.bajo)}</span><span>${esc(e.alto)}</span></div></div>`}).join('');
  setTimeout(() => $$('#az .tr i').forEach(e => e.style.left = e.dataset.v + '%'), 80);
  // Probabilidades de las cuatro estaciones y avisos de ambigüedad.
  const P = r.probabilidades;
  $('#nb').innerHTML = P.map(p => `<div class="nb${p.clave === r.estacion.clave ? ' on' : ''}"><span>${esc(p.nombre)}</span><b>${pct(p.probabilidad)}%</b><i style="width:${pct(p.probabilidad)}%"></i></div>`).join('');
  $('#nt').textContent = (P[0].probabilidad - P[1].probabilidad < .08 ? `Estás entre ${P[0].nombre} y ${P[1].nombre}: las dos son casi igual de probables. ` : '') + (P[0].probabilidad < .35 ? 'La probabilidad es moderada: repite con una luz más neutra, sin maquillaje, o confírmalo con un asesor.' : '');
  const m = r.modelo || {};
  const como = m.metodo === 'validacion' ? 'en el subconjunto de validación' : 'en validación cruzada';
  $('#prec').textContent = m.exactitud != null ? `Precisión del modelo ${como}: acierta la estación en el ${pct(m.exactitud)} % de los casos (IC 95 %: ${pct(m.ic95[0])}-${pct(m.ic95[1])} %) y la acertada está entre las dos más probables en el ${pct(m.top2)} % (n = ${m.n}).` : '';
  // Pestañas: mejores colores / evitar / maquillaje / joyería.
  $('#t-c').innerHTML = `<h3>Colores de acento</h3>${grid(g.favorecedores, c => c.estrella ? '<svg class="ic s st"><use href="#i-star"/></svg> ' : '')}<h3>Neutros base</h3>${grid(g.neutros)}<ul class="rules"><li>Colores estrella: úsalos cerca del rostro.</li><li>Estampados: si el 75 % o más de sus colores está en tu paleta, adelante.</li><li>${esc(g.consejo)}</li></ul>`;
  $('#t-a').innerHTML = `<h3>Mejor evitar cerca del rostro</h3>${grid(g.evitar)}<p class="sm">Úsalos lejos del rostro: restan luz a tu piel y endurecen tus rasgos.</p>`;
  const mq = g.maquillaje;
  $('#t-b').innerHTML = `<h3>Base</h3>${grid(mq.base)}<p class="sm">Calculada con el color medido de tu piel; pruébala en la mandíbula con luz neutra.</p><h3>Rubor</h3>${grid(mq.rubor)}<h3>Sombras</h3>${grid(mq.sombras)}<h3>Labial</h3>${grid(mq.labial)}<h3>Cabello</h3>${grid(g.cabello.tonos)}<p>${esc(mayuscula(g.cabello.consejos.join('; ')))}.</p>`;
  $('#t-j').innerHTML = `<h3>Metales que te favorecen</h3><div class="mts">${g.metales.map(x => `<div class="mt" style="--m1:${x.degradado[0]};--m2:${x.degradado[1]}"><i class="sph"><svg class="ic"><use href="#i-ring"/></svg></i><div><b>${esc(x.nombre)}</b><small><svg class="ic s"><use href="#i-ring"/></svg> <svg class="ic s"><use href="#i-neck"/></svg> Anillos y collares</small></div><em class="f2">Te favorece</em></div>`).join('')}</div>`;
  drape();
}
/** "Prueba en tu rostro": pinta el fondo del retrato con el color elegido. */
function drape() {
  if (st.dr === 'ab') return ab();
  const g = st.res.paleta, L = {best: g.favorecedores, neu: g.neutros, avoid: g.evitar}[st.dr], v = {best: 'Te favorece', neu: 'Neutro útil', avoid: 'Te favorece menos'}[st.dr], c = L[st.di] || L[0];
  $('#dr').innerHTML = L.map((q, i) => `<button class="dot" style="background:${q.hex}" data-i="${i}" aria-label="${esc(q.nombre)}"></button>`).join('');
  $('#dbox').style.background = c.hex; $('#dcap').textContent = `${c.nombre} · ${c.hex} · ${v}`;
}
/** Comparador A/B: mitad izquierda un color favorecedor, derecha uno a evitar. */
function ab() {const g = st.res.paleta, A = g.favorecedores[st.di] || g.favorecedores[0], B = g.evitar[st.dj] || g.evitar[0], d = (a, s) => `<div class="dots" style="flex:1 1 100%">${a.map((q, i) => `<button class="dot" style="background:${q.hex}" data-s="${s}" data-i="${i}" aria-label="${esc(q.nombre)}"></button>`).join('')}</div>`; $('#dr').innerHTML = d(g.favorecedores, 'a') + d(g.evitar, 'b'); $('#dbox').style.background = `linear-gradient(90deg,${A.hex} 50%,${B.hex} 50%)`; $('#dcap').textContent = `Mejor: ${A.nombre} · Evitar: ${B.nombre}`}
$('#dr').onclick = e => {const b = e.target.closest('.dot'); if (b) {st[b.dataset.s === 'b' ? 'dj' : 'di'] = +b.dataset.i; drape()}};
$$('[data-f]').forEach(b => b.onclick = () => {$$('[data-f]').forEach(x => x.classList.toggle('on', x === b)); st.dr = b.dataset.f; st.di = st.dj = 0; drape()});
$$('[data-t]').forEach(b => b.onclick = () => {$$('[data-t]').forEach(x => x.classList.toggle('on', x === b)); ['p', 'c', 'a', 'b', 'j'].forEach(k => $('#t-' + k).hidden = !(k === b.dataset.t || (k === 'p' && b.dataset.t === 'c')))});

/* 10. INFORME PDF Y CORREO ================================================= */
const correoValido = t => /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(t.trim());
/** Abre el modal del informe en modo "descargar" o "enviar". */
function abrirInforme(modo) {
  const enviar = modo === 'enviar';
  $('#rvt').textContent = enviar ? 'Enviar el informe a tu correo' : 'Descargar el informe en PDF';
  $('#rvm').hidden = !enviar; $('#rvsnd').hidden = !enviar; $('#rvd').hidden = enviar; $('#rvfm').hidden = !enviar;
  $('#rve').value = $('#em').value.trim(); $('#rva').checked = false; $('#rvf').checked = false; $('#rvs').textContent = ''; $('#rvs').className = 'est';
  actualizarEnvio(); $('#rv').hidden = false;
}
const actualizarEnvio = () => $('#rvsnd').disabled = !($('#rva').checked && correoValido($('#rve').value));
$('#rva').onchange = $('#rve').oninput = actualizarEnvio;
$('#dl').onclick = () => abrirInforme('descargar'); $('#mail').onclick = () => abrirInforme('enviar');
$('#rvx').onclick = () => $('#rv').hidden = true; $('#rv').onclick = e => {if (e.target.id === 'rv') $('#rv').hidden = true};
/** Datos del formulario del informe: el resultado oficial y, solo si se autorizó, la foto. */
function datosInforme(accion) {
  const fd = new FormData(); fd.append('resultado', JSON.stringify(st.res)); fd.append('accion', accion);
  if ($('#rvf').checked) {fd.append('incluir_foto', 'true'); fd.append('foto', st.foto, 'foto.jpg')}
  return fd;
}
const estado = (t, c = '') => {$('#rvs').textContent = t; $('#rvs').className = 'est ' + c};
$('#rvd').onclick = async () => {
  estado('Generando el PDF…');
  try {
    const b = await (await api('/api/informe', {method: 'POST', body: datosInforme('descargar')}, 30)).blob();
    const a = document.createElement('a'), u = URL.createObjectURL(b);
    a.href = u; a.download = `prisma-informe-${st.res.subtipo.clave.replace(/_/g, '-')}.pdf`; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(u), 10000); estado('Listo: el PDF quedó en tus descargas.', 'ok');
  } catch (e) {estado(e.message, 'err')}
};
$('#rvsnd').onclick = async () => {
  const fd = datosInforme('enviar'); fd.append('correo', $('#rve').value.trim()); fd.append('acepta_envio', 'true');
  estado('Enviando el informe…'); $('#rvsnd').disabled = true;
  try {await api('/api/informe', {method: 'POST', body: fd}, 45); estado('Enviado. Revisa tu correo (y la carpeta de spam).', 'ok'); $('#em').value = $('#rve').value.trim()}
  catch (e) {estado(e.message, 'err')} finally {actualizarEnvio()}
};

/* 11. ARRANQUE ============================================================== */
/** Si la URL trae #tel-CÓDIGO, abre la vista del celular con ese código; si no, el inicio. */
const m = location.hash.match(/^#tel-([A-Za-z0-9]{4,8})/);
if (m) {capUI(0); $('#pc').value = m[1].toUpperCase(); show('phone')} else show('home');
