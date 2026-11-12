# ------------------------------------------------------------
# PIANO VIRTUAL PRO (paso 3)
#
# Novedades respecto al paso 2:
#  * SIN LINEA NARANJA: la tecla suena cuando un dedo apoyado se levanta y
#    cae de golpe a la mesa. El detector mira cada dedo por separado,
#    se calcula respecto a tu mano (da igual donde la muevas) y es
#    independiente del tamano de la mano y de lo lejos que este de la camara.
#  * Sonido de piano de verdad (armonicos, ataque, caida, dos cuerdas,
#    estereo) y la nota se corta suave al levantar el dedo.
#  * Volumen segun la fuerza con la que pulsas.
#  * El piano colocado se guarda (piano_config.json) y se carga solo.
#  * Modo CANCION: te marca la tecla siguiente y avanza cuando la aciertas.
#  * Camara en un hilo aparte (menos retraso) y piano mas bonito.
#
# 1) Colocar el piano: con el dedo indice apuntas a la ESQUINA SUPERIOR
#    IZQUIERDA de donde quieras el piano y lo mantienes quieto 2 segundos.
#    Luego igual con la ESQUINA INFERIOR DERECHA.
# 2) Tocar: pon las manos en posicion de pianista, con los dedos apoyados en
#    la mesa. Para tocar una nota, LEVANTA ese dedo y dejalo CAER sobre la
#    tecla (como golpear la mesa). Suena al caer. Se detecta cada dedo,
#    tambien con las dos manos. Mientras el dedo sigue apoyado la tecla
#    sigue pulsada, y se suelta cuando lo vuelves a levantar.
#
# Teclas:
#   R = colocar el piano otra vez     D = piano por defecto
#   S = cambiar de cancion (guia)     L = nombres en todas las teclas
#   H = ver/ocultar las manos         B = modo depuracion (ver medidas)
#   + / - = sensibilidad              ESC = salir
# ------------------------------------------------------------
import os
import json
import math
import time
import threading
import urllib.request
from collections import deque

import numpy as np
import cv2
import pygame
import mediapipe as mp

# ------------------------------------------------------------
# Ajustes (puedes tocarlos)
# ------------------------------------------------------------
CARPETA = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CARPETA, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
CONFIG_PATH = os.path.join(CARPETA, "piano_config.json")

CAM_ANCHO, CAM_ALTO = 1280, 720   # resolucion que se pide a la camara
OPACIDAD_PIANO = 0.60             # 0 = invisible, 1 = opaco

# Detector de pulsacion (todo medido en "tamanos de mano": distancia muneca-nudillo)
# Gesto: dedo apoyado en la mesa -> se levanta -> CAE de golpe -> suena la nota
UMBRAL_LEVANTA = 0.04    # cuanto tiene que separarse el dedo de la mesa para contar como "levantado"
UMBRAL_BAJADA = 0.50     # velocidad minima de la caida (tamanos de mano por segundo)
T_REPOSO = 0.20          # segundos que el dedo debe estar apoyado antes de levantarse
T_CAIDA = 0.35           # la caida debe ocurrir en menos de esto desde el punto mas alto
VEL_MANO_MAX = 1.5       # si la mano entera viaja mas rapido, no se cuentan pulsaciones

# ------------------------------------------------------------
# Notas
# ------------------------------------------------------------
NOMBRES = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]
N_OCTAVAS = 3
N_BLANCAS = 7 * N_OCTAVAS
N_NOTAS = 12 * N_OCTAVAS                       # 3 octavas
FRECUENCIAS = [261.63 * 2 ** (n / 12) for n in range(N_NOTAS)]
BLANCAS_OCTAVA = [0, 2, 4, 5, 7, 9, 11]
NEGRA_TRAS_POS = (0, 1, 3, 4, 5)               # hay negra despues de DO RE FA SOL LA

PUNTAS = [4, 8, 12, 16, 20]                    # pulgar, indice, corazon, anular, menique
BASES = [2, 5, 9, 13, 17]                      # la articulacion de la que sale cada dedo
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# Canciones: cada numero es la nota (0 = DO central, 12 = DO de arriba...)
CANCIONES = [
    ("Himno a la alegria", [4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 4, 2, 2,
                            4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 2, 0, 0]),
    ("Estrellita", [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0,
                    7, 7, 5, 5, 4, 4, 2, 7, 7, 5, 5, 4, 4, 2,
                    0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0]),
    ("Cumpleanos feliz", [7, 7, 9, 7, 12, 11, 7, 7, 9, 7, 14, 12,
                          7, 7, 19, 16, 12, 11, 9, 17, 17, 16, 12, 14, 12]),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
ORO = (0, 200, 255)


# ------------------------------------------------------------
# Sonido
# ------------------------------------------------------------
SR = 44100


def sintetizar_nota(frecuencia, pan, duracion=3.0):
    """Sonido de piano por sintesis aditiva: suma de armonicos que se apagan a
    distinta velocidad (los agudos antes), dos cuerdas casi iguales por nota
    (da el brillo caracteristico), un golpe de martillo al principio y estereo."""
    n = int(SR * duracion)
    t = np.arange(n) / SR
    onda = np.zeros(n)
    caida = 0.9 + frecuencia / 450.0                # las notas agudas se apagan antes
    for k in range(1, 12):
        fk = frecuencia * k * math.sqrt(1 + 0.0004 * k * k)   # leve inarmonicidad
        if fk > 9000:
            break
        amp = 1.0 / (k ** 1.15)
        envolvente = np.exp(-t * caida * (1 + 0.55 * (k - 1)))
        cuerdas = np.sin(2 * np.pi * fk * t) + np.sin(2 * np.pi * fk * 1.0006 * t + k)
        onda += 0.5 * amp * cuerdas * envolvente
    rng = np.random.default_rng(int(frecuencia * 100))
    onda += rng.standard_normal(n) * np.exp(-t * 220) * 0.04      # golpe del martillo
    onda *= np.minimum(1.0, t / 0.004)                            # ataque de 4 ms
    onda *= np.minimum(1.0, (duracion - t) / 0.08)                # final sin click
    onda = onda / np.max(np.abs(onda)) * 0.55
    izq, der = math.cos(pan * math.pi / 2), math.sin(pan * math.pi / 2)
    estereo = np.stack([onda * izq, onda * der], axis=1) * 1.3
    return np.ascontiguousarray((estereo * 32767).astype(np.int16))


class Audio:
    def __init__(self):
        pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=512, allowedchanges=0)
        pygame.mixer.set_num_channels(32)
        print("Generando sonidos...")
        self.sonidos = [pygame.sndarray.make_sound(sintetizar_nota(f, 0.25 + 0.5 * i / (N_NOTAS - 1)))
                        for i, f in enumerate(FRECUENCIAS)]
        self.canales = {}

    def tocar(self, nota, volumen):
        self.soltar(nota, 40)
        canal = self.sonidos[nota].play()
        if canal is not None:
            canal.set_volume(volumen)
            self.canales[nota] = (canal, time.monotonic())

    def soltar(self, nota, ms=220):
        """Al levantar el dedo la nota se apaga suavemente (como el apagador del piano)."""
        dato = self.canales.pop(nota, None)
        if dato is not None and time.monotonic() - dato[1] < 2.9:
            dato[0].fadeout(ms)

    def soltar_todo(self):
        for nota in list(self.canales):
            self.soltar(nota, 100)


# ------------------------------------------------------------
# Piano (geometria + dibujo)
# ------------------------------------------------------------
def mezclar(a, b, k):
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))


def color_nota(n):
    hsv = np.uint8([[[(95 + n * 6) % 180, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


COLORES = [color_nota(n) for n in range(N_NOTAS)]


def texto(frame, t, x, y, tam=0.6, color=(255, 255, 255), grosor=1):
    cv2.putText(frame, t, (x + 1, y + 1), FONT, tam, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, t, (x, y), FONT, tam, color, grosor, cv2.LINE_AA)


class Piano:
    def __init__(self, x1, y1, x2, y2, ancho_frame, alto_frame):
        self.x1 = int(max(0, min(x1, x2)))
        self.x2 = int(min(ancho_frame, max(x1, x2)))
        self.y1 = int(max(0, min(y1, y2)))
        self.y2 = int(min(alto_frame, max(y1, y2)))
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.y_negra = self.y1 + int(self.h * 0.6)       # las negras son mas cortas
        aw = self.w / N_BLANCAS
        self.blancas, self.negras = [], []
        for i in range(N_BLANCAS):
            pos = i % 7
            nota = (i // 7) * 12 + BLANCAS_OCTAVA[pos]
            self.blancas.append((nota, self.x1 + i * aw, self.x1 + (i + 1) * aw))
            if pos in NEGRA_TRAS_POS:
                c = self.x1 + (i + 1) * aw
                self.negras.append((nota + 1, c - aw * 0.3, c + aw * 0.3))
        self.base = self._pintar_base()

    def rect(self):
        return self.x1, self.y1, self.x2, self.y2

    def _pintar_base(self):
        img = np.full((self.h, self.w, 3), 245, np.uint8)
        yn = self.y_negra - self.y1
        for _, xa, xb in self.blancas:
            cv2.rectangle(img, (int(xa - self.x1), 0), (int(xb - self.x1) - 1, self.h - 1), (80, 80, 80), 1)
        for _, xa, xb in self.negras:
            cv2.rectangle(img, (int(xa - self.x1), 0), (int(xb - self.x1), yn), (28, 28, 28), -1)
            cv2.line(img, (int(xa - self.x1) + 2, 0), (int(xa - self.x1) + 2, yn - 3), (85, 85, 85), 1)
        return img

    def tecla_en(self, x, y):
        """Que tecla hay en el punto (x, y)? Si no hay ninguna, None."""
        if not (self.x1 <= x < self.x2 and self.y1 - 0.2 * self.h <= y <= self.y2 + 0.1 * self.h):
            return None
        if y < self.y_negra:
            for nota, xa, xb in self.negras:
                if xa <= x < xb:
                    return nota
        for nota, xa, xb in self.blancas:
            if xa <= x < xb:
                return nota
        return None

    def dibujar(self, frame, brillo, objetivo, t, etiquetas):
        """brillo: {nota: 0..1} intensidad de cada tecla. objetivo: tecla guia (cancion)."""
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        capa = self.base.copy()
        yn = self.y_negra - y1
        pulso = 0.45 + 0.25 * math.sin(t * 6)

        def intensidad(nota):
            return brillo.get(nota, 0.0)

        for nota, xa, xb in self.blancas:
            k = intensidad(nota)
            col = None
            if k > 0.02:
                col = mezclar((245, 245, 245), COLORES[nota], k)
            elif nota == objetivo:
                col = mezclar((245, 245, 245), ORO, pulso)
            if col is not None:
                cv2.rectangle(capa, (int(xa - x1) + 1, 0), (int(xb - x1) - 2, self.h - 2), col, -1)
        for nota, xa, xb in self.negras:
            k = intensidad(nota)
            col = (28, 28, 28)
            if k > 0.02:
                col = mezclar(col, COLORES[nota], k)
            elif nota == objetivo:
                col = mezclar(col, ORO, pulso + 0.2)
            cv2.rectangle(capa, (int(xa - x1), 0), (int(xb - x1), yn), col, -1)

        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(capa, OPACIDAD_PIANO, roi, 1 - OPACIDAD_PIANO, 0)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (60, 60, 60), 2, cv2.LINE_AA)

        for nota, xa, xb in self.blancas:
            if nota % 12 == 0 or etiquetas:
                nombre = NOMBRES[nota % 12]
                if nota % 12 == 0:
                    nombre += str(4 + nota // 12)
                cv2.putText(frame, nombre, (int(xa) + 4, y2 - 8), FONT, 0.4, (40, 40, 40), 1, cv2.LINE_AA)


# ------------------------------------------------------------
# Detector de pulsaciones
# ------------------------------------------------------------
class Dedo:
    """Un dedo tiene 3 estados:
       REPOSO    = apoyado en la mesa (posicion de pianista)
       LEVANTADO = separado de la mesa
       ABAJO     = tecla pulsada (el dedo ha caido y sigue apoyado)
    La nota suena en el momento en que un dedo que estaba apoyado, se levanta y
    CAE de golpe de vuelta a la mesa. Para saber si un dedo esta "apoyado" se
    guarda su posicion de reposo (base) y se mide cuanto se aleja de ella (dev)."""

    def __init__(self):
        self.hist = deque(maxlen=24)   # (tiempo, dev, lateral)
        self.base = None               # medida del dedo cuando esta apoyado
        self.estado = "REPOSO"
        self.quieto_desde = None       # desde cuando esta quieto sobre la mesa
        self.armado = False            # llevaba suficiente rato apoyado antes de levantarse
        self.pico = 0.0                # lo maximo que se ha separado
        self.t_pico = 0.0
        self.t_lev = 0.0
        self.nota = None
        self.t_press = 0.0
        self.dev = 0.0                 # para depurar
        self.vret = 0.0

    @property
    def abajo(self):
        return self.estado == "ABAJO"

    def _serie(self, t, ventana):
        s = 1.0 if self.pico >= 0 else -1.0
        return [(ti, d * s, la) for ti, d, la in self.hist if t - ti <= ventana]

    def vel_retorno(self, t):
        """Velocidad maxima con la que el dedo ha vuelto hacia la mesa."""
        datos = self._serie(t, T_CAIDA + 0.05)
        v = 0.0
        for (t0, a, _), (t1, b, _) in zip(datos, datos[1:]):
            v = max(v, (a - b) / max(t1 - t0, 0.025))
        return v

    def lateral(self, t):
        datos = self._serie(t, T_CAIDA + 0.1)
        if not datos:
            return 0.0
        return max(d[2] for d in datos) - min(d[2] for d in datos)


class Mano:
    def __init__(self):
        self.dedos = [Dedo() for _ in range(5)]
        self.palma = None
        self.t = 0.0
        self.vel = 0.0
        self.pts = None


class Detector:
    def __init__(self):
        self.manos = {}
        self.siguiente_id = 0
        self.sens = 1.0

    def reset(self):
        self.manos = {}

    def _emparejar(self, t, lista):
        """Une cada mano detectada con la del frame anterior mas cercana (asi no
        dependemos de que MediaPipe diga siempre 'izquierda' o 'derecha')."""
        pares = []
        for i, (pts, palma, tam) in enumerate(lista):
            for mid, m in self.manos.items():
                d = float(np.linalg.norm(palma - m.palma))
                if d < 2.5 * tam:
                    pares.append((d, i, mid))
        pares.sort()
        usadas_i, usadas_m, res = set(), set(), {}
        for d, i, mid in pares:
            if i in usadas_i or mid in usadas_m:
                continue
            usadas_i.add(i)
            usadas_m.add(mid)
            res[i] = mid
        for i in range(len(lista)):
            if i not in res:
                res[i] = self.siguiente_id
                self.manos[self.siguiente_id] = Mano()
                self.siguiente_id += 1
        return res

    def procesar(self, t, manos_px, piano):
        """manos_px: lista de arrays 21x2 (pixeles). Devuelve (eventos, notas_pulsadas).
        Eventos: ('on', nota, volumen, dedo) y ('off', nota)."""
        eventos = []
        umbral = UMBRAL_LEVANTA / self.sens
        vmin = UMBRAL_BAJADA / self.sens

        lista = []
        for pts in manos_px:
            tam = float(np.linalg.norm(pts[9] - pts[0]))
            if tam < 15:
                continue
            lista.append((pts, pts[[0, 5, 9, 13, 17]].mean(axis=0), tam))
        asignacion = self._emparejar(t, lista)

        vistas = set()
        for i, (pts, palma, tam) in enumerate(lista):
            mid = asignacion[i]
            vistas.add(mid)
            mano = self.manos[mid]
            dt_m = max(t - mano.t, 0.025) if mano.palma is not None else 0.033
            if mano.palma is not None and t > mano.t:
                mano.vel = float(np.linalg.norm(palma - mano.palma)) / tam / dt_m
            mano.palma, mano.t, mano.pts = palma, t, pts
            eje = (pts[9] - pts[0]) / tam
            perp = np.array([-eje[1], eje[0]])
            movil = mano.vel >= VEL_MANO_MAX

            # Medida de cada dedo: cuanto "sale" desde su base en la direccion de la mano
            # (el pulgar: longitud aparente). Al levantar o bajar el dedo esta medida cambia.
            es, lats = [], []
            for f in range(5):
                v = pts[PUNTAS[f]] - pts[BASES[f]]
                es.append((float(np.linalg.norm(v)) if f == 0 else float(np.dot(v, eje))) / tam)
                lats.append(float(np.dot(v, perp)) / tam)
                if mano.dedos[f].base is None:
                    mano.dedos[f].base = es[f]
            raw = [es[f] - mano.dedos[f].base for f in range(5)]
            comun = float(np.median(raw[1:]))      # lo que cambia toda la mano (se descuenta)
            devs = [raw[0]] + [r - comun for r in raw[1:]]
            n_mov = 0
            for f in range(5):
                mano.dedos[f].hist.append((t, devs[f], lats[f]))
                mano.dedos[f].dev = devs[f]
                if f > 0 and abs(devs[f]) >= 0.6 * umbral:
                    n_mov += 1

            candidatos = []
            for f in range(5):
                dedo = mano.dedos[f]
                d = devs[f]
                L = umbral * (1.2 if f == 0 else 1.0)

                if dedo.estado == "ABAJO":
                    # la tecla se suelta cuando el dedo vuelve a levantarse
                    if (t - dedo.t_press > 0.08 and abs(d) >= 0.6 * L) or t - dedo.t_press > 3.0:
                        eventos.append(("off", dedo.nota))
                        dedo.nota = None
                        dedo.estado = "REPOSO"
                        dedo.quieto_desde = t - 1.0       # ya estaba apoyado: listo para otra
                    continue

                if dedo.estado == "REPOSO":
                    if abs(d) < 0.5 * L:
                        if dedo.quieto_desde is None:
                            dedo.quieto_desde = t
                        dedo.base += min(1.0, dt_m / 0.8) * d   # sigue cambios lentos de la mano
                    else:
                        if dedo.quieto_desde is not None:
                            dedo.armado = (t - dedo.quieto_desde) >= T_REPOSO
                            dedo.quieto_desde = None
                        if abs(d) >= L and dedo.armado and not movil:
                            dedo.estado = "LEVANTADO"
                            dedo.pico, dedo.t_pico, dedo.t_lev = d, t, t
                    continue

                # LEVANTADO
                s = 1.0 if dedo.pico >= 0 else -1.0
                cur = d * s
                if cur > abs(dedo.pico):
                    dedo.pico, dedo.t_pico = d, t
                if t - dedo.t_lev > 1.2:          # lleva mucho en el aire: nueva posicion de reposo
                    dedo.base += d
                    dedo.estado, dedo.quieto_desde, dedo.armado = "REPOSO", None, False
                    continue
                if cur <= min(0.35 * abs(dedo.pico), 0.45 * L):      # ha vuelto a la mesa
                    dedo.vret = dedo.vel_retorno(t)
                    if (t - dedo.t_pico <= T_CAIDA and dedo.vret >= vmin and not movil
                            and dedo.lateral(t) <= 0.6 * abs(dedo.pico) + 0.03):
                        candidatos.append((abs(dedo.pico), f))
                    else:                                             # bajada lenta: no suena
                        dedo.estado, dedo.quieto_desde, dedo.armado = "REPOSO", None, False

            if n_mov >= 3:       # media mano moviendose a la vez: no es una pulsacion
                for _, f in candidatos:
                    mano.dedos[f].estado = "REPOSO"
                    mano.dedos[f].quieto_desde, mano.dedos[f].armado = None, False
                candidatos = []
            if candidatos:
                mejor = max(c for c, _ in candidatos)
                for amp, f in candidatos:
                    dedo = mano.dedos[f]
                    x, y = pts[PUNTAS[f]]
                    nota = piano.tecla_en(x, y)
                    if amp < 0.55 * mejor or nota is None or nota in self.pulsadas():
                        dedo.estado, dedo.quieto_desde, dedo.armado = "REPOSO", None, False
                        continue
                    dedo.estado, dedo.nota, dedo.t_press = "ABAJO", nota, t
                    vol = float(np.clip(0.45 + 0.15 * dedo.vret, 0.45, 1.0))
                    eventos.append(("on", nota, vol, f))

        # manos que desaparecen: soltamos sus notas
        for mid in list(self.manos):
            if mid not in vistas and t - self.manos[mid].t > 0.15:
                for dedo in self.manos[mid].dedos:
                    if dedo.abajo:
                        eventos.append(("off", dedo.nota))
                del self.manos[mid]
        return eventos, self.pulsadas()

    def pulsadas(self):
        return {d.nota for m in self.manos.values() for d in m.dedos if d.abajo}


# ------------------------------------------------------------
# Modo cancion
# ------------------------------------------------------------
class Cancion:
    def __init__(self):
        self.i = -1            # -1 = modo libre
        self.pos = 0
        self.fallos = 0
        self.t_fin = None

    @property
    def activa(self):
        return self.i >= 0

    @property
    def nombre(self):
        return CANCIONES[self.i][0]

    @property
    def notas(self):
        return CANCIONES[self.i][1]

    @property
    def objetivo(self):
        if self.activa and self.t_fin is None:
            return self.notas[self.pos]
        return None

    def cambiar(self):
        self.i = self.i + 1 if self.i + 1 < len(CANCIONES) else -1
        self.pos, self.fallos, self.t_fin = 0, 0, None

    def tocada(self, nota, t):
        if self.objetivo is None:
            return
        if nota == self.objetivo:
            self.pos += 1
            if self.pos >= len(self.notas):
                self.t_fin = t
        else:
            self.fallos += 1

    def actualizar(self, t):
        if self.t_fin is not None and t - self.t_fin > 4:
            self.pos, self.fallos, self.t_fin = 0, 0, None


# ------------------------------------------------------------
# Camara en un hilo (siempre tenemos el ultimo frame, sin acumular retraso)
# ------------------------------------------------------------
class Camara:
    def __init__(self, indice=0):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(indice)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(indice)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_ANCHO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_ALTO)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.contador = 0
        self.activo = True
        self.lock = threading.Lock()
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while self.activo:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
                self.contador += 1

    def leer(self, ultimo):
        """Espera a un frame nuevo (distinto de 'ultimo') y lo devuelve."""
        limite = time.monotonic() + 2
        while time.monotonic() < limite:
            with self.lock:
                if self.frame is not None and self.contador != ultimo:
                    return self.frame.copy(), self.contador
            time.sleep(0.001)
        return None, ultimo

    def parar(self):
        self.activo = False
        self.hilo.join(timeout=1)
        self.cap.release()


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def esta_apuntando(pts):
    """True si solo tiene el indice estirado (funciona con la mano girada)."""
    d = lambda a: np.linalg.norm(pts[a] - pts[0])
    indice = d(8) > d(6) * 1.1
    otros = d(12) < d(10) * 1.05 and d(16) < d(14) * 1.05 and d(20) < d(18) * 1.05
    return indice and otros


def guardar_config(piano, ancho, alto):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump({"rect": list(piano.rect()), "res": [ancho, alto]}, f)
    except OSError:
        pass


def cargar_config(ancho, alto):
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        x1, y1, x2, y2 = cfg["rect"]
        rw, rh = cfg["res"]
        sx, sy = ancho / rw, alto / rh
        return Piano(x1 * sx, y1 * sy, x2 * sx, y2 * sy, ancho, alto)
    except (OSError, ValueError, KeyError):
        return None


def piano_por_defecto(ancho, alto):
    return Piano(20, int(alto * 0.55), ancho - 20, alto - 10, ancho, alto)


def barra(frame, alto_px):
    sub = frame[0:alto_px]
    frame[0:alto_px] = cv2.addWeighted(np.zeros_like(sub), 0.6, sub, 0.4, 0)


# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Descargando el modelo de manos...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    audio = Audio()
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    opciones = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.4)

    cam = Camara()
    detector = Detector()
    cancion = Cancion()

    with HandLandmarker.create_from_options(opciones) as landmarker:
        frame, contador = cam.leer(-1)
        if frame is None:
            print("No se pudo abrir la camara.")
            cam.parar()
            return
        alto, ancho = frame.shape[:2]

        piano = cargar_config(ancho, alto)
        estado = "TOCAR" if piano else "PUNTO_1"     # PUNTO_1 -> PUNTO_2 -> TOCAR
        candidato, t_inicio, esquina1 = None, 0.0, (0, 0)
        brillo = {}
        etiquetas, ver_manos, depurar = False, True, False
        ts, t_prev, fps = 0, time.monotonic(), 30.0
        ultima_nota, notas_tocadas, t_nota = "", 0, 0.0

        cv2.namedWindow("Piano", cv2.WINDOW_NORMAL)
        while True:
            frame, contador = cam.leer(contador)
            if frame is None:
                break
            frame = cv2.flip(frame, 1)               # efecto espejo
            t = time.monotonic()
            dt = max(t - t_prev, 1e-3)
            t_prev = t
            fps = 0.9 * fps + 0.1 / dt

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(ts + 1, int(t * 1000))
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            manos = [np.array([[p.x * ancho, p.y * alto] for p in m]) for m in res.hand_landmarks]

            # ---------------- Colocar el piano apuntando con el dedo ----------------
            if estado in ("PUNTO_1", "PUNTO_2"):
                apuntado = None
                for pts in manos:
                    if esta_apuntando(pts):
                        apuntado = (int(pts[8][0]), int(pts[8][1]))
                if apuntado is None:
                    candidato = None
                else:
                    if candidato is None or abs(apuntado[0] - candidato[0]) + abs(apuntado[1] - candidato[1]) > 30:
                        candidato, t_inicio = apuntado, t
                    progreso = min(1.0, (t - t_inicio) / 2)
                    cv2.circle(frame, candidato, 10, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.ellipse(frame, candidato, (25, 25), 0, 0, int(360 * progreso), (0, 255, 0), 4, cv2.LINE_AA)
                    if progreso >= 1.0:
                        if estado == "PUNTO_1":
                            esquina1, estado = candidato, "PUNTO_2"
                        else:
                            x1, y1 = esquina1
                            x2, y2 = candidato
                            if x2 - x1 > 200 and y2 - y1 > 60:
                                piano = Piano(x1, y1, x2, y2, ancho, alto)
                                guardar_config(piano, ancho, alto)
                                detector.reset()
                                estado = "TOCAR"
                            else:
                                estado = "PUNTO_1"
                        candidato = None
                barra(frame, 70)
                if estado == "PUNTO_1":
                    texto(frame, "Apunta con el indice a la esquina SUPERIOR IZQUIERDA del piano (2 s)", 12, 28, 0.6, (0, 255, 255))
                else:
                    cv2.circle(frame, esquina1, 8, (0, 255, 0), -1, cv2.LINE_AA)
                    texto(frame, "Ahora a la esquina INFERIOR DERECHA del piano (2 s)", 12, 28, 0.6, (0, 255, 255))
                texto(frame, "D = piano por defecto", 12, 56, 0.5)

            # ---------------- Tocar ----------------
            else:
                eventos, pulsadas = detector.procesar(t, manos, piano)
                for ev in eventos:
                    if ev[0] == "on":
                        _, nota, vol, _ = ev
                        audio.tocar(nota, vol)
                        notas_tocadas += 1
                        ultima_nota, t_nota = NOMBRES[nota % 12], t
                        cancion.tocada(nota, t)
                    else:
                        audio.soltar(ev[1])
                cancion.actualizar(t)

                for n in pulsadas:
                    brillo[n] = 1.0
                for n in list(brillo):
                    if n not in pulsadas:
                        brillo[n] -= dt / 0.4
                        if brillo[n] <= 0:
                            del brillo[n]

                piano.dibujar(frame, brillo, cancion.objetivo, t, etiquetas)

                # Barra superior
                barra(frame, 40)
                if cancion.activa:
                    if cancion.t_fin is not None:
                        izq = f"{cancion.nombre}: COMPLETADA!  Fallos: {cancion.fallos}"
                    else:
                        sig = " ".join(NOMBRES[n % 12] for n in cancion.notas[cancion.pos:cancion.pos + 5])
                        izq = f"{cancion.nombre} {cancion.pos}/{len(cancion.notas)}   Siguiente: {sig}   Fallos: {cancion.fallos}"
                else:
                    izq = f"MODO LIBRE   Notas: {notas_tocadas}"
                texto(frame, izq, 12, 27, 0.6, ORO if cancion.activa else (0, 255, 0))
                texto(frame, f"FPS {fps:4.0f}   Sens {detector.sens:.1f}", ancho - 210, 27, 0.55)
                if t - t_nota < 0.6 and ultima_nota:
                    texto(frame, ultima_nota, ancho // 2 - 40, 90, 1.6, (0, 255, 255), 3)
                if not manos:
                    texto(frame, "Pon las manos sobre el piano", ancho // 2 - 190, piano.y1 - 20, 0.7, (200, 200, 200))
                texto(frame, "R recolocar | D defecto | S cancion | L nombres | H manos | +/- sens | B depurar | ESC",
                      10, alto - 12, 0.45)

            # ---------------- Manos ----------------
            if ver_manos or estado != "TOCAR":
                for pts in manos:
                    if ver_manos:
                        for a, b in CONEXIONES:
                            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
                estados = {}      # (x, y) de la punta -> esta pulsando
                for m in detector.manos.values():
                    if m.pts is not None and t - m.t < 0.1:
                        for f, p in enumerate(PUNTAS):
                            estados[(int(m.pts[p][0]), int(m.pts[p][1]))] = m.dedos[f].estado
                for pts in manos:
                    for p in PUNTAS:
                        xy = (int(pts[p][0]), int(pts[p][1]))
                        est = estados.get(xy, "REPOSO")
                        col = {"ABAJO": (0, 255, 0), "LEVANTADO": (0, 255, 255)}.get(est, (0, 0, 255))
                        cv2.circle(frame, xy, 7 if est == "ABAJO" else 5, col, -1, cv2.LINE_AA)
            if depurar and estado == "TOCAR":
                for m in detector.manos.values():
                    if m.pts is None or t - m.t > 0.1:
                        continue
                    for f, d in enumerate(m.dedos):
                        x, y = m.pts[PUNTAS[f]]
                        texto(frame, f"{d.dev:+.2f} {d.estado[:3]}", int(x) - 20, int(y) - 14, 0.4, (0, 255, 255))
                    texto(frame, f"mano v={m.vel:.1f}", int(m.palma[0]) - 30, int(m.palma[1]) + 30, 0.4, (255, 200, 0))

            cv2.imshow("Piano", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or cv2.getWindowProperty("Piano", cv2.WND_PROP_VISIBLE) < 1:
                break
            if k in (ord("r"), ord("R")):
                estado, candidato = "PUNTO_1", None
                detector.reset()
                audio.soltar_todo()
                brillo.clear()
            elif k in (ord("d"), ord("D")):
                piano = piano_por_defecto(ancho, alto)
                estado = "TOCAR"
                detector.reset()
                audio.soltar_todo()
                brillo.clear()
            elif k in (ord("s"), ord("S")):
                cancion.cambiar()
            elif k in (ord("l"), ord("L")):
                etiquetas = not etiquetas
            elif k in (ord("h"), ord("H")):
                ver_manos = not ver_manos
            elif k in (ord("b"), ord("B")):
                depurar = not depurar
            elif k in (ord("+"), ord("=")):
                detector.sens = min(2.5, detector.sens + 0.1)
            elif k in (ord("-"), ord("_")):
                detector.sens = max(0.5, detector.sens - 0.1)

    audio.soltar_todo()
    cam.parar()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
