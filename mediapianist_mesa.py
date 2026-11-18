#!/usr/bin/env python3
"""
MediaPianist  -  piano virtual con vision por computador para practicar y rehabilitar las manos.

    pip install mediapipe opencv-python numpy pygame Pillow
    python mediapianist.py                  # con camara
    python mediapianist.py --demo           # SIN camara: una mano simulada toca sola
    python mediapianist.py --usuario ana    # cada usuario guarda su progreso aparte

Es UN SOLO ARCHIVO. Al ejecutarlo crea al lado una carpeta "datos/" (progreso, calibracion e
informe) y un "piano_config.json". Necesita el modelo "hand_landmarker.task": si esta junto
a este archivo (o en models/) lo usa, y si no lo descarga solo la primera vez.

Teclas: ESC volver | F pantalla completa | +/- sensibilidad | M mano activa | X izq/der |
        H ver manos | B depurar | L nombres de teclas | R recolocar piano | T calibrar mesa
"""
import math
from collections import deque, namedtuple
import numpy as np
import cv2
import time
import os
import unicodedata
import json
import re
from datetime import datetime
import html
import webbrowser
import random
import argparse
import sys
import threading
import urllib.request
import sys

# ======================================================================
# SECCION: deteccion
# MediaPianist - deteccion.py
# ===========================
# Todo lo que "entiende" la mano:
# 
#   * Obs            -> una mano observada en un frame (2D en pixeles + 3D en metros)
#   * Detector       -> detecta pulsaciones de tecla dedo a dedo
#   * Calibrador     -> mide como pulsa cada persona y ajusta el detector a ella
#   * funciones de medida para los ejercicios (pinza, apertura, separacion, temblor)
# 
# COMO DETECTA UNA PULSACION (por que es mucho mas fiable que antes)
# ------------------------------------------------------------------
# Para cada dedo se calculan 4 medidas que SUBEN cuando el dedo baja a pulsar:
# 
#    c0  flexion de la articulacion del nudillo (grados, 3D)
#    c1  flexion de las otras dos articulaciones (grados, 3D)
#    c2  acortamiento aparente del dedo en la imagen (2D)
#    c3  descenso de la punta respecto al nudillo en la imagen (2D)
# 
# Se fusionan en UNA senal x (0 = dedo en reposo, 1 = pulsacion completa) con
# pesos que se calibran por persona. Se dispara cuando x sube mas de un umbral
# dentro de una ventana corta de tiempo (un golpe rapido, no un movimiento lento).
# La tecla se decide con la posicion de la punta al INICIO del golpe.
# ======================================================================


PUNTAS = [4, 8, 12, 16, 20]            # pulgar, indice, corazon, anular, menique
BASES = [2, 5, 9, 13, 17]              # articulacion de la que sale cada dedo
CADENAS = {0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12),
           3: (13, 14, 15, 16), 4: (17, 18, 19, 20)}
NOMBRES_DEDOS = ["Pulgar", "Índice", "Corazón", "Anular", "Meñique"]
NC = 4                                  # numero de componentes de la senal

# valores por defecto (sirven sin calibrar)
A_DEF = np.array([25.0, 35.0, 0.16, 0.16])      # amplitud tipica de una pulsacion
W_DEF = np.array([0.30, 0.05, 0.325, 0.325])    # peso de cada componente (c1 es la mas ruidosa)
W_PULGAR = np.array([0.35, 0.10, 0.275, 0.275])
A_PULGAR = np.array([20.0, 30.0, 0.09, 0.09])   # el pulgar es mas corto: menos recorrido en 2D

# parametros del detector (todo en unidades de "x": 1 = pulsacion completa)
UMBRAL_ON = 0.35        # cuanto debe subir x para contar como pulsacion
VENTANA = 0.30          # s: el golpe tiene que ocurrir dentro de esta ventana
REFRACTARIO = 0.09      # s minimos entre dos pulsaciones del mismo dedo
SOLTAR_FRAC = 0.45      # fraccion que debe volver a bajar x para soltar la tecla
VEL_MANO_MAX = 2.5      # tamanos de mano/s: si la mano viaja mas rapido, no cuenta
MIN_TAM = 18            # px: manos mas pequenas se ignoran
HOLD_MAX = 3.0          # s: una tecla no se queda "pulsada" para siempre

Ev = namedtuple("Ev", "tipo nota vol dedo mano t amp")   # tipo: 'on' | 'off'


# ------------------------------------------------------------
# Una mano observada
# ------------------------------------------------------------
def _ang(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class Obs:
    """Una mano en un frame. pts: 21x2 pixeles. w3: 21x3 metros (o None)."""
    __slots__ = ("pts", "w3", "etiqueta", "tam", "palma", "eje", "t")

    def __init__(self, pts, w3=None, etiqueta=None, t=0.0):
        self.pts = np.asarray(pts, dtype=float)
        self.w3 = None if w3 is None else np.asarray(w3, dtype=float)
        self.etiqueta = etiqueta            # 'Left' | 'Right' | None
        self.tam = float(np.linalg.norm(self.pts[9] - self.pts[0]))
        self.palma = self.pts[[0, 5, 9, 13, 17]].mean(axis=0)
        self.eje = (self.pts[9] - self.pts[0]) / max(self.tam, 1e-6)
        self.t = t

    def tam3(self):
        if self.w3 is None:
            return None
        return float(np.linalg.norm(self.w3[9] - self.w3[0]))


def componentes(o):
    """Devuelve (C, ok3d): C es 5xNC con las medidas de cada dedo."""
    C = np.zeros((5, NC))
    ok3d = o.w3 is not None
    p, tam = o.pts, max(o.tam, 1e-6)
    for f in range(5):
        v = p[PUNTAS[f]] - p[BASES[f]]
        if f == 0:
            e = float(np.linalg.norm(v)) / tam
        else:
            e = float(np.dot(v, o.eje)) / tam
        C[f, 2] = -e
        C[f, 3] = v[1] / tam
        if ok3d:
            w = o.w3
            a, b, c, d = CADENAS[f]
            if f == 0:
                th1 = _ang(w[b] - w[a], w[c] - w[b])
                th2 = _ang(w[c] - w[b], w[d] - w[c])
                C[f, 0], C[f, 1] = th1, th2
            else:
                th1 = _ang(w[a] - w[0], w[b] - w[a])
                th2 = _ang(w[b] - w[a], w[c] - w[b])
                th3 = _ang(w[c] - w[b], w[d] - w[c])
                C[f, 0], C[f, 1] = th1, th2 + th3
    return C, ok3d


def normalizar_etiquetas(lista, invertir=False):
    """MediaPipe a veces repite 'Left'/'Right'. Con dos manos la de la izquierda
    de la pantalla (imagen en espejo) es la izquierda del usuario."""
    if len(lista) == 2 and lista[0].etiqueta == lista[1].etiqueta:
        a, b = sorted(lista, key=lambda o: o.palma[0])
        a.etiqueta, b.etiqueta = "Left", "Right"
    if invertir:
        for o in lista:
            if o.etiqueta in ("Left", "Right"):
                o.etiqueta = "Right" if o.etiqueta == "Left" else "Left"
    return lista


# ------------------------------------------------------------
# Detector de pulsaciones
# ------------------------------------------------------------
class Dedo:
    def __init__(self):
        self.hist = deque(maxlen=48)       # (t, x_suavizado, tipx, tipy, grados)
        self.raw = deque(maxlen=3)         # ultimas 3 x sin suavizar
        self.difs = deque(maxlen=60)       # diferencias consecutivas (estimar ruido)
        self.T = UMBRAL_ON                 # umbral efectivo (se adapta al ruido)
        self.x = 0.0
        self.rise = 0.0
        self.abajo = False
        self.nota = None
        self.t_press = 0.0
        self.t_ultimo = -1.0
        self.t_off = -1.0                  # instante de la ultima suelta
        self.x_base = 0.0
        self.x_pico = 0.0
        self.deg0 = 0.0
        self.amp_pico = 0.0


class Mano:
    def __init__(self):
        self.dedos = [Dedo() for _ in range(5)]
        self.palma = None
        self.t = 0.0
        self.vel = 0.0
        self.obs = None
        self.etiqueta = None


class Detector:
    def __init__(self):
        self.manos = {}
        self._sig = 0
        self.sens = 1.0
        self.W = np.tile(W_DEF, (5, 1))
        self.W[0] = W_PULGAR
        self.S = np.ones((5, NC))
        self.A = np.tile(A_DEF, (5, 1))
        self.A[0] = A_PULGAR
        self.SX = np.zeros(5)              # ruido de la senal fusionada (de la calibracion)
        self.calibrado = False

    # ---- perfil de calibracion ----
    def aplicar_perfil(self, perfil):
        """perfil: {"0": {"w": [...], "s": [...], "A": [...]}, ...}"""
        if not perfil:
            return
        for f in range(5):
            d = perfil.get(str(f))
            if not d:
                continue
            try:
                w, s, a = np.array(d["w"], float), np.array(d["s"], float), np.array(d["A"], float)
                if w.shape == (NC,) and s.shape == (NC,) and a.shape == (NC,) and w.sum() > 0:
                    self.W[f], self.S[f], self.A[f] = w / w.sum(), s, np.maximum(a, 1e-3)
                    self.SX[f] = float(d.get("sx", 0.0))
                    self.calibrado = True
            except (TypeError, ValueError, KeyError):
                continue

    def reset(self):
        self.manos = {}

    # ---- senal fusionada ----
    def senal(self, C, ok3d):
        W = self.W.copy()
        if not ok3d:
            W[:, 0:2] = 0.0
        tot = W.sum(axis=1, keepdims=True)
        tot[tot < 1e-9] = 1.0
        W = W / tot
        x = (W * self.S * C / self.A).sum(axis=1)
        deg = C[:, 0] + C[:, 1] if ok3d else np.zeros(5)
        return x, deg

    # ---- emparejar manos entre frames ----
    def _emparejar(self, lista):
        pares = []
        for i, o in enumerate(lista):
            for mid, m in self.manos.items():
                d = float(np.linalg.norm(o.palma - m.palma))
                if d < 2.5 * o.tam:
                    pares.append((d, i, mid))
        pares.sort()
        ui, um, res = set(), set(), {}
        for d, i, mid in pares:
            if i in ui or mid in um:
                continue
            ui.add(i)
            um.add(mid)
            res[i] = mid
        for i in range(len(lista)):
            if i not in res:
                res[i] = self._sig
                self.manos[self._sig] = Mano()
                self._sig += 1
        return res

    def procesar(self, t, lista, piano):
        """lista: [Obs]. Devuelve eventos Ev ('on'/'off')."""
        eventos = []
        lista = [o for o in lista if o.tam >= MIN_TAM]
        asign = self._emparejar(lista)
        T = UMBRAL_ON / max(self.sens, 0.2)
        vistas = set()

        for i, o in enumerate(lista):
            mid = asign[i]
            vistas.add(mid)
            m = self.manos[mid]
            if m.palma is not None and t > m.t:
                m.vel = float(np.linalg.norm(o.palma - m.palma)) / o.tam / max(t - m.t, 0.02)
            m.palma, m.t, m.obs, m.etiqueta = o.palma, t, o, o.etiqueta

            C, ok3d = componentes(o)
            x, deg = self.senal(C, ok3d)
            cand, moviendose = [], 0

            for f in range(5):
                d = m.dedos[f]
                tip = o.pts[PUNTAS[f]]
                if d.raw and t - (d.hist[-1][0] if d.hist else t) > 0.5:
                    d.raw.clear()
                    d.difs.clear()
                if d.raw:
                    d.difs.append(float(x[f]) - d.raw[-1])
                d.raw.append(float(x[f]))
                # suavizado (1,2,1)/4: quita el ruido de un solo frame
                d.x = (d.raw[0] + 2 * d.raw[1] + d.raw[2]) / 4.0 if len(d.raw) == 3 else float(x[f])
                d.hist.append((t, d.x, float(tip[0]), float(tip[1]), float(deg[f])))
                # umbral efectivo: nunca por debajo de 5 veces el ruido medido en directo
                sig = 0.0
                if len(d.difs) >= 15:
                    dd = np.asarray(d.difs)
                    sig = 1.4826 * float(np.median(np.abs(dd - np.median(dd)))) / math.sqrt(2.0)
                d.T = max(T, 5.0 * sig, 6.0 * self.SX[f] * 0.6)

                # ---- tecla ya pulsada: esperar a que se suelte ----
                if d.abajo:
                    d.x_pico = max(d.x_pico, d.x)
                    d.amp_pico = max(d.amp_pico, float(deg[f]) - d.deg0)
                    cae = d.x_pico - d.x
                    if ((t - d.t_press > 0.06 and cae >= SOLTAR_FRAC * max(d.x_pico - d.x_base, 0.2))
                            or t - d.t_press > HOLD_MAX):
                        d.abajo = False
                        d.t_off = t
                        eventos.append(Ev("off", d.nota, 0, f, m.etiqueta, t, d.amp_pico))
                        d.nota = None
                    d.rise = 0.0
                    continue

                datos = [h for h in d.hist if t - h[0] <= VENTANA and h[0] > d.t_off]
                if len(datos) < 3:
                    d.rise = 0.0
                    continue
                xs = [h[1] for h in datos]
                i_min = int(np.argmin(xs))
                d.rise = xs[-1] - xs[i_min]
                rise_prev = xs[-2] - xs[i_min]
                if f > 0 and d.rise >= 0.7 * d.T:
                    moviendose += 1
                if (d.rise >= d.T and rise_prev >= 0.5 * d.T and i_min <= len(xs) - 3
                        and t - d.t_ultimo > REFRACTARIO and m.vel < VEL_MANO_MAX):
                    dt = max(t - datos[i_min][0], 0.03)
                    cand.append((d.rise, f, i_min, datos, d.rise / dt))

            if moviendose >= 3:          # media mano a la vez: puño/barrido, no una pulsacion
                cand = []
            if not cand or piano is None:
                continue
            mejor = max(c[0] for c in cand)
            for rise, f, i_min, datos, vel in cand:
                if rise < 0.55 * mejor:  # dedos vecinos arrastrados por el que pulsa
                    continue
                d = m.dedos[f]
                _, _, px, py, deg0 = datos[i_min]
                nota = piano.tecla_en(px, py)
                if nota is None:
                    nota = piano.tecla_en(*o.pts[PUNTAS[f]])
                if nota is None or nota in self.pulsadas():
                    continue
                d.abajo, d.nota = True, nota
                d.t_press = d.t_ultimo = t
                d.x_base = datos[i_min][1]
                d.x_pico = d.x
                d.deg0 = deg0
                d.amp_pico = 0.0
                vol = float(np.clip(0.42 + 0.10 * vel, 0.42, 1.0))
                eventos.append(Ev("on", nota, vol, f, m.etiqueta, t, 0.0))

        for mid in list(self.manos):     # manos que desaparecen: soltar sus notas
            if mid not in vistas and t - self.manos[mid].t > 0.15:
                for f, d in enumerate(self.manos[mid].dedos):
                    if d.abajo:
                        eventos.append(Ev("off", d.nota, 0, f, self.manos[mid].etiqueta, t, d.amp_pico))
                del self.manos[mid]
        return eventos

    def pulsadas(self):
        return {d.nota for m in self.manos.values() for d in m.dedos if d.abajo}


# ------------------------------------------------------------
# Calibrador: aprende como pulsa CADA persona
# ------------------------------------------------------------
def construir_perfil_dedo(E, sigma):
    """E: (golpes, NC) subida de cada componente en cada golpe. sigma: ruido en reposo (NC).
    Devuelve el perfil del dedo o None si no hay datos fiables."""
    E = np.asarray(E, dtype=float)
    if len(E) < 3:
        return None
    med = np.median(E, axis=0)
    signo = np.where(med >= 0, 1.0, -1.0)
    amp = np.maximum(np.median(np.abs(E), axis=0), 1e-3)
    consistentes = np.mean(np.sign(E) == signo, axis=0) >= 0.8
    snr = amp / (np.asarray(sigma) + 1e-4)
    util = (snr >= 4.0) & consistentes
    if not util.any():
        return None
    w = np.where(util, np.minimum(snr, 15.0) ** 2, 0.0)
    w = w / w.sum()
    sx = float(np.sqrt(np.sum((w * np.asarray(sigma) / amp) ** 2)))     # ruido de la senal fusionada
    rom = float(max(med[0], 0) + max(med[1], 0)) if np.abs(med[:2]).max() > 1.0 else None   # grados (3D)
    return {"w": w.tolist(), "s": signo.tolist(), "A": amp.tolist(), "sx": sx,
            "snr": snr.tolist(), "rom_deg": rom, "golpes": int(len(E))}


class Calibrador:
    """Calibracion guiada por metronomo. Fases: reposo (2 s) y despues, para cada dedo,
    una cuenta atras y 5 'pulsa ahora' a intervalos fijos. Usa la mano mas grande.
    Como sabemos CUANDO debe pulsar, no hay que adivinar donde hay un golpe."""
    REPOSO_S = 2.0
    LEAD = 2.0                 # s de preparacion antes del primer pulso
    PERIODO = 1.2              # s entre pulsos
    N_PULSOS = 5
    VENTANA = 0.70             # s tras cada clic en los que se espera la pulsacion

    def __init__(self):
        self.fase = "reposo"              # reposo | dedo | fin
        self.dedo = 0
        self.t_fase = None
        self.filas = []                   # (t_rel, C[5,NC])
        self.sigma = None
        self.perfil = {}
        self.fallidos = []
        self.t_perdida = None
        self.msg = ""
        self.click = False                # el bucle principal lo lee y lo apaga (suena un clic)
        self._cue_visto = -1
        self.t_rel = 0.0

    @property
    def terminado(self):
        return self.fase == "fin"

    def progreso(self):
        if self.fase == "reposo":
            return 0.0
        if self.fase == "fin":
            return 1.0
        dur = self.LEAD + self.N_PULSOS * self.PERIODO
        return (self.dedo + min(self.t_rel / dur, 1.0)) / 5.0

    def cue(self):
        """Estado del metronomo para dibujarlo: (indice_pulso_actual, fraccion_hasta_el_siguiente, flash)."""
        if self.fase != "dedo" or self.t_rel < self.LEAD - self.PERIODO:
            return -1, 0.0, False
        k = (self.t_rel - self.LEAD) / self.PERIODO
        idx = int(math.floor(k)) + 1 if k >= 0 else 0          # siguiente cue
        frac = k - math.floor(k) if k >= 0 else (self.t_rel - (self.LEAD - self.PERIODO)) / self.PERIODO
        flash = 0 <= k < self.N_PULSOS and (k - math.floor(k)) * self.PERIODO < 0.15
        return min(idx, self.N_PULSOS), float(frac), bool(flash)

    def actualizar(self, t, lista):
        if self.fase == "fin":
            return
        lista = [o for o in lista if o.tam >= MIN_TAM]
        if not lista:
            if self.t_perdida is None:
                self.t_perdida = t
            if t - self.t_perdida > 0.7:
                self.t_fase = None
                self.filas = []
                self._cue_visto = -1
            self.msg = "Pon una mano delante de la cámara"
            return
        self.t_perdida = None
        o = max(lista, key=lambda q: q.tam)
        C, _ = componentes(o)
        if self.t_fase is None:
            self.t_fase = t
        self.t_rel = t - self.t_fase
        self.filas.append((self.t_rel, C))

        if self.fase == "reposo":
            self.msg = "Apoya la mano en la mesa y deja los dedos relajados"
            if self.t_rel >= self.REPOSO_S:
                arr = np.array([c for _, c in self.filas])           # (n,5,NC)
                self.sigma = np.maximum(arr.std(axis=0), 1e-4)       # (5,NC)
                self._fase("dedo", 0)
            return

        f = self.dedo
        # clic del metronomo
        k = (self.t_rel - self.LEAD) / self.PERIODO
        if k >= 0 and int(math.floor(k)) != self._cue_visto and int(math.floor(k)) < self.N_PULSOS:
            self._cue_visto = int(math.floor(k))
            self.click = True
        if self.t_rel < self.LEAD:
            self.msg = f"Prepárate: vas a pulsar con el {NOMBRES_DEDOS[f].upper()} cuando suene el clic"
        else:
            self.msg = f"Apoyando la mano, pulsa la mesa con el {NOMBRES_DEDOS[f].upper()} en cada clic"
        if self.t_rel >= self.LEAD + self.N_PULSOS * self.PERIODO + 0.2:
            self._cerrar_dedo(f)

    def _cerrar_dedo(self, f):
        T = np.array([r[0] for r in self.filas])
        X = np.array([r[1][f] for r in self.filas])                  # (n,NC)
        if len(X) >= 5:                                              # media de 3 frames: menos ruido
            k = np.ones(3) / 3.0
            X = np.stack([np.convolve(np.pad(X[:, j], 1, mode="edge"), k, mode="valid") for j in range(NC)], axis=1)
        base_sel = (T >= 0.2) & (T <= self.LEAD - 0.1)
        if base_sel.sum() >= 3:
            m = np.median(X[base_sel], axis=0)
            E = []
            for i in range(self.N_PULSOS):
                t0 = self.LEAD + i * self.PERIODO
                sel = (T >= t0) & (T <= t0 + self.VENTANA)
                if sel.sum() < 3:
                    continue
                dev = X[sel] - m
                idx = np.argmax(np.abs(dev), axis=0)
                E.append(dev[idx, np.arange(NC)])
            p = construir_perfil_dedo(E, self.sigma[f]) if E else None
        else:
            p = None
        if p is not None:
            self.perfil[str(f)] = p
        else:
            self.fallidos.append(f)
        if f < 4:
            self._fase("dedo", f + 1)
        else:
            self.fase = "fin"
            self.msg = "Calibración terminada"

    def _fase(self, fase, dedo):
        self.fase, self.dedo, self.t_fase = fase, dedo, None
        self.filas, self._cue_visto, self.t_rel = [], -1, 0.0


# ------------------------------------------------------------
# Medidas para los ejercicios sin piano
# ------------------------------------------------------------
def distancias_pinza(o):
    """Distancia pulgar-(indice, corazon, anular, menique) / tamano de palma."""
    if o.w3 is not None:
        base = max(o.tam3(), 1e-6)
        return np.array([np.linalg.norm(o.w3[4] - o.w3[p]) for p in PUNTAS[1:]]) / base
    return np.array([np.linalg.norm(o.pts[4] - o.pts[p]) for p in PUNTAS[1:]]) / max(o.tam, 1e-6)


def apertura(o):
    """Distancia media punta-muñeca / palma. Mano abierta ~1.7, puño ~0.8."""
    if o.w3 is not None:
        base = max(o.tam3(), 1e-6)
        return float(np.mean([np.linalg.norm(o.w3[p] - o.w3[0]) for p in PUNTAS[1:]]) / base)
    return float(np.mean([np.linalg.norm(o.pts[p] - o.pts[0]) for p in PUNTAS[1:]]) / max(o.tam, 1e-6))


def separacion(o):
    """Distancia indice-menique (puntas) / ancho de los nudillos. Juntos ~1, abanico ~2."""
    if o.w3 is not None:
        ancho = max(float(np.linalg.norm(o.w3[5] - o.w3[17])), 1e-6)
        return float(np.linalg.norm(o.w3[8] - o.w3[20])) / ancho
    ancho = max(float(np.linalg.norm(o.pts[5] - o.pts[17])), 1e-6)
    return float(np.linalg.norm(o.pts[8] - o.pts[20])) / ancho


def esta_apuntando(o):
    """True si solo tiene el indice estirado (funciona con la mano girada)."""
    p = o.pts
    d = lambda a: np.linalg.norm(p[a] - p[0])
    indice = d(8) > d(6) * 1.1
    otros = d(12) < d(10) * 1.05 and d(16) < d(14) * 1.05 and d(20) < d(18) * 1.05
    return indice and otros


# ------------------------------------------------------------
# Temblor
# ------------------------------------------------------------
def analizar_temblor(muestras, mm_por_palma=95.0, fs=30.0):
    """muestras: lista de (t, puntas[5,2] en px, tam_px). Devuelve dict con
    rms_mm (banda 4-12 Hz), frecuencia_hz, deriva_mm (<2 Hz) y fps_reales."""
    if len(muestras) < 30:
        return None
    t = np.array([m[0] for m in muestras])
    dur = t[-1] - t[0]
    if dur < 3:
        return None
    fps_real = (len(t) - 1) / dur
    tam = np.median([m[2] for m in muestras])
    P = np.array([m[1] for m in muestras]) / tam              # (n,5,2) en "palmas"
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    U = np.empty((len(tu), 5, 2))
    for j in range(5):
        for c in range(2):
            U[:, j, c] = np.interp(tu, t, P[:, j, c])
    n = len(tu)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    F = np.fft.rfft(U - U.mean(axis=0), axis=0)               # (nf,5,2)
    banda = (freqs >= 4) & (freqs <= 12)
    lenta = (freqs > 0.2) & (freqs < 2)
    Fb = np.where(banda[:, None, None], F, 0)
    Fl = np.where(lenta[:, None, None], F, 0)
    b = np.fft.irfft(Fb, n=n, axis=0)
    l = np.fft.irfft(Fl, n=n, axis=0)
    rms_b = float(np.sqrt(np.mean(np.sum(b ** 2, axis=2), axis=0)).mean())
    rms_l = float(np.sqrt(np.mean(np.sum(l ** 2, axis=2), axis=0)).mean())
    pot = (np.abs(F) ** 2).sum(axis=(1, 2))
    rango = (freqs >= 3) & (freqs <= 12)
    f_dom = float(freqs[rango][np.argmax(pot[rango])]) if rango.any() else 0.0
    return {"rms_mm": rms_b * mm_por_palma, "frecuencia_hz": f_dom,
            "deriva_mm": rms_l * mm_por_palma, "fps_reales": float(fps_real)}


# ======================================================================
# SECCION: piano
# MediaPianist - piano.py : geometria del piano, busqueda de tecla y dibujo.
# ======================================================================


NOMBRES = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]
N_OCTAVAS = 3
N_BLANCAS = 7 * N_OCTAVAS
N_NOTAS = 12 * N_OCTAVAS
BLANCAS_OCTAVA = [0, 2, 4, 5, 7, 9, 11]
NEGRA_TRAS_POS = (0, 1, 3, 4, 5)           # hay negra despues de DO RE FA SOL LA
ORO = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def mezclar(a, b, k):
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))


def color_nota(n):
    hsv = np.uint8([[[(95 + n * 6) % 180, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


COLORES = [color_nota(n) for n in range(N_NOTAS)]


class Piano:
    def __init__(self, x1, y1, x2, y2, ancho, alto, opacidad=0.62, quad=None):
        self.x1 = int(max(0, min(x1, x2)))
        self.x2 = int(min(ancho, max(x1, x2)))
        self.y1 = int(max(0, min(y1, y2)))
        self.y2 = int(min(alto, max(y1, y2)))
        self.w = max(self.x2 - self.x1, 10)
        self.h = max(self.y2 - self.y1, 10)
        self.opacidad = opacidad
        self.y_negra = self.y1 + int(self.h * 0.6)
        aw = self.w / N_BLANCAS
        self.aw = aw
        self.blancas, self.negras = [], []
        for i in range(N_BLANCAS):
            pos = i % 7
            nota = (i // 7) * 12 + BLANCAS_OCTAVA[pos]
            self.blancas.append((nota, self.x1 + i * aw, self.x1 + (i + 1) * aw))
            if pos in NEGRA_TRAS_POS:
                c = self.x1 + (i + 1) * aw
                self.negras.append((nota + 1, c - aw * 0.3, c + aw * 0.3))
        self._base = self._pintar_base()
        margen = int(self.w * 0.075)
        self.superficie = None if quad is None else np.asarray(quad, dtype=np.float32).reshape(4, 2)
        self._quad = self.superficie.copy() if self.superficie is not None else np.array([
            [self.x1 + margen, self.y1],
            [self.x2 - margen, self.y1],
            [self.x2, self.y2],
            [self.x1, self.y2]
        ], dtype=np.float32)
        destino = np.array([[self.x1, self.y1], [self.x2, self.y1],
                            [self.x2, self.y2], [self.x1, self.y2]], dtype=np.float32)
        self._a_plano = cv2.getPerspectiveTransform(self._quad, destino)

    def rect(self):
        return self.x1, self.y1, self.x2, self.y2

    def centro_tecla(self, nota):
        for n, xa, xb in self.blancas:
            if n == nota:
                return (xa + xb) / 2, self.y1 + self.h * 0.82
        for n, xa, xb in self.negras:
            if n == nota:
                return (xa + xb) / 2, self.y1 + self.h * 0.3
        return None

    def _pintar_base(self):
        img = np.full((self.h, self.w, 3), 245, np.uint8)
        yn = self.y_negra - self.y1
        for _, xa, xb in self.blancas:
            cv2.rectangle(img, (int(xa - self.x1), 0), (int(xb - self.x1) - 1, self.h - 1), (80, 80, 80), 1)
        for _, xa, xb in self.negras:
            cv2.rectangle(img, (int(xa - self.x1), 0), (int(xb - self.x1), yn), (28, 28, 28), -1)
            cv2.line(img, (int(xa - self.x1) + 2, 0), (int(xa - self.x1) + 2, yn - 3), (85, 85, 85), 1)
        return img

    def _transformar_punto(self, x, y):
        """Convierte un punto de la cámara a las coordenadas planas del teclado."""
        puntos = np.array([[[float(x), float(y)]]], dtype=np.float32)
        convertido = cv2.perspectiveTransform(puntos, self._a_plano)[0, 0]
        return float(convertido[0]), float(convertido[1])

    def tecla_en(self, x, y):
        """Busca la tecla usando la perspectiva del piano virtual en pantalla."""
        x, y = self._transformar_punto(x, y)
        m = 0.03 * self.w
        if x < self.x1 - m or x >= self.x2 + m:
            return None
        if y < self.y1 - 0.7 * self.h or y > self.y2 + 0.35 * self.h:
            return None
        x = min(max(x, self.x1), self.x2 - 1e-3)
        if y < self.y_negra:
            for nota, xa, xb in self.negras:
                if xa <= x < xb:
                    return nota
        for nota, xa, xb in self.blancas:
            if xa <= x < xb:
                return nota
        return None

    def _dibujar_plano(self, frame, brillo=None, resaltes=None, t=0.0, etiquetas=False, numeros=None):
        """brillo {nota: 0..1}: teclas pulsadas. resaltes {nota: (color, 0..1)}: guia.
        numeros {nota: 'texto'}: etiqueta grande (p. ej. numero de dedo)."""
        brillo, resaltes, numeros = brillo or {}, resaltes or {}, numeros or {}
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        capa = self._base.copy()
        yn = self.y_negra - y1

        for nota, xa, xb in self.blancas:
            col = None
            if brillo.get(nota, 0) > 0.02:
                col = mezclar((245, 245, 245), COLORES[nota], brillo[nota])
            elif nota in resaltes:
                c, k = resaltes[nota]
                col = mezclar((245, 245, 245), c, k)
            if col is not None:
                cv2.rectangle(capa, (int(xa - x1) + 1, 0), (int(xb - x1) - 2, self.h - 2), col, -1)
        for nota, xa, xb in self.negras:
            col = (28, 28, 28)
            if brillo.get(nota, 0) > 0.02:
                col = mezclar(col, COLORES[nota], brillo[nota])
            elif nota in resaltes:
                c, k = resaltes[nota]
                col = mezclar(col, c, min(1.0, k + 0.15))
            cv2.rectangle(capa, (int(xa - x1), 0), (int(xb - x1), yn), col, -1)

        roi = frame[y1:y2, x1:x2]
        if roi.shape[:2] == capa.shape[:2]:
            frame[y1:y2, x1:x2] = cv2.addWeighted(capa, self.opacidad, roi, 1 - self.opacidad, 0)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (60, 60, 60), 2, cv2.LINE_AA)

        esc = max(self.aw / 70.0, 0.5)
        for nota, xa, xb in self.blancas:
            if nota % 12 == 0 or etiquetas:
                nombre = NOMBRES[nota % 12]
                if nota % 12 == 0:
                    nombre += str(4 + nota // 12)
                cv2.putText(frame, nombre, (int(xa) + 4, y2 - 8), FONT, 0.4 * min(esc, 1.4),
                            (40, 40, 40), 1, cv2.LINE_AA)
        for nota, txt in numeros.items():
            c = self.centro_tecla(nota)
            if c is None:
                continue
            r = int(max(14, self.aw * 0.28))
            cv2.circle(frame, (int(c[0]), int(y2 - self.h * 0.28)), r, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(c[0]), int(y2 - self.h * 0.28)), r, (255, 255, 255), 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(txt, FONT, r / 22.0, 2)
            cv2.putText(frame, txt, (int(c[0] - tw / 2), int(y2 - self.h * 0.28 + th / 2)), FONT,
                        r / 22.0, (255, 255, 255), 2, cv2.LINE_AA)


    def dibujar(self, frame, brillo=None, resaltes=None, t=0.0, etiquetas=False, numeros=None):
        """Dibuja el piano como un teclado 3D flotando delante de la cámara."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = self.rect()

        # Si hay una mesa calibrada, el teclado se adapta a sus cuatro esquinas.
        # Si no, conserva la perspectiva 3D decorativa anterior.
        if self.superficie is not None:
            self._quad = self.superficie.copy()
        else:
            margen = int(self.w * 0.075)
            self._quad = np.array([
                [x1 + margen, y1], [x2 - margen, y1],
                [x2, y2], [x1, y2]
            ], dtype=np.float32)
        destino_plano = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        self._a_plano = cv2.getPerspectiveTransform(self._quad, destino_plano)

        frente = self._quad.astype(np.int32)
        if self.superficie is None:
            # Sombra y grosor inferior del teclado 3D original.
            desplazamiento = max(8, int(self.h * 0.055))
            parte_baja = np.array([
                frente[3], frente[2],
                [min(W - 1, frente[2][0] + 3), min(H - 1, frente[2][1] + desplazamiento)],
                [max(0, frente[3][0] - 3), min(H - 1, frente[3][1] + desplazamiento)]
            ], dtype=np.int32)
            cv2.fillPoly(frame, [parte_baja], (18, 22, 28), cv2.LINE_AA)
        else:
            # Borde fino para que se vea dónde termina la superficie calibrada.
            cv2.polylines(frame, [frente], True, (70, 190, 220), 2, cv2.LINE_AA)

        # Dibuja primero el teclado habitual y lo proyecta dentro del trapecio.
        temporal = frame.copy()
        self._dibujar_plano(temporal, brillo, resaltes, t, etiquetas, numeros)

        origen = np.array([
            [x1, y1], [x2 - 1, y1],
            [x2 - 1, y2 - 1], [x1, y2 - 1]
        ], dtype=np.float32)
        destino = self._quad.astype(np.float32)
        M = cv2.getPerspectiveTransform(origen, destino)
        proyectado = cv2.warpPerspective(temporal, M, (W, H))

        mascara_origen = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mascara_origen, origen.astype(np.int32), 255)
        mascara = cv2.warpPerspective(mascara_origen, M, (W, H))
        frame[mascara > 0] = proyectado[mascara > 0]

        cv2.polylines(frame, [frente], True, (225, 235, 245) if self.superficie is None else (70, 190, 220), 2, cv2.LINE_AA)
        if self.superficie is None:
            cv2.line(frame, tuple(frente[3]), tuple(parte_baja[3]), (90, 110, 130), 2, cv2.LINE_AA)
            cv2.line(frame, tuple(frente[2]), tuple(parte_baja[2]), (90, 110, 130), 2, cv2.LINE_AA)


def pulso(t, base=0.5, amp=0.25, vel=6.0):
    return base + amp * math.sin(t * vel)


# ======================================================================
# SECCION: audio
# MediaPianist - audio.py : sintesis de piano y sonidos de feedback.
# ======================================================================


SR = 44100
N_NOTAS = 36
FRECUENCIAS = [261.63 * 2 ** (n / 12) for n in range(N_NOTAS)]


def sintetizar_nota(frecuencia, pan, duracion=3.0):
    """Sonido de piano por sintesis aditiva: armonicos que se apagan a distinta
    velocidad, dos cuerdas casi iguales (brillo), golpe de martillo y estereo."""
    n = int(SR * duracion)
    t = np.arange(n) / SR
    onda = np.zeros(n)
    caida = 0.9 + frecuencia / 450.0
    for k in range(1, 12):
        fk = frecuencia * k * math.sqrt(1 + 0.0004 * k * k)
        if fk > 9000:
            break
        amp = 1.0 / (k ** 1.15)
        env = np.exp(-t * caida * (1 + 0.55 * (k - 1)))
        cuerdas = np.sin(2 * np.pi * fk * t) + np.sin(2 * np.pi * fk * 1.0006 * t + k)
        onda += 0.5 * amp * cuerdas * env
    rng = np.random.default_rng(int(frecuencia * 100))
    onda += rng.standard_normal(n) * np.exp(-t * 220) * 0.04
    onda *= np.minimum(1.0, t / 0.004)
    onda *= np.minimum(1.0, (duracion - t) / 0.08)
    onda = onda / np.max(np.abs(onda)) * 0.55
    izq, der = math.cos(pan * math.pi / 2), math.sin(pan * math.pi / 2)
    est = np.stack([onda * izq, onda * der], axis=1) * 1.3
    return np.ascontiguousarray((est * 32767).astype(np.int16))


def sintetizar_tono(freqs, dur=0.3, vol=0.5, decay=6.0):
    """Tono corto (para aciertos/errores). freqs: lista de (frecuencia, inicio_s)."""
    n = int(SR * (dur + max(s for _, s in freqs)))
    t = np.arange(n) / SR
    onda = np.zeros(n)
    for f, ini in freqs:
        tt = np.clip(t - ini, 0, None)
        env = np.where(t >= ini, np.exp(-tt * decay), 0.0) * np.minimum(1.0, tt / 0.005)
        onda += np.sin(2 * np.pi * f * t) * env + 0.3 * np.sin(4 * np.pi * f * t) * env
    onda = onda / max(np.max(np.abs(onda)), 1e-6) * vol
    est = np.stack([onda, onda], axis=1)
    return np.ascontiguousarray((est * 32767).astype(np.int16))


class Audio:
    def __init__(self, silencio=False):
        import pygame
        self.pg = pygame
        pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=512, allowedchanges=0)
        pygame.mixer.set_num_channels(40)
        self.silencio = silencio
        print("Generando sonidos...")
        mk = pygame.sndarray.make_sound
        self.sonidos = [mk(sintetizar_nota(f, 0.25 + 0.5 * i / (N_NOTAS - 1))) for i, f in enumerate(FRECUENCIAS)]
        self.s_ok = mk(sintetizar_tono([(880, 0.0), (1318.5, 0.09)], 0.25, 0.45, 7.0))
        self.s_mal = mk(sintetizar_tono([(196, 0.0), (155.6, 0.10)], 0.3, 0.45, 5.0))
        self.s_clic = mk(sintetizar_tono([(1500, 0.0)], 0.06, 0.5, 40.0))
        self.s_fin = mk(sintetizar_tono([(523.25, 0.0), (659.25, 0.12), (783.99, 0.24), (1046.5, 0.36)], 0.5, 0.5, 4.0))
        self.canales = {}

    def tocar(self, nota, volumen=0.8):
        if self.silencio or not (0 <= nota < len(self.sonidos)):
            return
        self.soltar(nota, 40)
        canal = self.sonidos[nota].play()
        if canal is not None:
            canal.set_volume(float(volumen))
            self.canales[nota] = (canal, time.monotonic())

    def soltar(self, nota, ms=220):
        """Al levantar el dedo la nota se apaga suavemente (apagador del piano)."""
        dato = self.canales.pop(nota, None)
        if dato is not None and time.monotonic() - dato[1] < 2.9:
            dato[0].fadeout(ms)

    def soltar_todo(self):
        for nota in list(self.canales):
            self.soltar(nota, 100)

    def _fx(self, s):
        if not self.silencio:
            s.play()

    def ok(self):
        self._fx(self.s_ok)

    def mal(self):
        self._fx(self.s_mal)

    def clic(self):
        self._fx(self.s_clic)

    def fin(self):
        self._fx(self.s_fin)

    def cerrar(self):
        try:
            self.soltar_todo()
            self.pg.mixer.quit()
        except Exception:
            pass


# ======================================================================
# SECCION: interfaz
# MediaPianist - interfaz.py
# ==========================
# Dibujo de la interfaz sobre el frame de OpenCV:
#   * texto con tildes y ñ (OpenCV no puede) mediante sprites de Pillow en cache
#   * paneles redondeados, botones, anillos de progreso
#   * Puntero: seleccion con raton o MANTENIENDO EL DEDO sobre un boton (sin teclado)
#   * pantalla del cuestionario SUS
# ======================================================================


try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:                                   # sin Pillow: texto ASCII con OpenCV
    PIL_OK = False

CARPETA = os.path.dirname(os.path.abspath(__file__))

# paleta (BGR)
BLANCO = (255, 255, 255)
GRIS = (175, 175, 175)
GRIS_OSC = (95, 90, 88)
FONDO = (34, 28, 26)
ORO = (0, 200, 255)
VERDE = (110, 205, 90)
ROJO = (85, 85, 235)
AZUL = (235, 165, 70)
MORADO = (205, 110, 175)
TURQUESA = (190, 200, 60)
NARANJA = (60, 150, 250)

_RUTAS_REG = [os.path.join(CARPETA, "assets", "DejaVuSans.ttf"),
              "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_RUTAS_NEG = [os.path.join(CARPETA, "assets", "DejaVuSans-Bold.ttf"),
              "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_fuentes, _sprites = {}, {}


def _ruta(lista):
    for r in lista:
        if os.path.exists(r):
            return r
    return None


def _fuente(px, negrita):
    clave = (int(px), negrita)
    if clave not in _fuentes:
        ruta = _ruta(_RUTAS_NEG if negrita else _RUTAS_REG) or _ruta(_RUTAS_REG)
        try:
            _fuentes[clave] = ImageFont.truetype(ruta, int(px)) if ruta else ImageFont.load_default(int(px))
        except Exception:
            _fuentes[clave] = ImageFont.load_default()
    return _fuentes[clave]


def _ascii(t):
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def _sprite(txt, px, color, negrita, borde):
    clave = (txt, int(px), color, negrita, borde)
    s = _sprites.get(clave)
    if s is None:
        f = _fuente(px, negrita)
        pad = 3
        bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), txt, font=f, stroke_width=2 if borde else 0)
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
        img = Image.new("RGBA", (max(w, 1), max(h, 1)), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((pad - bb[0], pad - bb[1]), txt, font=f, fill=(color[2], color[1], color[0], 255),
               stroke_width=2 if borde else 0, stroke_fill=(0, 0, 0, 255))
        arr = np.array(img)
        s = arr[:, :, [2, 1, 0, 3]].copy()
        if len(_sprites) > 700:
            _sprites.clear()
        _sprites[clave] = s
    return s


def _pegar(frame, s, x, y):
    H, W = frame.shape[:2]
    h, w = s.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    t = s[y0 - y:y1 - y, x0 - x:x1 - x]
    a = t[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1]
    roi[:] = (t[:, :, :3] * a + roi * (1.0 - a)).astype(np.uint8)


def texto(frame, txt, x, y, px=20, color=BLANCO, ancla="i", negrita=False, borde=True):
    """Dibuja texto con (x, y) = esquina superior del cuadro. ancla: i/c/d. Devuelve (ancho, alto)."""
    txt = str(txt)
    if not txt:
        return 0, 0
    if PIL_OK:
        s = _sprite(txt, px, tuple(int(c) for c in color), negrita, borde)
        h, w = s.shape[:2]
        xx = x if ancla == "i" else (x - w // 2 if ancla == "c" else x - w)
        _pegar(frame, s, int(xx), int(y))
        return w, h
    t = _ascii(txt)
    esc = px / 30.0
    (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, esc, 2 if negrita else 1)
    xx = x if ancla == "i" else (x - w // 2 if ancla == "c" else x - w)
    base = int(y + h)
    if borde:
        cv2.putText(frame, t, (int(xx), base), cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, t, (int(xx), base), cv2.FONT_HERSHEY_SIMPLEX, esc, color, 2 if negrita else 1, cv2.LINE_AA)
    return w, h


def texto_multilinea(frame, txt, x, y, ancho_max, px=20, color=BLANCO, interlineado=1.35, negrita=False):
    """Parte el texto en lineas que quepan en ancho_max. Devuelve la y final."""
    palabras, linea = str(txt).split(), ""
    cy = y
    for p in palabras:
        prueba = (linea + " " + p).strip()
        if medir(prueba, px, negrita) > ancho_max and linea:
            texto(frame, linea, x, cy, px, color, negrita=negrita)
            cy += int(px * interlineado)
            linea = p
        else:
            linea = prueba
    if linea:
        texto(frame, linea, x, cy, px, color, negrita=negrita)
        cy += int(px * interlineado)
    return cy


def medir(txt, px, negrita=False):
    if PIL_OK:
        return int(_fuente(px, negrita).getlength(str(txt))) + 6
    return cv2.getTextSize(_ascii(str(txt)), cv2.FONT_HERSHEY_SIMPLEX, px / 30.0, 1)[0][0]


_mascaras = {}


def _mascara(w, h, r):
    clave = (w, h, r)
    m = _mascaras.get(clave)
    if m is None:
        m = np.zeros((h, w), np.uint8)
        r = min(r, w // 2, h // 2)
        cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
        for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        if len(_mascaras) > 200:
            _mascaras.clear()
        _mascaras[clave] = m
    return m


def panel(frame, x, y, w, h, color=FONDO, alpha=0.7, radio=14, borde=None, grosor=2):
    """Rectangulo redondeado translucido."""
    H, W = frame.shape[:2]
    x, y, w, h = int(x), int(y), int(w), int(h)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or w < 4 or h < 4:
        return
    m = _mascara(w, h, int(radio))[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1]
    a = (m.astype(np.float32) / 255.0 * alpha)[:, :, None]
    roi[:] = (np.array(color, np.float32) * a + roi * (1 - a)).astype(np.uint8)
    if borde is not None:
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), borde, grosor, cv2.LINE_AA) if radio < 4 else \
            _borde_redondo(frame, x, y, w, h, int(radio), borde, grosor)


def _borde_redondo(frame, x, y, w, h, r, color, g):
    r = min(r, w // 2, h // 2)
    cv2.line(frame, (x + r, y), (x + w - r, y), color, g, cv2.LINE_AA)
    cv2.line(frame, (x + r, y + h), (x + w - r, y + h), color, g, cv2.LINE_AA)
    cv2.line(frame, (x, y + r), (x, y + h - r), color, g, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + r), (x + w, y + h - r), color, g, cv2.LINE_AA)
    for (cx, cy, a0) in ((x + r, y + r, 180), (x + w - r, y + r, 270), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90)):
        cv2.ellipse(frame, (cx, cy), (r, r), 0, a0, a0 + 90, color, g, cv2.LINE_AA)


def anillo(frame, centro, radio, progreso, color=VERDE, grosor=5):
    cv2.circle(frame, centro, radio, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    if progreso > 0:
        cv2.ellipse(frame, centro, (radio, radio), -90, 0, int(360 * min(progreso, 1.0)), color, grosor, cv2.LINE_AA)


def barra(frame, x, y, w, h, frac, color=VERDE, fondo=(60, 56, 54)):
    panel(frame, x, y, w, h, fondo, 0.85, h // 2)
    if frac > 0:
        panel(frame, x, y, max(int(w * min(frac, 1.0)), h), h, color, 1.0, h // 2)


# ------------------------------------------------------------
# Botones y puntero
# ------------------------------------------------------------
class Boton:
    def __init__(self, ident, titulo, sub="", color=AZUL, x=0, y=0, w=100, h=60, dwell=1.2, destacado=False):
        self.id, self.titulo, self.sub, self.color = ident, titulo, sub, color
        self.x, self.y, self.w, self.h = x, y, w, h
        self.dwell = dwell
        self.destacado = destacado

    def contiene(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def dibujar(self, frame, k, hover=False, prog=0.0, insignia=None):
        panel(frame, self.x, self.y, self.w, self.h, (52, 44, 40) if not hover else (80, 68, 60),
              0.88, int(14 * k), self.color if (hover or self.destacado) else GRIS_OSC, 3 if hover else 2)
        cv2.rectangle(frame, (self.x + int(6 * k), self.y + int(14 * k)),
                      (self.x + int(11 * k), self.y + self.h - int(14 * k)), self.color, -1)
        texto(frame, self.titulo, self.x + int(24 * k), self.y + int(10 * k), int(22 * k), BLANCO, negrita=True)
        if self.sub:
            texto_multilinea(frame, self.sub, self.x + int(24 * k), self.y + int(40 * k), self.w - int(36 * k),
                             int(15 * k), GRIS, 1.25)
        if insignia:
            texto(frame, insignia, self.x + self.w - int(12 * k), self.y + self.h - int(30 * k), int(14 * k), ORO, "d", True)
        if prog > 0:
            barra(frame, self.x + int(14 * k), self.y + self.h - int(10 * k), self.w - int(28 * k), int(6 * k), prog, VERDE)


class Puntero:
    """Selecciona botones con el raton (click) o manteniendo la punta del dedo encima."""

    def __init__(self):
        self.actual = None
        self.t_ini = 0.0
        self.t_bloqueo = 0.0
        self.progreso = 0.0

    def actualizar(self, t, botones, puntos_dedo, click=None):
        """puntos_dedo: [(x, y)]. Devuelve (id_seleccionado | None, id_hover | None)."""
        if click is not None:
            for b in botones:
                if b.contiene(*click):
                    self.t_bloqueo = t + 0.6
                    return b.id, b.id
        if t < self.t_bloqueo:
            self.actual, self.progreso = None, 0.0
            return None, None
        sobre = None
        for b in botones:
            if any(b.contiene(px, py) for px, py in puntos_dedo):
                sobre = b
                break
        if sobre is None:
            self.actual, self.progreso = None, 0.0
            return None, None
        if self.actual != sobre.id:
            self.actual, self.t_ini = sobre.id, t
        self.progreso = min((t - self.t_ini) / sobre.dwell, 1.0)
        if self.progreso >= 1.0:
            self.t_bloqueo = t + 1.0
            self.actual, self.progreso = None, 0.0
            return sobre.id, sobre.id
        return None, sobre.id


class Avisos:
    """Mensajes cortos centrados arriba (aciertos, errores, cambios de nivel...)."""

    def __init__(self):
        self.lista = []

    def poner(self, txt, t, color=BLANCO, dur=1.8):
        self.lista = [(txt, color, t + dur)]

    def dibujar(self, frame, t, k, y=None):
        self.lista = [a for a in self.lista if a[2] > t]
        H, W = frame.shape[:2]
        for txt, color, _ in self.lista:
            px = int((26 if len(txt) < 55 else 21) * k)
            maxw = int(W * 0.78)
            lineas, linea = [], ""
            for p in txt.split():
                prueba = (linea + " " + p).strip()
                if medir(prueba, px, True) > maxw and linea:
                    lineas.append(linea)
                    linea = p
                else:
                    linea = prueba
            lineas.append(linea)
            ancho = max(medir(l, px, True) for l in lineas)
            alto = int(len(lineas) * px * 1.35) + int(12 * k)
            yy = int(84 * k) if y is None else y
            panel(frame, W // 2 - ancho // 2 - int(18 * k), yy - int(6 * k), ancho + int(36 * k), alto, (20, 18, 16), 0.78, int(12 * k))
            for i, l in enumerate(lineas):
                texto(frame, l, W // 2, yy + int(i * px * 1.35), px, color, "c", True)


# ------------------------------------------------------------
# Cuestionario SUS (System Usability Scale)
# ------------------------------------------------------------
SUS_ITEMS = [
    "Creo que me gustaría usar este sistema con frecuencia.",
    "Encontré el sistema innecesariamente complejo.",
    "Pensé que el sistema era fácil de usar.",
    "Creo que necesitaría el apoyo de un técnico para poder usar este sistema.",
    "Encontré que las diversas funciones del sistema estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en este sistema.",
    "Imagino que la mayoría de la gente aprendería a usar este sistema muy rápidamente.",
    "Encontré el sistema muy incómodo de usar.",
    "Me sentí muy seguro/a usando el sistema.",
    "Necesité aprender muchas cosas antes de poder empezar con el sistema.",
]


def puntuacion_sus(resp):
    """resp: 10 valores 1..5. Devuelve 0..100."""
    tot = 0
    for i, r in enumerate(resp):
        tot += (r - 1) if i % 2 == 0 else (5 - r)
    return tot * 2.5


def interpretar_sus(p):
    if p >= 85:
        return "Excelente"
    if p >= 73:
        return "Buena"
    if p >= 68:
        return "Aceptable (media del sector)"
    if p >= 51:
        return "Mejorable"
    return "Mala: hay que rediseñar"


class PantallaSUS:
    def __init__(self):
        self.i = 0
        self.resp = []
        self.fin = False
        self.botones = []

    def _layout(self, W, H, k):
        self.botones = []
        bw, bh, gap = int(110 * k), int(70 * k), int(16 * k)
        x0 = W // 2 - (5 * bw + 4 * gap) // 2
        y = int(H * 0.62)
        for n in range(1, 6):
            self.botones.append(Boton(n, str(n), "", AZUL, x0 + (n - 1) * (bw + gap), y, bw, bh))

    def tecla(self, c):
        if ord("1") <= c <= ord("5"):
            self._responder(c - ord("0"))

    def click(self, x, y):
        for b in self.botones:
            if b.contiene(x, y):
                self._responder(b.id)

    def _responder(self, v):
        if self.fin:
            return
        self.resp.append(v)
        self.i += 1
        if self.i >= len(SUS_ITEMS):
            self.fin = True

    def dibujar(self, frame, k):
        H, W = frame.shape[:2]
        self._layout(W, H, k)
        panel(frame, 0, 0, W, H, (20, 16, 14), 0.88, 0)
        texto(frame, "Cuestionario de usabilidad (SUS)", W // 2, int(40 * k), int(34 * k), ORO, "c", True)
        if self.fin:
            p = puntuacion_sus(self.resp)
            texto(frame, f"Tu puntuación SUS: {p:.1f} / 100", W // 2, int(H * 0.38), int(40 * k), BLANCO, "c", True)
            texto(frame, interpretar_sus(p), W // 2, int(H * 0.38) + int(60 * k), int(28 * k), VERDE, "c")
            texto(frame, "Guardado. Pulsa cualquier tecla para volver (la media del sector es 68).",
                  W // 2, int(H * 0.38) + int(120 * k), int(20 * k), GRIS, "c")
            return
        texto(frame, f"Pregunta {self.i + 1} de {len(SUS_ITEMS)}", W // 2, int(100 * k), int(22 * k), GRIS, "c")
        barra(frame, W // 2 - int(300 * k), int(135 * k), int(600 * k), int(10 * k), self.i / len(SUS_ITEMS), ORO)
        texto_multilinea(frame, SUS_ITEMS[self.i], int(W * 0.18), int(H * 0.28), int(W * 0.64), int(34 * k), BLANCO, 1.4, True)
        texto(frame, "1 = Totalmente en desacuerdo", self.botones[0].x, self.botones[0].y - int(34 * k), int(18 * k), GRIS)
        texto(frame, "5 = Totalmente de acuerdo", self.botones[-1].x + self.botones[-1].w, self.botones[-1].y - int(34 * k),
              int(18 * k), GRIS, "d")
        for b in self.botones:
            b.dibujar(frame, k)
        texto(frame, "Responde con el teclado (1-5) o con el ratón.   ESC = cancelar", W // 2, H - int(40 * k), int(18 * k), GRIS, "c")


def vaciar_cache():
    _sprites.clear()


ui = sys.modules[__name__]          # 'ui.texto', 'ui.panel'... (todo vive en este archivo)


# ======================================================================
# SECCION: sesion
# MediaPianist - sesion.py
# ========================
# Persistencia y metricas:
#   * Almacen    -> carpeta datos/<usuario>/ con calibracion, niveles, SUS y sesiones
#   * Niveles    -> nivel de dificultad ADAPTATIVO de cada ejercicio (se recuerda entre dias)
#   * Adaptativa -> sube/baja el nivel segun precision y tiempo de reaccion
#   * Sesion     -> registra ensayos, pulsaciones y medidas extra de cada ejercicio
# ======================================================================


CARPETA = os.path.dirname(os.path.abspath(__file__))


def _limpio(nombre):
    return re.sub(r"[^A-Za-z0-9_-]", "_", nombre.strip()) or "invitado"


class Almacen:
    def __init__(self, usuario="invitado", base=None):
        self.usuario = _limpio(usuario)
        self.dir = os.path.join(base or os.path.join(CARPETA, "datos"), self.usuario)
        self.dir_sesiones = os.path.join(self.dir, "sesiones")
        os.makedirs(self.dir_sesiones, exist_ok=True)

    def ruta(self, nombre):
        return os.path.join(self.dir, nombre)

    def cargar(self, nombre, defecto=None):
        try:
            with open(self.ruta(nombre), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return defecto

    def guardar(self, nombre, obj):
        try:
            with open(self.ruta(nombre), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
            return True
        except OSError:
            return False

    def sesiones(self):
        res = []
        try:
            nombres = sorted(os.listdir(self.dir_sesiones))
        except OSError:
            return res
        for n in nombres:
            if n.endswith(".json"):
                try:
                    with open(os.path.join(self.dir_sesiones, n), encoding="utf-8") as f:
                        res.append(json.load(f))
                except (OSError, ValueError):
                    continue
        return res


# ------------------------------------------------------------
# Dificultad adaptativa
# ------------------------------------------------------------
class Niveles:
    def __init__(self, almacen):
        self.alm = almacen
        self.d = almacen.cargar("niveles.json", {}) or {}

    def get(self, clave, defecto=1):
        return int(self.d.get(clave, defecto))

    def set(self, clave, v):
        self.d[clave] = int(v)
        self.alm.guardar("niveles.json", self.d)


class Adaptativa:
    """Tras cada bloque de ensayos:
         sube de nivel si precision >= 85 % y el tiempo medio es el objetivo o menos,
         baja si la precision es <= 55 %,  si no se mantiene."""

    def __init__(self, niveles, clave, minimo=1, maximo=5):
        self.niveles, self.clave, self.min, self.max = niveles, clave, minimo, maximo

    @property
    def nivel(self):
        return max(self.min, min(self.max, self.niveles.get(self.clave, self.min)))

    def evaluar(self, precision, rt_medio, rt_objetivo):
        """Devuelve (cambio, nivel_nuevo) con cambio en {-1, 0, +1}."""
        n = self.nivel
        cambio = 0
        if precision >= 0.85 and (rt_medio is None or rt_medio <= rt_objetivo) and n < self.max:
            cambio = 1
        elif precision <= 0.55 and n > self.min:
            cambio = -1
        if cambio:
            self.niveles.set(self.clave, n + cambio)
        return cambio, n + cambio


# ------------------------------------------------------------
# Sesion
# ------------------------------------------------------------
class Sesion:
    def __init__(self, almacen):
        self.alm = almacen
        self.inicio = datetime.now()
        self.t0 = time.time()
        self.bloques = []
        self.b = None

    # ---- bloques (una ejecucion de un ejercicio) ----
    def iniciar_bloque(self, modo, titulo, nivel=None):
        self.cerrar_bloque()
        self.b = {"modo": modo, "titulo": titulo, "nivel": nivel,
                  "inicio": datetime.now().isoformat(timespec="seconds"), "t0": time.time(),
                  "dur_s": 0.0, "ensayos": [], "pulsaciones": [], "extra": {}}

    def cerrar_bloque(self):
        if self.b is None:
            return None
        self.b["dur_s"] = round(time.time() - self.b.pop("t0"), 1)
        b, self.b = self.b, None
        if b["ensayos"] or b["pulsaciones"] or b["extra"]:
            self.bloques.append(b)
        return b

    # ---- registro ----
    def ensayo(self, ok, rt=None, dedo=None, mano=None, dedo_ok=None):
        if self.b is not None:
            self.b["ensayos"].append({"ok": bool(ok), "rt": None if rt is None else round(float(rt), 3),
                                      "dedo": dedo, "mano": mano, "dedo_ok": dedo_ok})

    def pulsacion(self, dedo, mano, amp=None):
        if self.b is not None:
            self.b["pulsaciones"].append({"dedo": int(dedo), "mano": mano,
                                          "amp": None if amp is None else round(float(amp), 1)})

    def extra(self, **kw):
        if self.b is not None:
            self.b["extra"].update(kw)

    # ---- resumen ----
    @staticmethod
    def resumen(b):
        ens = b["ensayos"]
        ok = sum(1 for e in ens if e["ok"])
        rts = [e["rt"] for e in ens if e["ok"] and e["rt"] is not None]
        return {"ensayos": len(ens), "aciertos": ok, "fallos": len(ens) - ok,
                "precision": (ok / len(ens)) if ens else None,
                "rt_medio": float(np.mean(rts)) if rts else None,
                "pulsaciones": len(b["pulsaciones"]), "dur_s": b.get("dur_s", 0.0)}

    def tiene_datos(self):
        return bool(self.bloques) or (self.b is not None and (self.b["ensayos"] or self.b["pulsaciones"] or self.b["extra"]))

    def guardar(self):
        self.cerrar_bloque()
        if not self.bloques:
            return None
        datos = {"usuario": self.alm.usuario, "inicio": self.inicio.isoformat(timespec="seconds"),
                 "dur_s": round(time.time() - self.t0, 1), "bloques": self.bloques}
        nombre = self.inicio.strftime("%Y-%m-%d_%H%M%S") + ".json"
        try:
            with open(os.path.join(self.alm.dir_sesiones, nombre), "w", encoding="utf-8") as f:
                json.dump(datos, f, ensure_ascii=False, indent=1)
            return nombre
        except OSError:
            return None


# ======================================================================
# SECCION: informe
# MediaPianist - informe.py
# =========================
# Genera informe.html (autocontenido, con graficas SVG) con el progreso entre
# sesiones: precision, reaccion, rango de movimiento por dedo, simetria entre
# manos, temblor, memoria, niveles, calibracion y encuesta SUS.
# ======================================================================



CSS = """
:root{--bg:#f6f7f9;--card:#fff;--tx:#1b1f24;--mut:#667085;--ac:#0e8f8f;--ac2:#e0a100;--bad:#d9534f;--ok:#2e9e5b;--bd:#e3e6ea}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--card:#1d2226;--tx:#eceff2;--mut:#9aa4af;--bd:#2c343b;--ac:#35c2c2;--ac2:#f2b81f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:34px 0 10px;border-left:5px solid var(--ac);padding-left:10px}
.sub{color:var(--mut);margin:0 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px 16px}
.card b{display:block;font-size:26px;line-height:1.1}.card span{color:var(--mut);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:12px;overflow:hidden;font-size:14px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--bd)}th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase}
tr:last-child td{border-bottom:0}.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:var(--card);border:1px solid var(--bd);border-left:5px solid var(--ac2);border-radius:10px;padding:12px 14px;margin:12px 0}
.up{color:var(--ok);font-weight:600}.down{color:var(--bad);font-weight:600}
.chart{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:8px;overflow-x:auto}
svg text{fill:var(--mut);font-size:11px}svg .t{fill:var(--tx)}
.foot{margin-top:40px;color:var(--mut);font-size:13px}
"""


def e(x):
    return html.escape(str(x))


def fmt(v, nd=1, suf="", vacio="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return vacio
    return f"{v:.{nd}f}{suf}"


def med(lista):
    lista = [x for x in lista if x is not None]
    return float(np.median(lista)) if lista else None


def media(lista):
    lista = [x for x in lista if x is not None]
    return float(np.mean(lista)) if lista else None


# ------------------------------------------------------------
# Graficas SVG
# ------------------------------------------------------------
def svg_barras(etiquetas, valores, unidad="", color="var(--ac)", ymax=None, ancho=560, alto=190):
    if not any(v is not None for v in valores):
        return "<p class='sub'>Todavía no hay datos suficientes.</p>"
    top = ymax or max(v for v in valores if v is not None) * 1.15 or 1
    n = len(valores)
    pad_l, pad_b, pad_t = 38, 28, 14
    bw = (ancho - pad_l - 10) / n
    s = [f"<svg viewBox='0 0 {ancho} {alto}' width='100%' style='max-width:{ancho}px'>"]
    for i in range(5):
        y = pad_t + (alto - pad_b - pad_t) * i / 4
        val = top * (1 - i / 4)
        s.append(f"<line x1='{pad_l}' x2='{ancho - 6}' y1='{y:.1f}' y2='{y:.1f}' stroke='currentColor' opacity='.12'/>")
        s.append(f"<text x='{pad_l - 6}' y='{y + 4:.1f}' text-anchor='end'>{val:.0f}</text>")
    for i, (et, v) in enumerate(zip(etiquetas, valores)):
        x = pad_l + i * bw + bw * 0.18
        w = bw * 0.64
        if v is not None:
            h = (alto - pad_b - pad_t) * min(v / top, 1)
            y = alto - pad_b - h
            s.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='4' fill='{color}'/>")
            s.append(f"<text class='t' x='{x + w / 2:.1f}' y='{y - 4:.1f}' text-anchor='middle'>{v:.0f}{e(unidad)}</text>")
        s.append(f"<text x='{x + w / 2:.1f}' y='{alto - 10}' text-anchor='middle'>{e(et)}</text>")
    s.append("</svg>")
    return "".join(s)


def svg_lineas(etiquetas, valores, unidad="", color="var(--ac)", ymin=None, ymax=None, ancho=560, alto=190):
    pts = [(i, v) for i, v in enumerate(valores) if v is not None]
    if len(pts) < 2:
        return "<p class='sub'>Se necesitan al menos 2 sesiones con datos para ver la evolución.</p>"
    vs = [v for _, v in pts]
    lo = ymin if ymin is not None else min(vs) * 0.9
    hi = ymax if ymax is not None else max(vs) * 1.1
    if hi - lo < 1e-9:
        hi = lo + 1
    pad_l, pad_b, pad_t = 38, 28, 14
    n = len(valores)
    X = lambda i: pad_l + (ancho - pad_l - 14) * (i / max(n - 1, 1))
    Y = lambda v: pad_t + (alto - pad_b - pad_t) * (1 - (v - lo) / (hi - lo))
    s = [f"<svg viewBox='0 0 {ancho} {alto}' width='100%' style='max-width:{ancho}px'>"]
    for i in range(5):
        y = pad_t + (alto - pad_b - pad_t) * i / 4
        val = hi - (hi - lo) * i / 4
        s.append(f"<line x1='{pad_l}' x2='{ancho - 6}' y1='{y:.1f}' y2='{y:.1f}' stroke='currentColor' opacity='.12'/>")
        s.append(f"<text x='{pad_l - 6}' y='{y + 4:.1f}' text-anchor='end'>{val:.1f}</text>")
    d = " ".join(("M" if k == 0 else "L") + f"{X(i):.1f},{Y(v):.1f}" for k, (i, v) in enumerate(pts))
    s.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.5'/>")
    for i, v in pts:
        s.append(f"<circle cx='{X(i):.1f}' cy='{Y(v):.1f}' r='4' fill='{color}'/>")
        s.append(f"<text class='t' x='{X(i):.1f}' y='{Y(v) - 8:.1f}' text-anchor='middle'>{v:.1f}{e(unidad)}</text>")
    paso = max(1, n // 8)
    for i in range(0, n, paso):
        s.append(f"<text x='{X(i):.1f}' y='{alto - 10}' text-anchor='middle'>{e(etiquetas[i])}</text>")
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------
# Agregacion
# ------------------------------------------------------------
def _fecha_corta(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m %H:%M")
    except ValueError:
        return iso[:10]


def agregar(sesiones):
    por_sesion, dedos = [], {f: {"n": 0, "ok": 0, "dedo_ok": [], "rt": [], "amp": []} for f in range(5)}
    manos = {"Left": {"amp": [], "rt": [], "n": 0, "ok": 0}, "Right": {"amp": [], "rt": [], "n": 0, "ok": 0}}
    temblor, reps, simon, pinza = [], [], [], []
    for s in sesiones:
        n = ok = 0
        rts, pul = [], 0
        for b in s.get("bloques", []):
            pul += len(b.get("pulsaciones", []))
            for en in b.get("ensayos", []):
                n += 1
                ok += 1 if en["ok"] else 0
                if en["ok"] and en.get("rt") is not None:
                    rts.append(en["rt"])
                f = en.get("dedo")
                if f is not None and 0 <= f <= 4:
                    d = dedos[f]
                    d["n"] += 1
                    d["ok"] += 1 if en["ok"] else 0
                    if en.get("dedo_ok") is not None:
                        d["dedo_ok"].append(1.0 if en["dedo_ok"] else 0.0)
                    if en["ok"] and en.get("rt") is not None:
                        d["rt"].append(en["rt"])
                m = manos.get(en.get("mano"))
                if m is not None:
                    m["n"] += 1
                    m["ok"] += 1 if en["ok"] else 0
                    if en["ok"] and en.get("rt") is not None:
                        m["rt"].append(en["rt"])
            for p in b.get("pulsaciones", []):
                if p.get("amp") is not None and p["amp"] > 3 and 0 <= p["dedo"] <= 4:
                    dedos[p["dedo"]]["amp"].append(p["amp"])
                    m = manos.get(p.get("mano"))
                    if m is not None:
                        m["amp"].append(p["amp"])
            x = b.get("extra", {})
            if b["modo"] == "estabilidad" and "rms_mm" in x:
                temblor.append((s["inicio"], x))
            if b["modo"] in ("puno", "abanico") and "rango_bruto" in x:
                reps.append((s["inicio"], b["titulo"], x, Sesionres(b)))
            if b["modo"] == "simon" and "span_max" in x:
                simon.append((s["inicio"], x))
            if b["modo"] == "pinza":
                pinza.append((s["inicio"], Sesionres(b)))
        por_sesion.append({"inicio": s["inicio"], "dur": s.get("dur_s", 0), "ensayos": n,
                           "precision": (ok / n * 100) if n else None, "rt": media(rts), "pulsaciones": pul,
                           "modos": sorted({b["titulo"] for b in s.get("bloques", [])})})
    return por_sesion, dedos, manos, temblor, reps, simon, pinza


def Sesionres(b):
    ens = b.get("ensayos", [])
    ok = sum(1 for x in ens if x["ok"])
    rts = [x["rt"] for x in ens if x["ok"] and x.get("rt") is not None]
    return {"n": len(ens), "precision": (ok / len(ens) * 100) if ens else None, "rt": media(rts)}


def dedo_a_trabajar(dedos, calib):
    """Devuelve (indice, motivo) del dedo con peores resultados relativos, o None."""
    metricas = {}
    for f in range(5):
        d = dedos[f]
        rom = med(d["amp"])
        if rom is None and calib and str(f) in calib.get("perfil", {}):
            rom = calib["perfil"][str(f)].get("rom_deg")
        metricas[f] = {"rom": rom, "rt": med(d["rt"]), "prec": (d["ok"] / d["n"]) if d["n"] >= 4 else None,
                       "dedo_ok": media(d["dedo_ok"])}
    score = {f: 0.0 for f in range(5)}
    motivos = {f: [] for f in range(5)}
    usado = False
    for clave, signo, texto in (("rom", -1, "menor rango de movimiento"), ("rt", 1, "reacción más lenta"),
                                ("prec", -1, "más errores"), ("dedo_ok", -1, "confunde más este dedo")):
        vals = {f: m[clave] for f, m in metricas.items() if m[clave] is not None}
        if len(vals) < 3:
            continue
        arr = np.array(list(vals.values()), dtype=float)
        sd = arr.std()
        if sd < 1e-9:
            continue
        usado = True
        for f, v in vals.items():
            z = (v - arr.mean()) / sd * signo
            score[f] += z
            if z > 0.8:
                motivos[f].append(texto)
    if not usado:
        return None
    f = max(score, key=score.get)
    if score[f] < 0.8:
        return None
    return f, motivos[f]


# ------------------------------------------------------------
# HTML
# ------------------------------------------------------------
def generar(almacen):
    sesiones = almacen.sesiones()
    calib = almacen.cargar("calibracion.json")
    niveles = almacen.cargar("niveles.json", {}) or {}
    sus = almacen.cargar("sus.json", []) or []
    por_sesion, dedos, manos, temblor, reps, simon, pinza = agregar(sesiones)
    h = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>MediaPianist · Informe de {e(almacen.usuario)}</title><style>{CSS}</style></head><body><main>"]
    h.append(f"<h1>🎹 MediaPianist · Informe de {e(almacen.usuario)}</h1>")
    h.append(f"<p class='sub'>Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')} · {len(sesiones)} sesión(es) registradas</p>")

    if not sesiones:
        h.append("<div class='note'>Todavía no hay sesiones guardadas. Juega un rato con los ejercicios y vuelve a abrir el informe.</div>")
    else:
        dur = sum(s["dur"] for s in por_sesion)
        ens = sum(s["ensayos"] for s in por_sesion)
        prec = media([s["precision"] for s in por_sesion if s["precision"] is not None])
        rt = media([s["rt"] for s in por_sesion])
        h.append("<div class='grid'>")
        for valor, etq in ((str(len(sesiones)), "sesiones"), (fmt(dur / 60, 0, " min"), "tiempo total de práctica"),
                           (str(ens), "ensayos realizados"), (fmt(prec, 0, " %"), "precisión media"),
                           (fmt(rt, 2, " s"), "tiempo de reacción medio")):
            h.append(f"<div class='card'><b>{valor}</b><span>{etq}</span></div>")
        h.append("</div>")

        if len(por_sesion) >= 2:
            a, b = por_sesion[-2], por_sesion[-1]
            frases = []
            if a["precision"] is not None and b["precision"] is not None:
                d = b["precision"] - a["precision"]
                frases.append(f"Precisión: <span class='{'up' if d >= 0 else 'down'}'>{d:+.0f} puntos</span>")
            if a["rt"] is not None and b["rt"] is not None:
                d = b["rt"] - a["rt"]
                frases.append(f"Reacción: <span class='{'up' if d <= 0 else 'down'}'>{d:+.2f} s</span> (menos es mejor)")
            if frases:
                h.append(f"<div class='note'><b>Última sesión frente a la anterior:</b> {' · '.join(frases)}</div>")

        et = [_fecha_corta(s["inicio"]) for s in por_sesion]
        h.append("<h2>Evolución entre sesiones</h2><div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(300px,1fr))'>")
        h.append("<div><b>Precisión (%)</b><div class='chart'>" + svg_lineas(et, [s["precision"] for s in por_sesion], " %", "var(--ok)", 0, 100) + "</div></div>")
        h.append("<div><b>Tiempo de reacción (s)</b><div class='chart'>" + svg_lineas(et, [s["rt"] for s in por_sesion], " s", "var(--ac2)", 0) + "</div></div>")
        h.append("</div>")

        h.append("<h2>Rendimiento por dedo</h2>")
        filas = []
        for f in range(5):
            d = dedos[f]
            rom_cal = None
            if calib and str(f) in calib.get("perfil", {}):
                rom_cal = calib["perfil"][str(f)].get("rom_deg")
            filas.append((f, fmt(med(d["amp"]), 0, "°"), fmt(rom_cal, 0, "°"), fmt(med(d["rt"]), 2, " s"),
                          fmt(d["ok"] / d["n"] * 100, 0, " %") if d["n"] else "—",
                          fmt(media(d["dedo_ok"]) * 100 if d["dedo_ok"] else None, 0, " %"), str(d["n"])))
        h.append("<table><tr><th>Dedo</th><th class='num'>Recorrido al pulsar</th><th class='num'>Recorrido calibración</th><th class='num'>Reacción</th>"
                 "<th class='num'>Precisión</th><th class='num'>Dedo correcto</th><th class='num'>Ensayos</th></tr>")
        for f, a, b, c, d_, e_, n in filas:
            h.append(f"<tr><td>{NOMBRES_DEDOS[f]}</td><td class='num'>{a}</td><td class='num'>{b}</td><td class='num'>{c}</td>"
                     f"<td class='num'>{d_}</td><td class='num'>{e_}</td><td class='num'>{n}</td></tr>")
        h.append("</table>")
        h.append("<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(300px,1fr));margin-top:12px'>")
        h.append("<div><b>Recorrido de movimiento (°)</b><div class='chart'>" + svg_barras(
            NOMBRES_DEDOS, [med(dedos[f]["amp"]) or (calib or {}).get("perfil", {}).get(str(f), {}).get("rom_deg") for f in range(5)], "°") + "</div></div>")
        h.append("<div><b>Tiempo de reacción (ms)</b><div class='chart'>" + svg_barras(
            NOMBRES_DEDOS, [(med(dedos[f]["rt"]) * 1000 if med(dedos[f]["rt"]) else None) for f in range(5)], "", "var(--ac2)") + "</div></div>")
        h.append("</div>")
        w = dedo_a_trabajar(dedos, calib)
        if w:
            f, mot = w
            h.append(f"<div class='note'><b>Dedo a trabajar: {NOMBRES_DEDOS[f]}.</b> Destaca por: {e(', '.join(mot) or 'resultados algo inferiores')}. "
                     "Los ejercicios <i>Dedo a dedo</i> y <i>Pinza</i> son los más útiles para reforzarlo.</div>")

        h.append("<h2>Simetría entre manos</h2>")
        L, R = manos["Left"], manos["Right"]
        rl, rr = med(L["amp"]), med(R["amp"])
        tl, tr = med(L["rt"]), med(R["rt"])
        h.append("<table><tr><th></th><th class='num'>Mano izquierda</th><th class='num'>Mano derecha</th><th class='num'>Índice de simetría</th></tr>")
        sim_rom = (min(rl, rr) / max(rl, rr) * 100) if rl and rr else None
        sim_rt = (min(tl, tr) / max(tl, tr) * 100) if tl and tr else None
        h.append(f"<tr><td>Recorrido al pulsar</td><td class='num'>{fmt(rl, 0, '°')}</td><td class='num'>{fmt(rr, 0, '°')}</td><td class='num'>{fmt(sim_rom, 0, ' %')}</td></tr>")
        h.append(f"<tr><td>Tiempo de reacción</td><td class='num'>{fmt(tl, 2, ' s')}</td><td class='num'>{fmt(tr, 2, ' s')}</td><td class='num'>{fmt(sim_rt, 0, ' %')}</td></tr>")
        h.append(f"<tr><td>Ensayos</td><td class='num'>{L['n']}</td><td class='num'>{R['n']}</td><td></td></tr></table>")
        h.append("<p class='sub'>100 % = las dos manos rinden igual. Un índice bajo indica que una mano va por detrás de la otra.</p>")

        if reps:
            h.append("<h2>Movilidad: puño / mano abierta y abanico</h2><table><tr><th>Fecha</th><th>Ejercicio</th><th class='num'>Rango (unid.)</th>"
                     "<th class='num'>Pico medio</th><th class='num'>Reps</th><th class='num'>Tiempo/rep</th></tr>")
            for ini, tit, x, r in reps[-12:]:
                h.append(f"<tr><td>{_fecha_corta(ini)}</td><td>{e(tit)}</td><td class='num'>{fmt(x.get('rango_bruto'), 2)}</td>"
                         f"<td class='num'>{fmt(x.get('pico_medio'), 0, ' %')}</td><td class='num'>{r['n']}</td>"
                         f"<td class='num'>{fmt(x.get('t_rep'), 1, ' s')}</td></tr>")
            h.append("</table><p class='sub'>El rango se mide en la propia sesión (lo máximo que se consigue abrir/cerrar); "
                     "si crece de una sesión a otra, mejora la movilidad.</p>")

        if pinza:
            h.append("<h2>Pinza (oposición pulgar–dedo)</h2><table><tr><th>Fecha</th><th class='num'>Ensayos</th><th class='num'>Precisión</th><th class='num'>Tiempo</th></tr>")
            for ini, r in pinza[-10:]:
                h.append(f"<tr><td>{_fecha_corta(ini)}</td><td class='num'>{r['n']}</td><td class='num'>{fmt(r['precision'], 0, ' %')}</td><td class='num'>{fmt(r['rt'], 2, ' s')}</td></tr>")
            h.append("</table>")

        if simon:
            h.append("<h2>Memoria (Simón dice)</h2>")
            h.append("<div class='chart'>" + svg_lineas([_fecha_corta(i) for i, _ in simon], [float(x["span_max"]) for _, x in simon], " notas", "var(--ac)", 0) + "</div>")
            h.append("<p class='sub'>Longitud máxima de secuencia repetida correctamente en cada sesión.</p>")

        if temblor:
            h.append("<h2>Estabilidad de la mano (temblor)</h2><table><tr><th>Fecha</th><th>Mano</th><th class='num'>Temblor 4–12 Hz</th>"
                     "<th class='num'>Frecuencia</th><th class='num'>Deriva</th></tr>")
            for ini, x in temblor[-12:]:
                h.append(f"<tr><td>{_fecha_corta(ini)}</td><td>{e(x.get('mano', '—'))}</td><td class='num'>{fmt(x['rms_mm'], 2, ' mm')}</td>"
                         f"<td class='num'>{fmt(x['frecuencia_hz'], 1, ' Hz')}</td><td class='num'>{fmt(x.get('deriva_mm'), 2, ' mm')}</td></tr>")
            h.append("</table><p class='sub'>Medida orientativa con cámara: incluye ruido de la detección (≈0,3–0,6 mm). Sirve para comparar "
                     "la misma persona entre días, no como diagnóstico.</p>")

        h.append("<h2>Todas las sesiones</h2><table><tr><th>Fecha</th><th class='num'>Duración</th><th class='num'>Ensayos</th>"
                 "<th class='num'>Precisión</th><th class='num'>Reacción</th><th>Ejercicios</th></tr>")
        for s in reversed(por_sesion[-15:]):
            h.append(f"<tr><td>{_fecha_corta(s['inicio'])}</td><td class='num'>{fmt(s['dur'] / 60, 1, ' min')}</td><td class='num'>{s['ensayos']}</td>"
                     f"<td class='num'>{fmt(s['precision'], 0, ' %')}</td><td class='num'>{fmt(s['rt'], 2, ' s')}</td><td>{e(', '.join(s['modos']))}</td></tr>")
        h.append("</table>")

    if niveles:
        nombres = {"cancion": "Canciones", "dedo_a_dedo": "Dedo a dedo", "simon": "Simón dice", "pinza": "Pinza",
                   "puno": "Puño / mano abierta", "abanico": "Abanico"}
        h.append("<h2>Nivel de dificultad actual (adaptativo)</h2><div class='grid'>")
        for k, v in niveles.items():
            h.append(f"<div class='card'><b>Nivel {v}/5</b><span>{e(nombres.get(k, k))}</span></div>")
        h.append("</div><p class='sub'>Sube cuando la precisión supera el 85 % con buen tiempo y baja si cae del 55 %.</p>")

    if calib and calib.get("perfil"):
        h.append(f"<h2>Calibración ({e(calib.get('fecha', ''))[:10]})</h2><table><tr><th>Dedo</th><th class='num'>Recorrido medido</th><th class='num'>Calidad de señal</th></tr>")
        for f in range(5):
            p = calib["perfil"].get(str(f))
            if p:
                calidad = max(p.get("snr", [0]))
                h.append(f"<tr><td>{NOMBRES_DEDOS[f]}</td><td class='num'>{fmt(p.get('rom_deg'), 0, '°')}</td><td class='num'>{calidad:.0f}×</td></tr>")
            else:
                h.append(f"<tr><td>{NOMBRES_DEDOS[f]}</td><td class='num'>—</td><td class='num'>no calibrado</td></tr>")
        h.append("</table>")

    if sus:
        h.append("<h2>Usabilidad (cuestionario SUS)</h2><table><tr><th>Fecha</th><th class='num'>Puntuación</th><th>Valoración</th></tr>")
        for x in sus[-8:]:
            h.append(f"<tr><td>{e(x['fecha'][:10])}</td><td class='num'>{x['puntuacion']:.1f}</td><td>{e(interpretar_sus(x['puntuacion']))}</td></tr>")
        pm = media([x["puntuacion"] for x in sus])
        h.append(f"</table><p class='sub'>Media de {len(sus)} respuesta(s): <b>{pm:.1f}</b> (la media del sector es 68).</p>")

    h.append("<p class='foot'>MediaPianist es una herramienta de práctica y motivación basada en visión por computador. "
             "Sus medidas son orientativas y <b>no sustituyen la valoración de un profesional sanitario</b>.</p></main></body></html>")
    ruta = almacen.ruta("informe.html")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("".join(h))
    return ruta


def abrir(ruta):
    try:
        webbrowser.open("file:///" + os.path.abspath(ruta).replace("\\", "/"))
    except Exception:
        pass


informe = sys.modules[__name__]     # 'informe.generar' / 'informe.abrir'


# ======================================================================
# SECCION: modos
# MediaPianist - modos.py : todos los modos de juego / ejercicio.
# 
# Cada modo implementa:
#   iniciar(t)            al entrar
#   evento(ev, t)         pulsaciones del piano (Ev 'on' / 'off')
#   actualizar(t, obs)    cada frame (obs = manos ya filtradas)
#   resaltes(t), numeros()  como pintar el piano (solo modos con piano)
#   dibujar(frame, t)     HUD / guias encima del video
#   tecla(c)              teclas propias del modo
#   cerrar()              guarda medidas extra en la sesion
#   resumen()             lista (etiqueta, valor) para la pantalla final
# ======================================================================



CLAVES = [12, 14, 16, 17, 19]          # DO RE MI FA SOL de la octava central (5 dedos)
COLORES_SIMON = [ui.ROJO, ui.NARANJA, ui.VERDE, ui.AZUL, ui.MORADO]

CANCIONES = [
    ("Himno a la alegría", [4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 4, 2, 2,
                            4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 2, 0, 0]),
    ("Estrellita", [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0,
                    7, 7, 5, 5, 4, 4, 2, 7, 7, 5, 5, 4, 4, 2,
                    0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0]),
    ("Cumpleaños feliz", [7, 7, 9, 7, 12, 11, 7, 7, 9, 7, 14, 12,
                          7, 7, 19, 16, 12, 11, 9, 17, 17, 16, 12, 14, 12]),
]
NOMBRES_NOTAS = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]


def mano_principal(obs):
    return max(obs, key=lambda o: o.tam) if obs else None


def cabecera(frame, ctx, titulo, derecha=""):
    k = ctx.k
    W = frame.shape[1]
    ui.panel(frame, 0, 0, W, int(54 * k), (20, 16, 14), 0.72, 0)
    ui.texto(frame, titulo, int(16 * k), int(10 * k), int(26 * k), ui.ORO, negrita=True)
    if derecha:
        ui.texto(frame, derecha, W - int(250 * k), int(14 * k), int(18 * k), ui.BLANCO, "d")


class ModoBase:
    id = "base"
    titulo = ""
    usa_piano = True
    ayuda = ""

    def __init__(self, ctx):
        self.ctx = ctx
        self.ad = None

    def iniciar(self, t):
        pass

    def evento(self, ev, t):
        pass

    def actualizar(self, t, obs):
        pass

    def resaltes(self, t):
        return {}

    def numeros(self):
        return {}

    def dibujar(self, frame, t):
        pass

    def tecla(self, c):
        pass

    def cerrar(self):
        pass

    def nivel(self):
        return self.ad.nivel if self.ad else None

    def resumen(self):
        b = self.ctx.sesion.b
        if not b:
            return []
        r = Sesion.resumen(b)
        out = [("Duración", f"{r['dur_s'] if r['dur_s'] else 0:.0f} s")]
        if r["ensayos"]:
            out += [("Ensayos", str(r["ensayos"])), ("Precisión", f"{r['precision'] * 100:.0f} %")]
            if r["rt_medio"] is not None:
                out.append(("Reacción media", f"{r['rt_medio']:.2f} s"))
        if r["pulsaciones"]:
            out.append(("Notas tocadas", str(r["pulsaciones"])))
        if self.ad:
            out.append(("Nivel actual", f"{self.ad.nivel}/5"))
        return out


# ------------------------------------------------------------
# Piano libre
# ------------------------------------------------------------
class ModoLibre(ModoBase):
    id, titulo = "libre", "Piano libre"
    ayuda = "Toca como quieras. Baja el dedo como si pulsaras una tecla."

    def iniciar(self, t):
        self.notas, self.ultima, self.t_nota = 0, "", 0.0

    def evento(self, ev, t):
        if ev.tipo == "on":
            self.notas += 1
            self.ultima, self.t_nota = NOMBRES_NOTAS[ev.nota % 12], t

    def dibujar(self, frame, t):
        c = self.ctx
        cabecera(frame, c, "Piano libre", f"Notas: {self.notas}")
        if t - self.t_nota < 0.6 and self.ultima:
            ui.texto(frame, self.ultima, frame.shape[1] // 2, int(80 * c.k), int(64 * c.k), ui.ORO, "c", True)


# ------------------------------------------------------------
# Canciones guiadas (adaptativo)
# ------------------------------------------------------------
class ModoCancion(ModoBase):
    id, titulo = "cancion", "Canciones guiadas"
    ayuda = "Toca la tecla dorada. Cuanto mejor lo hagas, más larga y menos guiada será la canción."
    LONGS = [6, 10, 16, 24, 999]
    RT_OBJ = 3.0

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ad = Adaptativa(ctx.niveles, "cancion")
        self.idx = 0

    def iniciar(self, t):
        self._nueva(t)

    def _nueva(self, t):
        n = self.LONGS[self.ad.nivel - 1]
        self.notas = CANCIONES[self.idx][1][:n]
        self.pos, self.fallos, self.fin = 0, 0, None
        self.t_obj = t
        self.msg_nivel = ""

    def tecla(self, c):
        if c in (ord("s"), ord("S")):
            self.idx = (self.idx + 1) % len(CANCIONES)
            self._nueva(self.t_obj)

    def objetivo(self, t):
        if self.fin is not None:
            return None
        return self.notas[self.pos]

    def evento(self, ev, t):
        if ev.tipo != "on" or self.fin is not None:
            return
        S = self.ctx.sesion
        if ev.nota == self.notas[self.pos]:
            S.ensayo(True, min(t - self.t_obj, 10.0), ev.dedo, ev.mano)
            self.pos += 1
            self.t_obj = t
            if self.pos >= len(self.notas):
                total = len(self.notas)
                prec = total / (total + self.fallos)
                rts = [e["rt"] for e in S.b["ensayos"][-total:] if e["ok"] and e["rt"] is not None]
                cambio, nuevo = self.ad.evaluar(prec, float(np.mean(rts)) if rts else None, self.RT_OBJ)
                self.msg_nivel = {1: f"¡Subes al nivel {nuevo}!", -1: f"Bajamos al nivel {nuevo}: fragmento más corto",
                                  0: f"Mantienes el nivel {nuevo}"}[cambio]
                self.res = (prec, float(np.mean(rts)) if rts else 0.0)
                self.fin = t
                self.ctx.audio.fin()
        else:
            self.fallos += 1
            S.ensayo(False, None, ev.dedo, ev.mano)

    def _ayuda_visible(self, t):
        n = self.ad.nivel
        if n <= 2:
            return True
        return (t - self.t_obj) > (3.0 if n == 3 else 6.0)

    def resaltes(self, t):
        o = self.objetivo(t)
        if o is None or not self._ayuda_visible(t):
            return {}
        return {o: (ORO, pulso(t))}

    def dibujar(self, frame, t):
        c, k = self.ctx, self.ctx.k
        if self.fin is not None and t - self.fin > 4.5:
            self._nueva(t)
        cabecera(frame, c, f"{CANCIONES[self.idx][0]}", f"Nivel {self.ad.nivel}/5   Fallos: {self.fallos}")
        W = frame.shape[1]
        ui.barra(frame, int(16 * k), int(60 * k), W - int(32 * k), int(8 * k), self.pos / max(len(self.notas), 1), ui.ORO)
        if self.fin is not None:
            ui.panel(frame, W // 2 - int(280 * k), int(110 * k), int(560 * k), int(150 * k), (20, 16, 14), 0.85, int(16 * k), ui.VERDE)
            ui.texto(frame, "¡Completada!", W // 2, int(122 * k), int(36 * k), ui.VERDE, "c", True)
            ui.texto(frame, f"Precisión {self.res[0] * 100:.0f} %  ·  {self.res[1]:.1f} s por nota", W // 2, int(172 * k), int(22 * k), ui.BLANCO, "c")
            ui.texto(frame, self.msg_nivel, W // 2, int(208 * k), int(22 * k), ui.ORO, "c", True)
            return
        sig = "  ".join(NOMBRES_NOTAS[n % 12] for n in self.notas[self.pos:self.pos + 5])
        ui.texto(frame, "Siguientes:", int(16 * k), int(80 * k), int(18 * k), ui.GRIS)
        ui.texto(frame, sig, int(130 * k), int(76 * k), int(26 * k), ui.BLANCO, negrita=True)
        if not self._ayuda_visible(t):
            ui.texto(frame, "Modo memoria: sin tecla guía (la ayuda aparece si te atascas)", int(16 * k), int(112 * k), int(16 * k), ui.GRIS)
        ui.texto(frame, "S = cambiar canción", W - int(16 * k), int(80 * k), int(16 * k), ui.GRIS, "d")


# ------------------------------------------------------------
# Dedo a dedo (individuacion)
# ------------------------------------------------------------
class ModoDedoADedo(ModoBase):
    id, titulo = "dedo_a_dedo", "Dedo a dedo"
    ayuda = "Pon los 5 dedos sobre las teclas numeradas (1 = pulgar). Pulsa la tecla dorada con el dedo indicado."
    BLOQUE = 10
    LIMITE = [None, None, 8.0, 5.0, 3.2]
    RT_OBJ = [3.5, 3.0, 2.5, 2.0, 1.6]

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ad = Adaptativa(ctx.niveles, "dedo_a_dedo")

    def iniciar(self, t):
        self._bloque(t)

    def _secuencia(self):
        n, lvl = self.BLOQUE, self.ad.nivel
        if lvl == 1:
            return [i % 5 for i in range(n)]
        if lvl == 2:
            base = [0, 1, 2, 3, 4, 3, 2, 1]
            return [base[i % len(base)] for i in range(n)]
        seq, prev = [], -1
        for _ in range(n):
            x = random.choice([i for i in range(5) if i != prev])
            seq.append(x)
            prev = x
        return seq

    def _bloque(self, t):
        self.seq, self.i = self._secuencia(), 0
        self.ok = self.err = self.fuera = 0
        self.rts, self.fin, self.msg = [], None, ""
        self.t_msg = 0.0
        self._siguiente(t, 0.4)

    def _siguiente(self, t, pausa=0.35):
        self.obj = None
        self.t_ini = t + pausa

    def _esperado(self, obj, mano):
        return obj if mano != "Left" else 4 - obj

    def evento(self, ev, t):
        if ev.tipo != "on" or self.obj is None or self.fin is not None:
            return
        S = self.ctx.sesion
        if ev.nota == CLAVES[self.obj]:
            esp = self._esperado(self.obj, ev.mano)
            dedo_ok = (ev.dedo == esp) if ev.mano in ("Left", "Right") else (ev.dedo in (self.obj, 4 - self.obj))
            rt = t - self.t_ini
            S.ensayo(True, rt, esp, ev.mano, dedo_ok)
            self.rts.append(rt)
            self.ok += 1
            self.msg, self.t_msg = ("¡Bien!" if dedo_ok else "¡Tecla correcta! (pero con otro dedo)"), t
            self._avanzar(t)
        elif ev.nota in CLAVES:
            self.err += 1
            S.ensayo(False, None, self._esperado(self.obj, ev.mano), ev.mano, False)
            self.ctx.audio.mal()
            self.msg, self.t_msg = "Esa no: busca la tecla dorada", t

    def _avanzar(self, t):
        self.i += 1
        if self.i >= len(self.seq):
            tot = self.ok + self.err
            prec = self.ok / tot if tot else 0.0
            rt = float(np.mean(self.rts)) if self.rts else None
            cambio, nuevo = self.ad.evaluar(prec, rt, self.RT_OBJ[self.ad.nivel - 1])
            self.res = (prec, rt, cambio, nuevo)
            self.fin = t
            self.ctx.audio.fin()
        else:
            self._siguiente(t)

    def actualizar(self, t, obs):
        if self.fin is not None:
            if t - self.fin > 4.0:
                self._bloque(t)
            return
        if self.obj is None and t >= self.t_ini:
            self.obj = self.seq[self.i]
            self.t_ini = t
        lim = self.LIMITE[self.ad.nivel - 1]
        if self.obj is not None and lim and t - self.t_ini > lim:
            self.fuera += 1
            self.ctx.sesion.ensayo(False, None, self.obj, None, None)
            self.msg, self.t_msg = "Tiempo agotado", t
            self.ctx.audio.mal()
            self._avanzar(t)

    def resaltes(self, t):
        if self.obj is None or self.fin is not None:
            return {}
        return {CLAVES[self.obj]: (ORO, pulso(t))}

    def numeros(self):
        return {c: str(i + 1) for i, c in enumerate(CLAVES)}

    def dibujar(self, frame, t):
        c, k = self.ctx, self.ctx.k
        W = frame.shape[1]
        cabecera(frame, c, "Dedo a dedo", f"Nivel {self.ad.nivel}/5   {min(self.i, len(self.seq))}/{len(self.seq)}   ✓{self.ok} ✗{self.err + self.fuera}")
        if self.fin is not None:
            p, rt, cambio, nuevo = self.res
            ui.panel(frame, W // 2 - int(290 * k), int(90 * k), int(580 * k), int(150 * k), (20, 16, 14), 0.85, int(16 * k), ui.VERDE)
            ui.texto(frame, "¡Bloque completado!", W // 2, int(102 * k), int(34 * k), ui.VERDE, "c", True)
            ui.texto(frame, f"Precisión {p * 100:.0f} %" + (f"  ·  {rt:.1f} s por tecla" if rt else ""), W // 2, int(150 * k), int(22 * k), ui.BLANCO, "c")
            txt = {1: f"¡Subes al nivel {nuevo}!", -1: f"Bajamos al nivel {nuevo} para afianzar", 0: f"Mantienes el nivel {nuevo}"}[cambio]
            ui.texto(frame, txt, W // 2, int(190 * k), int(22 * k), ui.ORO, "c", True)
            return
        if self.obj is not None:
            m = f"Mano derecha: {NOMBRES_DEDOS[self.obj].upper()}   ·   Mano izquierda: {NOMBRES_DEDOS[4 - self.obj].upper()}"
            if c.filtro == "Right":
                m = f"Usa el dedo: {NOMBRES_DEDOS[self.obj].upper()}"
            elif c.filtro == "Left":
                m = f"Usa el dedo: {NOMBRES_DEDOS[4 - self.obj].upper()}"
            ui.panel(frame, W // 2 - int(330 * k), int(66 * k), int(660 * k), int(50 * k), (20, 16, 14), 0.8, int(12 * k))
            ui.texto(frame, m, W // 2, int(76 * k), int(26 * k), ui.ORO, "c", True)
        lim = self.LIMITE[self.ad.nivel - 1]
        if lim and self.obj is not None:
            ui.barra(frame, W // 2 - int(200 * k), int(124 * k), int(400 * k), int(8 * k), 1 - (t - self.t_ini) / lim, ui.AZUL)
        if t - self.t_msg < 1.2 and self.msg:
            ui.texto(frame, self.msg, W // 2, int(146 * k), int(22 * k), ui.BLANCO, "c", True)


# ------------------------------------------------------------
# Simon dice (memoria)
# ------------------------------------------------------------
class ModoSimon(ModoBase):
    id, titulo = "simon", "Simón dice"
    ayuda = "Memoriza la secuencia de notas y repítela. Cada acierto la alarga una nota."
    VEL = [0.9, 0.75, 0.62, 0.5, 0.42]
    RONDAS_BLOQUE = 4

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ad = Adaptativa(ctx.niveles, "simon")

    def iniciar(self, t):
        self.long = 2 + (self.ad.nivel - 1) // 2
        self.span_max = 0
        self.rondas = self.exitos = 0
        self._ronda(t)

    def _ronda(self, t):
        self.seq = [random.randrange(5) for _ in range(self.long)]
        self.fase, self.t_fase, self.k_mira = "espera", t + 0.9, 0
        self.pos, self.msg = 0, ""
        self.t_turno = 0.0

    def evento(self, ev, t):
        if ev.tipo != "on" or self.fase != "turno" or ev.nota not in CLAVES:
            return
        S = self.ctx.sesion
        idx = CLAVES.index(ev.nota)
        if idx == self.seq[self.pos]:
            self.pos += 1
            if self.pos >= len(self.seq):
                self.exitos += 1
                self.rondas += 1
                self.span_max = max(self.span_max, len(self.seq))
                S.ensayo(True, (t - self.t_turno) / len(self.seq), None, ev.mano)
                S.extra(span_max=self.span_max)
                self.msg, self.fase, self.t_fase = "¡Perfecto!", "feedback", t + 1.2
                self.long = min(self.long + 1, 9)
                self.ctx.audio.ok()
        else:
            self.rondas += 1
            S.ensayo(False, None, None, ev.mano)
            self.msg, self.fase, self.t_fase = "¡Ups! Vamos con una más corta", "feedback", t + 1.5
            self.long = max(2, self.long - 1)
            self.ctx.audio.mal()

    def actualizar(self, t, obs):
        v = self.VEL[self.ad.nivel - 1]
        if self.fase == "espera" and t >= self.t_fase:
            self.fase, self.t_fase, self.k_mira, self.sonado = "mira", t, 0, False
        elif self.fase == "mira":
            if t >= self.t_fase:
                if self.k_mira >= len(self.seq):
                    self.fase, self.pos, self.t_turno = "turno", 0, t
                else:
                    self.ctx.audio.tocar(CLAVES[self.seq[self.k_mira]], 0.8)
                    self.t_nota = t
                    self.k_mira += 1
                    self.t_fase = t + v
        elif self.fase == "turno":
            if t - self.t_turno - self.pos * 6.0 > 8.0 + 3.0 * len(self.seq):     # se ha quedado parado
                self.rondas += 1
                self.ctx.sesion.ensayo(False, None, None, None)
                self.msg, self.fase, self.t_fase = "Tiempo agotado", "feedback", t + 1.2
                self.long = max(2, self.long - 1)
        elif self.fase == "feedback" and t >= self.t_fase:
            if self.rondas and self.rondas % self.RONDAS_BLOQUE == 0:
                prec = self.exitos / self.rondas
                cambio, nuevo = self.ad.evaluar(prec, None, 99)
                self.ctx.avisos.poner({1: f"¡Nivel {nuevo}!  (más rápido)", -1: f"Nivel {nuevo}: más calma", 0: f"Nivel {nuevo}"}[cambio], t, ui.ORO, 2.2)
                self.rondas = self.exitos = 0
                if cambio:
                    self.long = max(2, self.long + cambio)
            self._ronda(t)

    def resaltes(self, t):
        r = {}
        for i, c in enumerate(CLAVES):
            r[c] = (COLORES_SIMON[i], 0.22)
        if self.fase == "mira" and self.k_mira > 0 and t - getattr(self, "t_nota", -9) < self.VEL[self.ad.nivel - 1] * 0.7:
            i = self.seq[self.k_mira - 1]
            r[CLAVES[i]] = (COLORES_SIMON[i], 1.0)
        return r

    def dibujar(self, frame, t):
        c, k = self.ctx, self.ctx.k
        W = frame.shape[1]
        cabecera(frame, c, "Simón dice", f"Nivel {self.ad.nivel}/5   Récord: {self.span_max} notas")
        txt = {"espera": "Prepárate...", "mira": "Memoriza", "turno": "¡Tu turno!", "feedback": self.msg}[self.fase]
        col = {"espera": ui.GRIS, "mira": ui.AZUL, "turno": ui.VERDE, "feedback": ui.ORO}[self.fase]
        ui.texto(frame, txt, W // 2, int(70 * k), int(40 * k), col, "c", True)
        n = len(self.seq)
        x0 = W // 2 - (n * int(26 * k)) // 2
        for i in range(n):
            lleno = (self.fase == "turno" and i < self.pos) or (self.fase == "mira" and i < self.k_mira)
            cx = x0 + i * int(26 * k) + int(13 * k)
            import cv2
            cv2.circle(frame, (cx, int(138 * k)), int(9 * k), COLORES_SIMON[self.seq[i]] if lleno else (90, 90, 90), -1 if lleno else 2, cv2.LINE_AA)

    def resumen(self):
        return super().resumen() + [("Secuencia más larga", f"{self.span_max} notas")]


# ------------------------------------------------------------
# Pinza (oposicion pulgar-dedo)
# ------------------------------------------------------------
class ModoPinza(ModoBase):
    id, titulo, usa_piano = "pinza", "Pinza", False
    ayuda = "Junta la punta del pulgar con la del dedo que te indique el juego."
    BLOQUE = 12
    LIMITE = [None, 10.0, 8.0, 6.0, 4.0]
    RT_OBJ = [4.0, 3.5, 3.0, 2.5, 2.0]
    ENTRA, SALE = 0.30, 0.45
    NOTAS = [12, 14, 16, 19]

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ad = Adaptativa(ctx.niveles, "pinza")

    def iniciar(self, t):
        self._bloque(t)

    def _bloque(self, t):
        lvl = self.ad.nivel
        if lvl <= 2:
            self.seq = [i % 4 for i in range(self.BLOQUE)]
        else:
            self.seq, prev = [], -1
            for _ in range(self.BLOQUE):
                x = random.choice([j for j in range(4) if j != prev])
                self.seq.append(x)
                prev = x
        self.i, self.ok, self.err = 0, 0, 0
        self.rts, self.fin, self.msg, self.t_msg = [], None, "", 0.0
        self.pinza = [False] * 4
        self.rearmar = False
        self.t_ini, self.t_libre = t + 0.5, 0.0
        self.obs = None

    def actualizar(self, t, obs):
        o = mano_principal(obs)
        self.obs = o
        if self.fin is not None:
            if t - self.fin > 4.0:
                self._bloque(t)
            return
        if o is None:
            self.t_ini = max(self.t_ini, t)       # sin mano no corre el tiempo
            return
        d = distancias_pinza(o)
        nuevos = []
        for j in range(4):
            if not self.pinza[j] and d[j] < self.ENTRA:
                self.pinza[j] = True
                nuevos.append(j)
            elif self.pinza[j] and d[j] > self.SALE:
                self.pinza[j] = False
        if self.rearmar:
            if not any(self.pinza) and t - self.t_libre > 0.4:
                self.rearmar = False
            return
        if t < self.t_ini:
            return
        obj = self.seq[self.i]
        S = self.ctx.sesion
        if nuevos:
            j = min(nuevos, key=lambda q: d[q])
            self.ctx.audio.tocar(self.NOTAS[j], 0.8)
            if j == obj:
                rt = t - self.t_ini
                S.ensayo(True, rt, obj + 1, o.etiqueta, True)
                self.rts.append(rt)
                self.ok += 1
                self.msg, self.t_msg = "¡Bien!", t
                self._avanzar(t)
            else:
                self.err += 1
                S.ensayo(False, None, obj + 1, o.etiqueta, False)
                self.msg, self.t_msg = "Ese no: mira el dedo marcado", t
                self.ctx.audio.mal()
                self.rearmar, self.t_libre = True, t
            return
        lim = self.LIMITE[self.ad.nivel - 1]
        if lim and t - self.t_ini > lim:
            S.ensayo(False, None, obj + 1, o.etiqueta, None)
            self.msg, self.t_msg = "Tiempo agotado", t
            self.ctx.audio.mal()
            self._avanzar(t)

    def _avanzar(self, t):
        self.i += 1
        self.rearmar, self.t_libre = True, t
        if self.i >= len(self.seq):
            tot = self.ok + self.err
            prec = self.ok / tot if tot else 0.0
            rt = float(np.mean(self.rts)) if self.rts else None
            cambio, nuevo = self.ad.evaluar(prec, rt, self.RT_OBJ[self.ad.nivel - 1])
            self.res = (prec, rt, cambio, nuevo)
            self.fin = t
            self.ctx.audio.fin()
        else:
            self.t_ini = t + 0.5

    def dibujar(self, frame, t):
        import cv2
        c, k = self.ctx, self.ctx.k
        W = frame.shape[1]
        cabecera(frame, c, "Pinza", f"Nivel {self.ad.nivel}/5   {min(self.i, len(self.seq))}/{len(self.seq)}   ✓{self.ok} ✗{self.err}")
        if self.fin is not None:
            p, rt, cambio, nuevo = self.res
            ui.panel(frame, W // 2 - int(290 * k), int(100 * k), int(580 * k), int(150 * k), (20, 16, 14), 0.85, int(16 * k), ui.VERDE)
            ui.texto(frame, "¡Bloque completado!", W // 2, int(112 * k), int(34 * k), ui.VERDE, "c", True)
            ui.texto(frame, f"Precisión {p * 100:.0f} %" + (f"  ·  {rt:.1f} s por pinza" if rt else ""), W // 2, int(160 * k), int(22 * k), ui.BLANCO, "c")
            ui.texto(frame, {1: f"¡Subes al nivel {nuevo}!", -1: f"Bajamos al nivel {nuevo}", 0: f"Mantienes el nivel {nuevo}"}[cambio], W // 2, int(200 * k), int(22 * k), ui.ORO, "c", True)
            return
        obj = self.seq[self.i]
        ui.panel(frame, W // 2 - int(330 * k), int(66 * k), int(660 * k), int(50 * k), (20, 16, 14), 0.8, int(12 * k))
        ui.texto(frame, f"Une el PULGAR con el {NOMBRES_DEDOS[obj + 1].upper()}", W // 2, int(76 * k), int(26 * k), ui.ORO, "c", True)
        lim = self.LIMITE[self.ad.nivel - 1]
        if lim:
            ui.barra(frame, W // 2 - int(200 * k), int(124 * k), int(400 * k), int(8 * k), max(0.0, 1 - (t - self.t_ini) / lim), ui.AZUL)
        if self.obs is not None:
            for p, col in ((4, ui.VERDE), (PUNTAS[obj + 1], ui.ORO)):
                x, y = self.obs.pts[p]
                r = int((20 + 6 * math.sin(t * 8)) * k)
                cv2.circle(frame, (int(x), int(y)), r, col, 4, cv2.LINE_AA)
        else:
            ui.texto(frame, "Pon la mano delante de la cámara", W // 2, int(150 * k), int(22 * k), ui.GRIS, "c")
        if t - self.t_msg < 1.2 and self.msg:
            ui.texto(frame, self.msg, W // 2, int(146 * k), int(22 * k), ui.BLANCO, "c", True)


# ------------------------------------------------------------
# Puño/mano abierta y abanico (repeticiones con rango propio)
# ------------------------------------------------------------
class ModoRepeticiones(ModoBase):
    usa_piano = False
    BLOQUE = 8
    HI = [0.55, 0.65, 0.75, 0.85, 0.90]
    HOLD = [0.0, 0.0, 0.4, 0.8, 1.2]
    RT_OBJ = [6.0, 5.0, 4.0, 3.5, 3.0]
    LO = 0.30
    MEDIR_S = 7.0

    def __init__(self, ctx, ident, titulo, fn, min_rango, alto, bajo, ayuda):
        super().__init__(ctx)
        self.id, self.titulo, self.fn, self.min_rango = ident, titulo, fn, min_rango
        self.alto, self.bajo, self.ayuda = alto, bajo, ayuda
        self.ad = Adaptativa(ctx.niveles, ident)

    def iniciar(self, t):
        self.fase, self.muestras, self.t_fase = "medir", [], None
        self.lo = self.hi = 0.0
        self.s = 0.0
        self.msg = ""
        self.reps_tot, self.picos_tot, self.dur_tot = 0, [], []
        self._bloque()

    def _bloque(self):
        self.estado, self.reps, self.picos, self.durs = "subir", 0, [], []
        self.t_rep = self.t_hold = None
        self.pico = 0.0
        self.fin = None

    def actualizar(self, t, obs):
        o = mano_principal(obs)
        if o is None:
            self.s_vis = None
            if self.fase == "medir":
                self.t_fase = None
                self.muestras = []
            return
        raw = self.fn(o)
        if self.fase == "medir":
            if self.t_fase is None:
                self.t_fase = t
            self.muestras.append(raw)
            self.s = 0.5
            if t - self.t_fase >= self.MEDIR_S:
                lo, hi = np.percentile(self.muestras, 5), np.percentile(self.muestras, 95)
                if hi - lo < self.min_rango:
                    self.msg = "Rango muy pequeño: repite el movimiento más amplio"
                    self.t_fase, self.muestras = None, []
                else:
                    self.lo, self.hi = float(lo), float(hi)
                    self.fase = "ejer"
                    self.msg = ""
                    self.ctx.audio.ok()
            return
        self.s = float(np.clip((raw - self.lo) / max(self.hi - self.lo, 1e-6), 0, 1.2))
        if self.fin is not None:
            if t - self.fin > 4.0:
                self._bloque()
            return
        lvl = self.ad.nivel
        if self.t_rep is None:
            self.t_rep = t
        if self.estado == "subir":
            if self.s >= self.HI[lvl - 1]:
                if self.t_hold is None:
                    self.t_hold = t
                    self.ctx.audio.tocar(16, 0.7)
                self.pico = max(self.pico, self.s)
                if t - self.t_hold >= self.HOLD[lvl - 1]:
                    self.estado = "bajar"
            else:
                self.t_hold = None
        else:
            self.pico = max(self.pico, self.s)
            if self.s <= self.LO:
                dur = t - self.t_rep
                ok = dur <= 2 * self.RT_OBJ[lvl - 1]
                self.ctx.sesion.ensayo(ok, dur, None, o.etiqueta)
                self.reps += 1
                self.picos.append(self.pico)
                self.durs.append(dur)
                self.picos_tot.append(self.pico)
                self.dur_tot.append(dur)
                self.ctx.audio.tocar(4, 0.7)
                self.estado, self.t_rep, self.t_hold, self.pico = "subir", t, None, 0.0
                if self.reps >= self.BLOQUE:
                    self._fin_bloque(t)

    def _fin_bloque(self, t):
        lvl = self.ad.nivel
        rapidas = sum(1 for d in self.durs if d <= 2 * self.RT_OBJ[lvl - 1]) / len(self.durs)
        cambio, nuevo = self.ad.evaluar(rapidas, float(np.mean(self.durs)), self.RT_OBJ[lvl - 1])
        self.res = (float(np.mean(self.picos)) * 100, float(np.mean(self.durs)), cambio, nuevo)
        self.fin = t
        self.ctx.audio.fin()

    def cerrar(self):
        if self.fase == "ejer" and self.picos_tot:
            self.ctx.sesion.extra(rango_bruto=round(self.hi - self.lo, 3), pico_medio=round(float(np.mean(self.picos_tot)) * 100, 1),
                                  t_rep=round(float(np.mean(self.dur_tot)), 2), reps=len(self.picos_tot))

    def dibujar(self, frame, t):
        import cv2
        c, k = self.ctx, self.ctx.k
        H, W = frame.shape[:2]
        cabecera(frame, c, self.titulo, f"Nivel {self.ad.nivel}/5   Repeticiones: {self.reps}/{self.BLOQUE}" if self.fase == "ejer" else "Medición de tu rango")
        if self.fase == "medir":
            ui.panel(frame, W // 2 - int(340 * k), int(80 * k), int(680 * k), int(120 * k), (20, 16, 14), 0.85, int(16 * k), ui.AZUL)
            ui.texto(frame, f"Primero medimos tu rango: {self.alto} y {self.bajo} al máximo, 2-3 veces", W // 2, int(94 * k), int(24 * k), ui.BLANCO, "c", True)
            frac = 0 if self.t_fase is None else (t - self.t_fase) / self.MEDIR_S
            ui.barra(frame, W // 2 - int(250 * k), int(146 * k), int(500 * k), int(12 * k), frac, ui.AZUL)
            if self.msg:
                ui.texto(frame, self.msg, W // 2, int(170 * k), int(18 * k), ui.ORO, "c")
            if self.t_fase is None:
                ui.texto(frame, "Pon la mano delante de la cámara", W // 2, int(170 * k), int(18 * k), ui.GRIS, "c")
            return
        if self.fin is not None:
            p, dur, cambio, nuevo = self.res
            ui.panel(frame, W // 2 - int(290 * k), int(90 * k), int(580 * k), int(150 * k), (20, 16, 14), 0.85, int(16 * k), ui.VERDE)
            ui.texto(frame, "¡Serie completada!", W // 2, int(102 * k), int(34 * k), ui.VERDE, "c", True)
            ui.texto(frame, f"Rango alcanzado {p:.0f} %  ·  {dur:.1f} s por repetición", W // 2, int(150 * k), int(22 * k), ui.BLANCO, "c")
            ui.texto(frame, {1: f"¡Subes al nivel {nuevo}!", -1: f"Bajamos al nivel {nuevo}", 0: f"Mantienes el nivel {nuevo}"}[cambio], W // 2, int(190 * k), int(22 * k), ui.ORO, "c", True)
        else:
            orden = (self.alto if self.estado == "subir" else self.bajo).upper()
            ui.panel(frame, W // 2 - int(220 * k), int(66 * k), int(440 * k), int(54 * k), (20, 16, 14), 0.8, int(12 * k))
            ui.texto(frame, orden, W // 2, int(74 * k), int(34 * k), ui.ORO if self.estado == "subir" else ui.AZUL, "c", True)
        # medidor vertical
        gx, gy, gh, gw = W - int(80 * k), int(150 * k), int(H * 0.5), int(36 * k)
        ui.panel(frame, gx, gy, gw, gh, (40, 36, 34), 0.85, int(10 * k), ui.GRIS_OSC)
        meta = self.HI[self.ad.nivel - 1]
        cv2.line(frame, (gx - 8, int(gy + gh * (1 - meta))), (gx + gw + 8, int(gy + gh * (1 - meta))), ui.ORO, 3)
        cv2.line(frame, (gx - 8, int(gy + gh * (1 - self.LO))), (gx + gw + 8, int(gy + gh * (1 - self.LO))), ui.AZUL, 3)
        h = int(gh * min(self.s, 1.0))
        if h > 2:
            ui.panel(frame, gx + 3, gy + gh - h, gw - 6, h, ui.VERDE, 0.95, int(6 * k))

    def resumen(self):
        out = super().resumen()
        if self.picos_tot:
            out += [("Rango medio alcanzado", f"{np.mean(self.picos_tot) * 100:.0f} %"), ("Repeticiones", str(len(self.picos_tot)))]
        return out


def modo_puno(ctx):
    return ModoRepeticiones(ctx, "puno", "Puño y mano abierta", apertura, 0.25, "Abre la mano", "Cierra el puño",
                            "Abre y cierra la mano completamente. Cada repetición suena.")


def modo_abanico(ctx):
    return ModoRepeticiones(ctx, "abanico", "Abanico", separacion, 0.30, "Separa los dedos", "Junta los dedos",
                            "Separa los dedos en abanico y júntalos de nuevo.")


# ------------------------------------------------------------
# Estabilidad (temblor)
# ------------------------------------------------------------
class ModoEstabilidad(ModoBase):
    id, titulo, usa_piano = "estabilidad", "Prueba de estabilidad", False
    ayuda = "Extiende la mano delante de la cámara y mantenla quieta 10 segundos."
    PREP_S, MEDIR_S = 3.0, 10.0

    def iniciar(self, t):
        self.fase, self.t_fase, self.muestras, self.res, self.etq = "prep", None, [], None, None
        self.err = ""
        self.o = None

    def tecla(self, c):
        if c in (ord("r"), ord("R")):
            self.iniciar(0.0)

    def actualizar(self, t, obs):
        o = mano_principal(obs)
        self.o = o
        if self.fase == "res":
            return
        if o is None:
            self.t_fase, self.muestras = None, []
            return
        if self.t_fase is None:
            self.t_fase = t
        self.etq = o.etiqueta
        if self.fase == "prep" and t - self.t_fase >= self.PREP_S:
            self.fase, self.t_fase, self.muestras = "medir", t, []
        if self.fase == "medir":
            self.muestras.append((t, o.pts[PUNTAS].copy(), o.tam))
            if t - self.t_fase >= self.MEDIR_S:
                r = analizar_temblor(self.muestras)
                if r is None:
                    self.err = "No se pudo medir. Repite con la mano bien visible."
                    self.fase, self.t_fase, self.muestras = "prep", None, []
                else:
                    self.res = r
                    self.fase = "res"
                    self.ctx.sesion.extra(mano=self.etq or "—", **{k: round(v, 3) for k, v in r.items()})
                    self.ctx.audio.fin()

    def dibujar(self, frame, t):
        c, k = self.ctx, self.ctx.k
        W = frame.shape[1]
        cabecera(frame, c, "Prueba de estabilidad")
        ui.panel(frame, W // 2 - int(330 * k), int(70 * k), int(660 * k), int(130 * k), (20, 16, 14), 0.85, int(16 * k), ui.AZUL)
        if self.fase == "prep":
            if self.t_fase is None:
                ui.texto(frame, "Extiende una mano delante de la cámara", W // 2, int(90 * k), int(26 * k), ui.BLANCO, "c", True)
                if self.err:
                    ui.texto(frame, self.err, W // 2, int(140 * k), int(18 * k), ui.ORO, "c")
            else:
                ui.texto(frame, f"Prepárate... {max(0, self.PREP_S - (t - self.t_fase)):.0f}", W // 2, int(90 * k), int(34 * k), ui.ORO, "c", True)
                ui.texto(frame, "Mano extendida y quieta", W // 2, int(140 * k), int(20 * k), ui.GRIS, "c")
        elif self.fase == "medir":
            ui.texto(frame, "¡Quieta! Midiendo...", W // 2, int(86 * k), int(32 * k), ui.VERDE, "c", True)
            ui.barra(frame, W // 2 - int(250 * k), int(140 * k), int(500 * k), int(12 * k), (t - self.t_fase) / self.MEDIR_S, ui.VERDE)
        else:
            r = self.res
            ui.texto(frame, f"Temblor (4-12 Hz): {r['rms_mm']:.2f} mm   ·   Frecuencia: {r['frecuencia_hz']:.1f} Hz", W // 2, int(84 * k), int(24 * k), ui.BLANCO, "c", True)
            ui.texto(frame, f"Deriva lenta: {r['deriva_mm']:.2f} mm   ·   cámara a {r['fps_reales']:.0f} fps", W // 2, int(122 * k), int(20 * k), ui.GRIS, "c")
            ui.texto(frame, "Orientativo (incluye ruido de la cámara ≈0,3-0,6 mm). R = repetir", W // 2, int(156 * k), int(16 * k), ui.GRIS, "c")

    def resumen(self):
        if not self.res:
            return []
        r = self.res
        return [("Temblor 4-12 Hz", f"{r['rms_mm']:.2f} mm"), ("Frecuencia", f"{r['frecuencia_hz']:.1f} Hz"), ("Deriva lenta", f"{r['deriva_mm']:.2f} mm")]


MODOS = {
    "libre": ModoLibre, "cancion": ModoCancion, "dedo_a_dedo": ModoDedoADedo, "simon": ModoSimon,
    "pinza": ModoPinza, "puno": modo_puno, "abanico": modo_abanico, "estabilidad": ModoEstabilidad,
}


# ======================================================================
# SECCION: demo sin camara
# ======================================================================
# ============================================================
# MODO DEMO (sin camara): una mano simulada toca sola. Sirve para ver la
# interfaz y oir los sonidos sin webcam:   python mediapianist.py --demo
# ============================================================
class ManoSim:
    _DEDOS = {1: ((-0.030, -0.090), (0.043, 0.025, 0.020)), 2: ((-0.008, -0.095), (0.048, 0.028, 0.021)),
              3: ((0.014, -0.090), (0.044, 0.026, 0.020)), 4: ((0.034, -0.080), (0.034, 0.019, 0.017))}
    _REPOSO = np.radians([8.0, 12.0, 6.0])
    _GOLPE = np.radians([20.0, 25.0, 8.0])

    def __init__(self, cx, cy, escala=1570.0, seed=0):
        self.cx, self.cy, self.S = cx, cy, escala
        self.rng = np.random.default_rng(seed)
        self.golpes = []          # (t0, dedo, dur)
        self.t_sig = 1.0

    @staticmethod
    def _dir(d0, phi):
        d = d0 * math.cos(phi) + np.array([0.0, 0.0, -1.0]) * math.sin(phi)
        return d / np.linalg.norm(d)

    def _puntos(self, ph):
        P = np.zeros((21, 3))
        P[1], P[2] = [-0.025, -0.015, 0], [-0.045, -0.040, 0]
        d0 = np.array([-0.5, -0.85, 0.0]) / np.linalg.norm([-0.5, -0.85])
        P[3] = P[2] + 0.035 * self._dir(d0, ph[0][0])
        P[4] = P[3] + 0.030 * self._dir(d0, ph[0][0] + ph[0][1])
        for f, ((mx, my), L) in self._DEDOS.items():
            i0 = 1 + 4 * f
            P[i0] = [mx, my, 0.0]
            ang = 0.0
            for s in range(3):
                ang += ph[f][s]
                P[i0 + 1 + s] = P[i0 + s] + L[s] * self._dir(np.array([0.0, -1.0, 0.0]), ang)
        return P

    def obs(self, t):
        if t >= self.t_sig:                       # programa un golpe nuevo cada ~0.5-0.9 s
            self.golpes.append((t, int(self.rng.integers(0, 5)), 0.22))
            self.t_sig = t + float(self.rng.uniform(0.5, 0.9))
        self.golpes = [g for g in self.golpes if t - g[0] < 0.5]
        ph = {f: self._REPOSO.copy() for f in range(5)}
        for t0, f, dur in self.golpes:
            if 0 <= t - t0 <= dur:
                ph[f] = ph[f] + self._GOLPE * 0.5 * (1 - math.cos(2 * math.pi * (t - t0) / dur))
        P = self._puntos(ph)
        dx = 380 * math.sin(t * 0.35)
        pts = np.stack([self.cx + dx + self.S * P[:, 0], self.cy + self.S * P[:, 1]], axis=1)
        pts += self.rng.normal(0, 1.0, pts.shape)
        w3 = P + self.rng.normal(0, 0.0015, P.shape)
        return Obs(pts, w3, "Right", t)


def main_demo(args):
    app = App(args.usuario, args.silencio)
    ancho, alto = 1280, 720
    mano = ManoSim(ancho / 2, alto * 0.97)
    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(NOMBRE_VENTANA, app.on_mouse)
    base = np.zeros((alto, ancho, 3), np.uint8)
    for y in range(alto):
        base[y, :] = (70 + y // 14, 60 + y // 16, 55 + y // 18)
    print("MODO DEMO: haz clic en un modo del menu. Una mano simulada toca sola. ESC para salir.")
    t0 = time.monotonic()
    while not app.salir:
        t = time.monotonic() - t0
        fr = base.copy()
        app.paso(fr, [mano.obs(t)], t)
        cv2.imshow(NOMBRE_VENTANA, fr)
        app.tecla(cv2.waitKey(16) & 0xFF, t)
        if cv2.getWindowProperty(NOMBRE_VENTANA, cv2.WND_PROP_VISIBLE) < 1:
            break
    app.guardar_sesion()
    if hasattr(app.audio, "cerrar"):
        app.audio.cerrar()
    cv2.destroyAllWindows()




# ======================================================================
# SECCION: app principal
# MediaPianist - mediapianist.py
# ==============================
# Piano virtual con vision por computador para practicar y rehabilitar las manos.
# 
#     python mediapianist.py                  (usuario "invitado")
#     python mediapianist.py --usuario ana    (cada usuario tiene su calibracion, niveles e informe)
# 
# Se maneja con RATON o con el DEDO: mantén la punta del índice sobre un botón ~1 s.
# Teclas utiles: ESC volver · F pantalla completa · +/- sensibilidad · M mano activa ·
# X intercambiar izquierda/derecha · H ver manos · B depurar · L nombres de teclas.
# ======================================================================



CARPETA = os.path.dirname(os.path.abspath(__file__))
NOMBRE_VENTANA = "MediaPianist"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_RUTAS = [os.path.join(CARPETA, "models", "hand_landmarker.task"), os.path.join(CARPETA, "hand_landmarker.task")]
CONFIG_PIANO = os.path.join(CARPETA, "piano_config.json")
CAM_ANCHO, CAM_ALTO = 1280, 720
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
COLOR_DEDOS = [ui.ROJO, ui.NARANJA, ui.VERDE, ui.AZUL, ui.MORADO]

# (id, titulo, descripcion, color, grupo)
MENU = [
    ("libre", "Piano libre", "Toca lo que quieras", ui.AZUL),
    ("cancion", "Canciones", "Melodías guiadas, dificultad adaptativa", ui.AZUL),
    ("simon", "Simón dice", "Memoria: repite la secuencia", ui.MORADO),
    ("dedo_a_dedo", "Dedo a dedo", "Un dedo por tecla (individuación)", ui.VERDE),
    ("pinza", "Pinza", "Pulgar con cada dedo (oposición)", ui.VERDE),
    ("puno", "Puño y mano abierta", "Abrir y cerrar con rango propio", ui.VERDE),
    ("abanico", "Abanico", "Separar y juntar los dedos", ui.VERDE),
    ("estabilidad", "Estabilidad", "Medir el temblor en 10 s", ui.NARANJA),
    ("calibrar", "Calibrar", "Ajusta la detección a tu mano", ui.ORO),
    ("colocar", "Colocar piano", "Elige dónde está el piano", ui.TURQUESA),
    ("mesa", "Calibrar mesa", "Corrige la perspectiva de la cámara", ui.TURQUESA),
    ("informe", "Mi progreso", "Informe con gráficas", ui.TURQUESA),
    ("sus", "Encuesta SUS", "Valora la usabilidad", ui.TURQUESA),
    ("salir", "Salir", "", ui.GRIS_OSC),
]
ADAPTATIVOS = {"cancion", "simon", "dedo_a_dedo", "pinza", "puno", "abanico"}


class SinAudio:
    """Audio mudo para pruebas sin altavoces."""
    def __getattr__(self, nombre):
        return lambda *a, **k: None


class Ctx:
    pass


class Camara:
    """Captura en un hilo aparte: siempre tenemos el ultimo frame (sin acumular retraso)."""

    def __init__(self, indice=0):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(indice)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(indice)
        try:                                                                  # MJPG: mas fps y menos latencia
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_ANCHO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_ALTO)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.t_cap, self.contador, self.activo = None, 0.0, 0, True
        self.lock = threading.Lock()
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while self.activo:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.monotonic()
            with self.lock:
                self.frame, self.t_cap = f, t
                self.contador += 1

    def leer(self, ultimo):
        limite = time.monotonic() + 2
        while time.monotonic() < limite:
            with self.lock:
                if self.frame is not None and self.contador != ultimo:
                    return self.frame.copy(), self.contador, self.t_cap
            time.sleep(0.001)
        return None, ultimo, 0.0

    def parar(self):
        self.activo = False
        self.hilo.join(timeout=1)
        self.cap.release()


# ------------------------------------------------------------
class App:
    def __init__(self, usuario="invitado", silencio=False, audio=None, headless=False):
        self.alm = Almacen(usuario)
        self.niveles = Niveles(self.alm)
        self.sesion = Sesion(self.alm)
        self.detector = Detector()
        calib = self.alm.cargar("calibracion.json")
        if calib:
            self.detector.aplicar_perfil(calib.get("perfil"))
        self.calibrado = bool(calib)
        self.audio = audio if audio is not None else (SinAudio() if headless else Audio(silencio))
        self.avisos = ui.Avisos()
        self.puntero = ui.Puntero()
        self.piano = None
        self.ancho = self.alto = 0
        self.estado = "MENU"
        self.modo = None
        self.cal = None
        self.sus = None
        self.resumen = []
        self.resumen_titulo = ""
        self.t_resumen = 0.0
        self.brillo = {}
        self.etiquetas, self.ver_manos, self.depurar = False, True, False
        self.filtro, self.invertir = "ambas", False
        self.mouse, self.t_mouse, self.click = (0, 0), -9.0, None
        self.drag = None
        self.fullscreen = False
        self.salir = False
        self.t_prev, self.fps = None, 30.0
        self.colocar = {"fase": 1, "cand": None, "t_ini": 0.0, "esq1": None}
        self.mesa_puntos = []
        self.ctx = Ctx()
        self.botones = []

    # ---------- piano ----------
    def _cargar_piano(self, ancho, alto):
        try:
            with open(CONFIG_PIANO) as f:
                cfg = json.load(f)
            x1, y1, x2, y2 = cfg["rect"]
            rw, rh = cfg["res"]
            quad = cfg.get("quad")
            if quad:
                quad = [[pt[0] * ancho / rw, pt[1] * alto / rh] for pt in quad]
            return Piano(x1 * ancho / rw, y1 * alto / rh, x2 * ancho / rw, y2 * alto / rh, ancho, alto, quad=quad)
        except (OSError, ValueError, KeyError):
            return self._piano_defecto(ancho, alto)

    def _piano_defecto(self, ancho, alto):
        return Piano(int(ancho * 0.04), int(alto * 0.48), int(ancho * 0.96), int(alto * 0.91), ancho, alto)

    def _guardar_piano(self):
        try:
            with open(CONFIG_PIANO, "w") as f:
                json.dump({"rect": list(self.piano.rect()), "res": [self.ancho, self.alto], "quad": self.piano.superficie.tolist() if self.piano.superficie is not None else None}, f)
        except OSError:
            pass

    def _preparar(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != (self.ancho, self.alto):
            self.ancho, self.alto = w, h
            self.piano = self._cargar_piano(w, h)
        k = h / 720.0
        c = self.ctx
        c.k, c.piano, c.audio, c.sesion, c.detector = k, self.piano, self.audio, self.sesion, self.detector
        c.niveles, c.avisos, c.filtro = self.niveles, self.avisos, self.filtro

    # ---------- navegacion ----------
    def ir(self, ident, t):
        if ident == "salir":
            self.salir = True
        elif ident in MODOS:
            self.iniciar_modo(ident, t)
        elif ident == "calibrar":
            self.cal, self.estado = Calibrador(), "CALIBRAR"
        elif ident == "colocar":
            self.colocar = {"fase": 1, "cand": None, "t_ini": 0.0, "esq1": None}
            self.estado = "COLOCAR"
        elif ident == "mesa":
            self.mesa_puntos = []
            self.estado = "MESA"
        elif ident == "informe":
            self.guardar_sesion()
            ruta = informe.generar(self.alm)
            informe.abrir(ruta)
            self.avisos.poner("Informe abierto en tu navegador", t, ui.VERDE, 3.0)
        elif ident == "sus":
            self.sus, self.estado = ui.PantallaSUS(), "SUS"

    def iniciar_modo(self, ident, t):
        self.modo = MODOS[ident](self.ctx)
        self.sesion.iniciar_bloque(ident, self.modo.titulo, self.modo.nivel())
        self.detector.reset()
        self.audio.soltar_todo()
        self.brillo.clear()
        self.estado = "MODO"
        self.modo.iniciar(t)
        if self.modo.ayuda:
            self.avisos.poner(self.modo.ayuda, t, ui.BLANCO, 4.5)

    def salir_modo(self, t):
        m = self.modo
        if m is None:
            self.estado = "MENU"
            return
        m.cerrar()
        self.sesion.cerrar_bloque()
        self.resumen, self.resumen_titulo = m.resumen(), m.titulo
        self.audio.soltar_todo()
        self.guardar_sesion()
        self.modo = None
        self.estado, self.t_resumen = "RESUMEN", t

    def guardar_sesion(self):
        if self.sesion.b is not None and (self.sesion.b["ensayos"] or self.sesion.b["pulsaciones"] or self.sesion.b["extra"]):
            pass
        self.sesion.guardar()

    # ---------- entrada ----------
    def on_mouse(self, evento, x, y, flags, param):
        self.mouse, self.t_mouse = (x, y), time.monotonic()
        if evento == cv2.EVENT_LBUTTONDOWN:
            self.drag = [x, y, x, y]
        elif evento == cv2.EVENT_MOUSEMOVE and self.drag is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            self.drag[2:] = [x, y]
        elif evento == cv2.EVENT_LBUTTONUP:
            if self.drag is not None:
                self.drag[2:] = [x, y]
            self.click = (x, y)
            if self.estado == "MESA":
                self.mesa_puntos.append((int(x), int(y)))
                if len(self.mesa_puntos) == 4:
                    puntos = np.asarray(self.mesa_puntos, dtype=np.float32)
                    contorno = puntos.astype(np.int32).reshape(-1, 1, 2)
                    area = abs(cv2.contourArea(contorno))
                    if cv2.isContourConvex(contorno) and area > 12000:
                        x1, y1 = puntos.min(axis=0)
                        x2, y2 = puntos.max(axis=0)
                        self.piano = Piano(x1, y1, x2, y2, self.ancho, self.alto, quad=puntos)
                        self._guardar_piano()
                        self.detector.reset()
                        self.estado = "MENU"
                        self.click = None
                        self.avisos.poner("Mesa calibrada: perspectiva corregida", time.monotonic(), ui.VERDE, 4.0)
                    else:
                        self.mesa_puntos = []
                        self.avisos.poner("Esquinas incorrectas. Vuelve a marcarlas en orden", time.monotonic(), ui.ROJO, 3.0)
                self.drag = None
            if self.estado == "COLOCAR" and self.drag is not None:
                x1, y1, x2, y2 = self.drag
                if abs(x2 - x1) > 200 and abs(y2 - y1) > 60:
                    self.piano = Piano(x1, y1, x2, y2, self.ancho, self.alto)
                    self._guardar_piano()
                    self.detector.reset()
                    self.estado = "MENU"
                    self.avisos.poner("Piano colocado y guardado", time.monotonic(), ui.VERDE)
                    self.click = None
            self.drag = None

    def tecla(self, c, t):
        if c == 255:
            return
        if self.estado == "SUS":
            if c == 27:
                self.estado = "MENU"
            elif self.sus.fin:
                self._guardar_sus()
                self.estado = "MENU"
            else:
                self.sus.tecla(c)
            return
        if self.estado == "RESUMEN" and t - self.t_resumen > 0.6:
            self.estado = "MENU"
            return
        if c == 27:
            if self.estado == "MENU":
                self.salir = True
            elif self.estado == "MODO":
                self.salir_modo(t)
            else:
                self.estado = "MENU"
            return
        ch = chr(c).lower() if 32 <= c < 127 else ""
        if ch == "f":
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(NOMBRE_VENTANA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif ch in ("+", "="):
            self.detector.sens = min(2.5, self.detector.sens + 0.1)
            self.avisos.poner(f"Sensibilidad {self.detector.sens:.1f}", t)
        elif ch in ("-", "_"):
            self.detector.sens = max(0.5, self.detector.sens - 0.1)
            self.avisos.poner(f"Sensibilidad {self.detector.sens:.1f}", t)
        elif ch == "h":
            self.ver_manos = not self.ver_manos
        elif ch == "b":
            self.depurar = not self.depurar
        elif ch == "l":
            self.etiquetas = not self.etiquetas
        elif ch == "x":
            self.invertir = not self.invertir
            self.avisos.poner("Izquierda/derecha intercambiadas" if self.invertir else "Izquierda/derecha normales", t)
        elif ch == "m":
            orden = ["ambas", "Right", "Left"]
            self.filtro = orden[(orden.index(self.filtro) + 1) % 3]
            self.avisos.poner({"ambas": "Ambas manos", "Right": "Solo mano derecha", "Left": "Solo mano izquierda"}[self.filtro], t, ui.ORO)
        elif ch == "t" and self.estado == "MENU":
            self.mesa_puntos = []
            self.estado = "MESA"
        elif ch == "d" and self.estado in ("MODO", "COLOCAR"): 
            self.piano = self._piano_defecto(self.ancho, self.alto)
            self.ctx.piano = self.piano
            self.detector.reset()
            if self.estado == "COLOCAR":
                self.estado = "MENU"
        elif ch == "r" and self.estado == "MODO" and self.modo.usa_piano:
            self.estado = "COLOCAR"
            self.colocar = {"fase": 1, "cand": None, "t_ini": 0.0, "esq1": None}
            self.salir_modo(t)
            self.estado = "COLOCAR"
        elif self.estado == "MODO":
            self.modo.tecla(c)

    def _guardar_sus(self):
        r = self.sus.resp
        if len(r) == 10:
            datos = self.alm.cargar("sus.json", []) or []
            datos.append({"fecha": datetime.now().isoformat(timespec="seconds"), "respuestas": r, "puntuacion": ui.puntuacion_sus(r)})
            self.alm.guardar("sus.json", datos)

    # ---------- un frame ----------
    def paso(self, frame, obs, t):
        """Logica + dibujo de UN frame (frame ya en espejo). obs: lista de Obs."""
        self._preparar(frame)
        k = self.ctx.k
        obs = normalizar_etiquetas(list(obs), self.invertir)
        if self.filtro != "ambas":
            obs = [o for o in obs if o.etiqueta in (None, self.filtro)]
        dt = 0.033 if self.t_prev is None else max(t - self.t_prev, 1e-3)
        self.t_prev = t
        self.fps = 0.92 * self.fps + 0.08 / dt
        click = self.click
        self.click = None
        usar_mouse = time.monotonic() - self.t_mouse < 3.0

        if self.estado == "MENU":
            self._menu(frame, obs, t, click, k)
        elif self.estado == "MODO":
            self._modo(frame, obs, t, dt, click, k)
        elif self.estado == "COLOCAR":
            self._colocar(frame, obs, t, k)
        elif self.estado == "MESA":
            self._calibrar_mesa(frame, k)
        elif self.estado == "CALIBRAR":
            self._calibrar(frame, obs, t, k)
        elif self.estado == "RESUMEN":
            self._resumen(frame, obs, t, click, k)
        elif self.estado == "SUS":
            if click:
                self.sus.click(*click)
            self.sus.dibujar(frame, k)

        if self.estado not in ("SUS",):
            self._dibujar_manos(frame, obs, t, k)
            if self.depurar and self.estado == "MODO":
                self._depuracion(frame, t, k)
            self.avisos.dibujar(frame, t, k, int(150 * k) if self.estado == "MODO" else None)
            ui.texto(frame, f"{self.fps:3.0f} fps · sens {self.detector.sens:.1f} · {'calibrado' if self.calibrado else 'sin calibrar'}"
                     + ("" if self.filtro == "ambas" else f" · {'derecha' if self.filtro == 'Right' else 'izquierda'}"),
                     self.ancho - int(12 * k), self.alto - int(26 * k), int(14 * k), ui.GRIS, "d")
        return frame

    # ---------- estados ----------
    def _puntas_dedo(self, obs):
        return [(int(o.pts[8][0]), int(o.pts[8][1])) for o in obs]

    def _menu(self, frame, obs, t, click, k):
        H, W = frame.shape[:2]
        ui.panel(frame, 0, 0, W, H, (24, 18, 16), 0.62, 0)
        ui.texto(frame, "MediaPianist", int(40 * k), int(26 * k), int(54 * k), ui.ORO, negrita=True)
        ui.texto(frame, f"Piano virtual para practicar y rehabilitar las manos  ·  usuario: {self.alm.usuario}",
                 int(44 * k), int(92 * k), int(20 * k), ui.GRIS)
        m, gap, cols = int(40 * k), int(16 * k), 4
        bw = (W - 2 * m - (cols - 1) * gap) // cols
        bh = int(104 * k)
        self.botones = []
        for i, (ident, tit, sub, col) in enumerate(MENU):
            b = ui.Boton(ident, tit, sub, col, m + (i % cols) * (bw + gap), int(140 * k) + (i // cols) * (bh + gap), bw, bh,
                         destacado=(ident == "calibrar" and not self.calibrado))
            self.botones.append(b)
        sel, hover = self.puntero.actualizar(t, self.botones, self._puntas_dedo(obs), click)
        usar_mouse = time.monotonic() - self.t_mouse < 3.0
        if usar_mouse and hover is None:
            for b in self.botones:
                if b.contiene(*self.mouse):
                    hover = b.id
        for b in self.botones:
            ins = None
            if b.id in ADAPTATIVOS:
                ins = f"Nv {self.niveles.get(b.id)}"
            elif b.id == "calibrar" and not self.calibrado:
                ins = "¡Recomendado!"
            b.dibujar(frame, k, hover == b.id, self.puntero.progreso if self.puntero.actual == b.id else 0.0, ins)
        if not obs:
            ui.texto(frame, "Haz clic, o pon la mano delante de la cámara y mantén el índice sobre un botón",
                     W // 2, H - int(60 * k), int(18 * k), ui.GRIS, "c")
        if sel:
            self.ir(sel, t)

    def _boton_volver(self, frame, obs, t, click, k):
        W = frame.shape[1]
        b = ui.Boton("volver", "Menú", "", ui.GRIS, W - int(150 * k), int(8 * k), int(140 * k), int(38 * k), dwell=1.6)
        sel, hover = self.puntero.actualizar(t, [b], self._puntas_dedo(obs), click)
        hover = hover or ("volver" if (time.monotonic() - self.t_mouse < 3.0 and b.contiene(*self.mouse)) else None)
        ui.panel(frame, b.x, b.y, b.w, b.h, (60, 52, 48) if hover else (30, 26, 24), 0.85, int(10 * k), ui.GRIS)
        ui.texto(frame, "Menú (ESC)", b.x + b.w // 2, b.y + int(8 * k), int(16 * k), ui.BLANCO, "c")
        if self.puntero.actual == "volver":
            ui.barra(frame, b.x + 6, b.y + b.h - int(8 * k), b.w - 12, int(5 * k), self.puntero.progreso, ui.VERDE)
        return sel == "volver"

    def _modo(self, frame, obs, t, dt, click, k):
        m = self.modo
        eventos = []
        if m.usa_piano:
            eventos = self.detector.procesar(t, obs, self.piano)
            for ev in eventos:
                if ev.tipo == "on":
                    self.audio.tocar(ev.nota, ev.vol)
                    self.brillo[ev.nota] = 1.0
                else:
                    self.audio.soltar(ev.nota)
                    self.sesion.pulsacion(ev.dedo, ev.mano, ev.amp)
                m.evento(ev, t)
            pulsadas = self.detector.pulsadas()
            for n in pulsadas:
                self.brillo[n] = 1.0
            for n in list(self.brillo):
                if n not in pulsadas:
                    self.brillo[n] -= dt / 0.4
                    if self.brillo[n] <= 0:
                        del self.brillo[n]
        m.actualizar(t, obs)
        if m.usa_piano:
            self.piano.dibujar(frame, self.brillo, m.resaltes(t), t, self.etiquetas, m.numeros())
        m.dibujar(frame, t)
        if m.usa_piano and not obs:
            ui.texto(frame, "Pon las manos sobre el piano", self.ancho // 2, max(self.piano.y1 - int(40 * k), int(200 * k)),
                     int(22 * k), ui.GRIS, "c")
        if self._boton_volver(frame, obs, t, click, k):
            self.salir_modo(t)

    def _calibrar_mesa(self, frame, k):
        H, W = frame.shape[:2]
        ui.panel(frame, 0, 0, W, int(112 * k), (20, 16, 14), 0.82, 0)
        ui.texto(frame, "Calibración de la mesa", int(18 * k), int(10 * k), int(25 * k), ui.ORO, negrita=True)
        ui.texto(frame, "Haz 4 CLICS en las esquinas de la zona de juego, en este orden:",
                 int(18 * k), int(45 * k), int(17 * k), ui.BLANCO)
        ui.texto(frame, "1 arriba-izquierda  ·  2 arriba-derecha  ·  3 abajo-derecha  ·  4 abajo-izquierda",
                 int(18 * k), int(72 * k), int(15 * k), ui.GRIS)
        ui.texto(frame, "La cámara debe mirar hacia la mesa. ESC cancela.",
                 int(18 * k), int(94 * k), int(14 * k), ui.GRIS)
        if self.mesa_puntos:
            for i, punto in enumerate(self.mesa_puntos):
                cv2.circle(frame, punto, int(10 * k), ui.VERDE, -1, cv2.LINE_AA)
                ui.texto(frame, str(i + 1), punto[0] + int(12 * k), punto[1] - int(10 * k),
                         int(22 * k), ui.BLANCO, negrita=True)
            if len(self.mesa_puntos) > 1:
                cv2.polylines(frame, [np.asarray(self.mesa_puntos, dtype=np.int32)], False, ui.VERDE, 2, cv2.LINE_AA)
        ui.texto(frame, f"Esquinas marcadas: {len(self.mesa_puntos)}/4", W - int(18 * k), H - int(18 * k),
                 int(16 * k), ui.BLANCO, "d")

    def _colocar(self, frame, obs, t, k):
        H, W = frame.shape[:2]
        c = self.colocar
        ui.panel(frame, 0, 0, W, int(90 * k), (20, 16, 14), 0.8, 0)
        apuntado = None
        for o in obs:
            if esta_apuntando(o):
                apuntado = (int(o.pts[8][0]), int(o.pts[8][1]))
        if apuntado is None:
            c["cand"] = None
        else:
            if c["cand"] is None or abs(apuntado[0] - c["cand"][0]) + abs(apuntado[1] - c["cand"][1]) > 30 * k:
                c["cand"], c["t_ini"] = apuntado, t
            prog = min(1.0, (t - c["t_ini"]) / 2.0)
            cv2.circle(frame, c["cand"], int(10 * k), (0, 255, 255), -1, cv2.LINE_AA)
            ui.anillo(frame, c["cand"], int(26 * k), prog, ui.VERDE)
            if prog >= 1.0:
                if c["fase"] == 1:
                    c["esq1"], c["fase"] = c["cand"], 2
                else:
                    x1, y1 = c["esq1"]
                    x2, y2 = c["cand"]
                    if x2 - x1 > 160 and y2 - y1 > 70:
                        self.piano = Piano(x1, y1, x2, y2, W, H)
                        self._guardar_piano()
                        self.detector.reset()
                        self.estado = "MENU"
                        self.avisos.poner("Piano colocado y guardado", t, ui.VERDE)
                    else:
                        c["fase"] = 1
                        self.avisos.poner("Rectángulo demasiado pequeño, repite", t, ui.ROJO)
                c["cand"] = None
        txt = ("Coloca el piano virtual: esquina SUPERIOR IZQUIERDA de la zona donde aparecerá (2 s)" if c["fase"] == 1
               else "Ahora señala la esquina INFERIOR DERECHA de esa zona (2 s)")
        ui.texto(frame, txt, int(16 * k), int(10 * k), int(24 * k), ui.ORO, negrita=True)
        ui.texto(frame, "O arrastra un rectángulo con el ratón  ·  D = piano por defecto  ·  ESC = cancelar", int(16 * k), int(50 * k), int(16 * k), ui.GRIS)
        if c["esq1"]:
            cv2.circle(frame, c["esq1"], int(8 * k), (0, 255, 0), -1, cv2.LINE_AA)
        if self.drag is not None:
            cv2.rectangle(frame, tuple(self.drag[:2]), tuple(self.drag[2:]), ui.VERDE, 3)
        elif self.piano is not None:
            self.piano.dibujar(frame, None, None, t, False)

    def _calibrar(self, frame, obs, t, k):
        H, W = frame.shape[:2]
        cal = self.cal
        cal.actualizar(t, obs)
        if cal.click:
            cal.click = False
            self.audio.clic()
        ui.panel(frame, 0, 0, W, H, (20, 16, 14), 0.45, 0)
        ui.panel(frame, W // 2 - int(380 * k), int(30 * k), int(760 * k), int(170 * k), (20, 16, 14), 0.85, int(16 * k), ui.ORO)
        ui.texto(frame, "Calibración de la detección", W // 2, int(40 * k), int(30 * k), ui.ORO, "c", True)
        ui.texto_multilinea(frame, cal.msg, W // 2 - int(350 * k), int(88 * k), int(700 * k), int(24 * k), ui.BLANCO, 1.3, True)
        ui.barra(frame, W // 2 - int(340 * k), int(170 * k), int(680 * k), int(12 * k), cal.progreso(), ui.ORO)
        if cal.fase == "dedo":
            f = cal.dedo
            ui.texto(frame, f"Dedo {f + 1} de 5: {NOMBRES_DEDOS[f]}", W - int(190 * k), int(H * 0.36), int(28 * k), COLOR_DEDOS[f], "c", True)
            idx, frac, flash = cal.cue()
            cx, cy = W - int(190 * k), int(H * 0.55)
            ui.anillo(frame, (cx, cy), int(60 * k), frac if idx >= 0 else 0, ui.AZUL, int(8 * k))
            if flash:
                cv2.circle(frame, (cx, cy), int(52 * k), ui.VERDE, -1, cv2.LINE_AA)
                ui.texto(frame, "¡YA!", cx, cy - int(20 * k), int(40 * k), (0, 0, 0), "c", True, False)
            ui.texto(frame, f"{min(max(idx, 0), cal.N_PULSOS)}/{cal.N_PULSOS}", cx, cy + int(80 * k), int(22 * k), ui.GRIS, "c")
        if cal.terminado:
            self.calibrado = True
            self.detector.aplicar_perfil(cal.perfil)
            self.alm.guardar("calibracion.json", {"fecha": datetime.now().isoformat(timespec="seconds"), "perfil": cal.perfil})
            ok = 5 - len(cal.fallidos)
            if cal.fallidos:
                nombres = ", ".join(NOMBRES_DEDOS[f] for f in cal.fallidos)
                self.avisos.poner(f"Calibrado {ok}/5. Repite para: {nombres}", t, ui.ORO, 5.0)
            else:
                self.avisos.poner("¡Calibración completa! 5/5 dedos", t, ui.VERDE, 3.5)
            self.audio.fin()
            self.estado = "MENU"

    def _resumen(self, frame, obs, t, click, k):
        H, W = frame.shape[:2]
        ui.panel(frame, 0, 0, W, H, (24, 18, 16), 0.7, 0)
        ph = int((130 + 46 * max(len(self.resumen), 1)) * k)
        ui.panel(frame, W // 2 - int(300 * k), int(90 * k), int(600 * k), ph, (30, 26, 24), 0.92, int(18 * k), ui.VERDE)
        ui.texto(frame, f"{self.resumen_titulo}: resumen", W // 2, int(104 * k), int(30 * k), ui.VERDE, "c", True)
        y = int(158 * k)
        for etq, val in self.resumen or [("Sin datos", "")]:
            ui.texto(frame, etq, W // 2 - int(260 * k), y, int(22 * k), ui.GRIS)
            ui.texto(frame, val, W // 2 + int(260 * k), y, int(22 * k), ui.BLANCO, "d", True)
            y += int(46 * k)
        b = ui.Boton("seguir", "Continuar", "", ui.VERDE, W // 2 - int(110 * k), int(90 * k) + ph + int(20 * k), int(220 * k), int(56 * k), dwell=1.2)
        sel, hover = self.puntero.actualizar(t, [b], self._puntas_dedo(obs), click)
        b.dibujar(frame, k, hover == "seguir", self.puntero.progreso if self.puntero.actual == "seguir" else 0.0)
        ui.texto(frame, "Sesión guardada. Pulsa una tecla o haz clic para continuar", W // 2, b.y + b.h + int(14 * k), int(16 * k), ui.GRIS, "c")
        if (sel or (click and b.contiene(*click))) and t - self.t_resumen > 0.6:
            self.estado = "MENU"

    # ---------- dibujo de manos y depuracion ----------
    def _dibujar_manos(self, frame, obs, t, k):
        if not self.ver_manos and self.estado == "MODO":
            return
        estado = {}
        for m in self.detector.manos.values():
            if m.obs is not None and t - m.t < 0.15:
                estado[id(m.obs)] = m
        for o in obs:
            if self.ver_manos:
                for a, b in CONEXIONES:
                    cv2.line(frame, tuple(o.pts[a].astype(int)), tuple(o.pts[b].astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
            m = estado.get(id(o))
            for f, p in enumerate(PUNTAS):
                abajo = bool(m and m.dedos[f].abajo)
                if self.estado == "MENU" and f != 1:
                    continue
                cv2.circle(frame, (int(o.pts[p][0]), int(o.pts[p][1])), int((8 if abajo else 5) * k),
                           ui.VERDE if abajo else ui.ROJO, -1, cv2.LINE_AA)

    def _depuracion(self, frame, t, k):
        x0, y0 = int(10 * k), int(self.alto * 0.34)
        ui.panel(frame, x0, y0, int(250 * k), int(150 * k), (20, 16, 14), 0.75, int(10 * k))
        ui.texto(frame, "Depuración: subida / umbral", x0 + int(8 * k), y0 + int(4 * k), int(14 * k), ui.ORO)
        for n, m in enumerate([m for m in self.detector.manos.values() if t - m.t < 0.2][:2]):
            for f, d in enumerate(m.dedos):
                bx = x0 + int(12 * k) + f * int(46 * k) + n * 0
                frac = min(d.rise / max(d.T, 1e-6), 2.0) / 2.0
                h = int(80 * k * frac)
                by = y0 + int(120 * k)
                cv2.rectangle(frame, (bx, by - h), (bx + int(30 * k), by), ui.VERDE if d.abajo else COLOR_DEDOS[f], -1)
                cv2.line(frame, (bx - 2, by - int(40 * k)), (bx + int(32 * k), by - int(40 * k)), (255, 255, 255), 1)
                ui.texto(frame, str(f + 1), bx + int(10 * k), by + int(2 * k), int(12 * k), ui.GRIS)
            break


# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------
def _buscar_modelo():
    for r in MODEL_RUTAS:
        if os.path.exists(r):
            return r
    destino = MODEL_RUTAS[0]
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    print("Descargando el modelo de manos...")
    urllib.request.urlretrieve(MODEL_URL, destino)
    return destino


def _a_obs(res, ancho, alto, t):
    lista = []
    mundo = getattr(res, "hand_world_landmarks", None) or []
    for i, m in enumerate(res.hand_landmarks):
        pts = np.array([[p.x * ancho, p.y * alto] for p in m])
        w3 = np.array([[p.x, p.y, p.z] for p in mundo[i]]) if i < len(mundo) else None
        etq = None
        try:
            etq = res.handedness[i][0].category_name
        except (IndexError, AttributeError):
            pass
        lista.append(Obs(pts, w3, etq, t))
    return lista


def main():
    ap = argparse.ArgumentParser(description="MediaPianist")
    ap.add_argument("--usuario", default="invitado", help="nombre del usuario (guarda su progreso aparte)")
    ap.add_argument("--camara", type=int, default=0, help="indice de la camara")
    ap.add_argument("--silencio", action="store_true", help="sin sonido")
    ap.add_argument("--demo", action="store_true", help="modo demo sin camara (mano simulada)")
    args = ap.parse_args()
    if args.demo:
        main_demo(args)
        return

    import mediapipe as mp
    modelo = _buscar_modelo()
    opciones = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=modelo),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.4)

    app = App(args.usuario, args.silencio)
    cam = Camara(args.camara)
    frame, cnt, _ = cam.leer(-1)
    if frame is None:
        print("No se pudo abrir la cámara.")
        cam.parar()
        return
    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(NOMBRE_VENTANA, app.on_mouse)
    ts = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(opciones) as lm:
        while not app.salir:
            frame, cnt, t_cap = cam.leer(cnt)
            if frame is None:
                break
            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
            esc = 960.0 / ancho if ancho > 960 else 1.0
            peque = cv2.resize(frame, (int(ancho * esc), int(alto * esc))) if esc < 1 else frame
            rgb = np.ascontiguousarray(cv2.cvtColor(peque, cv2.COLOR_BGR2RGB))
            ts = max(ts + 1, int(t_cap * 1000))
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            app.paso(frame, _a_obs(res, ancho, alto, t_cap), t_cap)
            cv2.imshow(NOMBRE_VENTANA, frame)
            c = cv2.waitKey(1) & 0xFF
            app.tecla(c, t_cap)
            if cv2.getWindowProperty(NOMBRE_VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break
    app.guardar_sesion()
    app.audio.cerrar() if hasattr(app.audio, "cerrar") else None
    cam.parar()
    cv2.destroyAllWindows()
    print("Sesión guardada en", app.alm.dir)




if __name__ == "__main__":
    main()
