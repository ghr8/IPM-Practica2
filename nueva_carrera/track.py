"""Pista: curvas de la carretera, minimapa, pastillas de turbo, obstaculos y puntos de control."""
import math
import random

import numpy as np

from config import *

# ------------------------------------------------------------------
# Mini-circuito tipo Mario Kart (circuito cerrado con curvas)
# ------------------------------------------------------------------
MAP_W, MAP_H = 270, 214
_CTRL = [(-0.85, 0.15), (-0.75, -0.45), (-0.35, -0.8), (0.15, -0.55), (0.3, -0.15), (0.7, -0.35),
         (0.92, 0.1), (0.62, 0.62), (0.12, 0.4), (-0.18, 0.78), (-0.62, 0.62)]


def _build_circuit():
    P = np.array(_CTRL, float)
    n = len(P)
    out = []
    for i in range(n):   # spline Catmull-Rom cerrada
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for t in np.linspace(0, 1, 24, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    pts = np.array(out)
    pts[:, 0] = (pts[:, 0] + 1) / 2 * (MAP_W - 52) + 26
    pts[:, 1] = (pts[:, 1] + 1) / 2 * 142 + 18
    closed = np.vstack([pts, pts[:1]])
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    return pts, closed, cum / cum[-1]


CIRCUIT, _CLOSED, _CUM = _build_circuit()
_tan = CIRCUIT[1] - CIRCUIT[0]
_tan = _tan / np.hypot(*_tan)
_nrm = np.array([-_tan[1], _tan[0]])
START_A, START_B = CIRCUIT[0] - _nrm * 10, CIRCUIT[0] + _nrm * 10


def circuit_pos(f):
    f = max(0.0, min(1.0, f))
    return float(np.interp(f, _CUM, _CLOSED[:, 0])), float(np.interp(f, _CUM, _CLOSED[:, 1]))



# Pastillas de turbo (misma posicion para los dos carriles)
_rng = random.Random(7)
PADS = []
_d = 900.0
while _d < TRACK_LENGTH - 700:
    PADS.append((_d, _rng.uniform(-0.28, 0.28) * ROAD_W))
    _d += _rng.uniform(900, 1300)

# Momentos (progreso) en los que puede aparecer un obstaculo. Siempre con margen antes de la meta.
_ro = random.Random(11)
OBST_TRIGGERS = []
_d = 1700.0
while _d < TRACK_LENGTH - OBST_DIST - 600:
    OBST_TRIGGERS.append(_d)
    _d += _ro.uniform(1100, 1600)


def sway(d):
    """Desplazamiento lateral del centro de la carretera a la distancia d (las curvas del circuito)."""
    k = 2 * math.pi / SWAY_LEN
    return SWAY_AMP * (0.7 * np.sin(d * k) + 0.3 * np.sin(d * k * 2.3 + 1.3))


def checkpoint_for(progress):
    """Punto de control al que vuelves si explotas con este progreso (al menos EXPLOSION_PENALTY por detras)."""
    return max(0.0, math.floor((progress - EXPLOSION_PENALTY) / CHECKPOINT_EVERY) * CHECKPOINT_EVERY)
