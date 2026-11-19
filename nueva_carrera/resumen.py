"""Resumen sencillo para el jugador: notas de 0 a 100, comparacion con la sesion anterior y evolucion.

Convierte las metricas tecnicas (grados, ms, % de temblor...) en 5-7 notas faciles de entender.
Son ORIENTATIVAS: sirven para comparar tu sesion de hoy con la anterior, no son un diagnostico.
"""
import math

import cv2

from config import *
from render import text, panel

# nombre -> explicacion corta (sin tildes: el texto se dibuja con cv2)
INFO = {
    "Movimiento": "Cuanto abres y cierras los dedos",
    "Rapidez": "Lo rapido que mueves los dedos",
    "Estabilidad": "Poco temblor en la mano",
    "Control": "Pocas explosiones en la carrera",
    "Reaccion": "Rapidez al frenar ante obstaculos",
    "Resistencia": "Mantienes el nivel hasta el final",
    "Equilibrio": "Las dos manos rinden por igual",
}
GREY = (120, 120, 120)


def _n(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _clamp(v):
    return max(0.0, min(100.0, v))


def _avg(row, key):
    vals = [_n(row.get(f"{p}_{key}")) for p in ("izq", "der", "mano")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def scores(row):
    """Fila plana del historial (metrics.flat o fila del CSV) -> {nota: 0..100}. Omite lo que no se pudo medir."""
    s = {}
    v = _avg(row, "rom_medio_deg")
    if v is not None:
        s["Movimiento"] = _clamp(v / 170.0 * 100)
    v = _avg(row, "vel_max_deg_s")
    if v is not None:
        s["Rapidez"] = _clamp(v / 600.0 * 100)
    v = _avg(row, "temblor_rms_pct")
    if v is not None:
        s["Estabilidad"] = _clamp(100 - v * 10)
    v = _n(row.get("explosiones"))
    if v is not None:
        s["Control"] = _clamp(100 - v * 12)
    v = _n(row.get("reaccion_media_ms"))
    if v is not None:
        s["Reaccion"] = _clamp(100.0 * (2500 - v) / 2000)
    drops = [_n(row.get("fatiga_caida_rom_pct")), _n(row.get("fatiga_caida_ritmo_pct"))]
    drops = [d for d in drops if d is not None]
    if drops:
        s["Resistencia"] = _clamp(100 - max(0.0, max(drops)) * 2)
    v = _n(row.get("simetria_rom_pct"))
    if v is not None:
        s["Equilibrio"] = _clamp(v)
    return s


def overall(sc, keys=None):
    vals = [v for k, v in sc.items() if keys is None or k in keys]
    return round(sum(vals) / len(vals)) if vals else None


def build(row, hist_rows):
    """row = sesion de hoy (fila plana); hist_rows = sesiones anteriores del mismo paciente y modo."""
    cur = scores(row)
    prev = scores(hist_rows[-1]) if hist_rows else None
    delta = None
    if prev:
        common = set(cur) & set(prev)
        if common:
            delta = overall(cur, common) - overall(prev, common)
    trend = [overall(scores(r)) for r in hist_rows[-5:]]
    trend = [t for t in trend if t is not None] + [overall(cur)]
    return dict(cur=cur, prev=prev, delta=delta, total=overall(cur), trend=trend, tiempo=_n(row.get("tiempo_carrera_s")))


# ------------------------------------------------------------------
def level_color(v):
    return GREEN if v >= 70 else YELLOW if v >= 45 else RED


def _message(sm):
    d = sm["delta"]
    if d is None:
        return "Primera sesion: te sirve de punto de partida para comparar", WHITE
    if d >= 3:
        return f"Muy bien! Has mejorado {d} puntos respecto a la sesion anterior", GREEN
    if d <= -3:
        return "Hoy ha costado un poco mas que la ultima vez. Sigue practicando", YELLOW
    return "Nivel parecido al de la sesion anterior. La constancia es lo que cuenta", WHITE


def _delta_tag(canvas, x, y, d):
    if d >= 3:
        text(canvas, f"+{d}", x, y, 0.8, GREEN, 2)
    elif d <= -3:
        text(canvas, f"{d}", x, y, 0.8, (80, 80, 255), 2)
    else:
        text(canvas, "=", x, y, 0.8, (190, 190, 190), 2)


def draw_summary(canvas, name, sm):
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.94)
    text(canvas, f"TU PROGRESO - {name.upper()}", W // 2, 52, 1.1, YELLOW, 3, "c")
    msg, mc = _message(sm)
    text(canvas, msg, W // 2, 92, 0.75, mc, 2, "c")

    cur, prev = sm["cur"], sm["prev"]
    names = list(cur)
    # ---- barras: sesion anterior (gris) y hoy (color) ----
    BX, BW = 360, 330
    rh = min(78, 470 // max(1, len(names)))
    for i, k in enumerate(names):
        yt = 135 + i * rh
        text(canvas, k, 50, yt + 24, 0.8, WHITE, 2)
        text(canvas, INFO[k], 50, yt + 48, 0.48, (190, 190, 190), 1)
        v = cur[k]
        if prev and k in prev:
            cv2.rectangle(canvas, (BX, yt + 4), (BX + BW, yt + 18), (45, 45, 45), -1)
            cv2.rectangle(canvas, (BX, yt + 4), (BX + int(BW * prev[k] / 100), yt + 18), GREY, -1)
            by = yt + 24
        else:
            by = yt + 14
        cv2.rectangle(canvas, (BX, by), (BX + BW, by + 22), (45, 45, 45), -1)
        cv2.rectangle(canvas, (BX, by), (BX + int(BW * v / 100), by + 22), level_color(v), -1)
        text(canvas, str(round(v)), BX + BW + 14, by + 20, 0.8, WHITE, 2)
        if prev and k in prev:
            _delta_tag(canvas, BX + BW + 70, by + 20, round(v) - round(prev[k]))

    # ---- nota global + evolucion ----
    panel(canvas, 790, 120, 1240, 655, (30, 30, 30), 0.6)
    text(canvas, "NOTA GLOBAL", 1015, 152, 0.75, WHITE, 2, "c")
    cx, cy, r = 1015, 245, 66
    tot = sm["total"] or 0
    cv2.circle(canvas, (cx, cy), r, (55, 55, 55), 12, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (r, r), 0, -90, -90 + 360 * tot / 100, level_color(tot), 12, cv2.LINE_AA)
    text(canvas, str(tot), cx, cy + 18, 2.0, WHITE, 4, "c")
    if sm["delta"] is not None:
        d = sm["delta"]
        text(canvas, ("+" if d > 0 else "") + f"{d} vs sesion anterior", cx, 345, 0.65,
             GREEN if d >= 3 else (80, 80, 255) if d <= -3 else (190, 190, 190), 2, "c")

    text(canvas, "TUS ULTIMAS SESIONES", 1015, 392, 0.6, WHITE, 1, "c")
    gx1, gx2, gy1, gy2 = 840, 1205, 415, 545          # y: 100 arriba, 0 abajo
    for v in (0, 50, 100):
        y = int(gy2 - (gy2 - gy1) * v / 100)
        cv2.line(canvas, (gx1, y), (gx2, y), (70, 70, 70), 1, cv2.LINE_AA)
        text(canvas, str(v), gx1 - 8, y + 5, 0.45, (170, 170, 170), 1, "r")
    tr = sm["trend"]
    n = len(tr)
    pts = [(int((gx1 + gx2) / 2) if n == 1 else int(gx1 + 20 + (gx2 - gx1 - 40) * k / (n - 1)),
            int(gy2 - (gy2 - gy1) * v / 100)) for k, v in enumerate(tr)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(canvas, a, b, YELLOW, 3, cv2.LINE_AA)
    for k, (p, v) in enumerate(zip(pts, tr)):
        last = k == n - 1
        cv2.circle(canvas, p, 8 if last else 5, level_color(v) if last else YELLOW, -1, cv2.LINE_AA)
        text(canvas, str(v), p[0], p[1] - 14, 0.5, WHITE, 1, "c")
        text(canvas, "hoy" if last else f"-{n - 1 - k}", p[0], gy2 + 22, 0.45, (190, 190, 190), 1, "c")

    best, worst = max(cur, key=cur.get), min(cur, key=cur.get)
    text(canvas, f"Punto fuerte: {best}", 810, 610, 0.62, GREEN, 1)
    text(canvas, f"A mejorar: {worst}", 810, 640, 0.62, YELLOW, 1)

    # ---- leyenda y teclas ----
    y = H - 62
    if prev:
        cv2.rectangle(canvas, (50, y - 14), (74, y), GREY, -1)
        text(canvas, "Sesion anterior", 82, y, 0.55, WHITE, 1)
    cv2.rectangle(canvas, (260, y - 14), (284, y), GREEN, -1)
    text(canvas, "Hoy:  verde = muy bien   amarillo = regular   rojo = a mejorar", 292, y, 0.55, WHITE, 1)
    text(canvas, "M: volver   D: detalle tecnico   ENTER: revancha   ESC: salir      Notas orientativas, no son un diagnostico",
         W // 2, H - 26, 0.55, (190, 190, 190), 1, "c")
