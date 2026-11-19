"""Dibujo de la escena: carretera en perspectiva, coche, obstaculos, manos y minimapa."""
import math
import random

import cv2
import numpy as np

import config as cfg
from config import *
from effects import draw_particles, RINGS
from track import sway, PADS, CIRCUIT, START_A, START_B, circuit_pos, MAP_W, MAP_H

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]

def text(img, s, x, y, scale=0.8, color=WHITE, thick=2, anchor="l"):
    size = cv2.getTextSize(s, FONT, scale, thick)[0]
    if anchor == "c":
        x -= size[0] // 2
    elif anchor == "r":
        x -= size[0]
    x, y = int(x), int(y)
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def panel(img, x1, y1, x2, y2, color, alpha):
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def rot_rect(cx, cy, hw, hh, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = [(int(cx + dx * c - dy * s), int(cy + dx * s + dy * c))
           for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    return np.array(pts, np.int32)


def local(cx, cy, ang, dx, dy):
    c, s = math.cos(ang), math.sin(ang)
    return cx + dx * c - dy * s, cy + dx * s + dy * c



# --- Precalculos de la perspectiva (solo se hacen una vez) ---
LW = W // 2
HZ = int(H * 0.36)                       # fila del horizonte
NR = (H - HZ) // 2                       # filas de carretera (de 2 en 2 px)
Y0 = H - 2 * NR
Z_K = 59000.0
TREE_GAP = 200.0
FOG_T = (235, 205, 175)
SY = (Y0 + 2 * np.arange(NR) + 1 - HZ).astype(np.float64)
S_ROW = SY / (CAR_Y - HZ)
DZ = Z_K * (1.0 / SY - 1.0 / (CAR_Y - HZ))
GAINV = 1.0 + 1.6 * (1.0 - np.clip(S_ROW, 0, 1))
RW_ROW = ROAD_W / 2 * S_ROW
CW_ROW = RW_ROW * 0.12 + 0.8
DW_ROW = np.minimum(RW_ROW * 0.4, np.maximum(0.6, RW_ROW * 0.02))
FOG_ROW = np.clip(1.0 - SY / 140.0, 0, 1) * 0.92
XS = (np.arange(LW // 2) * 2 + 1)[None, :]
ROWIDX = np.arange(NR)[:, None]
GRASS_A, GRASS_B = np.array((40, 120, 45.0)), np.array((30, 100, 35.0))
ROAD_A, ROAD_B = np.array((74, 74, 78.0)), np.array((64, 64, 68.0))
RED_A, WHITE_A, DASH_A = np.array(RED, float), np.array(WHITE, float), np.array((225, 225, 225.0))
FOG_A = np.array(FOG_T, float)

SKY = np.zeros((Y0, LW, 3), np.uint8)
_t = np.linspace(0, 1, Y0)[:, None]
for _k, (_a, _b) in enumerate(zip((140, 70, 25), (235, 200, 150))):
    SKY[:, :, _k] = (_a + (_b - _a) * _t).astype(np.uint8)

_rm = random.Random(3)
MTN = np.array([(x, Y0 - _rm.randint(20, 80)) for x in range(-250, LW + 251, 45)] +
               [(LW + 250, Y0 + 1), (-250, Y0 + 1)], np.int32)


def mixc(col, f):
    return tuple(int(c * (1 - f) + fc * f) for c, fc in zip(col, FOG_T))


def project(dz, xd, xp):
    """Proyecta un punto a distancia dz por delante. Devuelve (y, escala, x_centro_carretera)."""
    inv = dz / Z_K + 1.0 / (CAR_Y - HZ)
    if inv <= 0:
        return None
    sy = 1.0 / inv
    if sy < 8 or HZ + sy > H + 90:
        return None
    s = sy / (CAR_Y - HZ)
    g = 1.0 + 1.6 * (1.0 - min(1.0, s))
    return HZ + sy, s, LW / 2 + xp + (xd - xp) * g


def shade(col, k):
    return tuple(int(max(0, min(255, c * k))) for c in col)



def draw_car(img, cx, by, color, ext, vel, boosting, blink, now):
    """Coche visto desde atras y un poco desde arriba. Ruedas: verde = dedo extendido."""
    if blink:
        return
    lean = max(-22.0, min(22.0, vel * 0.09))

    def P(dx, dy, lf=0.0):
        return (int(cx + dx + lean * lf), int(by + dy))

    def poly(pts, col, edge=True):
        a = np.array(pts, np.int32)
        cv2.fillPoly(img, [a], col)
        if edge:
            cv2.polylines(img, [a], True, (20, 20, 20), 2, cv2.LINE_AA)

    def wheel(dx, dy, w, h, lf, on):
        poly([P(dx - w / 2, dy - h / 2, lf), P(dx + w / 2, dy - h / 2, lf),
              P(dx + w / 2, dy + h / 2, lf), P(dx - w / 2, dy + h / 2, lf)], GREEN if on else RED)

    cv2.ellipse(img, (int(cx), int(by + 2)), (88, 14), 0, 0, 360, (12, 12, 12), -1, cv2.LINE_AA)

    if boosting:   # llamas del turbo
        fl = 30 + int(14 * math.sin(now * 50))
        for sx in (-1, 1):
            poly([P(sx * 32, -12), P(sx * 16, -12), P(sx * 24, fl)], (0, 150, 255), False)
            poly([P(sx * 28, -12), P(sx * 20, -12), P(sx * 24, fl * 0.55)], (0, 235, 255), False)

    wheel(-66, -104, 20, 34, 1.0, ext[0])      # ruedas delanteras (al fondo)
    wheel(66, -104, 20, 34, 1.0, ext[2])

    poly([P(-58, -60), P(58, -60), P(50, -128, 1.0), P(-50, -128, 1.0)], shade(color, 1.15))  # capo/techo
    poly([P(-40, -62, 0.1), P(40, -62, 0.1), P(33, -98, 0.6), P(-33, -98, 0.6)], (70, 50, 35))  # luna trasera
    cv2.line(img, P(-26, -72, 0.2), P(-8, -90, 0.5), (190, 170, 140), 2, cv2.LINE_AA)
    poly([P(-33, -98, 0.6), P(33, -98, 0.6), P(29, -122, 0.9), P(-29, -122, 0.9)], shade(color, 1.05))  # techo

    poly([P(-62, -60), P(62, -60), P(64, -10), P(-64, -10)], color)                          # trasera
    poly([P(-64, -22), P(64, -22), P(64, -10), P(-64, -10)], (40, 40, 45))                   # paragolpes
    poly([P(-16, -38), P(16, -38), P(16, -24), P(-16, -24)], (235, 235, 235), False)         # matricula
    glow = (90, 90, 255) if boosting else (30, 30, 235)
    for sx in (-1, 1):
        poly([P(sx * 56, -54), P(sx * 28, -54), P(sx * 28, -40), P(sx * 56, -40)], glow)     # pilotos
    poly([P(-60, -78), P(60, -78), P(60, -68), P(-60, -68)], shade(color, 0.45))             # alerón
    for sx in (-1, 1):
        cv2.line(img, P(sx * 46, -68), P(sx * 46, -60), (20, 20, 20), 3)

    wheel(-74, -18, 22, 38, 0.0, ext[1])       # ruedas traseras (delante en pantalla)
    wheel(74, -18, 22, 38, 0.0, ext[3])



def draw_lane3d(lane, p, now):
    prog = p["progress"]
    xp = float(sway(prog))

    # Cielo, sol y montanas (se mueven un poco segun la curva que viene)
    lane[:Y0] = cv2.addWeighted(lane[:Y0], 0.55, SKY, 0.45, 0)
    cv2.circle(lane, (int(LW * 0.72), Y0 - 52), 34, (70, 200, 255), -1, cv2.LINE_AA)
    pts = MTN.copy()
    pts[:, 0] += int(-0.25 * (float(sway(prog + 900)) - xp))
    cv2.fillPoly(lane, [pts], (100, 72, 62))

    # Carretera en perspectiva (calculada por filas con numpy)
    d = prog + DZ
    c = LW / 2 + xp + (sway(d) - xp) * GAINV
    idx = np.zeros((NR, LW // 2), np.int8)
    for b in (c - RW_ROW - CW_ROW, c - RW_ROW, c - DW_ROW, c + DW_ROW, c + RW_ROW, c + RW_ROW + CW_ROW):
        idx += (XS >= b[:, None])

    dd = np.floor(d)
    gb = (dd // 90 % 2 == 0)[:, None]
    cb = (dd // 25 % 2 == 0)[:, None]
    rb = (dd // 50 % 2 == 0)[:, None]
    dash = (dd // 60 % 2 == 0)[:, None]
    grass = np.where(gb, GRASS_A, GRASS_B)
    curb = np.where(cb, RED_A, WHITE_A)
    road = np.where(rb, ROAD_A, ROAD_B)
    pal = np.empty((NR, 7, 3))
    pal[:, 0] = pal[:, 6] = grass
    pal[:, 1] = pal[:, 5] = curb
    pal[:, 2] = pal[:, 4] = road
    pal[:, 3] = np.where(dash, DASH_A, road)
    f = FOG_ROW[:, None, None]
    pal = (pal * (1 - f) + FOG_A * f).astype(np.uint8)

    small = pal[ROWIDX, idx]
    big = cv2.resize(small, (LW, NR * 2), interpolation=cv2.INTER_LINEAR)
    lane[Y0:] = cv2.addWeighted(big, 0.78, lane[Y0:], 0.22, 0)

    # Pastillas de turbo
    for k, (pd, off) in enumerate(PADS):
        if k in p["used"]:
            continue
        r = project(pd - prog, float(sway(pd)), xp)
        if r is None:
            continue
        y, s, ccx = r
        x = ccx + off * s
        th = max(1, int(7 * s))
        for j in range(3):
            yy = y - j * 22 * s
            a = np.array([(x - 30 * s, yy + 14 * s), (x, yy - 6 * s), (x + 30 * s, yy + 14 * s)], np.int32)
            cv2.polylines(lane, [a], False, YELLOW, th, cv2.LINE_AA)

    # Linea de meta con arco
    r = project(TRACK_LENGTH - prog, float(sway(TRACK_LENGTH)), xp)
    if r is not None:
        y, s, ccx = r
        rw = ROAD_W / 2 * s
        n = 14
        cw = 2 * rw / n
        hh = max(2.0, 14 * s)
        for row in range(2):
            for q in range(n):
                col = WHITE if (q + row) % 2 == 0 else (20, 20, 20)
                cv2.rectangle(lane, (int(ccx - rw + q * cw), int(y - hh + row * hh)),
                              (int(ccx - rw + (q + 1) * cw), int(y + row * hh)), col, -1)
        if s > 0.12:
            for sx in (-1, 1):
                cv2.line(lane, (int(ccx + sx * rw), int(y)), (int(ccx + sx * rw), int(y - 200 * s)),
                         (200, 200, 200), max(2, int(10 * s)))
            cv2.rectangle(lane, (int(ccx - rw), int(y - 200 * s)), (int(ccx + rw), int(y - 150 * s)),
                          (30, 30, 30), -1)
            text(lane, "META", ccx, y - 160 * s, 1.8 * s, WHITE, max(1, int(3 * s)), "c")

    # Obstaculo: barrera de rayas rojas y blancas con cartel de aviso
    ob = p["obst"]["active"]
    if ob is not None:
        r = project(ob["pos"] - prog, float(sway(ob["pos"])), xp)
        if r is not None:
            y, s, ccx = r
            half = ROAD_W * 0.46 * s
            bh = max(5.0, 52 * s)
            top, bot = y - bh * 1.9, y - bh * 0.9
            for sx in (-1, 1):                                   # patas
                cv2.rectangle(lane, (int(ccx + sx * half * 0.85 - 3 * s - 1), int(bot)),
                              (int(ccx + sx * half * 0.85 + 3 * s + 1), int(y)), (40, 40, 40), -1)
            nseg = 8
            sw = 2 * half / nseg
            for q in range(nseg):
                col = RED if q % 2 == 0 else WHITE
                cv2.rectangle(lane, (int(ccx - half + q * sw), int(top)), (int(ccx - half + (q + 1) * sw), int(bot)),
                              col, -1)
            cv2.rectangle(lane, (int(ccx - half), int(top)), (int(ccx + half), int(bot)), (20, 20, 20), max(1, int(3 * s)))
            tri = np.array([(ccx - 30 * s, top - 6 * s), (ccx + 30 * s, top - 6 * s), (ccx, top - 58 * s)], np.int32)
            cv2.fillPoly(lane, [tri], YELLOW)
            cv2.polylines(lane, [tri], True, (20, 20, 20), max(1, int(3 * s)), cv2.LINE_AA)
            text(lane, "!", ccx, top - 14 * s, max(0.3, 1.1 * s), (20, 20, 20), max(1, int(3 * s)), "c")

    # Arboles a los lados (de lejos a cerca)
    n0 = max(0, int((prog - 60) // TREE_GAP))
    n1 = int((prog + 5600) // TREE_GAP)
    for n in range(n1, n0 - 1, -1):
        dk = n * TREE_GAP
        r = project(dk - prog, float(sway(dk)), xp)
        if r is None:
            continue
        y, s, ccx = r
        side = -1 if n % 2 == 0 else 1
        x = ccx + side * ROAD_W * 0.80 * s
        fg = max(0.0, 1.0 - (y - HZ) / 140.0) * 0.92
        h, w, tr = 190 * s, 90 * s, 34 * s
        cv2.rectangle(lane, (int(x - 5 * s - 0.5), int(y - tr)), (int(x + 5 * s + 0.5), int(y)),
                      mixc((30, 60, 95), fg), -1)
        cv2.fillPoly(lane, [np.array([(x - w / 2, y - tr), (x + w / 2, y - tr),
                                      (x, y - tr - h * 0.62)], np.int32)], mixc((32, 105, 38), fg))
        cv2.fillPoly(lane, [np.array([(x - w * 0.38, y - tr - h * 0.38), (x + w * 0.38, y - tr - h * 0.38),
                                      (x, y - h)], np.int32)], mixc((45, 135, 50), fg))

    # Lineas de velocidad que salen del horizonte
    sp = p["speed"] / MAX_SPEED
    if sp > 0.55:
        for _ in range(int((sp - 0.4) * 14)):
            x = random.uniform(0, LW)
            y = random.uniform(HZ + 40, H)
            vx, vy = x - LW / 2, y - HZ
            ln = (20 + 70 * sp) * (y - HZ) / 450
            nrm = math.hypot(vx, vy) or 1.0
            cv2.line(lane, (int(x), int(y)), (int(x + vx / nrm * ln), int(y + vy / nrm * ln)),
                     (230, 230, 230), 1, cv2.LINE_AA)



def draw_hand(canvas, hand, color, show_thumb=False):
    px = [(int(x * W), int(y * H)) for x, y in hand["pts"]]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], color, 3, cv2.LINE_AA)
    for pt in px:
        cv2.circle(canvas, pt, 4, WHITE, -1, cv2.LINE_AA)
    for k, tip in enumerate(hand["order"]):
        col = GREEN if hand["ext"][k] else RED
        cv2.circle(canvas, px[tip], 11, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, px[tip], 11, WHITE, 2, cv2.LINE_AA)
    if show_thumb:                                  # pulgar: naranja = metido en la palma (nitro)
        col = (0, 140, 255) if hand.get("thumb") else (230, 230, 230)
        cv2.circle(canvas, px[4], 13, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, px[4], 13, (20, 20, 20), 2, cv2.LINE_AA)



def draw_minimap(canvas, players, now, race_t):
    x0, y0 = W // 2 - MAP_W // 2, 10
    panel(canvas, x0, y0, x0 + MAP_W, y0 + MAP_H, (0, 0, 0), 0.62)
    cv2.rectangle(canvas, (x0, y0), (x0 + MAP_W, y0 + MAP_H), (200, 200, 200), 2)
    off = np.array([x0, y0])
    pts = (CIRCUIT + off).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (20, 20, 20), 16, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], True, (195, 195, 195), 11, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], True, (115, 115, 120), 3, cv2.LINE_AA)
    cv2.line(canvas, tuple((START_A + off).astype(int)), tuple((START_B + off).astype(int)), (255, 255, 255), 5)
    cv2.line(canvas, tuple((START_A + off).astype(int)), tuple((START_B + off).astype(int)), (20, 20, 20), 1)

    order = sorted(range(len(players)), key=lambda k: (-players[k]["progress"], k))
    for i in reversed(order):                       # el ultimo se dibuja debajo
        fx, fy = circuit_pos(players[i]["progress"] / TRACK_LENGTH)
        c = (int(fx + x0), int(fy + y0))
        if len(players) == 2 and i == order[0]:
            cv2.circle(canvas, c, int(14 + 3 * math.sin(now * 8)), YELLOW, 2, cv2.LINE_AA)
        cv2.circle(canvas, c, 9, COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(canvas, c, 9, WHITE, 2, cv2.LINE_AA)

    if len(players) == 2:
        for rank, i in enumerate(order):            # ranking en directo
            xx = x0 + MAP_W * (0.27 + 0.46 * rank)
            text(canvas, f"{rank + 1}o {NAMES[i][:9].upper()}", xx, y0 + 186, 0.55, COLORS[i], 1, "c")
    else:
        text(canvas, NAMES[0][:12].upper(), x0 + MAP_W // 2, y0 + 186, 0.55, COLORS[0], 1, "c")
    text(canvas, f"{race_t:05.1f}s", x0 + MAP_W // 2, y0 + 207, 0.55, WHITE, 1, "c")



def draw_scene(canvas, players, hands, now, show_cars=True):
    for i, p in enumerate(players):
        x0 = LANE_X0[i]
        lane = canvas[:, x0:x0 + LW].copy()
        draw_lane3d(lane, p, now)
        canvas[:, x0:x0 + LW] = lane
    if len(players) == 2:
        cv2.line(canvas, (W // 2, 0), (W // 2, H), (15, 15, 15), 6)
    else:
        for xx in (LANE_X0[0], LANE_X0[0] + LW):
            cv2.line(canvas, (xx, 0), (xx, H), (15, 15, 15), 6)

    for i, p in enumerate(players):
        cx = LANE_CX[i] + p["x"]
        if not show_cars:
            continue
        if now >= p["exploded_until"]:
            blink = now < p["immune_until"] and int(now * 10) % 2 == 0
            by = CAR_Y + int(2 * math.sin(now * 40) * min(1.0, p["speed"] / MAX_SPEED))
            draw_car(canvas, cx, by, COLORS[i], p["ext"], p["vel"], now < p["boost_until"], blink, now)
            if p["off_time"] > 0:   # aviso: te sales de la carretera
                a = 0.20 + 0.12 * math.sin(now * 18)
                panel(canvas, LANE_X0[i], 0, LANE_X0[i] + LW, H, (0, 0, 255), a)
                text(canvas, "VUELVE A LA CARRETERA!", LANE_CX[i], int(H * 0.30), 0.9, (80, 80, 255), 3, "c")
        else:
            msg = {"off": "FUERA DE PISTA!", "obstacle": "CHOQUE!"}.get(p["reason"], "BOOM!")
            text(canvas, msg, cx, CAR_Y - 170 - 10 * math.sin(now * 20), 1.4, YELLOW, 4, "c")

    draw_particles(canvas)

    for r in RINGS[:]:
        t = now - r[2]
        if t > 0.5:
            RINGS.remove(r)
            continue
        col = int(255 * (1 - t / 0.5))
        cv2.circle(canvas, (int(r[0]), int(r[1])), int(t * 650), (0, col // 2 + 40, col), 5, cv2.LINE_AA)
    for i, p in enumerate(players):
        if now < p["flash_until"]:
            a = 0.7 * (p["flash_until"] - now) / 0.18
            panel(canvas, LANE_X0[i], 0, LANE_X0[i] + LW, H, (255, 255, 255), a)

    for i, hd in enumerate(hands):
        if hd is not None:
            draw_hand(canvas, hd, COLORS[i], cfg.NITRO)

