"""
CARRERA DE COCHES CON LOS DEDOS  -  IPM Practica 2
---------------------------------------------------
- 2 jugadores: una mano cada uno, jugador 1 a la izquierda y jugador 2 a la derecha.
- Cada dedo (indice, corazon, anular, menique) es una rueda. Verde = extendido.
- Mas ruedas empujando = mas velocidad. Mas ruedas a la izquierda que a la derecha = giras a la derecha.
- Cruza indice y corazon = el coche explota.
- Pastillas amarillas = turbo.
Teclas: ESC salir | ENTER revancha | R reiniciar | F pantalla completa
"""
import os
import time
import math
import random
import threading

import cv2
import numpy as np
import requests
import mediapipe as mp

# ------------------------------------------------------------------
# Modelo de manos (se descarga solo la primera vez)
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Descargando hand_landmarker.task ...")
    r = requests.get(MODEL_URL)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)


ensure_model()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ------------------------------------------------------------------
# Ajustes del juego (toca estos numeros para cambiar la dificultad)
# ------------------------------------------------------------------
W, H = 1280, 720
FPS_TARGET = 60
WIN = "Carrera de coches"
FONT = cv2.FONT_HERSHEY_DUPLEX

COUNTDOWN_SECONDS = 3
READY_SECONDS = 0.8        # tiempo con la mano abierta para estar listo
EXPLOSION_SECONDS = 2.0
IMMUNE_SECONDS = 1.2       # tiempo sin poder explotar tras reaparecer
BOOST_SECONDS = 1.4
EXPLOSION_PENALTY = 150    # pixeles que retrocedes al explotar

ROAD_W = W * 0.17
SWAY_AMP = W * 0.055
SWAY_LEN = 1100.0
CLAMP_X = W / 4 - 40
MAX_SPEED = 420.0          # px/s con las 4 ruedas
STEER_RATE = W * 0.16      # px/s por cada rueda de diferencia
TRACK_LENGTH = 9000.0
CAR_Y = int(H * 0.78)
LANE_CX = [W * 0.25, W * 0.75]
CAR_HW, CAR_HH = int(W * 0.02), int(H * 0.07)
STEP = 8

COLORS = [(255, 170, 40), (60, 120, 255)]  # BGR jugador 1 y 2
RED, WHITE, GREEN, YELLOW = (40, 40, 230), (240, 240, 240), (60, 220, 60), (0, 220, 255)

WAITING, COUNTDOWN, RACING, FINISHED = range(4)

# (base, articulacion media, punta): indice, corazon, anular, menique
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]

# Vineta para oscurecer la camara (se aplica en el hilo de camara, es gratis)
_yy, _xx = np.mgrid[0:H, 0:W]
_r = np.sqrt(((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2)
_v = np.clip(0.62 - 0.30 * _r ** 2, 0.18, 0.62)
VIG = np.dstack([(_v * 255).astype(np.uint8)] * 3)

YS = np.arange(0, H + STEP, STEP)

# Pastillas de turbo (misma posicion para los dos carriles)
_rng = random.Random(7)
PADS = []
_d = 900.0
while _d < TRACK_LENGTH - 700:
    PADS.append((_d, _rng.uniform(-0.28, 0.28) * ROAD_W))
    _d += _rng.uniform(900, 1300)


def sway(d):
    k = 2 * math.pi / SWAY_LEN
    return SWAY_AMP * (0.7 * np.sin(d * k) + 0.3 * np.sin(d * k * 2.3 + 1.3))


# ------------------------------------------------------------------
# Analisis de la mano
# ------------------------------------------------------------------
def dst(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def analyze_hand(pts):
    """Devuelve (dedos extendidos de izq. a der., puntas en ese orden, dedos cruzados)."""
    wrist = pts[0]
    items = []
    for mcp, pip, tip in FINGERS:
        items.append((pts[mcp][0], tip, dst(pts[tip], wrist) > dst(pts[pip], wrist)))
    items.sort()
    ext = [it[2] for it in items]
    order = [it[1] for it in items]

    sep_tips = pts[8][0] - pts[12][0]
    sep_base = pts[5][0] - pts[9][0]
    crossed = (dst(pts[8], wrist) > dst(pts[6], wrist) and
               dst(pts[12], wrist) > dst(pts[10], wrist) and
               sep_tips * sep_base < 0 and abs(sep_tips) > 0.015)
    return ext, order, crossed


# ------------------------------------------------------------------
# Hilo de camara + deteccion (asi el juego va fluido a 60 fps)
# ------------------------------------------------------------------
class VisionWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.running = True
        self.ready = threading.Event()
        self.failed = False
        self.bg = np.zeros((H, W, 3), np.uint8)
        self.hands = [None, None]
        self.det_fps = 0.0
        self.last_pts = [None, None]
        self.last_order = [[8, 12, 16, 20]] * 2
        self.last_seen = [0.0, 0.0]
        self.ext_state = [[False] * 4 for _ in range(2)]
        self.ext_cnt = [[0] * 4 for _ in range(2)]
        self.cross_cnt = [0, 0]

    def prepare(self, frame):
        frame = cv2.flip(frame, 1)
        h0, w0 = frame.shape[:2]
        target_h = int(w0 * 9 / 16)
        if target_h <= h0:                     # recorte vertical a 16:9
            y0 = (h0 - target_h) // 2
            frame = frame[y0:y0 + target_h]
        else:                                  # recorte horizontal a 16:9
            tw = int(h0 * 16 / 9)
            x0 = (w0 - tw) // 2
            frame = frame[:, x0:x0 + tw]
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))
        return frame

    def process(self, res):
        now = time.time()
        slots = [None, None]
        for lm in res.hand_landmarks:
            pts = [(p.x, p.y) for p in lm]
            i = 0 if pts[0][0] < 0.5 else 1   # mitad izquierda = jugador 1
            if slots[i] is None:
                slots[i] = pts

        out = [None, None]
        for i in range(2):
            if slots[i] is not None:
                self.last_pts[i] = slots[i]
                self.last_seen[i] = now
                raw, order, crossed = analyze_hand(slots[i])
                for k in range(4):             # anti-parpadeo: hay que verlo 2 frames seguidos
                    if raw[k] != self.ext_state[i][k]:
                        self.ext_cnt[i][k] += 1
                        if self.ext_cnt[i][k] >= 2:
                            self.ext_state[i][k] = raw[k]
                            self.ext_cnt[i][k] = 0
                    else:
                        self.ext_cnt[i][k] = 0
                self.cross_cnt[i] = self.cross_cnt[i] + 1 if crossed else 0
                self.last_order[i] = order
                out[i] = dict(pts=slots[i], ext=list(self.ext_state[i]), order=order,
                              crossed=self.cross_cnt[i] >= 3)
            elif self.last_pts[i] is not None and now - self.last_seen[i] < 0.25:
                out[i] = dict(pts=self.last_pts[i], ext=list(self.ext_state[i]),
                              order=self.last_order[i], crossed=False)
            else:
                self.last_pts[i] = None
                self.ext_state[i] = [False] * 4
                self.ext_cnt[i] = [0] * 4
                self.cross_cnt[i] = 0
        return out

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            self.failed = True
            self.ready.set()
            return

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        with HandLandmarker.create_from_options(options) as landmarker:
            start = time.time()
            last_ts = -1
            t_prev = time.time()
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = self.prepare(frame)

                small = cv2.resize(frame, (640, 360))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                ts = max(int((time.time() - start) * 1000), last_ts + 1)
                last_ts = ts
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                hands = self.process(res)
                bg = cv2.multiply(frame, VIG, scale=1 / 255.0)

                t_now = time.time()
                fps = 1.0 / max(t_now - t_prev, 1e-4)
                t_prev = t_now
                with self.lock:
                    self.bg = bg
                    self.hands = hands
                    self.det_fps = 0.9 * self.det_fps + 0.1 * fps
                self.ready.set()
        cap.release()


# ------------------------------------------------------------------
# Dibujo: utilidades
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Particulas
# ------------------------------------------------------------------
PARTS = []
RINGS = []
SHAKE = [0.0]


def spawn(x, y, vx, vy, life, color, size, g=0.0):
    if len(PARTS) < 700:
        PARTS.append([x, y, vx, vy, life, life, color, size, g])


def update_particles(dt):
    global PARTS
    alive = []
    for p in PARTS:
        p[4] -= dt
        if p[4] > 0:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += p[8] * dt
            alive.append(p)
    PARTS = alive


def draw_particles(img):
    for p in PARTS:
        f = p[4] / p[5]
        col = (int(p[6][0] * f), int(p[6][1] * f), int(p[6][2] * f))
        r = max(1, int(p[7] * (0.4 + 0.6 * f)))
        cv2.circle(img, (int(p[0]), int(p[1])), r, col, -1)


# ------------------------------------------------------------------
# Jugadores y fisica
# ------------------------------------------------------------------
def new_player():
    return dict(x=float(sway(0.0)), vel=0.0, speed=0.0, progress=0.0, explosions=0,
                exploded_until=0.0, immune_until=0.0, boost_until=0.0, flash_until=0.0,
                used=set(), finish=None, ext=[False] * 4, crossed=False)


def explode(i, p, now):
    cx = LANE_CX[i] + p["x"]
    for _ in range(110):
        a = random.uniform(0, 2 * math.pi)
        v = random.uniform(80, 560)
        col = random.choice([(0, 80, 255), (0, 160, 255), (0, 230, 255), (255, 255, 255), (60, 60, 60)])
        spawn(cx, CAR_Y, math.cos(a) * v, math.sin(a) * v, random.uniform(0.5, 1.3),
              col, random.uniform(3, 9), 160)
    RINGS.append([cx, CAR_Y, now])
    SHAKE[0] = 22.0
    p["exploded_until"] = now + EXPLOSION_SECONDS
    p["immune_until"] = p["exploded_until"] + IMMUNE_SECONDS
    p["flash_until"] = now + 0.18
    p["explosions"] += 1
    p["speed"] = 0.0
    p["vel"] = 0.0
    p["progress"] = max(0.0, p["progress"] - EXPLOSION_PENALTY)
    p["x"] = float(sway(p["progress"]))


def update_player(i, p, hand, dt, now):
    """Devuelve True si el jugador ha llegado a la meta."""
    if hand is None:
        p["ext"], p["crossed"] = [False] * 4, False
    else:
        p["ext"], p["crossed"] = hand["ext"], hand["crossed"]

    cx = LANE_CX[i] + p["x"]

    if now < p["exploded_until"]:
        if random.random() < 0.5:   # humo del coche destrozado
            spawn(cx + random.uniform(-14, 14), CAR_Y + random.uniform(-10, 10),
                  random.uniform(-25, 25), random.uniform(60, 140), 0.8, (90, 90, 90), 8)
        return False

    L, R = sum(p["ext"][:2]), sum(p["ext"][2:])
    target = MAX_SPEED * (L + R) / 4
    boosting = now < p["boost_until"]
    if boosting:
        target = max(target, MAX_SPEED * 0.5) * 1.5

    road_c = float(sway(p["progress"]))
    off_road = abs(p["x"] - road_c) > ROAD_W / 2 - 6
    if off_road:
        target *= 0.4

    rate = 2.2 if target > p["speed"] else 4.0
    p["speed"] += (target - p["speed"]) * min(1.0, rate * dt)

    steer_target = (L - R) * STEER_RATE * (0.35 + 0.65 * min(1.0, p["speed"] / MAX_SPEED))
    p["vel"] += (steer_target - p["vel"]) * min(1.0, 9 * dt)
    p["x"] += p["vel"] * dt
    if abs(p["x"]) > CLAMP_X:
        p["x"] = math.copysign(CLAMP_X, p["x"])
        p["vel"] = 0.0
    p["progress"] += p["speed"] * dt

    # Turbo
    for idx, (pd, off) in enumerate(PADS):
        if idx not in p["used"] and abs(pd - p["progress"]) < 45 and \
                abs(p["x"] - (float(sway(pd)) + off)) < 55:
            p["used"].add(idx)
            p["boost_until"] = now + BOOST_SECONDS
            for _ in range(25):
                spawn(cx, CAR_Y, random.uniform(-200, 200), random.uniform(-200, 100), 0.5,
                      YELLOW, 4)

    # Particulas del coche
    sp = p["speed"] / MAX_SPEED
    if random.random() < min(1.0, sp + 0.1):
        spawn(cx + random.uniform(-10, 10), CAR_Y + CAR_HH, random.uniform(-20, 20),
              120 + p["speed"] * 0.35, 0.35, (170, 170, 170), 4)
    if boosting:
        for _ in range(3):
            spawn(cx + random.uniform(-9, 9), CAR_Y + CAR_HH + 4, random.uniform(-30, 30),
                  260 + random.uniform(0, 160), 0.3, random.choice([(0, 120, 255), (0, 220, 255)]), 5)
    if off_road and p["speed"] > 40 and random.random() < 0.8:
        spawn(cx + random.uniform(-20, 20), CAR_Y + CAR_HH * 0.6, random.uniform(-70, 70),
              random.uniform(80, 200), 0.5, (40, 90, 130), 4)

    # Dedos en cruz = explosion
    if p["crossed"] and now > p["immune_until"]:
        explode(i, p, now)
        return False

    return p["progress"] >= TRACK_LENGTH


# ------------------------------------------------------------------
# Dibujo: escena
# ------------------------------------------------------------------
def draw_car(img, cx, cy, ang, color, ext, boosting, blink):
    if blink:
        return
    cw, ch = CAR_HW, CAR_HH
    cv2.fillPoly(img, [rot_rect(cx + 7, cy + 9, cw + 8, ch, ang)], (15, 15, 15))  # sombra

    wheel_pos = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for k, (sx, sy) in enumerate(wheel_pos):
        wx, wy = local(cx, cy, ang, sx * (cw + 6), sy * ch * 0.55)
        col = GREEN if ext[k] else RED
        cv2.fillPoly(img, [rot_rect(wx, wy, 8, ch * 0.22, ang)], col)
        cv2.polylines(img, [rot_rect(wx, wy, 8, ch * 0.22, ang)], True, (0, 0, 0), 1)

    cv2.fillPoly(img, [rot_rect(cx, cy, cw, ch, ang)], color)
    cv2.polylines(img, [rot_rect(cx, cy, cw, ch, ang)], True, WHITE, 2, cv2.LINE_AA)
    sx_, sy_ = local(cx, cy, ang, 0, 0)
    cv2.fillPoly(img, [rot_rect(sx_, sy_, cw * 0.16, ch * 0.98, ang)], WHITE)
    wx, wy = local(cx, cy, ang, 0, -ch * 0.18)
    cv2.fillPoly(img, [rot_rect(wx, wy, cw * 0.7, ch * 0.2, ang)], (50, 35, 25))
    for sgn in (-1, 1):
        hx, hy = local(cx, cy, ang, sgn * cw * 0.6, -ch * 0.92)
        cv2.circle(img, (int(hx), int(hy)), 5, YELLOW, -1)
        rx, ry = local(cx, cy, ang, sgn * cw * 0.6, ch * 0.92)
        cv2.circle(img, (int(rx), int(ry)), 4, (0, 0, 255), -1)


def draw_lane(canvas, i, p, now):
    # Geometria de la carretera
    d = p["progress"] + (CAR_Y - YS)
    c = LANE_CX[i] + sway(d)
    left, right = c - ROAD_W / 2, c + ROAD_W / 2

    # Pianos (rojo/blanco) y linea central discontinua
    for k in range(len(YS) - 1):
        col = RED if int(d[k] // 45) % 2 == 0 else WHITE
        y1, y2 = int(YS[k]), int(YS[k + 1])
        cv2.line(canvas, (int(left[k]), y1), (int(left[k + 1]), y2), col, 9)
        cv2.line(canvas, (int(right[k]), y1), (int(right[k + 1]), y2), col, 9)
        if int(d[k] // 70) % 2 == 0:
            cv2.line(canvas, (int(c[k]), y1), (int(c[k + 1]), y2), (200, 200, 200), 4)

    # Pastillas de turbo
    for idx, (pd, off) in enumerate(PADS):
        if idx in p["used"]:
            continue
        y = CAR_Y - (pd - p["progress"])
        if -60 < y < H + 60:
            x = LANE_CX[i] + float(sway(pd)) + off
            for j in range(3):
                yy = y + j * 18
                pts = np.array([(x - 26, yy + 12), (x, yy - 6), (x + 26, yy + 12)], np.int32)
                cv2.polylines(canvas, [pts], False, YELLOW, 6, cv2.LINE_AA)

    # Linea de meta
    yf = int(CAR_Y - (TRACK_LENGTH - p["progress"]))
    if -40 < yf < H + 40:
        cf = LANE_CX[i] + float(sway(TRACK_LENGTH))
        x0 = int(cf - ROAD_W / 2)
        n = 12
        wcell = int(ROAD_W / n) + 1
        for r in range(2):
            for q in range(n):
                col = WHITE if (q + r) % 2 == 0 else (20, 20, 20)
                cv2.rectangle(canvas, (x0 + q * wcell, yf - 14 + r * 14),
                              (x0 + (q + 1) * wcell, yf + r * 14), col, -1)
        text(canvas, "META", cf, yf - 28, 0.9, WHITE, 2, "c")

    # Lineas de velocidad
    if p["speed"] > MAX_SPEED * 0.5:
        for _ in range(int(p["speed"] / MAX_SPEED * 8)):
            x = random.uniform(i * W / 2 + 10, (i + 1) * W / 2 - 10)
            y = random.uniform(0, H)
            cv2.line(canvas, (int(x), int(y)), (int(x), int(y + 30 + p["speed"] * 0.12)),
                     (210, 210, 210), 1)


def draw_hand(canvas, hand, color):
    px = [(int(x * W), int(y * H)) for x, y in hand["pts"]]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], color, 3, cv2.LINE_AA)
    for pt in px:
        cv2.circle(canvas, pt, 4, WHITE, -1, cv2.LINE_AA)
    for k, tip in enumerate(hand["order"]):
        col = GREEN if hand["ext"][k] else RED
        cv2.circle(canvas, px[tip], 11, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, px[tip], 11, WHITE, 2, cv2.LINE_AA)


def draw_hud(canvas, players, now, race_t):
    lead = None
    if players[0]["progress"] != players[1]["progress"]:
        lead = 0 if players[0]["progress"] > players[1]["progress"] else 1

    for i, p in enumerate(players):
        x0 = int(i * W / 2)
        panel(canvas, x0 + 14, 62, x0 + 300, 130, (0, 0, 0), 0.45)
        text(canvas, f"JUGADOR {i + 1}", x0 + 26, 90, 0.8, COLORS[i], 2)
        text(canvas, f"Explosiones: {p['explosions']}", x0 + 26, 118, 0.6, WHITE, 1)
        if lead is not None:
            lab = "1o" if lead == i else "2o"
            text(canvas, lab, x0 + W / 2 - 40, 100, 1.6, YELLOW if lead == i else (190, 190, 190), 3, "r")

        kmh = int(p["speed"] / MAX_SPEED * 180)
        text(canvas, f"{kmh}", x0 + 30, H - 40, 1.6, WHITE, 3)
        text(canvas, "km/h", x0 + 30 + 100, H - 40, 0.7, (200, 200, 200), 1)

        # Ruedas / dedos
        for k in range(4):
            cxk = int(x0 + W / 4 - 75 + k * 50)
            col = GREEN if p["ext"][k] else RED
            cv2.circle(canvas, (cxk, H - 55), 17, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cxk, H - 55), 17, WHITE, 2, cv2.LINE_AA)
        text(canvas, "ruedas", x0 + W / 4, H - 14, 0.55, (210, 210, 210), 1, "c")

        if now < p["boost_until"]:
            text(canvas, "TURBO!", LANE_CX[i] + p["x"], CAR_Y - 90, 1.0, YELLOW, 3, "c")

    # Barra de progreso central
    b1, b2, by = W // 2 - 280, W // 2 + 280, 32
    panel(canvas, b1 - 24, 8, b2 + 70, 66, (0, 0, 0), 0.5)
    cv2.line(canvas, (b1, by), (b2, by), (180, 180, 180), 4)
    cv2.rectangle(canvas, (b2 - 2, by - 14), (b2 + 8, by + 14), WHITE, -1)
    for i, p in enumerate(players):
        fx = b1 + (b2 - b1) * min(1.0, p["progress"] / TRACK_LENGTH)
        cv2.circle(canvas, (int(fx), by + (-8 if i == 0 else 8)), 9, COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(fx), by + (-8 if i == 0 else 8)), 9, WHITE, 2, cv2.LINE_AA)
    text(canvas, f"{race_t:05.1f}s", W // 2, 62, 0.6, WHITE, 1, "c")


def draw_scene(canvas, players, hands, now):
    # Hierba translucida a cada lado
    for i in range(2):
        panel(canvas, i * W / 2, 0, (i + 1) * W / 2, H, (10, 55, 10), 0.30)

    # Asfalto
    overlay = canvas.copy()
    for i, p in enumerate(players):
        d = p["progress"] + (CAR_Y - YS)
        c = LANE_CX[i] + sway(d)
        left = np.stack([c - ROAD_W / 2, YS], 1)
        right = np.stack([c + ROAD_W / 2, YS], 1)
        poly = np.vstack([left, right[::-1]]).astype(np.int32)
        cv2.fillPoly(overlay, [poly], (62, 62, 66))
    canvas[:] = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)

    for i, p in enumerate(players):
        draw_lane(canvas, i, p, now)

    for i, p in enumerate(players):
        cx = LANE_CX[i] + p["x"]
        if now >= p["exploded_until"]:
            ang = max(-0.45, min(0.45, math.atan2(p["vel"], max(p["speed"], 200.0))))
            blink = now < p["immune_until"] and int(now * 10) % 2 == 0
            draw_car(canvas, cx, CAR_Y, ang, COLORS[i], p["ext"], now < p["boost_until"], blink)
        else:
            text(canvas, "BOOM!", cx, CAR_Y - 90 - 12 * math.sin(now * 20), 1.5, YELLOW, 4, "c")

    draw_particles(canvas)

    # Ondas de choque y destello
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
            panel(canvas, i * W / 2, 0, (i + 1) * W / 2, H, (255, 255, 255), a)

    for i, hd in enumerate(hands):
        if hd is not None:
            draw_hand(canvas, hd, COLORS[i])


def draw_waiting(canvas, ready, hands, now):
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.35)
    pulse = 1.0 + 0.04 * math.sin(now * 4)
    text(canvas, "CARRERA DE COCHES", W // 2, 130, 2.2 * pulse, YELLOW, 5, "c")
    text(canvas, "Pon tu mano ABIERTA en tu lado de la pantalla", W // 2, 190, 0.95, WHITE, 2, "c")

    for i in range(2):
        cx, cy = int(LANE_CX[i]), int(H * 0.50)
        col = GREEN if ready[i] >= 1.0 else COLORS[i]
        cv2.circle(canvas, (cx, cy), 82, (40, 40, 40), 10, cv2.LINE_AA)
        if ready[i] > 0:
            cv2.ellipse(canvas, (cx, cy), (82, 82), 0, -90, -90 + 360 * ready[i], col, 10, cv2.LINE_AA)
        text(canvas, f"J{i + 1}", cx, cy + 14, 1.6, COLORS[i], 3, "c")
        if ready[i] >= 1.0:
            msg, mc = "LISTO!", GREEN
        elif hands[i] is not None:
            msg, mc = "Abre la mano...", YELLOW
        else:
            msg, mc = "Esperando mano...", (170, 170, 170)
        text(canvas, msg, cx, cy + 135, 0.9, mc, 2, "c")

    text(canvas, "Dedos = ruedas  |  Mas ruedas a la izquierda = giras a la derecha",
         W // 2, H - 110, 0.75, WHITE, 1, "c")
    text(canvas, "Cruza indice y corazon = BOOM  |  Pastillas amarillas = turbo",
         W // 2, H - 75, 0.75, WHITE, 1, "c")
    text(canvas, "ESC salir   F pantalla completa", W // 2, H - 35, 0.6, (190, 190, 190), 1, "c")


def draw_countdown(canvas, elapsed):
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.25)
    n = COUNTDOWN_SECONDS - int(elapsed)
    frac = elapsed - int(elapsed)
    scale = 9.0 - 4.0 * frac
    col = [(0, 0, 255), (0, 170, 255), (0, 255, 255)][min(2, max(0, n - 1))]
    text(canvas, str(n), W // 2, int(H * 0.5 + 60 * scale / 9), scale, col, 18, "c")
    text(canvas, "PREPARADOS", W // 2, 140, 1.6, WHITE, 3, "c")


def draw_finished(canvas, players, winner, now):
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.55)
    text(canvas, f"GANA EL JUGADOR {winner + 1}!", W // 2, 200, 2.0, COLORS[winner], 5, "c")
    for i, p in enumerate(players):
        x = int(LANE_CX[i])
        t = f"{p['finish']:.1f} s" if p["finish"] is not None else \
            f"{int(100 * p['progress'] / TRACK_LENGTH)}%"
        text(canvas, f"Jugador {i + 1}", x, 320, 1.0, COLORS[i], 2, "c")
        text(canvas, t, x, 390, 1.6, WHITE, 3, "c")
        text(canvas, f"Explosiones: {p['explosions']}", x, 440, 0.8, (220, 220, 220), 1, "c")
    text(canvas, "ENTER: revancha     ESC: salir", W // 2, H - 90, 1.0, YELLOW, 2, "c")

    # Confeti
    for _ in range(3):
        spawn(random.uniform(0, W), -10, random.uniform(-60, 60), random.uniform(120, 300), 4.0,
              random.choice([(0, 255, 255), (255, 80, 80), (80, 255, 80), (255, 255, 255), (255, 0, 255)]),
              5, 120)


# ------------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------------
def main():
    worker = VisionWorker()
    worker.start()
    print("Arrancando camara y modelo ...")
    worker.ready.wait(timeout=20)
    if worker.failed or not worker.ready.is_set():
        print("No se pudo abrir la camara (o el modelo ha tardado demasiado).")
        worker.running = False
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    fullscreen = False

    state = WAITING
    players = [new_player(), new_player()]
    ready = [0.0, 0.0]
    countdown_start = race_start = 0.0
    lost = 0.0
    winner = None
    fps = 60.0
    last = time.time()

    while True:
        t_frame = time.time()
        now = t_frame
        dt = min(now - last, 0.05)
        real_dt = max(now - last, 1e-4)
        last = now
        fps = 0.95 * fps + 0.05 * (1.0 / real_dt)

        with worker.lock:
            bg = worker.bg
            hands = list(worker.hands)
            det_fps = worker.det_fps

        present = [h is not None for h in hands]

        # ---------- Logica ----------
        if state == WAITING:
            for i in range(2):
                open_hand = hands[i] is not None and sum(hands[i]["ext"]) >= 3
                if open_hand:
                    ready[i] = min(1.0, ready[i] + dt / READY_SECONDS)
                else:
                    ready[i] = max(0.0, ready[i] - dt * 2.0)
            if ready[0] >= 1.0 and ready[1] >= 1.0:
                state = COUNTDOWN
                countdown_start = now
                lost = 0.0
                players = [new_player(), new_player()]
                PARTS.clear()

        elif state == COUNTDOWN:
            lost = 0.0 if all(present) else lost + dt
            if lost > 0.5:
                state = WAITING
                ready = [0.0, 0.0]
            elif now - countdown_start >= COUNTDOWN_SECONDS:
                state = RACING
                race_start = now

        elif state == RACING:
            for i, p in enumerate(players):
                if update_player(i, p, hands[i], dt, now):
                    p["finish"] = now - race_start
                    winner = i
                    state = FINISHED
                    break

        elif state == FINISHED:
            for p in players:
                p["ext"] = [False] * 4

        update_particles(dt)
        SHAKE[0] *= math.exp(-9 * dt)

        # ---------- Dibujo ----------
        canvas = bg.copy()
        draw_scene(canvas, players, hands, now)
        race_t = (now - race_start) if state == RACING else (
            players[winner]["finish"] if state == FINISHED and winner is not None else 0.0)
        if state in (RACING, FINISHED):
            draw_hud(canvas, players, now, race_t)

        if state == WAITING:
            draw_waiting(canvas, ready, hands, now)
        elif state == COUNTDOWN:
            draw_countdown(canvas, now - countdown_start)
        elif state == RACING and now - race_start < 0.9:
            text(canvas, "YA!", W // 2, int(H * 0.45), 6 + 4 * (now - race_start), GREEN, 14, "c")
        elif state == FINISHED:
            draw_finished(canvas, players, winner, now)
            draw_particles(canvas)

        if SHAKE[0] > 0.8:
            M = np.float32([[1, 0, random.uniform(-SHAKE[0], SHAKE[0])],
                            [0, 1, random.uniform(-SHAKE[0], SHAKE[0])]])
            canvas = cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_REFLECT)

        cv2.putText(canvas, f"{int(fps)} fps | deteccion {int(det_fps)} fps", (W - 280, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)

        # ---------- Teclas y control de fps ----------
        wait = max(1, int(1000 / FPS_TARGET - (time.time() - t_frame) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        if key == 27:
            break
        if key == 13 and state == FINISHED:
            players = [new_player(), new_player()]
            ready = [0.0, 0.0]
            winner = None
            PARTS.clear()
            state = WAITING
        if key in (ord("r"), ord("R")):
            players = [new_player(), new_player()]
            ready = [0.0, 0.0]
            winner = None
            PARTS.clear()
            state = WAITING
        if key in (ord("f"), ord("F")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    worker.running = False
    worker.join(timeout=2)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
