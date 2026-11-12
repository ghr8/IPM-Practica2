"""
CARRERA DE COCHES CON LOS DEDOS (VISTA 3D)  -  IPM Practica 2
-------------------------------------------------------------
- 2 jugadores: una mano cada uno, jugador 1 a la izquierda y jugador 2 a la derecha.
- Cada dedo (indice, corazon, anular, menique) es una rueda. Verde = extendido.
- Mas ruedas empujando = mas velocidad. Mas ruedas a la izquierda que a la derecha = giras a la derecha.
- Si te sales de la carretera, EXPLOTAS. Cruza indice y corazon = tambien explotas.
- Pastillas amarillas = turbo.
- Mini-circuito en el centro con los dos coches y ranking de mejores tiempos (ranking.json).
Teclas: ESC salir | ENTER revancha | R reiniciar | F pantalla completa
"""
import os
import time
import math
import random
import threading
import json
import unicodedata

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

ROAD_W = W * 0.24
SWAY_AMP = W * 0.08
SWAY_LEN = 3000.0
CLAMP_X = W / 4 - 50
MAX_SPEED = 520.0          # px/s con las 4 ruedas
STEER_RATE = W * 0.16      # px/s por cada rueda de diferencia
TRACK_LENGTH = 12000.0
CAR_Y = int(H * 0.86)
LANE_CX = [W * 0.25, W * 0.75]
CAR_HW, CAR_HH = int(W * 0.02), int(H * 0.07)
STEP = 8
OFF_ROAD_GRACE = 0.35      # segundos fuera de la carretera antes de explotar

COLORS = [(255, 170, 40), (60, 120, 255)]  # BGR jugador 1 y 2
RED, WHITE, GREEN, YELLOW = (40, 40, 230), (240, 240, 240), (60, 220, 60), (0, 220, 255)

WAITING, COUNTDOWN, RACING, FINISHED = range(4)

# ------------------------------------------------------------------
# Nombres y ranking de mejores tiempos (se guarda en ranking.json)
# ------------------------------------------------------------------
NAMES = ["Jugador 1", "Jugador 2"]
RANK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ranking.json")


def clean_name(s, default):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip()
    return (s or default)[:12]


def ask_names():
    names = []
    for i in range(2):
        try:
            n = input(f"Nombre del jugador {i + 1} (Enter = Jugador {i + 1}): ")
        except EOFError:
            n = ""
        names.append(clean_name(n, f"Jugador {i + 1}"))
    return names


def load_ranking():
    try:
        with open(RANK_FILE, "r", encoding="utf-8") as f:
            rows = json.load(f)
        rows = [r for r in rows if isinstance(r, dict) and "time" in r and "name" in r]
        return sorted(rows, key=lambda r: r["time"])
    except Exception:
        return []


def add_record(name, t):
    """Guarda el tiempo y devuelve (ranking, posicion del nuevo tiempo)."""
    rows = load_ranking()
    new = {"name": name, "time": round(float(t), 2), "date": time.strftime("%d/%m/%Y")}
    rows.append(new)
    rows.sort(key=lambda r: r["time"])
    rows = rows[:10]
    try:
        with open(RANK_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    idx = next((k for k, r in enumerate(rows) if r is new), -1)
    return rows, idx


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
                used=set(), finish=None, ext=[False] * 4, crossed=False,
                off_time=0.0, reason="")


def explode(i, p, now, reason):
    cx = LANE_CX[i] + p["x"]
    cy = CAR_Y - 55
    for _ in range(110):
        a = random.uniform(0, 2 * math.pi)
        v = random.uniform(80, 560)
        col = random.choice([(0, 80, 255), (0, 160, 255), (0, 230, 255), (255, 255, 255), (60, 60, 60)])
        spawn(cx, cy, math.cos(a) * v, math.sin(a) * v, random.uniform(0.5, 1.3),
              col, random.uniform(3, 9), 160)
    RINGS.append([cx, cy, now])
    SHAKE[0] = 22.0
    p["exploded_until"] = now + EXPLOSION_SECONDS
    p["immune_until"] = p["exploded_until"] + IMMUNE_SECONDS
    p["flash_until"] = now + 0.18
    p["explosions"] += 1
    p["speed"] = 0.0
    p["vel"] = 0.0
    p["off_time"] = 0.0
    p["reason"] = reason
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
            spawn(cx + random.uniform(-14, 14), CAR_Y - 55 + random.uniform(-10, 10),
                  random.uniform(-25, 25), random.uniform(-90, -30), 0.8, (90, 90, 90), 8)
        return False

    L, R = sum(p["ext"][:2]), sum(p["ext"][2:])
    target = MAX_SPEED * (L + R) / 4
    boosting = now < p["boost_until"]
    if boosting:
        target = max(target, MAX_SPEED * 0.5) * 1.5

    road_c = float(sway(p["progress"]))
    off_road = abs(p["x"] - road_c) > ROAD_W / 2 + 12
    if off_road:
        target *= 0.5
        p["off_time"] += dt
    else:
        p["off_time"] = 0.0

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
        if idx not in p["used"] and abs(pd - p["progress"]) < 60 and \
                abs(p["x"] - (float(sway(pd)) + off)) < 60:
            p["used"].add(idx)
            p["boost_until"] = now + BOOST_SECONDS
            for _ in range(25):
                spawn(cx, CAR_Y - 50, random.uniform(-220, 220), random.uniform(-220, 100), 0.5,
                      YELLOW, 4)

    # Particulas del coche
    sp = p["speed"] / MAX_SPEED
    if random.random() < min(1.0, sp + 0.1):
        spawn(cx + random.uniform(-14, 14), CAR_Y - 8, random.uniform(-25, 25),
              120 + p["speed"] * 0.3, 0.35, (170, 170, 170), 4)
    if boosting:
        for _ in range(3):
            spawn(cx + random.uniform(-20, 20), CAR_Y + 6, random.uniform(-30, 30),
                  260 + random.uniform(0, 160), 0.3, random.choice([(0, 120, 255), (0, 220, 255)]), 5)
    if off_road and p["speed"] > 40 and random.random() < 0.8:
        spawn(cx + random.uniform(-40, 40), CAR_Y - 4, random.uniform(-70, 70),
              random.uniform(80, 200), 0.5, (40, 90, 130), 5)

    # Explosiones: salirse de la carretera o dedos en cruz
    if now > p["immune_until"]:
        if p["off_time"] > OFF_ROAD_GRACE:
            explode(i, p, now, "off")
            return False
        if p["crossed"]:
            explode(i, p, now, "cross")
            return False

    return p["progress"] >= TRACK_LENGTH


# ------------------------------------------------------------------
# Dibujo: escena
# ------------------------------------------------------------------
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

    order = sorted(range(2), key=lambda k: (-players[k]["progress"], k))
    for i in reversed(order):                       # el ultimo se dibuja debajo
        fx, fy = circuit_pos(players[i]["progress"] / TRACK_LENGTH)
        c = (int(fx + x0), int(fy + y0))
        if i == order[0]:
            cv2.circle(canvas, c, int(14 + 3 * math.sin(now * 8)), YELLOW, 2, cv2.LINE_AA)
        cv2.circle(canvas, c, 9, COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(canvas, c, 9, WHITE, 2, cv2.LINE_AA)

    for rank, i in enumerate(order):                # ranking en directo
        xx = x0 + MAP_W * (0.27 + 0.46 * rank)
        text(canvas, f"{rank + 1}o {NAMES[i][:9].upper()}", xx, y0 + 186, 0.55, COLORS[i], 1, "c")
    text(canvas, f"{race_t:05.1f}s", x0 + MAP_W // 2, y0 + 207, 0.55, WHITE, 1, "c")


def draw_hud(canvas, players, now, race_t, show_map=True):
    lead = None
    if players[0]["progress"] != players[1]["progress"]:
        lead = 0 if players[0]["progress"] > players[1]["progress"] else 1

    for i, p in enumerate(players):
        x0 = int(i * W / 2)
        px0 = 14 if i == 0 else W - 300           # paneles en los bordes exteriores
        panel(canvas, px0, 14, px0 + 286, 82, (0, 0, 0), 0.45)
        text(canvas, NAMES[i].upper(), px0 + 12, 42, 0.8, COLORS[i], 2)
        text(canvas, f"Explosiones: {p['explosions']}", px0 + 12, 70, 0.6, WHITE, 1)
        if lead is not None:
            lab = "1o" if lead == i else "2o"
            text(canvas, lab, px0 + 274, 72, 1.4, YELLOW if lead == i else (190, 190, 190), 3, "r")

        kmh = int(p["speed"] / MAX_SPEED * 180)
        text(canvas, f"{kmh}", x0 + 30, H - 40, 1.6, WHITE, 3)
        text(canvas, "km/h", x0 + 30 + 100, H - 40, 0.7, (200, 200, 200), 1)

        for k in range(4):                         # ruedas / dedos
            cxk = int(x0 + W / 4 - 75 + k * 50)
            col = GREEN if p["ext"][k] else RED
            cv2.circle(canvas, (cxk, H - 55), 17, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cxk, H - 55), 17, WHITE, 2, cv2.LINE_AA)
        text(canvas, "ruedas", x0 + W / 4, H - 14, 0.55, (210, 210, 210), 1, "c")

        if now < p["boost_until"]:
            text(canvas, "TURBO!", LANE_CX[i] + p["x"], CAR_Y - 175, 1.0, YELLOW, 3, "c")

    if show_map:
        draw_minimap(canvas, players, now, race_t)


def draw_scene(canvas, players, hands, now):
    for i, p in enumerate(players):
        x0 = i * LW
        lane = canvas[:, x0:x0 + LW].copy()
        draw_lane3d(lane, p, now)
        canvas[:, x0:x0 + LW] = lane
    cv2.line(canvas, (W // 2, 0), (W // 2, H), (15, 15, 15), 6)

    for i, p in enumerate(players):
        cx = LANE_CX[i] + p["x"]
        if now >= p["exploded_until"]:
            blink = now < p["immune_until"] and int(now * 10) % 2 == 0
            by = CAR_Y + int(2 * math.sin(now * 40) * min(1.0, p["speed"] / MAX_SPEED))
            draw_car(canvas, cx, by, COLORS[i], p["ext"], p["vel"], now < p["boost_until"], blink, now)
            if p["off_time"] > 0:   # aviso: te sales de la carretera
                a = 0.20 + 0.12 * math.sin(now * 18)
                panel(canvas, i * LW, 0, (i + 1) * LW, H, (0, 0, 255), a)
                text(canvas, "VUELVE A LA CARRETERA!", LANE_CX[i], int(H * 0.30), 0.9, (80, 80, 255), 3, "c")
        else:
            msg = "FUERA DE PISTA!" if p["reason"] == "off" else "BOOM!"
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
            panel(canvas, i * W / 2, 0, (i + 1) * W / 2, H, (255, 255, 255), a)

    for i, hd in enumerate(hands):
        if hd is not None:
            draw_hand(canvas, hd, COLORS[i])


def draw_waiting(canvas, ready, hands, now, ranking):
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
        text(canvas, NAMES[i].upper(), cx, cy - 100, 0.8, COLORS[i], 2, "c")
        if ready[i] >= 1.0:
            msg, mc = "LISTO!", GREEN
        elif hands[i] is not None:
            msg, mc = "Abre la mano...", YELLOW
        else:
            msg, mc = "Esperando mano...", (170, 170, 170)
        text(canvas, msg, cx, cy + 135, 0.9, mc, 2, "c")

    text(canvas, "MEJORES TIEMPOS", W // 2, 545, 0.7, YELLOW, 2, "c")
    if ranking:
        line = "   ".join(f"{k + 1}. {r['name']} {r['time']:.1f}s" for k, r in enumerate(ranking[:3]))
    else:
        line = "Aun no hay tiempos: se el primero!"
    text(canvas, line, W // 2, 578, 0.7, WHITE, 1, "c")

    text(canvas, "Dedos = ruedas  |  Mas ruedas a la izquierda = giras a la derecha",
         W // 2, H - 110, 0.75, WHITE, 1, "c")
    text(canvas, "Cruza indice y corazon = BOOM  |  Fuera de la carretera = BOOM  |  Amarillo = turbo",
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


def draw_finished(canvas, players, winner, now, ranking, new_idx):
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.65)
    text(canvas, f"GANA {NAMES[winner].upper()}!", W // 2, 120, 2.0, COLORS[winner], 5, "c")
    if new_idx == 0:
        text(canvas, "NUEVO RECORD!", W // 2, 172, 1.1, YELLOW, 3, "c")
    for i, p in enumerate(players):
        x = int(LANE_CX[i])
        t = f"{p['finish']:.1f} s" if p["finish"] is not None else \
            f"{int(100 * p['progress'] / TRACK_LENGTH)}%"
        text(canvas, NAMES[i].upper(), x, 250, 1.0, COLORS[i], 2, "c")
        text(canvas, t, x, 312, 1.5, WHITE, 3, "c")
        text(canvas, f"Explosiones: {p['explosions']}", x, 352, 0.7, (220, 220, 220), 1, "c")

    text(canvas, "RANKING - MEJORES TIEMPOS", W // 2, 420, 0.9, YELLOW, 2, "c")
    if not ranking:
        text(canvas, "Aun no hay tiempos", W // 2, 470, 0.8, WHITE, 1, "c")
    for k, r in enumerate(ranking[:5]):
        y = 458 + k * 34
        col = YELLOW if k == new_idx else WHITE
        text(canvas, f"{k + 1}.", W // 2 - 230, y, 0.8, col, 2, "r")
        text(canvas, r["name"].upper(), W // 2 - 215, y, 0.8, col, 2)
        text(canvas, f"{r['time']:.1f}s", W // 2 + 130, y, 0.8, col, 2, "r")
        text(canvas, str(r.get("date", "")), W // 2 + 240, y, 0.6, (190, 190, 190), 1, "r")
    text(canvas, "ENTER: revancha     ESC: salir", W // 2, H - 30, 1.0, YELLOW, 2, "c")

    # Confeti
    for _ in range(3):
        spawn(random.uniform(0, W), -10, random.uniform(-60, 60), random.uniform(120, 300), 4.0,
              random.choice([(0, 255, 255), (255, 80, 80), (80, 255, 80), (255, 255, 255), (255, 0, 255)]),
              5, 120)


# ------------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------------
def main():
    NAMES[:] = ask_names()
    ranking = load_ranking()
    new_idx = -1
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
            for p in players:   # la carretera avanza sola en la pantalla de espera
                p["progress"] = (now * 160) % 4000
                p["x"] = float(sway(p["progress"]))
                p["speed"] = MAX_SPEED * 0.5
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
                    ranking, new_idx = add_record(NAMES[i], p["finish"])
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
            draw_hud(canvas, players, now, race_t, state == RACING)

        if state == WAITING:
            draw_waiting(canvas, ready, hands, now, ranking)
        elif state == COUNTDOWN:
            draw_countdown(canvas, now - countdown_start)
        elif state == RACING and now - race_start < 0.9:
            text(canvas, "YA!", W // 2, int(H * 0.45), 6 + 4 * (now - race_start), GREEN, 14, "c")
        elif state == FINISHED:
            draw_finished(canvas, players, winner, now, ranking, new_idx)
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
