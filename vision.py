"""Vision por computador: modelo de MediaPipe, analisis de la mano (dedos, pulgar, angulos) e hilo de camara."""
import math
import os
import threading
import time

import cv2
import numpy as np
import requests
import mediapipe as mp

from config import *

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


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# (base, articulacion media, punta): indice, corazon, anular, menique
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]

# Vineta para oscurecer la camara (se aplica en el hilo de camara, es gratis)
_yy, _xx = np.mgrid[0:H, 0:W]
_r = np.sqrt(((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2)
_v = np.clip(0.62 - 0.30 * _r ** 2, 0.18, 0.62)

# Vineta para oscurecer la camara (se aplica en el hilo de camara, es gratis)
_yy, _xx = np.mgrid[0:H, 0:W]
_r = np.sqrt(((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2)
_v = np.clip(0.62 - 0.30 * _r ** 2, 0.18, 0.62)
VIG = np.dstack([(_v * 255).astype(np.uint8)] * 3)
def dst(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dpx(a, b):
    """Distancia en pixeles de pantalla (corrige la proporcion 16:9 de las coordenadas normalizadas)."""
    return math.hypot((a[0] - b[0]) * W, (a[1] - b[1]) * H)


def analyze_hand(pts):
    """Devuelve (dedos extendidos de izq. a der., puntas en ese orden, dedos cruzados, ratio del pulgar).

    ratio del pulgar = distancia punta del pulgar - base del menique, dividida por el largo de la palma.
    Pulgar suelto ~1.1 o mas; pulgar metido en la palma ~0.5 o menos.
    """
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

    palm = max(dpx(pts[0], pts[9]), 1e-6)
    thumb_ratio = dpx(pts[4], pts[17]) / palm
    return ext, order, crossed, thumb_ratio



# Articulacion interfalangica proximal (PIP) de cada dedo: (base, PIP, falange distal) -> indice, corazon, anular, menique
PIP_TRIPLETS = [(5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]


def finger_angles(pts):
    """Angulo de flexion (grados) de cada dedo en el PIP: 0 = dedo recto, ~90-100 = doblado del todo."""
    out = []
    for a, b, c in PIP_TRIPLETS:
        v1 = ((pts[a][0] - pts[b][0]) * W, (pts[a][1] - pts[b][1]) * H)
        v2 = ((pts[c][0] - pts[b][0]) * W, (pts[c][1] - pts[b][1]) * H)
        n = math.hypot(*v1) * math.hypot(*v2)
        if n < 1e-6:
            out.append(0.0)
            continue
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
        out.append(180.0 - math.degrees(math.acos(cosang)))
    return out


class VisionWorker(threading.Thread):
    def __init__(self, cam=0):
        super().__init__(daemon=True)
        self.cam = cam
        self.lock = threading.Lock()
        self.running = True
        self.ready = threading.Event()
        self.failed = False
        self.bg = np.zeros((H, W, 3), np.uint8)
        self.hands = [None, None]
        self.det_fps = 0.0
        self.last_pts = [None, None]
        self.last_order = [[8, 12, 16, 20]] * 2
        self.last_ratio = [1.0, 1.0]
        self.last_seen = [0.0, 0.0]
        self.ext_state = [[False] * 4 for _ in range(2)]
        self.ext_cnt = [[0] * 4 for _ in range(2)]
        self.cross_cnt = [0, 0]
        self.thumb_state = [False, False]

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
        # Las manos se asignan por posicion: la de mas a la izquierda es la 0 y la otra la 1.
        found = sorted(([lm[0].x, [(p.x, p.y) for p in lm]] for lm in res.hand_landmarks),
                       key=lambda t: t[0])
        slots = [None, None]
        if len(found) >= 2:
            slots[0], slots[1] = found[0][1], found[-1][1]
        elif len(found) == 1:
            slots[0 if found[0][0] < 0.5 else 1] = found[0][1]

        out = [None, None]
        for i in range(2):
            if slots[i] is not None:
                self.last_pts[i] = slots[i]
                self.last_seen[i] = now
                raw, order, crossed, tr = analyze_hand(slots[i])
                for k in range(4):             # anti-parpadeo: hay que verlo 2 frames seguidos
                    if raw[k] != self.ext_state[i][k]:
                        self.ext_cnt[i][k] += 1
                        if self.ext_cnt[i][k] >= 2:
                            self.ext_state[i][k] = raw[k]
                            self.ext_cnt[i][k] = 0
                    else:
                        self.ext_cnt[i][k] = 0
                if tr < THUMB_IN:              # histeresis del pulgar
                    self.thumb_state[i] = True
                elif tr > THUMB_OUT:
                    self.thumb_state[i] = False
                self.cross_cnt[i] = self.cross_cnt[i] + 1 if crossed else 0
                self.last_order[i] = order
                self.last_ratio[i] = tr
                out[i] = dict(pts=slots[i], ext=list(self.ext_state[i]), order=order,
                              crossed=self.cross_cnt[i] >= 3, thumb=self.thumb_state[i], thumb_ratio=tr,
                              t=now, fresh=True)
            elif self.last_pts[i] is not None and now - self.last_seen[i] < 0.25:
                out[i] = dict(pts=self.last_pts[i], ext=list(self.ext_state[i]),
                              order=self.last_order[i], crossed=False,
                              thumb=self.thumb_state[i], thumb_ratio=self.last_ratio[i],
                              t=self.last_seen[i], fresh=False)
            else:
                self.last_pts[i] = None
                self.ext_state[i] = [False] * 4
                self.ext_cnt[i] = [0] * 4
                self.cross_cnt[i] = 0
                self.thumb_state[i] = False
        return out

    def run(self):
        cap = cv2.VideoCapture(self.cam)
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

