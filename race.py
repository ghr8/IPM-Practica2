import os
import time
import math
import cv2
import numpy as np
import requests
import mediapipe as mp

# ---------------------------------------------------------------
# Modelo de manos (se descarga solo la primera vez)
# ---------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
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

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

# ---------------------------------------------------------------
# Ajustes del juego
# ---------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
COUNTDOWN_SECONDS = 3
EXPLOSION_SECONDS = 2.0
COLORS = [(255, 160, 0), (0, 130, 255)]  # BGR jugador 1 y jugador 2

WAITING, COUNTDOWN, RACING, FINISHED = range(4)

# (base, articulación media, punta) de índice, corazón, anular y meñique
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]


# ---------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def analyze_hand(lm):
    """Devuelve (4 dedos extendidos ordenados de izquierda a derecha, dedos cruzados)."""
    wrist = lm[0]
    fingers = []
    for mcp, pip, tip in FINGERS:
        extended = dist(lm[tip], wrist) > dist(lm[pip], wrist)
        fingers.append((lm[mcp].x, extended))
    fingers.sort(key=lambda f: f[0])  # de izquierda a derecha en pantalla
    ext = [f[1] for f in fingers]

    # Dedos en cruz: índice y corazón extendidos con las puntas "intercambiadas"
    sep_tips = lm[8].x - lm[12].x
    sep_base = lm[5].x - lm[9].x
    crossed = (dist(lm[8], wrist) > dist(lm[6], wrist) and
               dist(lm[12], wrist) > dist(lm[10], wrist) and
               sep_tips * sep_base < 0 and abs(sep_tips) > 0.015)
    return ext, crossed


def put_center(img, text, y, scale=1.0, color=(255, 255, 255), thick=2):
    size = cv2.getTextSize(text, FONT, scale, thick)[0]
    x = (img.shape[1] - size[0]) // 2
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick)


def new_player():
    return {"x": 0.0, "progress": 0.0, "ext": [False] * 4, "crossed": False,
            "exploded_until": 0.0, "explosions": 0}


# ---------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, frame = cap.read()
    h, w = frame.shape[:2]

    # Medidas que dependen del tamaño de pantalla
    ROAD_W = w * 0.20
    SWAY_AMP = w * 0.06
    SWAY_FREQ = 5.8 / h
    CLAMP_X = w / 4 - 30
    MAX_SPEED = h * 0.30        # px/s con las 4 ruedas
    STEER_RATE = w * 0.12       # px/s por rueda de diferencia
    TRACK_LENGTH = h * 8
    CAR_Y = int(h * 0.80)
    LANE_CX = [w * 0.25, w * 0.75]

    def sway(d):
        return SWAY_AMP * math.sin(d * SWAY_FREQ)

    players = [new_player(), new_player()]
    state = WAITING
    countdown_start = 0.0
    winner = None
    start = time.time()
    last_ts = -1
    last_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - last_time
        last_time = now

        # ---------- Detección de manos ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((now - start) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        # Mano en la mitad izquierda -> jugador 1, derecha -> jugador 2
        hands = [None, None]
        for lm in result.hand_landmarks:
            idx = 0 if lm[0].x < 0.5 else 1
            if hands[idx] is None:
                hands[idx] = lm

        present = all(hd is not None for hd in hands)

        for i in range(2):
            if hands[i] is not None:
                players[i]["ext"], players[i]["crossed"] = analyze_hand(hands[i])
            else:
                players[i]["ext"], players[i]["crossed"] = [False] * 4, False

        # ---------- Lógica según el estado ----------
        if state == WAITING:
            if present:
                state = COUNTDOWN
                countdown_start = now

        elif state == COUNTDOWN:
            if not present:
                state = WAITING
            elif now - countdown_start >= COUNTDOWN_SECONDS:
                state = RACING

        elif state == RACING:
            for i, p in enumerate(players):
                if now < p["exploded_until"]:
                    continue

                left = sum(p["ext"][:2])
                right = sum(p["ext"][2:])
                speed = MAX_SPEED * (left + right) / 4

                # Más ruedas izquierdas empujando -> el coche gira a la derecha
                p["x"] += (left - right) * STEER_RATE * dt
                p["x"] = max(-CLAMP_X, min(CLAMP_X, p["x"]))

                road_c = sway(p["progress"])
                if abs(p["x"] - road_c) > ROAD_W / 2:
                    speed *= 0.35  # fuera de la carretera

                p["progress"] += speed * dt

                if p["crossed"]:
                    p["exploded_until"] = now + EXPLOSION_SECONDS
                    p["explosions"] += 1
                    p["x"] = sway(p["progress"])  # reaparece en el centro

                if p["progress"] >= TRACK_LENGTH:
                    winner = i
                    state = FINISHED
                    break

        # ---------- Dibujo ----------
        frame = cv2.convertScaleAbs(frame, alpha=0.6)  # oscurece la cámara
        overlay = frame.copy()

        for i, p in enumerate(players):
            pts_l, pts_r = [], []
            for y in range(0, h + 8, 8):
                d = p["progress"] + (CAR_Y - y)
                c = LANE_CX[i] + sway(d)
                pts_l.append((int(c - ROAD_W / 2), y))
                pts_r.append((int(c + ROAD_W / 2), y))
            poly = np.array(pts_l + pts_r[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (70, 70, 70))
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        for i, p in enumerate(players):
            pts_l, pts_r = [], []
            for y in range(0, h + 8, 8):
                d = p["progress"] + (CAR_Y - y)
                c = LANE_CX[i] + sway(d)
                pts_l.append((int(c - ROAD_W / 2), y))
                pts_r.append((int(c + ROAD_W / 2), y))
            cv2.polylines(frame, [np.array(pts_l, dtype=np.int32)], False, (255, 255, 255), 3)
            cv2.polylines(frame, [np.array(pts_r, dtype=np.int32)], False, (255, 255, 255), 3)

            # Línea de meta
            yf = int(CAR_Y - (TRACK_LENGTH - p["progress"]))
            if 0 <= yf < h:
                cf = LANE_CX[i] + sway(TRACK_LENGTH)
                cv2.rectangle(frame, (int(cf - ROAD_W / 2), yf - 8),
                              (int(cf + ROAD_W / 2), yf + 8), (255, 255, 255), -1)
                cv2.putText(frame, "META", (int(cf - 35), yf - 15), FONT, 0.8, (255, 255, 255), 2)

            # Coche
            cx = int(LANE_CX[i] + p["x"])
            cw, ch = int(w * 0.022), int(h * 0.07)
            ww, wh = int(cw * 0.55), int(ch * 0.45)

            if now < p["exploded_until"]:
                frac = 1 - (p["exploded_until"] - now) / EXPLOSION_SECONDS
                r = int(20 + frac * 90)
                cv2.circle(frame, (cx, CAR_Y), r, (0, 100, 255), -1)
                cv2.circle(frame, (cx, CAR_Y), int(r * 0.6), (0, 200, 255), -1)
                cv2.circle(frame, (cx, CAR_Y), int(r * 0.3), (255, 255, 255), -1)
                put_center(frame, "BOOM!", CAR_Y - 110, 1.2, (0, 200, 255), 3)
            else:
                cv2.rectangle(frame, (cx - cw, CAR_Y - ch), (cx + cw, CAR_Y + ch), COLORS[i], -1)
                cv2.rectangle(frame, (cx - cw + 4, CAR_Y - ch + 6), (cx + cw - 4, CAR_Y - ch // 3),
                              (230, 230, 230), -1)  # parabrisas
                # 4 ruedas = 4 dedos (verde = dedo extendido)
                wheel_pos = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for k, (sx, sy) in enumerate(wheel_pos):
                    wx = cx + sx * (cw + ww // 2)
                    wy = CAR_Y + int(sy * ch * 0.55)
                    color = (0, 220, 0) if p["ext"][k] else (0, 0, 220)
                    cv2.rectangle(frame, (wx - ww // 2, wy - wh // 2),
                                  (wx + ww // 2, wy + wh // 2), color, -1)

            # Marcador
            pct = int(100 * min(p["progress"], TRACK_LENGTH) / TRACK_LENGTH)
            cv2.putText(frame, f"Jugador {i + 1}: {pct}%  Explosiones: {p['explosions']}",
                        (int(LANE_CX[i] - w * 0.22), 35), FONT, 0.8, COLORS[i], 2)

        # Puntos de las manos para ver qué detecta la cámara
        for i, hd in enumerate(hands):
            if hd is not None:
                for lm in hd:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, COLORS[i], -1)

        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

        # ---------- Mensajes según el estado ----------
        if state == WAITING:
            put_center(frame, "CARRERA DE COCHES", int(h * 0.30), 1.6, (0, 255, 255), 3)
            put_center(frame, "Poned una mano cada uno, uno a cada lado de la pantalla",
                       int(h * 0.40), 0.9)
            for i in range(2):
                ok_txt = "LISTO" if hands[i] is not None else "esperando mano..."
                col = (0, 255, 0) if hands[i] is not None else (0, 0, 255)
                put_center(frame, f"Jugador {i + 1}: {ok_txt}", int(h * 0.50 + i * 45), 0.9, col)

        elif state == COUNTDOWN:
            remaining = COUNTDOWN_SECONDS - (now - countdown_start)
            put_center(frame, str(int(math.ceil(remaining))), int(h * 0.5), 6, (0, 255, 255), 10)

        elif state == RACING:
            if now - countdown_start < COUNTDOWN_SECONDS + 1:
                put_center(frame, "YA!", int(h * 0.5), 5, (0, 255, 0), 8)

        elif state == FINISHED:
            put_center(frame, f"Gana el Jugador {winner + 1}!", int(h * 0.40), 1.8, COLORS[winner], 4)
            put_center(frame, "Enter: otra carrera   ESC: salir", int(h * 0.52), 1.0)

        cv2.imshow("Carrera de coches", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 13 and state == FINISHED:  # Enter
            players = [new_player(), new_player()]
            winner = None
            state = WAITING

    cap.release()
    cv2.destroyAllWindows()
