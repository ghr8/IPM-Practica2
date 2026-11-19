"""Menu de seleccion de modo de juego, interactivo con el dedo indice (tambien raton y teclas 1-3).

Se apunta con la punta del indice sobre una tarjeta y se mantiene ~1.2 s: el aro de progreso se llena y se elige.
"""
import time

import cv2
import numpy as np

import config as cfg
from config import *
from render import text, panel, draw_hand
from storage import clean_name

DWELL_SECONDS = 1.2
CARD_W, CARD_H, CARD_GAP, CARD_Y = 360, 370, 40, 190
CARD_X0 = (W - (3 * CARD_W + 2 * CARD_GAP)) // 2

CARDS = [
    ("bimanual", "DOS MANOS", "1 jugador", "Las dos manos al volante", "Ideal para coordinar ambas manos"),
    ("unimanual", "UNA MANO", "1 jugador", "Solo una mano", "Para rehabilitar la mano que elijas"),
    ("duelo", "DUELO", "2 jugadores", "Dos jugadores a la vez", "Una mano cada uno, pantalla dividida"),
]

_FS = [False]


def toggle_fullscreen():
    _FS[0] = not _FS[0]
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if _FS[0] else cv2.WINDOW_NORMAL)


def _rect(i):
    x = CARD_X0 + i * (CARD_W + CARD_GAP)
    return x, CARD_Y, x + CARD_W, CARD_Y + CARD_H


def _card_at(pos):
    if pos is None:
        return None
    for i in range(len(CARDS)):
        x1, y1, x2, y2 = _rect(i)
        if x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2:
            return i
    return None


def _dots(canvas, cx, cy, color):
    for k in range(4):
        c = (int(cx - 45 + k * 30), int(cy))
        cv2.circle(canvas, c, 11, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, c, 11, WHITE, 2, cv2.LINE_AA)


def _icon(canvas, mode, cx, cy):
    if mode == "bimanual":
        _dots(canvas, cx - 75, cy, COLORS[0])
        _dots(canvas, cx + 75, cy, COLORS[0])
    elif mode == "unimanual":
        _dots(canvas, cx, cy, COLORS[0])
    else:
        _dots(canvas, cx - 75, cy, COLORS[0])
        _dots(canvas, cx + 75, cy, COLORS[1])


def _draw_menu(canvas, hover, dwell, cursor, hand, now):
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.55)
    text(canvas, "CARRERA DE COCHES", W // 2, 90, 2.0, YELLOW, 5, "c")
    text(canvas, "Elige como quieres jugar: apunta con el dedo indice a una opcion", W // 2, 145, 0.8, WHITE, 2, "c")

    for i, (mode, title, who, l1, l2) in enumerate(CARDS):
        x1, y1, x2, y2 = _rect(i)
        on = hover == i
        panel(canvas, x1, y1, x2, y2, (70, 55, 30) if on else (25, 25, 25), 0.85 if on else 0.7)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), YELLOW if on else (150, 150, 150), 5 if on else 2, cv2.LINE_AA)
        cx = (x1 + x2) // 2
        text(canvas, title, cx, y1 + 75, 1.3, WHITE, 3, "c")
        text(canvas, who, cx, y1 + 120, 0.85, YELLOW, 2, "c")
        _icon(canvas, mode, cx, y1 + 200)
        text(canvas, l1, cx, y1 + 280, 0.65, WHITE, 1, "c")
        text(canvas, l2, cx, y1 + 312, 0.55, (200, 200, 200), 1, "c")
        if on and dwell > 0:
            cv2.rectangle(canvas, (x1 + 20, y2 - 28), (x2 - 20, y2 - 14), (60, 60, 60), -1)
            cv2.rectangle(canvas, (x1 + 20, y2 - 28), (int(x1 + 20 + (CARD_W - 40) * dwell), y2 - 14), GREEN, -1)

    if hand is not None:
        draw_hand(canvas, hand, COLORS[0])
    else:
        if int(now * 2) % 2 == 0:
            text(canvas, "Muestra tu mano a la camara y apunta con el indice", W // 2, 610, 0.8, (80, 200, 255), 2, "c")
    if cursor is not None:
        c = (int(cursor[0]), int(cursor[1]))
        cv2.circle(canvas, c, 26, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.circle(canvas, c, 26, YELLOW, 3, cv2.LINE_AA)
        if hover is not None and dwell > 0:
            cv2.ellipse(canvas, c, (26, 26), 0, -90, -90 + 360 * dwell, GREEN, 8, cv2.LINE_AA)
        cv2.circle(canvas, c, 5, YELLOW, -1, cv2.LINE_AA)

    text(canvas, "Manten el dedo quieto sobre la opcion unos segundos para elegirla", W // 2, H - 90, 0.7, WHITE, 1, "c")
    text(canvas, "Tambien: teclas 1, 2, 3 o clic con el raton   |   F pantalla completa   |   ESC salir",
         W // 2, H - 55, 0.6, (190, 190, 190), 1, "c")


def choose_mode(worker):
    """Bucle del menu. Devuelve el modo elegido o None si se pulsa ESC."""
    mouse = {"pos": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    cv2.setMouseCallback(WIN, on_mouse)
    cursor, hover, dwell = None, None, 0.0
    last = time.time()
    try:
        while True:
            t0 = time.time()
            dt = min(t0 - last, 0.05)
            last = t0
            with worker.lock:
                bg = worker.bg.copy()
                hands = list(worker.hands)
            hand = next((h for h in hands if h is not None), None)

            if hand is not None:
                tip = hand["pts"][8]                         # punta del indice
                target = np.array([tip[0] * W, tip[1] * H])
                cursor = target if cursor is None else cursor * 0.6 + target * 0.4   # suavizado
            else:
                cursor = None

            if cursor is not None:
                h = _card_at(cursor)
                dwell = dwell + dt / DWELL_SECONDS if (h is not None and h == hover) else 0.0
                hover = h
                if hover is not None and dwell >= 1.0:
                    return CARDS[hover][0]
            else:
                hover, dwell = _card_at(mouse["pos"]), 0.0

            if mouse["click"] is not None:
                c = _card_at(mouse["click"])
                mouse["click"] = None
                if c is not None:
                    return CARDS[c][0]

            _draw_menu(bg, hover, min(dwell, 1.0), cursor, hand, t0)
            cv2.imshow(WIN, bg)
            key = cv2.waitKey(max(1, int(1000 / FPS_TARGET - (time.time() - t0) * 1000))) & 0xFF
            if key == 27:
                return None
            if key in (ord("1"), ord("2"), ord("3")):
                return CARDS[key - ord("1")][0]
            if key in (ord("f"), ord("F")):
                toggle_fullscreen()
    finally:
        cv2.setMouseCallback(WIN, lambda *a: None)


def ask_names_window(worker, n=1):
    """Pide el nombre de cada jugador escribiendo en la ventana. Devuelve [nombre1, nombre2] o None si se pulsa ESC."""
    names = []
    for i in range(n):
        buf = ""
        title = "ESCRIBE TU NOMBRE" if n == 1 else f"NOMBRE DEL JUGADOR {i + 1}"
        while True:
            t0 = time.time()
            with worker.lock:
                bg = worker.bg.copy()
            panel(bg, 0, 0, W, H, (0, 0, 0), 0.7)
            text(bg, title, W // 2, 210, 1.6, YELLOW, 4, "c")
            if n == 1:
                text(bg, "Con este nombre se guarda tu progreso y se compara con tus sesiones anteriores",
                     W // 2, 262, 0.7, WHITE, 1, "c")
            x1, y1, x2, y2 = W // 2 - 260, 310, W // 2 + 260, 390
            panel(bg, x1, y1, x2, y2, (40, 40, 40), 0.9)
            cv2.rectangle(bg, (x1, y1), (x2, y2), YELLOW, 3, cv2.LINE_AA)
            if buf:
                text(bg, buf, x1 + 20, y1 + 55, 1.3, WHITE, 3)
            else:
                text(bg, f"Jugador {i + 1}", x1 + 20, y1 + 55, 1.3, (130, 130, 130), 2)
            if int(t0 * 2) % 2 == 0:                      # cursor parpadeante
                cx = x1 + 20 + (cv2.getTextSize(buf, FONT, 1.3, 3)[0][0] if buf else 0) + 4
                cv2.line(bg, (cx, y1 + 18), (cx, y1 + 62), YELLOW, 3, cv2.LINE_AA)
            text(bg, "Escribe con el teclado (maximo 12 letras)", W // 2, 440, 0.65, (200, 200, 200), 1, "c")
            text(bg, "ENTER: continuar     RETROCESO: borrar     ESC: salir", W // 2, 480, 0.65, (200, 200, 200), 1, "c")
            cv2.imshow(WIN, bg)
            key = cv2.waitKey(max(1, int(1000 / FPS_TARGET - (time.time() - t0) * 1000))) & 0xFF
            if key == 27:
                return None
            if key in (13, 10):
                break
            if key in (8, 127):
                buf = buf[:-1]
            elif 32 <= key < 127 and len(buf) < 12:
                buf += chr(key)
        names.append(clean_name(buf, f"Jugador {i + 1}"))
    while len(names) < 2:
        names.append(names[0])
    return names
