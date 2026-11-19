"""Ajustes del juego y modo de juego (toca estos numeros para cambiar la dificultad)."""
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

W, H = 1280, 720
FPS_TARGET = 60
WIN = "Carrera de coches"
FONT = cv2.FONT_HERSHEY_DUPLEX

COUNTDOWN_SECONDS = 3
READY_SECONDS = 0.8        # tiempo con la mano abierta para estar listo
EXPLOSION_SECONDS = 2.0
IMMUNE_SECONDS = 1.2       # tiempo sin poder explotar tras reaparecer
BOOST_SECONDS = 1.4
EXPLOSION_PENALTY = 150    # minimo de pixeles que retrocedes al explotar
CHECKPOINT_EVERY = 1000.0  # al explotar reapareces en el punto de control anterior (ver players.explode)

ROAD_W = W * 0.24
SWAY_AMP = W * 0.20        # desvio de la carretera: mayor que ROAD_W/2, asi que ir recto SIEMPRE te saca
SWAY_LEN = 2500.0         # longitud de una curva completa (menor = curvas mas seguidas)
CLAMP_X = W / 4 - 50
MAX_SPEED = 350.0          # px/s con las 4 ruedas
STEER_RATE = W * 0.16      # px/s por cada rueda de diferencia
TRACK_LENGTH = 12000.0
CAR_Y = int(H * 0.86)
LANE_X0 = [0, W // 2]       # borde izquierdo de cada carril (cambia segun el modo)
LANE_CX = [W * 0.25, W * 0.75]
CAR_HW, CAR_HH = int(W * 0.02), int(H * 0.07)
STEP = 8
OFF_ROAD_GRACE = 0.35      # segundos fuera de la carretera (acumulados) antes de explotar
OFF_ROAD_LEAK = 0.5        # segundos que se perdonan por cada segundo que vuelves a estar en la carretera
NITRO_MAX = 3              # cargas de nitro acumulables (modos bimanual/unimanual)
THUMB_IN, THUMB_OUT = 0.60, 0.85   # pulgar metido / suelto (distancia punta pulgar - base menique / palma)

# Obstaculos (modos de un jugador): aparecen de golpe, hay que frenar y quedarse quieto 1 s
OBST_DIST = 800.0          # a que distancia por delante aparece el obstaculo (px de carretera)
OBST_HOLD = 1.0            # segundos quieto para que el obstaculo desaparezca
OBST_BRAKE_FRAC = 0.25     # se considera "frenar" cuando los dedos extendidos <= 25 % del total
OBST_STOP_FRAC = 0.05      # se considera "quieto" por debajo del 5 % de la velocidad maxima
OBST_MIN_SPEED = 0.40      # el obstaculo solo aparece si vas al menos al 40 % de la velocidad maxima
OBST_HIT = 25.0            # distancia a la que el coche choca con el obstaculo

COLORS = [(255, 170, 40), (60, 120, 255)]  # BGR jugador 1 y 2
RED, WHITE, GREEN, YELLOW = (40, 40, 230), (240, 240, 240), (60, 220, 60), (0, 220, 255)

WAITING, COUNTDOWN, RACING, FINISHED = range(4)

MODES = ("bimanual", "unimanual", "duelo")
NAMES = ["Jugador 1", "Jugador 2"]

# Estado del modo actual (se rellena con configure_mode; usar siempre cfg.MODE, cfg.NP...)
MODE, NP, NITRO, OBSTACLES = "duelo", 2, False, False


def configure_mode(mode, turbo_auto=False, obstacles=True):
    """Ajusta la disposicion de pantalla segun el modo de juego."""
    global MODE, NP, NITRO, OBSTACLES
    MODE = mode
    NP = 2 if mode == "duelo" else 1
    NITRO = mode != "duelo" and not turbo_auto
    OBSTACLES = mode != "duelo" and obstacles
    LANE_X0[:] = [0, W // 2] if NP == 2 else [W // 4]
    LANE_CX[:] = [x0 + W / 4 for x0 in LANE_X0]


def hands_for(i, hands):
    """Manos que controlan al jugador i (lista de 1 o 2 elementos, pueden ser None)."""
    if MODE == "duelo":
        return [hands[i]]
    if MODE == "bimanual":
        return [hands[0], hands[1]]
    return [next((h for h in hands if h is not None), None)]


def hand_labels():
    """Etiqueta de cada grupo de dedos del jugador 1 (para las metricas)."""
    return ["izquierda", "derecha"] if MODE == "bimanual" else ["mano"]


def ready_inputs(hands):
    """Un bool por cada 'mano abierta' necesaria para empezar."""
    op = [h is not None and sum(h["ext"]) >= 3 for h in hands]
    return [any(op)] if MODE == "unimanual" else op


def hands_present(hands):
    pr = [h is not None for h in hands]
    return [any(pr)] if MODE == "unimanual" else pr


def ready_layout():
    """(posiciones x, etiquetas, iniciales) de los aros de 'listo' de la pantalla de espera."""
    if MODE == "unimanual":
        return [W / 2], [NAMES[0]], ["M"]
    if MODE == "bimanual":
        return [W * 0.25, W * 0.75], ["MANO IZQUIERDA", "MANO DERECHA"], ["I", "D"]
    return list(LANE_CX), list(NAMES), ["J1", "J2"]
