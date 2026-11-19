"""Pantallas y HUD: marcador, espera, cuenta atras, final y informe de sesion."""
import math
import random

import cv2
import numpy as np

import config as cfg
from config import *
from effects import spawn
from render import text, panel, draw_minimap

def draw_hud(canvas, players, now, race_t, show_map=True):
    lead = None
    if len(players) == 2 and players[0]["progress"] != players[1]["progress"]:
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
        if len(players) == 2:
            text(canvas, f"{kmh}", x0 + 30, H - 40, 1.6, WHITE, 3)
            text(canvas, "km/h", x0 + 30 + 100, H - 40, 0.7, (200, 200, 200), 1)
        else:                                      # un jugador: velocidad y nitro en el panel derecho
            rx = W - 300
            panel(canvas, rx, 14, rx + 286, 82, (0, 0, 0), 0.45)
            text(canvas, f"{kmh}", rx + 14, 68, 1.4, WHITE, 3)
            text(canvas, "km/h", rx + 120, 68, 0.7, (200, 200, 200), 1)
            if cfg.NITRO:
                panel(canvas, rx, 90, rx + 286, 140, (0, 0, 0), 0.45)
                text(canvas, "NITRO", rx + 12, 125, 0.8, YELLOW, 2)
                for k in range(NITRO_MAX):
                    c = (rx + 150 + k * 46, 116)
                    if k < p["nitro"]:
                        cv2.circle(canvas, c, 15, YELLOW, -1, cv2.LINE_AA)
                    cv2.circle(canvas, c, 15, WHITE, 2, cv2.LINE_AA)
                panel(canvas, 14, 90, 300, 140, (0, 0, 0), 0.45)
                text(canvas, f"Pulgar: {p['thumb_reps']} rep.", 26, 125, 0.7, WHITE, 1)

        groups = p["fingers"]
        two = len(groups) == 2
        centers = [LANE_CX[i] - 125, LANE_CX[i] + 125] if two else [LANE_CX[i]]
        for g, (grp, cc) in enumerate(zip(groups, centers)):
            for k in range(4):                     # ruedas / dedos
                cxk = int(cc - 57 + k * 38) if two else int(cc - 75 + k * 50)
                rad = 14 if two else 17
                col = GREEN if grp[k] else RED
                cv2.circle(canvas, (cxk, H - 55), rad, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cxk, H - 55), rad, WHITE, 2, cv2.LINE_AA)
            if two:
                text(canvas, "mano izq." if g == 0 else "mano der.", cc, H - 14, 0.55, (210, 210, 210), 1, "c")
            if cfg.NITRO and g < len(p["thumbs"]):     # estado del pulgar (y ratio para calibrar)
                tb, tr = p["thumbs"][g]
                cv2.circle(canvas, (int(cc) - 45, H - 92), 9, (0, 140, 255) if tb else (150, 150, 150), -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(cc) - 45, H - 92), 9, WHITE, 1, cv2.LINE_AA)
                text(canvas, "pulgar " + ("--" if tr is None else f"{tr:.2f}"), cc - 30, H - 86, 0.5,
                     (210, 210, 210), 1)
        if not two:
            text(canvas, "ruedas", LANE_CX[i], H - 14, 0.55, (210, 210, 210), 1, "c")

        if now < p["boost_until"]:
            text(canvas, "TURBO!", LANE_CX[i] + p["x"], CAR_Y - 175, 1.0, YELLOW, 3, "c")

        ob = p["obst"]["active"]                   # obstaculo: frena y quedate quieto 1 s
        if ob is not None:
            if ob["brake_t"] is None:
                if int(now * 4) % 2 == 0:
                    text(canvas, "OBSTACULO! FRENA", LANE_CX[i], 262, 1.0, (80, 80, 255), 3, "c")
            elif ob["still_since"] is not None:
                left = max(0.0, OBST_HOLD - (now - ob["still_since"]))
                text(canvas, f"QUIETO... {left:.1f}s", LANE_CX[i], 262, 1.0, GREEN, 3, "c")
            else:
                text(canvas, "Frena del todo", LANE_CX[i], 262, 0.9, YELLOW, 2, "c")

    if show_map:
        draw_minimap(canvas, players, now, race_t)



HELP = {
    "duelo": ("Pon tu mano ABIERTA en tu lado de la pantalla",
              "Dedos = ruedas  |  Mas ruedas a la izquierda que a la derecha = giras a la derecha",
              "Cruza indice y corazon = BOOM  |  Fuera de la carretera = BOOM  |  Amarillo = turbo"),
    "bimanual": ("Pon las DOS manos ABIERTAS: la izquierda a la izquierda y la derecha a la derecha",
                 "Mas dedos en la mano IZQUIERDA que en la DERECHA = giras a la derecha",
                 "Pulgar metido en la palma = NITRO (lo cargan las pastillas amarillas)  |  Fuera de la carretera = BOOM"),
    "unimanual": ("Pon UNA mano ABIERTA delante de la camara",
                  "Dedos = ruedas  |  Mas ruedas a la izquierda que a la derecha = giras a la derecha",
                  "Pulgar metido en la palma = NITRO  |  Cruza indice y corazon = BOOM  |  Fuera de la carretera = BOOM"),
}


def draw_waiting(canvas, ready, present, now, ranking):
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.35)
    pulse = 1.0 + 0.04 * math.sin(now * 4)
    text(canvas, "CARRERA DE COCHES", W // 2, 130, 2.2 * pulse, YELLOW, 5, "c")
    sub, l1, l2 = HELP[cfg.MODE]
    text(canvas, sub, W // 2, 190, 0.95 if cfg.MODE == "duelo" or cfg.MODE == "unimanual" else 0.8, WHITE, 2, "c")

    pos, labels, tags = ready_layout()
    for i, (px, lab, tag) in enumerate(zip(pos, labels, tags)):
        cx, cy = int(px), int(H * 0.50)
        col = GREEN if ready[i] >= 1.0 else COLORS[i]
        cv2.circle(canvas, (cx, cy), 82, (40, 40, 40), 10, cv2.LINE_AA)
        if ready[i] > 0:
            cv2.ellipse(canvas, (cx, cy), (82, 82), 0, -90, -90 + 360 * ready[i], col, 10, cv2.LINE_AA)
        text(canvas, tag, cx, cy + 14, 1.6, COLORS[i], 3, "c")
        text(canvas, lab.upper(), cx, cy - 100, 0.8, COLORS[i], 2, "c")
        if ready[i] >= 1.0:
            msg, mc = "LISTO!", GREEN
        elif present[i]:
            msg, mc = "Abre la mano...", YELLOW
        else:
            msg, mc = "Esperando mano...", (170, 170, 170)
        text(canvas, msg, cx, cy + 135, 0.9, mc, 2, "c")

    text(canvas, "MEJORES TIEMPOS", W // 2, 520, 0.7, YELLOW, 2, "c")
    if ranking:
        line = "   ".join(f"{k + 1}. {r['name']} {r['time']:.1f}s" for k, r in enumerate(ranking[:3]))
    else:
        line = "Aun no hay tiempos: se el primero!"
    text(canvas, line, W // 2, 550, 0.7, WHITE, 1, "c")

    lc = "SIGUE LAS CURVAS: si no giras te sales de la carretera y explotas"
    if cfg.OBSTACLES:
        lc += "  |  OBSTACULO: frena y quieto 1 s"
    text(canvas, lc, W // 2, H - 125, 0.7, (80, 200, 255), 1, "c")
    text(canvas, l1, W // 2, H - 92, 0.75, WHITE, 1, "c")
    text(canvas, l2, W // 2, H - 62, 0.75, WHITE, 1, "c")
    text(canvas, "ESC salir   F pantalla completa", W // 2, H - 30, 0.6, (190, 190, 190), 1, "c")


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
    if len(players) == 2:
        text(canvas, f"GANA {NAMES[winner].upper()}!", W // 2, 120, 2.0, COLORS[winner], 5, "c")
    else:
        text(canvas, f"META! {NAMES[0].upper()}", W // 2, 120, 2.0, COLORS[0], 5, "c")
    if new_idx == 0:
        text(canvas, "NUEVO RECORD!", W // 2, 172, 1.1, YELLOW, 3, "c")
    for i, p in enumerate(players):
        x = int(LANE_CX[i])
        t = f"{p['finish']:.1f} s" if p["finish"] is not None else \
            f"{int(100 * p['progress'] / TRACK_LENGTH)}%"
        text(canvas, NAMES[i].upper(), x, 250, 1.0, COLORS[i], 2, "c")
        text(canvas, t, x, 312, 1.5, WHITE, 3, "c")
        text(canvas, f"Explosiones: {p['explosions']}", x, 352, 0.7, (220, 220, 220), 1, "c")
        if cfg.NITRO:
            text(canvas, f"Repeticiones de pulgar: {p['thumb_reps']}", x, 384, 0.7, (220, 220, 220), 1, "c")

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
    if len(players) == 1:
        text(canvas, "M: informe de sesion", W // 2, H - 68, 0.9, (80, 200, 255), 2, "c")
    text(canvas, "ENTER: revancha     ESC: salir", W // 2, H - 30, 1.0, YELLOW, 2, "c")

    # Confeti
    for _ in range(3):
        spawn(random.uniform(0, W), -10, random.uniform(-60, 60), random.uniform(120, 300), 4.0,
              random.choice([(0, 255, 255), (255, 80, 80), (80, 255, 80), (255, 255, 255), (255, 0, 255)]),
              5, 120)



def _f(v, suf="", nd=0):
    return "--" if v is None else (f"{v:.{nd}f}{suf}" if nd else f"{round(v)}{suf}")


def draw_report(canvas, rep, paths=None):
    """Informe de la sesion (modos de un jugador). Se abre con la tecla M en la pantalla final."""
    panel(canvas, 0, 0, W, H, (0, 0, 0), 0.94)
    text(canvas, f"INFORME DE SESION - {rep['paciente'].upper()}", W // 2, 52, 1.1, YELLOW, 3, "c")
    y = [100]

    def line(s, col=WHITE, sc=0.62, x=50):
        text(canvas, s, x, y[0], sc, col, 1)
        y[0] += 29

    line(f"Modo {rep['modo']}  |  Tiempo {rep['tiempo_carrera_s']:.1f} s  |  Explosiones {rep['explosiones']} "
         f"(fuera {rep['fuera_de_pista']}, cruce {rep['dedos_cruzados']}, choque {rep['choques_obstaculo']})")
    if cfg.NITRO:
        line(f"Pulgar (oposicion): {rep['pulgar_repeticiones']} repeticiones")
    y[0] += 6
    for lab, m in rep["manos"].items():
        line(f"MANO {lab.upper()}" if lab != "mano" else "MANO", YELLOW, 0.7)
        if m["rom_deg"] is None:
            line("Pocos datos de la mano (detecciones insuficientes)", (190, 190, 190), x=70)
            continue
        r = m["rom_deg"]
        line(f"Rango de movimiento medio {_f(m['rom_medio_deg'], ' deg')}   (indice {_f(r['indice'])}  corazon "
             f"{_f(r['corazon'])}  anular {_f(r['anular'])}  menique {_f(r['menique'])})", x=70)
        line(f"Velocidad apertura/cierre {_f(m['vel_max_deg_s'], ' deg/s')}  |  {_f(m['ciclos_por_s'], ' ciclos/s por dedo', 2)}"
             f"  |  mano detectada {_f(m['deteccion_pct'], ' %')}", x=70)
        t = m["temblor"]
        line("Temblor 4-6 Hz: " + ("pocos datos" if t is None else
             f"{t['rms_pct_palma']:.2f} % de la palma  (ratio {t['ratio_4_6hz']:.2f}, frecuencia dominante {t['f_dominante_hz']} Hz)"), x=70)
    y[0] += 6
    r = rep["reaccion"]
    if r["n_obstaculos"]:
        line(f"REACCION ante {r['n_obstaculos']} obstaculos ({r['choques']} choques): media {_f(r['media_ms'], ' ms')}  "
             f"mediana {_f(r['mediana_ms'], ' ms')}  min {_f(r['min_ms'])}  max {_f(r['max_ms'], ' ms')}", YELLOW, 0.62)
        ev = "  ".join((_f(e["reaccion_ms"]) + ("*" if e["resultado"] == "choque" else "")) for e in r["eventos"][:12])
        line(f"Reacciones (ms, * = choque): {ev}", x=70)
        line(f"Tiempo medio hasta pararte: {_f(r['parada_media_ms'], ' ms')}   |   cambio en la 2a mitad: {_f(r['cambio_2a_mitad_ms'], ' ms')}", x=70)
    else:
        line("REACCION: no hubo obstaculos en esta sesion", YELLOW, 0.62)
    fa = rep.get("fatiga")
    if fa:
        line(f"Fatiga (ultimo tercio vs primero; + = baja): rango {_f(fa['caida_rom_pct'], ' %')}  |  ritmo {_f(fa['caida_ritmo_pct'], ' %')}")
    sim = rep.get("simetria")
    if sim:
        line(f"Simetria entre manos (100 % = igual): rango {_f(sim['rom_pct'], ' %')}  |  actividad {_f(sim['actividad_pct'], ' %')}")
    if paths:
        text(canvas, "Guardado en: " + paths["estado"].replace("\\", "/")[-70:], 50, H - 62, 0.5, (160, 220, 160), 1)
    text(canvas, "M: volver    ENTER: revancha    ESC: salir        Metricas orientativas, no son un diagnostico",
         W // 2, H - 28, 0.6, (190, 190, 190), 1, "c")
