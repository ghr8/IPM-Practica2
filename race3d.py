"""CARRERA TERAPEUTICA CON LAS MANOS (VISTA 3D)  -  IPM Practica 2   (punto de entrada)
----------------------------------------------------------------------------------
Tres modos (se elige al arrancar o con --modo):
  bimanual   1 jugador, DOS manos: la izquierda mueve las ruedas izquierdas y la derecha las derechas.
  unimanual  1 jugador, UNA mano (la que quieras rehabilitar): los 4 dedos son las 4 ruedas.
  duelo      2 jugadores, una mano cada uno (pantalla dividida).
Dedos extendidos (verde) = ruedas que empujan. Mas ruedas a la izquierda que a la derecha = giras a la derecha.
Hay que SEGUIR LAS CURVAS: la carretera se desvia mas que su anchura, asi que ir recto te saca y explotas
(y vuelves al punto de control anterior).
Obstaculos (un jugador): aparecen de golpe; frena (baja los dedos) y quedate quieto 1 s. Se mide tu reaccion.
Pulgar metido en la palma = NITRO (bimanual/unimanual; las pastillas amarillas cargan el nitro).
Teclas: ESC salir | ENTER revancha | R reiniciar | F pantalla completa | M informe (al acabar)

Modulos:  config (ajustes)  vision (camara/mano)  players (fisica)  obstacles (reaccion)  track (carretera)
          metrics (informe)  storage (ranking y pacientes)  render (escena)  screens (HUD y pantallas)
"""
import math
import random
import time
import argparse

import numpy as np
import cv2

import config as cfg
from config import *
from effects import PARTS, SHAKE, update_particles, draw_particles
from players import fresh_players, update_player
from render import draw_scene, text
from screens import draw_hud, draw_waiting, draw_countdown, draw_finished, draw_report
from resumen import draw_summary
from storage import load_ranking, add_record, save_session, history
import metrics as metrics_mod
import resumen
from menu import choose_mode, ask_names_window, toggle_fullscreen
from metrics import SessionMetrics
from vision import VisionWorker, ensure_model
from track import sway


def parse_args():
    ap = argparse.ArgumentParser(description="Carrera terapeutica con las manos (IPM Practica 2)")
    ap.add_argument("--modo", choices=MODES, help="bimanual | unimanual | duelo (si no, se pregunta)")
    ap.add_argument("--camara", type=int, default=0, help="indice de la camara (por defecto 0)")
    ap.add_argument("--turbo-auto", action="store_true",
                    help="las pastillas dan turbo directo (sin usar el pulgar) en bimanual/unimanual")
    ap.add_argument("--sin-obstaculos", action="store_true",
                    help="desactiva los obstaculos (y por tanto la medida del tiempo de reaccion)")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_model()
    worker = VisionWorker(args.camara)
    worker.start()
    print("Arrancando camara y modelo ...")
    worker.ready.wait(timeout=20)
    if worker.failed or not worker.ready.is_set():
        print("No se pudo abrir la camara (o el modelo ha tardado demasiado).")
        worker.running = False
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    mode = args.modo or choose_mode(worker)      # menu interactivo con el dedo
    if mode is None:                             # ESC en el menu
        worker.running = False
        worker.join(timeout=2)
        cv2.destroyAllWindows()
        return
    cfg.configure_mode(mode, args.turbo_auto, not args.sin_obstaculos)
    names = ask_names_window(worker, cfg.NP)     # nombre escrito en la propia ventana
    if names is None:                            # ESC
        worker.running = False
        worker.join(timeout=2)
        cv2.destroyAllWindows()
        return
    NAMES[:] = names
    ranking = load_ranking(cfg.MODE)
    new_idx = -1

    state = WAITING
    players = fresh_players()
    ready = [0.0, 0.0]
    countdown_start = race_start = 0.0
    lost = 0.0
    winner = None
    metrics = report = paths = summary = None
    show_report = show_detail = False
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

        present = hands_present(hands)

        # ---------- Logica ----------
        if state == WAITING:
            for p in players:   # la carretera avanza sola en la pantalla de espera
                p["progress"] = (now * 160) % 4000
                p["x"] = float(sway(p["progress"]))
                p["speed"] = MAX_SPEED * 0.5
            op = ready_inputs(hands)
            for i in range(len(op)):
                if op[i]:
                    ready[i] = min(1.0, ready[i] + dt / READY_SECONDS)
                else:
                    ready[i] = max(0.0, ready[i] - dt * 2.0)
            if all(r >= 1.0 for r in ready[:len(op)]):
                state = COUNTDOWN
                countdown_start = now
                lost = 0.0
                players = fresh_players()
                PARTS.clear()

        elif state == COUNTDOWN:
            lost = 0.0 if all(present) else lost + dt
            if lost > 0.5:
                state = WAITING
                ready = [0.0, 0.0]
            elif now - countdown_start >= COUNTDOWN_SECONDS:
                state = RACING
                race_start = now
                if cfg.NP == 1:
                    metrics = SessionMetrics(NAMES[0], cfg.MODE, hand_labels())
                    metrics.start(now)
                    report = paths = summary = None
                    show_report = show_detail = False

        elif state == RACING:
            if metrics is not None:
                metrics.sample(hands_for(0, hands), now)
            for i, p in enumerate(players):
                if update_player(i, p, hands_for(i, hands), dt, now):
                    p["finish"] = now - race_start
                    winner = i
                    state = FINISHED
                    ranking, new_idx = add_record(NAMES[i], p["finish"], cfg.MODE)
                    if metrics is not None:             # un jugador: informe de la sesion + estado del paciente
                        try:
                            report = metrics.finish(p, p["finish"])
                            prev_rows = history(NAMES[0], cfg.MODE)      # antes de guardar la de hoy
                            paths = save_session(report)
                            summary = resumen.build(metrics_mod.flat(report), prev_rows)
                            print("Informe guardado en", paths["estado"])
                        except Exception as e:           # nunca romper el final de la carrera por un fallo al guardar
                            print("No se pudo guardar el informe:", e)
                    break

        elif state == FINISHED:
            for p in players:
                p["ext"] = [False] * 4
                p["fingers"] = [[False] * 4 for _ in p["fingers"]]
                p["thumbs"] = [(False, None) for _ in p["thumbs"]]

        update_particles(dt)
        SHAKE[0] *= math.exp(-9 * dt)

        # ---------- Dibujo ----------
        canvas = bg.copy()
        draw_scene(canvas, players, hands, now, show_cars=not (state == WAITING and cfg.NP == 1))
        race_t = (now - race_start) if state == RACING else (
            players[winner]["finish"] if state == FINISHED and winner is not None else 0.0)
        if state in (RACING, FINISHED):
            draw_hud(canvas, players, now, race_t, state == RACING)

        if state == WAITING:
            draw_waiting(canvas, ready, present, now, ranking)
        elif state == COUNTDOWN:
            draw_countdown(canvas, now - countdown_start)
        elif state == RACING and now - race_start < 0.9:
            text(canvas, "YA!", W // 2, int(H * 0.45), 6 + 4 * (now - race_start), GREEN, 14, "c")
        elif state == FINISHED:
            if show_detail and report is not None:
                draw_report(canvas, report, paths)
            elif show_report and summary is not None:
                draw_summary(canvas, NAMES[0], summary)
            else:
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
        if key in (ord("m"), ord("M")) and state == FINISHED and summary is not None:
            show_report, show_detail = not (show_report or show_detail), False
        if key in (ord("d"), ord("D")) and state == FINISHED and report is not None and (show_report or show_detail):
            show_detail = not show_detail
            show_report = not show_detail
        if (key == 13 and state == FINISHED) or key in (ord("r"), ord("R")):
            players = fresh_players()
            ready = [0.0, 0.0]
            winner = None
            metrics = report = paths = summary = None
            show_report = show_detail = False
            PARTS.clear()
            state = WAITING
        if key in (ord("f"), ord("F")):
            toggle_fullscreen()

    worker.running = False
    worker.join(timeout=2)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
