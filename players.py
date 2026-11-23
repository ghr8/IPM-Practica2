"""Jugadores: estado, fisica del coche, explosiones y puntos de control."""
import math
import random

import config as cfg
from config import *
from effects import spawn, RINGS, SHAKE
from track import PADS, sway, checkpoint_for
import obstacles


def new_player():
    ng = 2 if cfg.MODE == "bimanual" else 1      # grupos de 4 dedos (una mano = un grupo)
    return dict(x=float(sway(0.0)), vel=0.0, speed=0.0, progress=0.0, explosions=0,
                exploded_until=0.0, immune_until=0.0, boost_until=0.0, flash_until=0.0,
                used=set(), finish=None, ext=[False] * 4, crossed=False,
                off_time=0.0, reason="",
                fingers=[[False] * 4 for _ in range(ng)], thumbs=[(False, None)] * ng,
                nitro=0, thumb_prev=False, thumb_last=0.0, thumb_reps=0,
                obst=obstacles.new_state(), reasons=[], off_acc=0.0)


def fresh_players():
    return [new_player() for _ in range(cfg.NP)]


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
    p["off_acc"] = 0.0
    p["reason"] = reason
    p["reasons"].append(reason)
    if reason == "obstacle":                       # choque: solo retrocedes un poco
        p["progress"] = max(0.0, p["progress"] - EXPLOSION_PENALTY)
    else:                                          # fuera de pista / cruce: vuelves al punto de control anterior
        p["progress"] = checkpoint_for(p["progress"])
        obstacles.abort(p["obst"])
    p["x"] = float(sway(p["progress"]))


def update_player(i, p, hs, dt, now):
    """hs = manos que controlan a este jugador (1 o 2, pueden ser None). True si llega a la meta."""
    if cfg.MODE == "bimanual":
        fl = list(hs[0]["ext"]) if hs[0] else [False] * 4
        fr = list(hs[1]["ext"]) if hs[1] else [False] * 4
        p["fingers"] = [fl, fr]
        # 8 dedos -> 4 ruedas (cada rueda se ilumina si alguno de sus 2 dedos esta extendido)
        p["ext"] = [fl[0] or fl[1], fl[2] or fl[3], fr[0] or fr[1], fr[2] or fr[3]]
        L, R, n = sum(fl), sum(fr), 8
    else:
        ext = list(hs[0]["ext"]) if hs[0] else [False] * 4
        p["fingers"] = [ext]
        p["ext"] = ext
        L, R, n = sum(ext[:2]), sum(ext[2:]), 4
    p["crossed"] = any(h is not None and h["crossed"] for h in hs)
    p["thumbs"] = [(bool(h and h.get("thumb")), h.get("thumb_ratio") if h else None) for h in hs]
    thumb_now = cfg.NITRO and any(t for t, _ in p["thumbs"])

    cx = LANE_CX[i] + p["x"]

    if now < p["exploded_until"]:
        p["thumb_prev"] = thumb_now
        if random.random() < 0.5:   # humo del coche destrozado
            spawn(cx + random.uniform(-14, 14), CAR_Y - 55 + random.uniform(-10, 10),
                  random.uniform(-25, 25), random.uniform(-90, -30), 0.8, (90, 90, 90), 8)
        return False

    # Pulgar: al METERLO en la palma (flanco) se cuenta una repeticion y se gasta una carga de nitro
    if thumb_now and not p["thumb_prev"] and now - p["thumb_last"] > 0.35:
        p["thumb_last"] = now
        p["thumb_reps"] += 1
        if p["nitro"] > 0 and now >= p["boost_until"]:
            p["nitro"] -= 1
            p["boost_until"] = now + BOOST_SECONDS
            for _ in range(25):
                spawn(cx, CAR_Y - 50, random.uniform(-220, 220), random.uniform(-220, 100), 0.5,
                      YELLOW, 4)
    p["thumb_prev"] = thumb_now

    ob = p["obst"]["active"]
    if ob is not None and (L + R) <= OBST_BRAKE_FRAC * n:   # frenar ante un obstaculo cancela el turbo
        p["boost_until"] = 0.0

    target = MAX_SPEED * (L + R) / n
    boosting = now < p["boost_until"]
    if boosting:
        target = max(target, MAX_SPEED * 0.5) * 1.5

    road_c = float(sway(p["progress"]))
    off_road = abs(p["x"] - road_c) > ROAD_W / 2 + 12
    if off_road:
        target *= 0.5
        p["off_time"] += dt
        p["off_acc"] += dt                         # tiempo fuera acumulado (para el aviso se usa off_time)
    else:
        p["off_time"] = 0.0
        p["off_acc"] = max(0.0, p["off_acc"] - OFF_ROAD_LEAK * dt)   # se "perdona" poco a poco, no de golpe

    rate = 2.2 if target > p["speed"] else 4.0
    p["speed"] += (target - p["speed"]) * min(1.0, rate * dt)

    steer_target = (L - R) * STEER_RATE * (4.0 / n) * (0.35 + 0.65 * min(1.0, p["speed"] / MAX_SPEED))
    p["vel"] += (steer_target - p["vel"]) * min(1.0, 9 * dt)
    p["x"] += p["vel"] * dt
    if abs(p["x"]) > CLAMP_X:
        p["x"] = math.copysign(CLAMP_X, p["x"])
        p["vel"] = 0.0
    p["progress"] += p["speed"] * dt

    # Obstaculos (solo un jugador): hay que frenar y quedarse quieto 1 s
    if cfg.OBSTACLES and obstacles.update(p["obst"], p, (L + R) / n, now) == "hit":
        explode(i, p, now, "obstacle")
        return False

    # Pastillas amarillas: en duelo dan turbo directo; en los demas modos cargan el nitro
    for idx, (pd, off) in enumerate(PADS):
        if idx not in p["used"] and abs(pd - p["progress"]) < 60 and \
                abs(p["x"] - (float(sway(pd)) + off)) < 60:
            p["used"].add(idx)
            if cfg.NITRO:
                p["nitro"] = min(NITRO_MAX, p["nitro"] + 1)
            else:
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
        if p["off_acc"] > OFF_ROAD_GRACE:
            explode(i, p, now, "off")
            return False
        if p["crossed"]:
            explode(i, p, now, "cross")
            return False

    return p["progress"] >= TRACK_LENGTH

