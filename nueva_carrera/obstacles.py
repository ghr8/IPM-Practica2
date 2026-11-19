"""Obstaculos y tiempo de reaccion.

Un obstaculo aparece de golpe en la carretera (OBST_DIST por delante). El jugador debe FRENAR (bajar dedos) y
quedarse quieto OBST_HOLD segundos; entonces el obstaculo desaparece. Si llega hasta el, choca (BOOM).

Tiempo de reaccion = desde que aparece hasta el primer instante en que frena (dedos extendidos <= OBST_BRAKE_FRAC).
Cada obstaculo deja un evento en st["events"] que luego recoge metrics.py.
"""
from config import *
from track import OBST_TRIGGERS


def new_state():
    return dict(active=None, done=set(), events=[], min_progress=0.0)


def abort(st):
    """El jugador exploto por otra causa: el obstaculo se cancela y se volvera a lanzar mas adelante."""
    st["active"] = None


def _try_spawn(st, p, now):
    if p["speed"] < OBST_MIN_SPEED * MAX_SPEED:
        return
    for idx, trig in enumerate(OBST_TRIGGERS):
        if idx in st["done"]:
            continue
        if p["progress"] >= max(trig, st["min_progress"]):
            st["active"] = dict(idx=idx, pos=p["progress"] + OBST_DIST, t0=now, speed0=p["speed"] / MAX_SPEED,
                                brake_t=None, brake_dist=None, stop_t=None, still_since=None)
        return                      # solo mira el primer obstaculo pendiente


def _finish(st, ob, result, p):
    st["events"].append(dict(
        n=len(st["events"]) + 1, t_inicio=ob["t0"], resultado=result,
        reaccion_ms=None if ob["brake_t"] is None else round((ob["brake_t"] - ob["t0"]) * 1000),
        parada_ms=None if ob["stop_t"] is None else round((ob["stop_t"] - ob["t0"]) * 1000),
        vel_inicial_pct=round(ob["speed0"] * 100),
        dist_al_frenar_px=None if ob["brake_dist"] is None else round(ob["brake_dist"])))
    st["done"].add(ob["idx"])
    st["min_progress"] = p["progress"] + 700.0      # separa los obstaculos entre si
    st["active"] = None


def update(st, p, frac, now):
    """frac = fraccion de dedos extendidos (0-1). Devuelve 'hit' si choca, 'clear' si lo supera, o None."""
    ob = st["active"]
    if ob is None:
        _try_spawn(st, p, now)
        return None
    if ob["brake_t"] is None and frac <= OBST_BRAKE_FRAC:
        ob["brake_t"] = now
        ob["brake_dist"] = ob["pos"] - p["progress"]
    quieto = ob["brake_t"] is not None and p["speed"] <= OBST_STOP_FRAC * MAX_SPEED
    if quieto:
        if ob["still_since"] is None:
            ob["still_since"] = now
            if ob["stop_t"] is None:
                ob["stop_t"] = now
        elif now - ob["still_since"] >= OBST_HOLD:
            _finish(st, ob, "ok", p)
            return "clear"
    else:
        ob["still_since"] = None
    if ob["pos"] - p["progress"] <= OBST_HIT:
        _finish(st, ob, "choque", p)
        return "hit"
    return None
