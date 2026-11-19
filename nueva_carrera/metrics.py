"""Metricas de la sesion (modos de un jugador). Todas son ORIENTATIVAS, no un diagnostico.

Durante la carrera SessionMetrics.sample() guarda, por cada mano, los angulos de los dedos, la posicion de la palma
y el estado de los dedos (solo con detecciones nuevas, no repetidas). Al llegar a meta finish() calcula:
  - rango de movimiento (ROM) por dedo: angulo de flexion del PIP, percentil 95 - percentil 5
  - velocidad de apertura/cierre (percentil 90 de la velocidad angular) y ciclos de dedo por segundo
  - temblor: amplitud de la oscilacion de la palma en la banda 4-6 Hz
  - fatiga: caida del ROM y del ritmo entre el primer y el ultimo tercio de la carrera
  - simetria entre manos (bimanual)
  - tiempo de reaccion ante los obstaculos (ver obstacles.py)
"""
import math
import time

import numpy as np

from config import W, H
from vision import finger_angles, dpx

FINGER_NAMES = ["indice", "corazon", "anular", "menique"]
AVISO = ("Metricas orientativas obtenidas con una camara web; no son un diagnostico. "
         "Sirven para comparar sesiones del mismo paciente en las mismas condiciones.")


class SessionMetrics:
    def __init__(self, patient, mode, labels):
        self.patient, self.mode, self.labels = patient, mode, list(labels)
        self.g = [dict(t=[], ang=[], pos=[], ext=[], frames=0, seen=0, last_t=-1.0) for _ in self.labels]
        self.t0 = None

    def start(self, now):
        self.t0 = now

    def sample(self, hs, now):
        """hs = manos del jugador (una por grupo, pueden ser None). Llamar una vez por frame durante la carrera."""
        for g, h in zip(self.g, hs):
            g["frames"] += 1
            if h is None:
                continue
            g["seen"] += 1
            t = h.get("t", now)
            if not h.get("fresh", True) or t <= g["last_t"]:
                continue                         # misma deteccion que el frame anterior
            g["last_t"] = t
            pts = h["pts"]
            palm = max(dpx(pts[0], pts[9]), 1e-6)
            g["t"].append(t)
            g["ang"].append(finger_angles(pts))
            g["pos"].append((pts[9][0] * W / palm, pts[9][1] * H / palm))   # palma, en "largos de palma"
            g["ext"].append(tuple(bool(e) for e in h["ext"]))

    # ------------------------------------------------------------------
    def finish(self, player, race_time):
        """Devuelve el informe completo (dict listo para guardar como JSON)."""
        t_end = max([g["t"][-1] for g in self.g if g["t"]] + [self.t0 or 0.0])
        manos = {lab: _hand_report(g) for lab, g in zip(self.labels, self.g)}
        reasons = player["reasons"]
        rep = dict(
            paciente=self.patient, fecha=time.strftime("%Y-%m-%d %H:%M:%S"), modo=self.mode,
            tiempo_carrera_s=round(race_time, 2),
            explosiones=len(reasons), fuera_de_pista=reasons.count("off"),
            dedos_cruzados=reasons.count("cross"), choques_obstaculo=reasons.count("obstacle"),
            pulgar_repeticiones=player["thumb_reps"],
            manos=manos,
            reaccion=_reaction(player["obst"]["events"], self.t0 or 0.0),
            fatiga=_fatigue(self.g, self.t0 or 0.0, t_end),
            aviso=AVISO)
        if len(self.g) == 2:
            rep["simetria"] = _symmetry(manos, self.g)
        return rep


# ------------------------------------------------------------------
def _smooth(ang, k=3):
    ker = np.ones(k) / k
    return np.vstack([np.convolve(ang[:, j], ker, mode="same") for j in range(ang.shape[1])]).T


def _rom(sm):
    return np.percentile(sm, 95, axis=0) - np.percentile(sm, 5, axis=0)


def _hand_report(g):
    n = len(g["t"])
    rep = dict(muestras=n, deteccion_pct=round(100.0 * g["seen"] / max(1, g["frames"]), 1),
               rom_deg=None, rom_medio_deg=None, vel_max_deg_s=None, ciclos_por_s=None, temblor=None)
    if n < 15:
        return rep
    t = np.array(g["t"])
    sm = _smooth(np.array(g["ang"]))
    rom = _rom(sm)
    rep["rom_deg"] = {name: round(float(v), 1) for name, v in zip(FINGER_NAMES, rom)}
    rep["rom_medio_deg"] = round(float(rom.mean()), 1)

    dt = np.diff(t)
    vel = np.abs(np.diff(sm, axis=0)) / np.maximum(dt, 1e-6)[:, None]
    vel = vel[(dt > 0.005) & (dt < 0.2)]
    if vel.size:
        rep["vel_max_deg_s"] = round(float(np.percentile(vel, 90)), 1)

    ext = np.array(g["ext"], int)
    cambios = np.abs(np.diff(ext, axis=0)).sum()
    rep["ciclos_por_s"] = round(float(cambios / 2 / 4 / max(t[-1] - t[0], 1e-6)), 2)   # ciclos abrir-cerrar por dedo
    rep["temblor"] = _tremor(t, np.array(g["pos"]))
    return rep


def _tremor(t, pos, fs=30.0):
    """Amplitud RMS (en % del largo de la palma) de la oscilacion de la palma entre 4 y 6 Hz."""
    if len(t) < 60 or t[-1] - t[0] < 3.0 or 1.0 / np.median(np.diff(t)) < 12.0:
        return None                      # muy pocas muestras / camara demasiado lenta para ver 4-6 Hz
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    N = len(grid)
    w = np.hanning(N)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    P = np.zeros(len(freqs))
    for k in range(2):
        s = np.interp(grid, t, pos[:, k])
        s = s - np.polyval(np.polyfit(grid, s, 1), grid)      # quita la deriva lenta
        P += np.abs(np.fft.rfft(s * w)) ** 2

    def band(lo, hi):
        return float(P[(freqs >= lo) & (freqs <= hi)].sum())

    rms = math.sqrt(2 * band(4, 6)) / (N * math.sqrt(float(np.mean(w ** 2))))
    total = band(1, 12)
    sel = (freqs >= 3) & (freqs <= 12)
    return dict(rms_pct_palma=round(rms * 100, 3),
                ratio_4_6hz=round(band(4, 6) / total, 3) if total > 0 else 0.0,
                f_dominante_hz=round(float(freqs[sel][np.argmax(P[sel])]), 1))


def _reaction(events, t0):
    rts = [e["reaccion_ms"] for e in events if e["reaccion_ms"] is not None]
    stops = [e["parada_ms"] for e in events if e["parada_ms"] is not None]
    rep = dict(n_obstaculos=len(events), choques=sum(e["resultado"] == "choque" for e in events),
               media_ms=None, mediana_ms=None, min_ms=None, max_ms=None, parada_media_ms=None,
               cambio_2a_mitad_ms=None, eventos=[])
    if rts:
        rep.update(media_ms=round(float(np.mean(rts))), mediana_ms=round(float(np.median(rts))),
                   min_ms=min(rts), max_ms=max(rts))
    if stops:
        rep["parada_media_ms"] = round(float(np.mean(stops)))
    if len(rts) >= 4:                    # si la reaccion empeora al final de la sesion (fatiga/atencion)
        h = len(rts) // 2
        rep["cambio_2a_mitad_ms"] = round(float(np.mean(rts[h:]) - np.mean(rts[:h])))
    for e in events:
        ev = {k: v for k, v in e.items() if k != "t_inicio"}
        ev["t_inicio_s"] = round(e["t_inicio"] - t0, 2)
        rep["eventos"].append(ev)
    return rep


def _fatigue(groups, t_start, t_end):
    """Compara el primer y el ultimo tercio de la carrera (ROM y ritmo de apertura/cierre)."""
    if t_end - t_start < 9:
        return None
    edges = np.linspace(t_start, t_end, 4)
    rom_t, rate_t = [], []
    for k in range(3):
        roms, cnt = [], 0.0
        for g in groups:
            t = np.array(g["t"])
            if len(t) < 15:
                continue
            m = (t >= edges[k]) & (t < edges[k + 1] + 1e-9)
            if m.sum() >= 8:
                roms.append(float(_rom(_smooth(np.array(g["ang"])[m])).mean()))
            ext = np.array(g["ext"], int)
            mm = m[1:]
            cnt += float(np.abs(np.diff(ext, axis=0)).sum(axis=1)[mm].sum()) / 2 / 4
        rom_t.append(float(np.mean(roms)) if roms else None)
        rate_t.append(cnt / ((edges[k + 1] - edges[k]) * len(groups)))

    def drop(v):
        return None if v[0] is None or v[2] is None or v[0] <= 0 else round(100.0 * (v[0] - v[2]) / v[0], 1)

    return dict(rom_por_tercio_deg=[None if v is None else round(v, 1) for v in rom_t],
                ritmo_por_tercio=[round(v, 2) for v in rate_t],
                caida_rom_pct=drop(rom_t), caida_ritmo_pct=drop(rate_t))


def _symmetry(manos, groups):
    """100 % = las dos manos rinden igual. Solo modo bimanual."""
    out = dict(rom_pct=None, actividad_pct=None)
    r = [m["rom_medio_deg"] for m in manos.values()]
    if all(v for v in r):
        out["rom_pct"] = round(100.0 * min(r) / max(r), 1)
    act = [float(np.mean(np.array(g["ext"], float))) for g in groups if g["ext"]]
    if len(act) == 2 and max(act) > 0:
        out["actividad_pct"] = round(100.0 * min(act) / max(act), 1)
    return out


# ------------------------------------------------------------------
# Version "plana" (una fila por sesion) para el CSV y para comparar con la sesion anterior
# ------------------------------------------------------------------
PREFIX = {"izquierda": "izq", "derecha": "der", "mano": "mano"}
HAND_KEYS = [("rom_medio_deg", None), ("rom_indice", "indice"), ("rom_corazon", "corazon"),
             ("rom_anular", "anular"), ("rom_menique", "menique"), ("vel_max_deg_s", None),
             ("ciclos_por_s", None), ("temblor_rms_pct", None), ("temblor_ratio_4_6hz", None),
             ("temblor_f_hz", None), ("deteccion_pct", None)]
GENERAL = ["tiempo_carrera_s", "explosiones", "fuera_de_pista", "dedos_cruzados", "choques_obstaculo",
           "pulgar_repeticiones", "reaccion_n", "reaccion_media_ms", "reaccion_mediana_ms", "reaccion_min_ms",
           "reaccion_max_ms", "parada_media_ms", "reaccion_cambio_2a_mitad_ms", "fatiga_caida_rom_pct",
           "fatiga_caida_ritmo_pct", "simetria_rom_pct", "simetria_actividad_pct"]
COLUMNS = ["fecha", "paciente", "modo"] + GENERAL + [f"{p}_{k}" for p in PREFIX.values() for k, _ in HAND_KEYS]


def flat(rep):
    f = dict(fecha=rep["fecha"], paciente=rep["paciente"], modo=rep["modo"])
    for k in ("tiempo_carrera_s", "explosiones", "fuera_de_pista", "dedos_cruzados", "choques_obstaculo",
              "pulgar_repeticiones"):
        f[k] = rep[k]
    r = rep["reaccion"]
    f.update(reaccion_n=r["n_obstaculos"], reaccion_media_ms=r["media_ms"], reaccion_mediana_ms=r["mediana_ms"],
             reaccion_min_ms=r["min_ms"], reaccion_max_ms=r["max_ms"], parada_media_ms=r["parada_media_ms"],
             reaccion_cambio_2a_mitad_ms=r["cambio_2a_mitad_ms"])
    fa = rep.get("fatiga") or {}
    f.update(fatiga_caida_rom_pct=fa.get("caida_rom_pct"), fatiga_caida_ritmo_pct=fa.get("caida_ritmo_pct"))
    si = rep.get("simetria") or {}
    f.update(simetria_rom_pct=si.get("rom_pct"), simetria_actividad_pct=si.get("actividad_pct"))
    for lab, m in rep["manos"].items():
        p = PREFIX[lab]
        rom = m["rom_deg"] or {}
        tr = m["temblor"] or {}
        f.update({f"{p}_rom_medio_deg": m["rom_medio_deg"], f"{p}_vel_max_deg_s": m["vel_max_deg_s"],
                  f"{p}_ciclos_por_s": m["ciclos_por_s"], f"{p}_deteccion_pct": m["deteccion_pct"],
                  f"{p}_temblor_rms_pct": tr.get("rms_pct_palma"), f"{p}_temblor_ratio_4_6hz": tr.get("ratio_4_6hz"),
                  f"{p}_temblor_f_hz": tr.get("f_dominante_hz")})
        for name in FINGER_NAMES:
            f[f"{p}_rom_{name}"] = rom.get(name)
    return f
