"""Pruebas SIN camara: simula manos sinteticas y comprueba fisica, obstaculos, metricas, informe y bucle principal.
Uso:  python test_sin_camara.py          (guarda capturas en tests_out/)"""
import math
import os
import random
import shutil
import sys
import tempfile
import types

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from config import *
import players as P
import storage
from track import sway
from vision import analyze_hand

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_out")
os.makedirs(OUT, exist_ok=True)
TMP = tempfile.mkdtemp()
storage.PATIENTS_DIR = os.path.join(TMP, "pacientes")
storage.RANK_FILE = os.path.join(TMP, "ranking.json")
fails = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FALLA"), name, extra)
    if not cond:
        fails.append(name)


# ------------------------------------------------------------------ mano sintetica
def synth_pts(cx_px, cy_px, flex, thumb_out=True, jitter=(0.0, 0.0)):
    """21 landmarks normalizados. flex = angulo de flexion del PIP (grados) de cada uno de los 4 dedos."""
    jx, jy = jitter
    wrist = np.array([cx_px + jx, cy_px + 150 + jy])
    pts = [None] * 21
    pts[0] = wrist
    for k, dx in enumerate((-45, -15, 15, 45)):                     # indice, corazon, anular, menique
        base = 5 + 4 * k
        mcp = wrist + np.array([dx * 1.0, -130.0])
        ang1 = math.radians(0.35 * flex[k])
        d1 = np.array([math.sin(-ang1), -math.cos(ang1)])           # segmento 1 (mcp->pip)
        pip = mcp + d1 * 52
        a2 = ang1 + math.radians(flex[k])
        d2 = np.array([math.sin(-a2), -math.cos(a2)])
        dip = pip + d2 * 32
        a3 = a2 + math.radians(0.6 * flex[k])
        d3 = np.array([math.sin(-a3), -math.cos(a3)])
        tip = dip + d3 * 26
        pts[base], pts[base + 1], pts[base + 2], pts[base + 3] = mcp, pip, dip, tip
    pts[1], pts[2], pts[3] = wrist + [-30, -30], wrist + [-60, -70], wrist + [-85, -105]
    pts[4] = wrist + ([-110, -140] if thumb_out else [-5, -60])
    return [(float(p[0]) / W, float(p[1]) / H) for p in pts]


def hand_from_ext(ext, cx, t, ext_state=None, jitter=(0, 0)):
    flex = [8.0 if e else 88.0 for e in ext]
    pts = synth_pts(cx, 300, flex, True, jitter)
    raw, order, crossed, tr = analyze_hand(pts)
    return dict(pts=pts, ext=list(ext), order=order, crossed=False, thumb=False, thumb_ratio=tr, t=t, fresh=True)


# ------------------------------------------------------------------ 1) fisica: nadie gana sin girar
print("1) Anti-autopiloto")
cfg.configure_mode("bimanual", turbo_auto=True, obstacles=False)


def run(mode, policy, obst=False, max_t=600, seed=0):
    random.seed(seed)
    cfg.configure_mode(mode, turbo_auto=True, obstacles=obst)
    p = P.new_player()
    now, t, dt = 1000.0, 0.0, 1 / 60
    while t < max_t:
        now += dt
        t += dt
        if P.update_player(0, p, policy(p, now), dt, now):
            return True, t, p
    return False, t, p


def hands_steer(mode, d):
    top = 4 if mode == "bimanual" else 2
    d = max(-top, min(top, d))
    L, R = (top, top - d) if d >= 0 else (top + d, top)
    mk = lambda n: dict(ext=[i < n for i in range(4)], crossed=False, thumb=False, thumb_ratio=1.0)
    if mode == "bimanual":
        return [mk(L), mk(R)]
    return [dict(ext=[k < L for k in range(2)] + [k < R for k in range(2)], crossed=False, thumb=False, thumb_ratio=1.0)]


for mode in ("bimanual", "unimanual"):
    allopen = lambda p, now, mode=mode: hands_steer(mode, 0)
    ok, t, p = run(mode, allopen, max_t=900)
    check(f"{mode}: todos los dedos abiertos NO llegan a meta en 15 min", not ok,
          f"(progreso {p['progress']:.0f}/{TRACK_LENGTH:.0f}, {p['explosions']} explosiones)")
    top = 4 if mode == "bimanual" else 2
    ok_any = [run(mode, lambda p, now, d=d: hands_steer(mode, d), max_t=300)[0] for d in range(-top, top + 1)]
    check(f"{mode}: ningun giro constante llega a meta", not any(ok_any))


def follower(mode, react=0.5, look=40, gain=22.0):
    st = dict(seen=None)

    def pol(p, now):
        ob = p["obst"]["active"]
        if ob is None:
            st["seen"] = None
        else:
            st["seen"] = st["seen"] or now
            if now - st["seen"] >= react:
                return [dict(ext=[False] * 4, crossed=False, thumb=False, thumb_ratio=1.0)] * (2 if mode == "bimanual" else 1)
        return hands_steer(mode, int(round((float(sway(p["progress"] + look)) - p["x"]) / gain)))
    return pol


for mode in ("bimanual", "unimanual"):
    ok, t, p = run(mode, follower(mode), obst=False, max_t=300)
    check(f"{mode}: seguir las curvas SI llega a meta", ok, f"({t:.0f} s, {p['explosions']} explosiones)")

# ------------------------------------------------------------------ 2) obstaculos / reaccion
print("2) Obstaculos y tiempo de reaccion")
for mode in ("bimanual", "unimanual"):
    for react in (0.3, 0.8):
        ok, t, p = run(mode, follower(mode, react=react), obst=True, max_t=300)
        ev = p["obst"]["events"]
        rts = [e["reaccion_ms"] for e in ev if e["reaccion_ms"] is not None]
        oks = [e for e in ev if e["resultado"] == "ok"]
        check(f"{mode} reaccion simulada {int(react*1000)} ms: llega y mide ~{int(react*1000)} ms",
              ok and len(ev) >= 6 and rts and abs(np.mean(rts) - react * 1000) < 80 and len(oks) == len(ev),
              f"(medido {np.mean(rts):.0f} ms, {len(ev)} obstaculos, {len(oks)} superados)")
        check(f"{mode}: parar tarda >= 1 s quieto", all(e["parada_ms"] is not None for e in oks))
# sin frenar -> choque
ok, t, p = run("unimanual", lambda p, now: hands_steer("unimanual", int(round((float(sway(p["progress"] + 40)) - p["x"]) / 22))),
               obst=True, max_t=120)
check("sin frenar: choca con el obstaculo", "obstacle" in p["reasons"] and
      any(e["resultado"] == "choque" and e["reaccion_ms"] is None for e in p["obst"]["events"]))

# ------------------------------------------------------------------ 3) metricas con manos sinteticas
print("3) Metricas (ROM, temblor, informe)")
from metrics import SessionMetrics
import metrics as M


def stream(tremor_hz=None, amp_palm=0.0, secs=40, fps=30.0, rom_amp=35.0):
    m = SessionMetrics("Test", "unimanual", ["mano"])
    m.start(0.0)
    pl = P.new_player()
    for i in range(int(secs * fps)):
        t = i / fps
        flex = [45 + rom_amp * math.sin(2 * math.pi * 0.5 * t + k * 0.3) for k in range(4)]
        j = amp_palm * 150 * math.sin(2 * math.pi * tremor_hz * t) if tremor_hz else 0.0
        pts = synth_pts(640, 300, flex, True, (j, j * 0.5))
        raw, order, crossed, tr = analyze_hand(pts)
        ext = [f < 40 for f in flex]
        m.sample([dict(pts=pts, ext=ext, t=t, fresh=True)], t)
        m.sample([dict(pts=pts, ext=ext, t=t, fresh=False)], t + 0.016)      # frame repetido: no debe contar
    return m, pl


m, pl = stream()
rep = m.finish(pl, 40.0)
h = rep["manos"]["mano"]
check("ROM medio ~ 2*35*0.95 = 66 deg", 58 < h["rom_medio_deg"] < 72, f"(medido {h['rom_medio_deg']})")
check("no cuenta detecciones repetidas", abs(h["muestras"] - 1200) <= 2, f"({h['muestras']} muestras)")
check("ciclos/s ~ 0.5-1", h["ciclos_por_s"] and 0.3 < h["ciclos_por_s"] < 1.3, f"({h['ciclos_por_s']})")
t0 = h["temblor"]
m2, pl2 = stream(5.0, 0.01)
t5 = m2.finish(pl2, 40.0)["manos"]["mano"]["temblor"]
check("temblor 5 Hz detectado", t5 and t5["f_dominante_hz"] in (4.9, 5.0, 5.1, 5.2) and t5["ratio_4_6hz"] > 0.6 and
      t5["rms_pct_palma"] > 3 * max(t0["rms_pct_palma"], 0.01), f"(sin: {t0}; con: {t5})")
check("fatiga calculada", rep["fatiga"] is not None and rep["fatiga"]["caida_rom_pct"] is not None)

# ------------------------------------------------------------------ 4) bucle principal completo con camara falsa
print("4) Bucle principal completo (camara falsa)")
import race3d


class Clock:
    t = 5000.0

    @classmethod
    def time(cls):
        return cls.t


def run_main(mode, name, react, shots):
    random.seed(1)
    Clock.t = 5000.0
    store = {}
    orig_fresh = race3d.fresh_players

    def fresh():
        store["p"] = orig_fresh()
        return store["p"]
    race3d.fresh_players = fresh
    race3d.time = types.SimpleNamespace(time=Clock.time)
    race3d.ask_names = lambda n=2: [name, name]

    class FakeWorker:
        def __init__(self, cam=0):
            self.lock = __import__("threading").Lock()
            self.running = True
            self.ready = __import__("threading").Event()
            self.failed = False
            self.bg = np.full((H, W, 3), 60, np.uint8)
            self.hands = [None, None]
            self.det_fps = 30.0
            self.pol = follower("bimanual" if mode == "bimanual" else "unimanual", react=react, look=40, gain=22.0)
            self.last = -1

        def start(self):
            self.ready.set()

        def join(self, timeout=None):
            pass

        def update(self):
            if Clock.t - self.last < 1 / 30 - 1e-6:
                return
            self.last = Clock.t
            p = store.get("p", [None])[0]
            if p is None or not store.get("racing"):
                ext = [True] * 4
                hs = [dict(ext=ext)] * 2
            else:
                hs = self.pol(p, Clock.t)
            jx = 0.0
            if mode == "bimanual":
                cxs = [320, 960]
            else:
                cxs = [640, 640]
            out = []
            for k in range(2):
                if mode == "unimanual" and k == 1:
                    out.append(None)
                    continue
                e = hs[min(k, len(hs) - 1)]["ext"]
                if mode == "unimanual":
                    e = [e[0], e[1], e[2], e[3]]
                out.append(hand_from_ext(e, cxs[k], Clock.t))
            with self.lock:
                self.hands = out
    fw = FakeWorker()
    race3d.VisionWorker = lambda cam=0: fw
    race3d.ensure_model = lambda: None
    race3d.load_ranking = storage.load_ranking

    frames = dict(n=0, last=None, phase=[])
    real_show, real_wait = cv2.imshow, cv2.waitKey
    cv2.namedWindow = lambda *a, **k: None
    cv2.resizeWindow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    state = dict(sent_m=False, finished_frames=0)

    def imshow(w, img):
        frames["last"] = img

    def wait(ms):
        Clock.t += 1 / 60
        frames["n"] += 1
        fw.update()
        pl = store.get("p")
        if pl and pl[0]["speed"] > 0 and frames["n"] > 200:
            store["racing"] = True
        ob = pl[0]["obst"]["active"] if pl else None
        if ob is not None and ob["brake_t"] is not None and "obst" not in shots:
            shots["obst"] = frames["last"].copy()
        if frames["n"] == 100 and "wait" not in shots:
            shots["wait"] = frames["last"].copy()
        # al terminar: pantalla final, luego M, luego ESC
        if pl and pl[0]["finish"] is not None:
            state["finished_frames"] += 1
            if state["finished_frames"] == 30:
                shots["fin"] = frames["last"].copy()
                return ord("m")
            if state["finished_frames"] == 60:
                shots["rep"] = frames["last"].copy()
                return 27
        if frames["n"] > 60 * 200:
            return 27
        return -1
    cv2.imshow, cv2.waitKey = imshow, wait
    sys.argv = ["race3d.py", "--modo", mode]
    try:
        race3d.main()
    finally:
        cv2.imshow, cv2.waitKey = real_show, real_wait
    return frames["n"] / 60.0


for mode in ("unimanual", "bimanual"):
    shots = {}
    dur = run_main(mode, "Paciente Prueba", 0.45, shots)
    for k, img in shots.items():
        cv2.imwrite(os.path.join(OUT, f"{mode}_{k}.png"), img)
    folder = os.path.join(storage.PATIENTS_DIR, "paciente_prueba")
    st_path = os.path.join(folder, "estado_actual.json")
    check(f"{mode}: carrera completada y estado_actual.json creado", os.path.exists(st_path), f"({dur:.0f} s simulados)")
    if os.path.exists(st_path):
        import json
        st = json.load(open(st_path, encoding="utf-8"))
        mm = st["metricas_actuales"]
        check(f"{mode}: estado_actual incluye reaccion y ROM",
              mm.get("reaccion_media_ms") and any(k.endswith("rom_medio_deg") and v for k, v in mm.items()),
              f"(reaccion {mm.get('reaccion_media_ms')} ms, sesiones {st['sesiones_totales']})")
        print("      claves:", sorted(mm)[:8], "...")
    check(f"{mode}: capturas final e informe", "fin" in shots and "rep" in shots)

# segunda sesion: deltas
shots = {}
run_main("unimanual", "Paciente Prueba", 0.7, shots)
st = __import__("json").load(open(os.path.join(storage.PATIENTS_DIR, "paciente_prueba", "estado_actual.json"), encoding="utf-8"))
check("2a sesion: cambio vs anterior calculado", bool(st["cambio_vs_sesion_anterior"]) and st["sesiones_totales"] == 3,
      f"(reaccion {st['cambio_vs_sesion_anterior'].get('reaccion_media_ms')} ms, {st['sesiones_totales']} filas)")

shutil.rmtree(TMP, ignore_errors=True)
print("\nRESULTADO:", "TODO OK" if not fails else f"{len(fails)} FALLOS: {fails}")
sys.exit(1 if fails else 0)
