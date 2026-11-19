"""Almacenamiento: ranking de tiempos, entrada por consola y carpeta de pacientes (informes, historial, estado actual)."""
import csv
import json
import os
import re
import time
import unicodedata

from config import *
import metrics

RANK_FILE = os.path.join(BASE_DIR, "ranking.json")
PATIENTS_DIR = os.path.join(BASE_DIR, "pacientes")

def clean_name(s, default):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip()
    return (s or default)[:12]


def ask_mode():
    print("Modo de juego:")
    print("  1) bimanual  - 1 jugador, DOS manos (izquierda = ruedas izquierdas, derecha = derechas)")
    print("  2) unimanual - 1 jugador, UNA mano (para rehabilitar una sola mano)")
    print("  3) duelo     - 2 jugadores, una mano cada uno")
    try:
        s = input("Elige 1-3 (Enter = 1): ").strip()
    except EOFError:
        s = ""
    return MODES[int(s) - 1] if s in ("1", "2", "3") else MODES[0]


def ask_names(n=2):
    names = []
    for i in range(n):
        label = f"del jugador {i + 1}" if n > 1 else "del jugador"
        try:
            nm = input(f"Nombre {label} (Enter = Jugador {i + 1}): ")
        except EOFError:
            nm = ""
        names.append(clean_name(nm, f"Jugador {i + 1}"))
    while len(names) < 2:
        names.append(names[0])
    return names


def _load_all():
    try:
        with open(RANK_FILE, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return [r for r in rows if isinstance(r, dict) and "time" in r and "name" in r]
    except Exception:
        return []


def load_ranking(mode=None):
    rows = sorted(_load_all(), key=lambda r: r["time"])
    if mode is not None:
        rows = [r for r in rows if r.get("mode", "duelo") == mode]
    return rows


def add_record(name, t, mode="duelo"):
    """Guarda el tiempo (top 10 por modo) y devuelve (ranking del modo, posicion del nuevo tiempo)."""
    rows = _load_all()
    new = {"name": name, "time": round(float(t), 2), "date": time.strftime("%d/%m/%Y"), "mode": mode}
    rows.append(new)
    rows.sort(key=lambda r: r["time"])
    kept, cnt = [], {}
    for r in rows:
        m = r.get("mode", "duelo")
        cnt[m] = cnt.get(m, 0) + 1
        if cnt[m] <= 10:
            kept.append(r)
    try:
        with open(RANK_FILE, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    mine = [r for r in kept if r.get("mode", "duelo") == mode]
    idx = next((k for k, r in enumerate(mine) if r is new), -1)
    return mine, idx



# ------------------------------------------------------------------
# Pacientes: pacientes/<nombre>/{sesiones/*.json, historial.csv, estado_actual.json}
# ------------------------------------------------------------------
def slug(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "paciente"


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _last_row(csv_path, mode):
    """Ultima sesion guardada del mismo modo (None si no hay)."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("modo") == mode]
    except Exception:
        return None
    return rows[-1] if rows else None


def save_session(rep):
    """Guarda el informe y actualiza el estado actual del paciente. Devuelve las rutas escritas (dict)."""
    folder = os.path.join(PATIENTS_DIR, slug(rep["paciente"]))
    os.makedirs(os.path.join(folder, "sesiones"), exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    p_json = os.path.join(folder, "sesiones", f"{stamp}_{rep['modo']}.json")
    p_csv = os.path.join(folder, "historial.csv")
    p_state = os.path.join(folder, "estado_actual.json")

    row = metrics.flat(rep)
    prev = _last_row(p_csv, rep["modo"])         # antes de anadir la fila nueva

    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    new_file = not os.path.exists(p_csv)
    with open(p_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics.COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    with open(p_csv, newline="", encoding="utf-8") as f:
        n_sessions = max(0, sum(1 for _ in f) - 1)

    deltas = {}
    if prev:
        for k, v in row.items():
            a, b = _to_num(v), _to_num(prev.get(k))
            if k not in ("fecha", "paciente", "modo") and a is not None and b is not None:
                deltas[k] = round(a - b, 2)
    r = rep["reaccion"]
    state = dict(
        paciente=rep["paciente"], actualizado=rep["fecha"], sesiones_totales=n_sessions, modo=rep["modo"],
        metricas_actuales={k: v for k, v in row.items() if v is not None and k not in ("paciente",)},
        reaccion={k: v for k, v in r.items() if k != "eventos"},
        fatiga=rep.get("fatiga"), simetria=rep.get("simetria"),
        cambio_vs_sesion_anterior=deltas or None,
        comparado_con=prev["fecha"] if prev else None,
        aviso=rep["aviso"])
    with open(p_state, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    return dict(informe=p_json, historial=p_csv, estado=p_state)

