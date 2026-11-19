"""Descarga el modelo de manos de MediaPipe en models/ (solo si no existe)."""
import os
import sys

import requests

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


def main():
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
        print("Modelo ya presente:", DEST)
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print("Descargando hand_landmarker.task ...")
    try:
        r = requests.get(URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print("ERROR al descargar el modelo:", e)
        print("Descargalo manualmente desde:\n ", URL, "\ny guardalo en:\n ", DEST)
        return 1
    with open(DEST, "wb") as f:
        f.write(r.content)
    print("Modelo guardado en", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
