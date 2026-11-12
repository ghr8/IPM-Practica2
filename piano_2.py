# ------------------------------------------------------------
# PIANO VIRTUAL CON LAS MANOS (paso 2)
#
# 1) Colocar el piano: con el dedo indice apuntas a la ESQUINA SUPERIOR
#    IZQUIERDA de donde quieras el piano (por ejemplo en la mesa) y lo
#    mantienes quieto 2 segundos. Luego igual con la ESQUINA INFERIOR DERECHA.
#    (Si no quieres colocarlo, pulsa la tecla D y sale un piano por defecto.)
# 2) Tocar: el piano tiene 2 octavas (con teclas negras). Una tecla suena
#    cuando una punta de dedo esta encima de ella y cruza hacia abajo la
#    linea naranja (como si tocaras la mesa).
#
# Teclas: R = colocar el piano otra vez, D = piano por defecto, ESC = salir.
# ------------------------------------------------------------
import os
import time
import urllib.request
import numpy as np
import cv2
import pygame
import mediapipe as mp

# --- Modelo de manos (si no lo tenemos lo descargamos) ---
CARPETA = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CARPETA, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- Configuracion de la tarea (como en app.py pero con manos) ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2  # podemos tocar con las dos manos
)

# --- Notas ---
# En una octava hay 12 notas (7 teclas blancas y 5 negras). Con 2 octavas son 24 notas.
NOMBRES = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]
# Cada nota es 2^(1/12) mas aguda que la anterior. Empezamos en el do central (261.63 Hz)
FRECUENCIAS = [261.63 * 2 ** (n / 12) for n in range(24)]

BLANCAS_OCTAVA = [0, 2, 4, 5, 7, 9, 11]   # que notas de la octava son teclas blancas

# Puntas de los dedos de MediaPipe: pulgar, indice, corazon, anular, menique
PUNTAS = [4, 8, 12, 16, 20]

# --- Sonido con pygame ---
pygame.mixer.init(frequency=44100, size=-16, channels=1)
pygame.mixer.set_num_channels(20)   # para que puedan sonar muchas notas a la vez


def crear_nota(frecuencia, duracion=0.7):
    """Crea el sonido de una nota. Una nota es una onda senoidal: seno(2*pi*frecuencia*tiempo)."""
    muestras = int(44100 * duracion)
    t = np.linspace(0, duracion, muestras, False)       # el tiempo de cada muestra
    onda = np.sin(2 * np.pi * frecuencia * t)           # la onda de la nota
    onda = onda * np.linspace(1, 0, muestras)           # va bajando el volumen para que no "corte"
    onda = (onda * 0.4 * 32767).astype(np.int16)        # pasamos a enteros de 16 bits
    return pygame.sndarray.make_sound(onda)


# Creamos el sonido de las 24 notas una sola vez al empezar
sonidos = [crear_nota(f) for f in FRECUENCIAS]


def construir_teclas(x1, y1, x2, y2):
    """Calcula donde va cada tecla dentro del rectangulo (x1,y1)-(x2,y2).
    Devuelve dos listas de tuplas (nota, x_inicio, x_fin): teclas blancas y negras."""
    blancas = []
    negras = []
    ancho_blanca = (x2 - x1) / 14          # 2 octavas = 14 teclas blancas
    for i in range(14):
        octava = i // 7                    # 0 = primera octava, 1 = segunda
        pos = i % 7                        # posicion dentro de la octava (0 = DO ... 6 = SI)
        nota = octava * 12 + BLANCAS_OCTAVA[pos]
        blancas.append((nota, x1 + i * ancho_blanca, x1 + (i + 1) * ancho_blanca))

        # La tecla negra va en la union con la blanca siguiente. Hay negra despues
        # de DO, RE, FA, SOL y LA (no hay despues de MI ni de SI)
        if pos in (0, 1, 3, 4, 5):
            centro = x1 + (i + 1) * ancho_blanca
            negras.append((nota + 1, centro - ancho_blanca * 0.3, centro + ancho_blanca * 0.3))
    return blancas, negras


def esta_apuntando(mano):
    """True si solo tiene el indice estirado (la mano con los dedos hacia arriba)."""
    indice = mano[8].y < mano[6].y
    otros_doblados = (mano[12].y > mano[10].y and mano[16].y > mano[14].y
                      and mano[20].y > mano[18].y)
    return indice and otros_doblados


def texto(frame, t, x, y, tam, color):
    cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, tam, color, 2)


with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)  # abrir camara
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_ms = int(1000 / fps)
    timestamp = 0

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    estado = "PUNTO_1"        # PUNTO_1 -> PUNTO_2 -> TOCAR
    candidato = None          # posicion donde estamos apuntando
    t_inicio = 0              # cuando empezamos a apuntar ahi
    esquina1 = (0, 0)

    blancas, negras = [], []
    y_arriba = y_abajo = y_negra = linea_y = 0
    pulsadas = set()          # notas que estaban pulsadas en el frame anterior
    notas_tocadas = 0
    ultima_nota = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # efecto espejo
        ahora = time.time()

        # Detectar las manos
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms

        # Posicion (en pixeles) de todas las puntas de los dedos
        puntas = []
        for mano in result.hand_landmarks:
            for p in PUNTAS:
                puntas.append((int(mano[p].x * ancho), int(mano[p].y * alto)))

        # ------------------------------------------------------
        # Colocar el piano apuntando con el dedo
        # ------------------------------------------------------
        if estado in ("PUNTO_1", "PUNTO_2"):
            # Buscamos una mano que este apuntando
            apuntado = None
            for mano in result.hand_landmarks:
                if esta_apuntando(mano):
                    apuntado = (int(mano[8].x * ancho), int(mano[8].y * alto))

            if apuntado is None:
                candidato = None   # ya no apunta: empezamos de cero
            else:
                # Si se ha movido mucho, empezamos a contar de nuevo
                if candidato is None or abs(apuntado[0] - candidato[0]) + abs(apuntado[1] - candidato[1]) > 30:
                    candidato = apuntado
                    t_inicio = ahora
                progreso = min(1.0, (ahora - t_inicio) / 2)   # 2 segundos quieto
                cv2.circle(frame, candidato, 10, (0, 255, 255), -1)
                cv2.ellipse(frame, candidato, (25, 25), 0, 0, int(360 * progreso), (0, 255, 0), 4)

                if progreso >= 1.0:
                    if estado == "PUNTO_1":
                        esquina1 = candidato
                        estado = "PUNTO_2"
                    else:
                        x1, y1 = esquina1
                        x2, y2 = candidato
                        if x2 - x1 > 200 and y2 - y1 > 60:   # rectangulo valido
                            blancas, negras = construir_teclas(x1, y1, x2, y2)
                            y_arriba, y_abajo = y1, y2
                            y_negra = y1 + int((y2 - y1) * 0.6)    # las negras son mas cortas
                            linea_y = y1 + int((y2 - y1) * 0.5)    # linea que hay que cruzar
                            estado = "TOCAR"
                        else:
                            estado = "PUNTO_1"   # punto mal puesto: repetimos
                    candidato = None

            if estado == "PUNTO_1":
                texto(frame, "Apunta con el indice a la esquina", 10, 30, 0.7, (0, 255, 255))
                texto(frame, "SUPERIOR IZQUIERDA del piano (2 s)", 10, 60, 0.7, (0, 255, 255))
            elif estado == "PUNTO_2":
                cv2.circle(frame, esquina1, 8, (0, 255, 0), -1)
                texto(frame, "Ahora a la esquina INFERIOR", 10, 30, 0.7, (0, 255, 255))
                texto(frame, "DERECHA del piano (2 s)", 10, 60, 0.7, (0, 255, 255))
            texto(frame, "D = piano por defecto", 10, alto - 15, 0.5, (255, 255, 255))

        # ------------------------------------------------------
        # Tocar el piano
        # ------------------------------------------------------
        elif estado == "TOCAR":
            # Miramos que tecla toca cada punta que esta por debajo de la linea
            activas = set()
            for (x, y) in puntas:
                if y > linea_y:
                    tecla = None
                    # primero las negras (estan encima de las blancas)
                    for nota, xa, xb in negras:
                        if xa <= x < xb and y < y_negra:
                            tecla = nota
                    if tecla is None:
                        for nota, xa, xb in blancas:
                            if xa <= x < xb:
                                tecla = nota
                    if tecla is not None:
                        activas.add(tecla)

            # Solo suena cuando una nota pasa de "no pulsada" a "pulsada"
            # (asi no se repite mientras mantenemos el dedo abajo)
            for nota in activas:
                if nota not in pulsadas:
                    sonidos[nota].play()
                    notas_tocadas += 1
                    ultima_nota = NOMBRES[nota % 12]
            pulsadas = activas

            # Dibujar el piano semitransparente (en una copia que luego mezclamos)
            capa = frame.copy()
            for nota, xa, xb in blancas:
                color = (0, 255, 0) if nota in pulsadas else (255, 255, 255)
                cv2.rectangle(capa, (int(xa), y_arriba), (int(xb), y_abajo), color, -1)
                cv2.rectangle(capa, (int(xa), y_arriba), (int(xb), y_abajo), (0, 0, 0), 1)
                if nota % 12 == 0:   # escribimos el nombre solo en los DO
                    texto(capa, "DO", int(xa) + 5, y_abajo - 8, 0.4, (0, 0, 0))
            for nota, xa, xb in negras:
                color = (0, 200, 0) if nota in pulsadas else (30, 30, 30)
                cv2.rectangle(capa, (int(xa), y_arriba), (int(xb), y_negra), color, -1)
            frame = cv2.addWeighted(capa, 0.65, frame, 0.35, 0)

            # Linea que hay que cruzar
            cv2.line(frame, (0, linea_y), (ancho, linea_y), (255, 128, 0), 2)

            texto(frame, f"Notas tocadas: {notas_tocadas}", 10, 30, 0.7, (0, 255, 0))
            texto(frame, f"Ultima nota: {ultima_nota}", 10, 60, 0.7, (0, 255, 255))
            texto(frame, "R = colocar piano otra vez", 10, alto - 15, 0.5, (255, 255, 255))

        # Puntas de los dedos
        for (x, y) in puntas:
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        cv2.imshow("Piano", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:                  # ESC para salir
            break
        if tecla in (ord("r"), ord("R")):  # volver a colocar el piano
            estado = "PUNTO_1"
            candidato = None
            pulsadas = set()
        if tecla in (ord("d"), ord("D")):  # piano por defecto (abajo de la pantalla)
            x1, y1, x2, y2 = 20, int(alto * 0.55), ancho - 20, alto - 10
            blancas, negras = construir_teclas(x1, y1, x2, y2)
            y_arriba, y_abajo = y1, y2
            y_negra = y1 + int((y2 - y1) * 0.6)
            linea_y = y1 + int((y2 - y1) * 0.5)
            estado = "TOCAR"
            pulsadas = set()

    cap.release()
    cv2.destroyAllWindows()
