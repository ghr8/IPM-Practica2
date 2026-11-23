# Carrera terapéutica con las manos (IPM - Práctica 2)

Videojuego serio controlado por visión por computador (MediaPipe Hand Landmarker) pensado como
ejercicio de **rehabilitación de la mano**. Cada dedo (índice, corazón, anular, meñique) es una rueda
del coche: **verde = dedo extendido (empuja)**, rojo = dedo doblado. El pulgar activa el nitro.

> **Autores:** Germán Hurtado Rodríguez y Alejandro Párraga Sanchez
> **Asignatura:** Interacción Persona-Máquina (IPM) - Práctica 2

## 1. Propósito del juego

Este videojuego serio está pensado como apoyo a la **rehabilitación de la mano**, por ejemplo tras un ictus o en personas mayores con pérdida de movilidad fina y fragilidad. Los ejercicios de rehabilitación suelen ser repetitivos y poco motivadores, y eso baja la adherencia al tratamiento. Convertirlos en una carrera de coches, con una meta y un ranking, busca que el paciente quiera repetir la sesión.

El coche se controla solo con las manos, sin mando, usando la cámara web y MediaPipe. Cada dedo es una rueda, y cada mecánica trabaja un ejercicio real:

- **Flexión y extensión individual de los dedos:** abrir y cerrar cada dedo para acelerar y girar. Seguir las curvas obliga a graduar el movimiento, así que no se puede ganar con la mano quieta.
- **Coordinación bimanual:** en modo bimanual la mano izquierda gobierna las ruedas izquierdas y la derecha las derechas. En modo unimanual se rehabilita una sola mano.
- **Oposición del pulgar:** meter el pulgar en la palma gasta una carga de nitro.
- **Tiempo de reacción e inhibición del movimiento:** ante un obstáculo hay que frenar y quedarse quieto 1 s.
- **Individualización de dedos:** cruzar índice y corazón penaliza.

Mientras se juega, el sistema registra métricas orientativas: rango de movimiento por dedo, velocidad y ritmo de apertura y cierre, temblor, fatiga a lo largo de la sesión, simetría entre manos y tiempo de reacción. Se guardan en un historial y en un fichero de estado actual por paciente, para que un profesional pueda comparar sesiones. Estas medidas se inspiran en pruebas conocidas como el *finger tapping test* o el *Nine-Hole Peg Test*, y la lentitud motora es uno de los criterios de fragilidad de Fried.

**Limitaciones:** el juego no diagnostica ni sustituye a un fisioterapeuta ni a las pruebas clínicas validadas. Con una webcam y landmarks en 2D las medidas sirven para **comparar sesiones del mismo paciente en las mismas condiciones**, no para dar valores clínicos absolutos. Sobre la fragilidad solo medimos indicadores indirectos de la mano (lentitud, fatiga), no la fragilidad completa.

## 2. Requisitos

- Python **3.10, 3.11 o 3.12** (con `pip` y `venv`).
- Webcam (idealmente 720p) y buena iluminación frontal.
- Conexión a internet **solo** si `models/hand_landmarker.task` no está en la carpeta
  (el script de instalación lo descarga).
- Probado con: mediapipe 1.2.3, opencv-python 5.1, numpy 2.5 (ver `requirements.txt`).

## 3. Instalación (equipo limpio)

Descomprime el `.zip`, abre una terminal en la carpeta y ejecuta:

**Windows**
```bat
setup.bat
```

**Linux / macOS**
```bash
chmod +x setup.sh run.sh
./setup.sh
```

El script crea un entorno virtual en `.venv/`, instala `requirements.txt` y descarga el modelo
a `models/` si no existe.

<details>
<summary>Instalación manual (equivalente)</summary>

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
```
</details>

## 4. Ejecución

**Windows:** `run.bat`  |  **Linux/macOS:** `./run.sh`  |  o bien `python race3d.py`

Al arrancar pregunta el modo y el nombre. También se puede indicar por argumentos:

```bash
python race3d.py --modo bimanual            # bimanual | unimanual | duelo
python race3d.py --modo unimanual --camara 1   # usar la segunda cámara
python race3d.py --modo bimanual --turbo-auto  # las pastillas dan turbo sin usar el pulgar
```

## 5. Modos de juego y cómo se juega

| Modo | Jugadores | Manos | Pensado para |
|------|-----------|-------|--------------|
| `bimanual` | 1 | Las dos: la izquierda mueve las ruedas izquierdas, la derecha las derechas | Coordinación bimanual, simetría entre manos |
| `unimanual` | 1 | Una (la que se quiera rehabilitar; sirve cualquiera) | Rehabilitar una sola mano (p. ej. tras un ictus) |
| `duelo` | 2 | Una cada uno, pantalla dividida | Modo competitivo / social |

**Controles comunes**

1. Pon la(s) **mano(s) abierta(s)** hasta que se llene el círculo. Cuenta atrás y ¡a correr!
2. **Velocidad:** más dedos extendidos = más velocidad.
3. **Dirección:** más dedos a la izquierda que a la derecha = giras a la derecha, y viceversa.
   En bimanual hay 9 niveles de giro (0-4 dedos por mano), mucho más fino que con una sola mano.
4. **Hay que seguir las curvas.** La carretera se desvía más de lo que mide de ancho, así que con todos los dedos
   abiertos (o sin girar) te sales y explotas: no se puede ganar sin bajar dedos. Si te sales de la carretera o
   **cruzas índice y corazón**, el coche explota y vuelves al **punto de control** anterior (cada 1000 px).
5. **Pastillas amarillas:** en `duelo` dan turbo directo. En `bimanual`/`unimanual` **cargan nitro** (hasta 3).
6. **Pulgar (bimanual/unimanual):** metiendo el pulgar en la palma (hacia la base del meñique) gastas una carga
   de nitro. Es el ejercicio de **oposición/aducción del pulgar**; cada repetición se cuenta en pantalla.
7. **Obstáculos (bimanual/unimanual):** aparece de golpe una barrera en la carretera. Hay que **frenar** (bajar los
   dedos hasta quedar con 25 % o menos extendidos) y **quedarse quieto 1 s**; entonces desaparece. Si no frenas,
   choca. Con `--sin-obstaculos` se desactivan. Se mide el **tiempo de reacción** (aparición → primer frenazo).
8. Gana quien llegue antes a la meta. Los mejores tiempos se guardan en `ranking.json` (top 10 por modo).

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir |
| `ENTER` | Revancha (al terminar) |
| `R` | Reiniciar |
| `F` | Pantalla completa |
| `M` | Informe de la sesión (al terminar, un jugador) |

**Calibrar el pulgar.** Bajo cada mano se muestra `pulgar X.XX`: es la distancia de la punta del pulgar a la
base del meñique dividida por el largo de la palma (suelto ≈ 1.1-1.8, metido ≈ 0.2-0.5). El círculo se vuelve
naranja al detectarlo "metido". Si te cuesta activarlo o salta solo, ajusta `THUMB_IN` y `THUMB_OUT`
(en `config.py`; por defecto 0.60 y 0.85).

## 6. Estructura del proyecto

```
race3d.py            Punto de entrada: argumentos, bucle principal y estados (espera/cuenta atrás/carrera/final)
config.py            Todos los ajustes (dificultad, curvas, obstáculos, pulgar) y la configuración de cada modo
vision.py            Cámara + MediaPipe, análisis de dedos/pulgar y ángulos de flexión
players.py           Física del coche, explosiones y puntos de control
obstacles.py         Obstáculos y tiempo de reacción
track.py             Curvas de la carretera, pastillas, minimapa, posiciones de los obstáculos
metrics.py           Métricas de la sesión (ROM, velocidad, temblor, fatiga, simetría, reacción)
storage.py           Ranking, entrada por consola y carpeta de pacientes (informes, historial, estado)
render.py            Dibujo de la escena 3D, coche, obstáculo, manos y minimapa
screens.py           HUD, pantallas de espera/cuenta atrás/final e informe
effects.py           Partículas y temblor de pantalla
download_model.py    Descarga el modelo de MediaPipe
requirements.txt     Dependencias            setup.* / run.*   Instalación y ejecución
models/              hand_landmarker.task
pacientes/           Se crea al terminar una carrera de un jugador (ver sección 7)
ranking.json         Se crea al terminar la primera carrera
```

## 7. Métricas, informe y estado del paciente (modos de un jugador)

Al llegar a meta se calcula el informe (pulsa `M` para verlo) y se guarda en `pacientes/<nombre>/`:

| Archivo | Contenido |
|---------|-----------|
| `estado_actual.json` | **Estado actual del paciente**: métricas de la última sesión, reacción, fatiga, simetría y cambio respecto a la sesión anterior del mismo modo |
| `historial.csv` | Una fila por sesión (para hacer la gráfica de evolución) |
| `sesiones/<fecha>_<modo>.json` | Informe completo, con cada obstáculo (reacción, tiempo de parada, distancia al frenar) |

Métricas (todas **orientativas**, no son un diagnóstico):

- **Rango de movimiento (ROM)** por dedo: ángulo de flexión de la articulación PIP (percentil 95 - percentil 5).
- **Velocidad de apertura/cierre** (percentil 90 de la velocidad angular) y **ciclos por segundo** por dedo (parecido a un *finger tapping test*).
- **Temblor**: amplitud de la oscilación de la palma entre 4 y 6 Hz, en % del largo de la palma.
- **Fatiga**: caída del ROM y del ritmo entre el primer y el último tercio de la carrera.
- **Simetría** entre mano izquierda y derecha (bimanual).
- **Tiempo de reacción**: media, mediana, mínimo y máximo (ms), choques y tiempo hasta parar.
- **Pulgar**: repeticiones de oposición.

Se mide con una webcam y landmarks en 2D: sirve para **comparar sesiones del mismo paciente en las mismas condiciones**
(misma cámara, distancia y luz), no para dar valores clínicos absolutos.

**Dificultad** (en `config.py`): `SWAY_AMP` y `SWAY_LEN` (cuánto se curva la carretera), `OFF_ROAD_GRACE`,
`CHECKPOINT_EVERY`, `OBST_DIST` (más distancia = más tiempo para reaccionar) y `OBST_BRAKE_FRAC`.

## 8. Problemas frecuentes

| Problema | Solución |
|----------|----------|
| "No se pudo abrir la cámara" | Cierra otras apps que usen la webcam o prueba otra con `--camara 1`. |
| El nitro se activa solo / no se activa | Mira el valor `pulgar X.XX` en pantalla y ajusta `THUMB_IN` / `THUMB_OUT`. |
| No detecta la mano | Más luz frontal, fondo no demasiado desordenado, mano a 40-80 cm de la cámara. |
| Va lento | Cierra otros programas; la detección corre en un hilo aparte (se muestra "deteccion X fps" abajo a la derecha). |
| Error al descargar el modelo | Descárgalo a mano (URL en `download_model.py`) y guárdalo en `models/hand_landmarker.task`. |
