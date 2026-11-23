"""Particulas, anillos de explosion y temblor de pantalla."""
import cv2

PARTS = []
RINGS = []
SHAKE = [0.0]


def spawn(x, y, vx, vy, life, color, size, g=0.0):
    if len(PARTS) < 700:
        PARTS.append([x, y, vx, vy, life, life, color, size, g])


def update_particles(dt):
    alive = []
    for p in PARTS:
        p[4] -= dt
        if p[4] > 0:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += p[8] * dt
            alive.append(p)
    PARTS[:] = alive


def draw_particles(img):
    for p in PARTS:
        f = p[4] / p[5]
        col = (int(p[6][0] * f), int(p[6][1] * f), int(p[6][2] * f))
        r = max(1, int(p[7] * (0.4 + 0.6 * f)))
        cv2.circle(img, (int(p[0]), int(p[1])), r, col, -1)

