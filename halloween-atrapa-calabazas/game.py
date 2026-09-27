"""
Halloween: Atrapa Calabazas - juego de camara para proyectar en una pared
==========================================================================

Tema Halloween pensado para proyectar sobre una pared:
    - Calabazas (naranjas)  -> ATRAPARLAS suma puntos.
    - Murcielagos (aletean, se mueven en zigzag) -> EVITARLOS, si los tocas
      te asustan (sonido + flash), pero las vidas son INFINITAS: nunca
      termina el juego por esto.
    - Fantasma (aparece de vez en cuando, translucido, flota) -> EVITARLO,
      si lo tocas te asusta mas fuerte (la pantalla se sacude un instante)
      y suena un efecto espeluznante.
    - Sombrero de bruja magico (grande, aparece rara vez) -> ATRAPARLO te
      da un power-up de VELOCIDAD (agranda el radio de atrape) por unos
      segundos, y dispara una animacion de estrellas, sapitos y pociones
      alrededor del sombrero.

Al iniciar se muestra una pantalla de inicio (solo una vez, con teclado)
para elegir la velocidad de caida: Lenta / Normal / Rapida. Esa pantalla
NO vuelve a aparecer durante el juego proyectado.

Seguimiento de manos: MediaPipe Hands (21 puntos por mano), dibujados
como un esqueleto de hueso (blanco hueso) sobre fondo negro -- no se
muestra el video real de la camara, solo el esqueleto y el juego.

Sonido: se sintetiza en el momento con numpy y se reproduce con
sounddevice (no se necesitan archivos de audio externos). Si la
computadora no tiene salida de audio disponible, el juego sigue
funcionando normalmente pero en silencio.

Controles:
    Q / ESC   -> salir
    R         -> reiniciar puntaje
    F         -> pantalla completa on/off (util para proyectar en la pared)
"""

import sys
import os
import wave
import time
import random
import math
import threading

import cv2
import numpy as np
import mediapipe as mp

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except Exception:
    _HAS_SOUNDDEVICE = False


# ----------------------------------------------------------------------------
# Configuracion general
# ----------------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 1280, 720

CATCH_RADIUS = 55
CATCH_LANDMARKS = [4, 8, 12, 16, 20, 0]   # yemas de los 5 dedos + muneca
POWERUP_CATCH_BONUS = 25

SPEED_PRESETS = [
    ("Lenta", 3.0),
    ("Normal", 4.2),
    ("Rapida", 6.0),
]
SPEED_INCREASE_PER_10_POINTS = 0.6

MAX_OBJECTS_ON_SCREEN = 6
SPAWN_INTERVAL_FRAMES = 40

# Probabilidades de que, al spawnear, el objeto sea de cada tipo
SPAWN_WEIGHTS = {
    "pumpkin": 0.55,
    "vampire": 0.32,
    "ghost": 0.07,
    "witch_hat": 0.06,
}

POWERUP_DURATION_SEC = 6.0
GHOST_MIN_INTERVAL_SEC = 7.0   # tiempo minimo entre apariciones de fantasma

# Colores en BGR
COLOR_PUMPKIN = (0, 140, 255)
COLOR_PUMPKIN_DARK = (0, 90, 180)
COLOR_PUMPKIN_LIGHT = (40, 180, 255)
COLOR_PUMPKIN_GLOW = (10, 190, 255)
COLOR_STEM = (30, 90, 20)
COLOR_BAT = (90, 40, 70)
COLOR_BAT_LIGHT = (140, 80, 120)
COLOR_GHOST = (255, 245, 235)
COLOR_GHOST_GLOW = (60, 150, 255)
COLOR_HAT = (70, 25, 55)
COLOR_HAT_LIGHT = (130, 60, 110)
COLOR_HAT_OUTLINE = (200, 130, 220)
COLOR_HAT_BAND = (0, 165, 255)
COLOR_HAT_BUCKLE = (0, 215, 255)

SIZE_SCALE = 1.15  # todos los personajes son un 15% mas grandes
EXTRA_SIZE_BOOST = 1.20  # +20% adicional para calabaza/fantasma/vampirito
SECOND_SIZE_BOOST = 1.20  # +20% mas, esta vez para TODOS (incluido el sombrero)

MAGIC_COLORS = [
    (200, 30, 200), (255, 120, 0), (0, 200, 120),
    (0, 165, 255), (255, 220, 0), (180, 60, 255),
]

# Fondo de proyeccion (en vez de mostrar la imagen real de la camara) y
# colores del esqueleto de manos ("esqueleto de hueso") dibujado sobre ese fondo.
BACKGROUND_COLOR = (10, 6, 4)            # casi negro, se funde con la pared
SKELETON_BONE_COLOR = (235, 240, 245)    # blanco hueso
SKELETON_BONE_OUTLINE = (55, 45, 45)     # contorno oscuro sutil
SKELETON_BONE_THICKNESS = 5
SKELETON_JOINT_RADIUS = 7


# ----------------------------------------------------------------------------
# Utilidades de dibujo
# ----------------------------------------------------------------------------
def draw_text(img, text, org, scale=0.9, color=(255, 255, 255), thickness=2, shadow=True):
    if shadow:
        cv2.putText(img, text, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def apply_glow(img, x, y, radius, color, strength=0.5):
    """Dibuja un resplandor suave (glow) centrado en (x, y), sumado con brillo
    sobre lo que ya esta dibujado -- da sensacion de luz interior/3D."""
    x, y = int(x), int(y)
    r = max(2, int(radius))
    margin = r // 2 + 6  # margen extra para que el desenfoque se apague antes del borde del parche
    half = r + margin
    x0, x1 = max(0, x - half), min(img.shape[1], x + half)
    y0, y1 = max(0, y - half), min(img.shape[0], y + half)
    if x1 <= x0 or y1 <= y0:
        return
    glow_layer = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    cv2.circle(glow_layer, (x - x0, y - y0), r, color, -1, cv2.LINE_AA)
    k = max(3, ((margin // 2) * 2) + 1)
    glow_layer = cv2.GaussianBlur(glow_layer, (k, k), 0, borderType=cv2.BORDER_CONSTANT)
    roi = img[y0:y1, x0:x1]
    blended = cv2.addWeighted(glow_layer, strength, roi, 1.0, 0)
    img[y0:y1, x0:x1] = blended


def shade_ellipse(img, center, axes, color, alpha=0.35):
    """Superpone una elipse de color a baja opacidad -- util para sombras o
    brillos suaves que dan volumen sin tapar el dibujo de abajo."""
    overlay = img.copy()
    cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_pumpkin(img, x, y, size):
    x, y = int(x), int(y)
    body_w, body_h = size, int(size * 0.8)

    # sombra de contacto (ancla el objeto, sensacion de peso/volumen)
    shade_ellipse(img, (x, y + int(body_h * 0.56)), (int(body_w * 0.45), int(body_h * 0.1)), (0, 0, 0), alpha=0.35)

    for dx_frac in (-0.35, 0.0, 0.35):
        cv2.ellipse(img, (x + int(dx_frac * body_w * 0.6), y), (int(body_w * 0.42), int(body_h * 0.5)),
                    0, 0, 360, COLOR_PUMPKIN, -1, cv2.LINE_AA)

    # sombreado inferior-derecho (mas oscuro) y brillo superior-izquierdo (mas
    # claro) para dar sensacion de esfera 3D en vez de disco plano
    shade_ellipse(img, (x + int(body_w * 0.12), y + int(body_h * 0.16)),
                  (int(body_w * 0.42), int(body_h * 0.38)), COLOR_PUMPKIN_DARK, alpha=0.30)
    shade_ellipse(img, (x - int(body_w * 0.18), y - int(body_h * 0.22)),
                  (int(body_w * 0.20), int(body_h * 0.16)), COLOR_PUMPKIN_LIGHT, alpha=0.40)

    cv2.ellipse(img, (x, y), (int(body_w * 0.5), int(body_h * 0.5)), 0, 0, 360,
                COLOR_PUMPKIN_DARK, 2, cv2.LINE_AA)
    for dx_frac in (-0.25, 0.25):
        px = x + int(dx_frac * body_w)
        cv2.line(img, (px, y - int(body_h * 0.45)), (px, y + int(body_h * 0.45)), COLOR_PUMPKIN_DARK, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x - 4, y - int(body_h * 0.5) - 12), (x + 4, y - int(body_h * 0.5) + 2),
                  COLOR_STEM, -1, cv2.LINE_AA)

    # cara tallada Jack-o-lantern: ojos triangulares + boca dentada, con
    # resplandor calido saliendo de los cortes (como una vela encendida adentro)
    eye_off = int(size * 0.2)
    apply_glow(img, x - eye_off, y - int(size * 0.02), size * 0.55, COLOR_PUMPKIN_GLOW, strength=0.55)
    apply_glow(img, x + eye_off, y - int(size * 0.02), size * 0.55, COLOR_PUMPKIN_GLOW, strength=0.55)

    tri_l = np.array([
        [x - eye_off, y - int(size * 0.14)],
        [x - eye_off - int(size * 0.11), y + int(size * 0.06)],
        [x - eye_off + int(size * 0.11), y + int(size * 0.06)],
    ])
    tri_r = np.array([
        [x + eye_off, y - int(size * 0.14)],
        [x + eye_off - int(size * 0.11), y + int(size * 0.06)],
        [x + eye_off + int(size * 0.11), y + int(size * 0.06)],
    ])
    cv2.fillPoly(img, [tri_l], (5, 5, 5), cv2.LINE_AA)
    cv2.fillPoly(img, [tri_r], (5, 5, 5), cv2.LINE_AA)

    mouth_y = y + int(size * 0.20)
    mouth_w = size * 0.56
    n_teeth = 5
    apply_glow(img, x, mouth_y + int(size * 0.06), size * 0.6, COLOR_PUMPKIN_GLOW, strength=0.5)
    pts = [(int(x - mouth_w / 2), mouth_y)]
    for i in range(n_teeth):
        mx = x - mouth_w / 2 + (i + 0.5) * (mouth_w / n_teeth)
        my = mouth_y + (int(size * 0.16) if i % 2 == 0 else int(size * 0.04))
        pts.append((int(mx), int(my)))
    pts.append((int(x + mouth_w / 2), mouth_y))
    cv2.fillPoly(img, [np.array(pts)], (5, 5, 5), cv2.LINE_AA)


def draw_bat(img, x, y, size, wing_phase):
    x, y = int(x), int(y)
    flap = math.sin(wing_phase) * 0.9
    body_r = max(2, int(size * 0.28))
    for side in (-1, 1):
        wing_len = size
        tip_x = x + side * int(wing_len * (0.9 + 0.1 * flap))
        tip_y = y - int(wing_len * 0.35 * flap)
        mid_x = x + side * int(wing_len * 0.5)
        mid_y = y - int(wing_len * 0.15)
        pts = np.array([[x, y], [mid_x, mid_y], [tip_x, tip_y], [mid_x, y + int(wing_len * 0.15)]])
        cv2.fillPoly(img, [pts], COLOR_BAT, cv2.LINE_AA)
        # nervaduras del ala (dan textura/profundidad, no solo silueta plana)
        cv2.line(img, (x, y), (tip_x, tip_y), COLOR_BAT_LIGHT, 1, cv2.LINE_AA)
        cv2.line(img, (x, y), (mid_x, y + int(wing_len * 0.15)), COLOR_BAT_LIGHT, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (body_r, int(body_r * 1.1)), 0, 0, 360, COLOR_BAT, -1, cv2.LINE_AA)
    shade_ellipse(img, (x - int(body_r * 0.3), y - int(body_r * 0.3)), (int(body_r * 0.5), int(body_r * 0.5)),
                  COLOR_BAT_LIGHT, alpha=0.45)
    eye_off = max(1, int(size * 0.08))
    cv2.circle(img, (x - eye_off, y - eye_off), 2, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x + eye_off, y - eye_off), 2, (0, 0, 255), -1, cv2.LINE_AA)


def draw_ghost(img, x, y, size, alpha, bob_phase):
    x = int(x)
    y = int(y + math.sin(bob_phase) * 6)
    apply_glow(img, x, y + int(size * 0.25), size * 0.9, COLOR_GHOST_GLOW, strength=0.25 * alpha)
    overlay = img.copy()
    body_top = y - int(size * 0.6)
    cv2.ellipse(overlay, (x, body_top + int(size * 0.5)), (int(size * 0.55), int(size * 0.55)),
                0, 180, 360, COLOR_GHOST, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x - int(size * 0.55), body_top + int(size * 0.5)),
                  (x + int(size * 0.55), y + int(size * 0.35)), COLOR_GHOST, -1, cv2.LINE_AA)
    n_waves = 4
    wave_w = int(size * 1.1 / n_waves)
    base_y = y + int(size * 0.35)
    pts = [(x - int(size * 0.55), base_y)]
    for i in range(n_waves):
        wx = x - int(size * 0.55) + i * wave_w + wave_w // 2
        wy = base_y + (14 if i % 2 == 0 else -6)
        pts.append((wx, wy))
    pts.append((x + int(size * 0.55), base_y))
    cv2.fillPoly(overlay, [np.array(pts)], COLOR_GHOST, cv2.LINE_AA)
    # tinte calido suave hacia abajo (como una vela por debajo) para dar profundidad
    cv2.ellipse(overlay, (x, y + int(size * 0.15)), (int(size * 0.5), int(size * 0.3)), 0, 0, 360,
                COLOR_GHOST_GLOW, -1, cv2.LINE_AA)
    eye_off = int(size * 0.18)
    cv2.circle(overlay, (x - eye_off, body_top + int(size * 0.35)), max(2, int(size * 0.09)), (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + eye_off, body_top + int(size * 0.35)), max(2, int(size * 0.09)), (40, 40, 40), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_witch_hat(img, x, y, size, wobble=0.0):
    """Sombrero de bruja grande, con brillo violeta y banda dorada."""
    x, y = int(x), int(y + math.sin(wobble) * 4)
    brim_w, brim_h = int(size * 0.95), int(size * 0.28)
    cone_h = int(size * 1.15)
    tip_dx = int(math.sin(wobble * 0.7) * size * 0.12)

    # sombra suave debajo del ala
    cv2.ellipse(img, (x, y + int(brim_h * 0.15)), (brim_w, int(brim_h * 0.9)), 0, 0, 360,
                (0, 0, 0), -1, cv2.LINE_AA)
    # ala del sombrero
    cv2.ellipse(img, (x, y), (brim_w, brim_h), 0, 0, 360, COLOR_HAT, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (brim_w, brim_h), 0, 0, 360, COLOR_HAT_OUTLINE, 2, cv2.LINE_AA)

    # cono
    base_l = (x - int(size * 0.42), y - int(size * 0.05))
    base_r = (x + int(size * 0.42), y - int(size * 0.05))
    tip = (x + tip_dx, y - cone_h)
    pts = np.array([base_l, base_r, tip])
    cv2.fillPoly(img, [pts], COLOR_HAT, cv2.LINE_AA)
    # resaltado de un lado del cono (da volumen: parece redondo, no plano)
    highlight_pts = np.array([base_l, (x - int(size * 0.10), y - int(size * 0.05)), tip])
    overlay = img.copy()
    cv2.fillPoly(overlay, [highlight_pts], COLOR_HAT_LIGHT, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
    cv2.polylines(img, [pts], True, COLOR_HAT_OUTLINE, 2, cv2.LINE_AA)

    # banda con hebilla
    band_y = y - int(size * 0.28)
    band_half_w = int(size * 0.34)
    cv2.line(img, (x - band_half_w, band_y), (x + band_half_w, band_y), COLOR_HAT_BAND, max(3, int(size * 0.09)), cv2.LINE_AA)
    cv2.rectangle(img, (x - 8, band_y - 7), (x + 8, band_y + 7), COLOR_HAT_BUCKLE, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x - 8, band_y - 7), (x + 8, band_y + 7), (60, 60, 60), 1, cv2.LINE_AA)

    # brillo magico: un par de destellos
    for k in range(2):
        sp_x = x + int(math.cos(wobble * 2 + k * 2.1) * size * 0.6)
        sp_y = y - cone_h + int(math.sin(wobble * 2 + k * 2.1) * size * 0.3) + int(size * 0.3)
        cv2.circle(img, (sp_x, sp_y), 2, (255, 255, 255), -1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Escena de fondo (luna, arboles retorcidos, cementerio) -- se dibuja UNA
# sola vez al arrancar y se cachea; no se recalcula cada frame.
# ----------------------------------------------------------------------------
COLOR_SKY_TOP = np.array([70, 20, 35], dtype=np.float32)       # violeta oscuro (BGR)
COLOR_SKY_HORIZON = np.array([40, 70, 130], dtype=np.float32)  # naranja apagado (BGR)
COLOR_GROUND = np.array([8, 6, 10], dtype=np.float32)
COLOR_SILHOUETTE = (18, 12, 20)


def draw_tree_silhouette(img, x, horizon_y, height, side=1, seed=0):
    trunk_w = max(6, int(height * 0.045))
    top_y = int(horizon_y * 0.35)
    pts = np.array([
        (x - trunk_w, horizon_y), (x - int(trunk_w * 0.35), top_y),
        (x + int(trunk_w * 0.35), top_y), (x + trunk_w, horizon_y),
    ])
    cv2.fillPoly(img, [pts], COLOR_SILHOUETTE, cv2.LINE_AA)
    rng = random.Random(seed)
    for _ in range(6):
        by = top_y + rng.randint(0, int(horizon_y * 0.45))
        bx = x + side * rng.randint(int(trunk_w * 0.2), int(trunk_w * 1.2))
        ex = bx + side * rng.randint(25, 70)
        ey = by - rng.randint(15, 60)
        mx = (bx + ex) // 2 + side * rng.randint(-15, 15)
        my = (by + ey) // 2
        thickness = max(2, int(trunk_w * 0.18))
        cv2.line(img, (bx, by), (mx, my), COLOR_SILHOUETTE, thickness, cv2.LINE_AA)
        cv2.line(img, (mx, my), (ex, ey), COLOR_SILHOUETTE, max(1, thickness - 2), cv2.LINE_AA)


def draw_gravestone(img, x, y, size, cross=False):
    if cross:
        cv2.rectangle(img, (x - 2, y - size), (x + 2, y), COLOR_SILHOUETTE, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x - int(size * 0.32), y - int(size * 0.68)),
                      (x + int(size * 0.32), y - int(size * 0.56)), COLOR_SILHOUETTE, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (x, y - int(size * 0.5)), (int(size * 0.36), int(size * 0.5)), 0, 180, 360,
                    COLOR_SILHOUETTE, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x - int(size * 0.36), y - int(size * 0.5)), (x + int(size * 0.36), y),
                      COLOR_SILHOUETTE, -1, cv2.LINE_AA)


def build_background(width, height):
    """Genera la escena de fondo (cielo con luna, arboles, cementerio con
    calabazas) inspirada en un clima de Halloween. Se llama una sola vez;
    el resultado se reutiliza (copy()) en cada frame, asi que no cuesta
    rendimiento en el loop del juego."""
    bg = np.zeros((height, width, 3), dtype=np.float32)
    horizon_y = int(height * 0.62)

    for row in range(horizon_y):
        t = row / max(1, horizon_y - 1)
        bg[row, :] = COLOR_SKY_TOP * (1 - t) + COLOR_SKY_HORIZON * t
    for row in range(horizon_y, height):
        t = (row - horizon_y) / max(1, height - horizon_y - 1)
        bg[row, :] = COLOR_SKY_HORIZON * (1 - t) + COLOR_GROUND * t
    bg = bg.astype(np.uint8)

    # nubes oscuras suaves cruzando el cielo
    for i in range(4):
        cx = int(width * (0.12 + i * 0.24))
        cy = int(height * (0.16 + (i % 2) * 0.10))
        shade_ellipse(bg, (cx, cy), (int(width * 0.15), int(height * 0.03)), (35, 15, 30), alpha=0.4)

    # luna grande con resplandor
    moon_x, moon_y, moon_r = int(width * 0.6), int(height * 0.26), int(height * 0.12)
    apply_glow(bg, moon_x, moon_y, moon_r * 2.4, (60, 150, 255), strength=0.45)
    cv2.circle(bg, (moon_x, moon_y), moon_r, (170, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(bg, (moon_x, moon_y), moon_r, (140, 190, 230), 2, cv2.LINE_AA)

    # murcielagos decorativos (fijos, no interactivos) cerca de la luna
    for (dx, dy, s) in ((-100, -30, 16), (-45, -65, 13), (75, -25, 15), (30, 55, 12)):
        draw_bat(bg, moon_x + dx, moon_y + dy, s, wing_phase=1.0)

    # arboles retorcidos a los costados
    draw_tree_silhouette(bg, int(width * 0.05), horizon_y, height, side=1, seed=1)
    draw_tree_silhouette(bg, int(width * 0.95), horizon_y, height, side=-1, seed=2)

    # lapidas y cruces sobre el horizonte
    rng = random.Random(7)
    for i in range(6):
        gx = int(width * (0.18 + i * 0.13) + rng.uniform(-15, 15))
        gy = horizon_y - rng.randint(0, 15)
        draw_gravestone(bg, gx, gy, rng.randint(18, 30), cross=(i % 2 == 0))

    # calabazas brillando en el piso, decorativas (las que caen son aparte)
    draw_pumpkin(bg, int(width * 0.09), int(height * 0.88), int(38 * SIZE_SCALE))
    draw_pumpkin(bg, int(width * 0.91), int(height * 0.90), int(34 * SIZE_SCALE))

    return bg


BACKGROUND_IMAGE_FILENAME = "background.jpg"


def load_background(width, height):
    """Busca background.png junto a este script y lo usa como fondo del
    juego (redimensionado si hace falta). Si no esta, o no se puede leer,
    usa la escena dibujada a mano (build_background) como respaldo -- el
    juego nunca se rompe por falta de este archivo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, BACKGROUND_IMAGE_FILENAME)
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            return img
        print(f"Aviso: no se pudo leer {BACKGROUND_IMAGE_FILENAME}, uso el fondo dibujado por Claude.")
    return build_background(width, height)


# ----------------------------------------------------------------------------
# Sprites animados (calabaza, fantasma, sombrero, vampirito): 12 frames cada
# uno, en /sprites/<nombre>/00.png .. 11.png, con transparencia real. Si la
# carpeta no esta o esta vacia, se usa el dibujo a mano como respaldo -- el
# juego nunca se rompe por faltar estos archivos.
# ----------------------------------------------------------------------------
SPRITES_DIRNAME = "sprites"
ANIM_PHASE_STEP = 0.4  # cuanto avanza self.phase por cada frame de animacion
_SPRITE_CACHE = {}


def load_sprite_sequence(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, SPRITES_DIRNAME, name)
    frames = []
    if os.path.isdir(folder):
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".png"):
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 4:
                    frames.append(img)
    return frames


def get_sprite_frames(name):
    if name not in _SPRITE_CACHE:
        _SPRITE_CACHE[name] = load_sprite_sequence(name)
    return _SPRITE_CACHE[name]


def paste_sprite(canvas, sprite_rgba, cx, cy, target_size):
    """Pega (con transparencia real) un frame de sprite centrado en (cx, cy),
    redimensionado para que su lado mas largo mida target_size pixeles."""
    sh, sw = sprite_rgba.shape[:2]
    if sh == 0 or sw == 0 or target_size <= 0:
        return
    scale = target_size / max(sw, sh)
    new_w, new_h = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(sprite_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0, y0 = int(cx - new_w / 2), int(cy - new_h / 2)
    ch, cw = canvas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(cw, x0 + new_w), min(ch, y0 + new_h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

    roi = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    sprite_crop = resized[sy0:sy1, sx0:sx1].astype(np.float32)
    alpha = sprite_crop[:, :, 3:4] / 255.0
    blended = sprite_crop[:, :, :3] * alpha + roi * (1 - alpha)
    canvas[dy0:dy1, dx0:dx1] = blended.astype(np.uint8)


# ----------------------------------------------------------------------------
# Particulas magicas (estrellas, sapitos, pociones) al atrapar el sombrero
# ----------------------------------------------------------------------------
def draw_mini_star(img, x, y, size, angle_deg, color):
    x, y = int(x), int(y)
    pts = []
    for i in range(10):
        r = size if i % 2 == 0 else size * 0.42
        a = math.radians(angle_deg + i * 36)
        pts.append((int(x + math.cos(a) * r), int(y + math.sin(a) * r)))
    cv2.fillPoly(img, [np.array(pts)], color, cv2.LINE_AA)


def draw_mini_frog(img, x, y, size):
    x, y = int(x), int(y)
    green = (60, 190, 90)
    dark = (30, 110, 50)
    cv2.ellipse(img, (x, y), (int(size * 0.6), int(size * 0.45)), 0, 0, 360, green, -1, cv2.LINE_AA)
    for side in (-1, 1):
        cv2.circle(img, (x + side * int(size * 0.35), y - int(size * 0.35)), max(2, int(size * 0.22)), green, -1, cv2.LINE_AA)
        cv2.circle(img, (x + side * int(size * 0.35), y - int(size * 0.35)), max(1, int(size * 0.1)), (10, 10, 10), -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y + int(size * 0.1)), (int(size * 0.3), int(size * 0.12)), 0, 0, 180, dark, 1, cv2.LINE_AA)


def draw_mini_potion(img, x, y, size, liquid_color):
    x, y = int(x), int(y)
    glass = (210, 220, 225)
    neck_w = max(2, int(size * 0.22))
    body_r = max(3, int(size * 0.42))
    cv2.rectangle(img, (x - neck_w // 2, y - int(size * 0.7)), (x + neck_w // 2, y - int(size * 0.25)), glass, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), body_r, glass, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y + int(body_r * 0.15)), int(body_r * 0.8), liquid_color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), body_r, (120, 120, 130), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x - neck_w // 2 - 2, y - int(size * 0.78)), (x + neck_w // 2 + 2, y - int(size * 0.68)),
                  (90, 60, 40), -1, cv2.LINE_AA)


PARTICLE_KINDS = ("star", "frog", "potion")
PARTICLE_WEIGHTS = (0.6, 0.2, 0.2)  # que salten mas estrellas que sapitos/pociones
COLOR_WITCH_SMOKE = (60, 200, 90)  # verde bruja (BGR)


class SmokeParticle:
    """Humo verde grande y suave (tipo bruja) -- se expande y se desvanece.
    Tiene la misma interfaz que Particle (update/alive/draw) asi que se
    puede mezclar en la misma lista de un Burst."""

    def __init__(self, x, y):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(20, 70)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 30
        self.life = random.uniform(1.0, 1.6)
        self.max_life = self.life
        self.size = random.randint(45, 75)
        self.growth = random.uniform(1.3, 1.8)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.97
        self.vy *= 0.97
        self.life -= dt

    def alive(self):
        return self.life > 0

    def draw(self, img):
        t = 1 - max(0.0, min(1.0, self.life / self.max_life))
        alpha = max(0.0, min(1.0, self.life / self.max_life)) * 0.6
        cur_size = self.size * (1 + t * self.growth)
        apply_glow(img, self.x, self.y, cur_size, COLOR_WITCH_SMOKE, strength=alpha)


class Particle:
    def __init__(self, x, y):
        self.kind = random.choices(PARTICLE_KINDS, weights=PARTICLE_WEIGHTS, k=1)[0]
        angle = random.uniform(0, math.tau)
        speed = random.uniform(70, 190)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 60
        self.life = random.uniform(0.7, 1.15)
        self.max_life = self.life
        self.size = random.randint(9, 16)
        self.color = random.choice(MAGIC_COLORS)
        self.angle_deg = random.uniform(0, 360)
        self.spin = random.uniform(-220, 220)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 140 * dt
        self.vx *= 0.985
        self.angle_deg += self.spin * dt
        self.life -= dt

    def alive(self):
        return self.life > 0

    def draw(self, img):
        alpha = max(0.0, min(1.0, self.life / self.max_life))
        overlay = img.copy()
        if self.kind == "star":
            draw_mini_star(overlay, self.x, self.y, self.size, self.angle_deg, self.color)
        elif self.kind == "frog":
            draw_mini_frog(overlay, self.x, self.y, self.size)
        else:
            draw_mini_potion(overlay, self.x, self.y, self.size, self.color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


class Burst:
    def __init__(self, x, y, count=16, smoke_count=0):
        self.particles = [Particle(x, y) for _ in range(count)]
        self.particles += [SmokeParticle(x, y) for _ in range(smoke_count)]

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive()]

    def draw(self, img):
        for p in self.particles:
            p.draw(img)

    def finished(self):
        return len(self.particles) == 0


# ----------------------------------------------------------------------------
# Objetos que caen / se mueven
# ----------------------------------------------------------------------------
class FallingObject:
    def __init__(self, kind, width, speed):
        self.kind = kind
        self.width = width
        self.x = random.uniform(70, max(71, width - 70))
        self.y = -70.0
        self.speed = speed
        if kind == "ghost":
            self.size = int(random.randint(55, 75) * SIZE_SCALE * EXTRA_SIZE_BOOST * SECOND_SIZE_BOOST)
        elif kind == "witch_hat":
            self.size = int(random.randint(75, 100) * SIZE_SCALE * SECOND_SIZE_BOOST)
        else:
            self.size = int(random.randint(30, 50) * SIZE_SCALE * EXTRA_SIZE_BOOST * SECOND_SIZE_BOOST)
        self.phase = random.uniform(0, math.tau)
        self.zigzag_speed = random.uniform(1.5, 3.0)
        self.zigzag_amp = random.uniform(1.5, 3.5)
        self.alpha = 0.0

    def update(self, dt):
        self.phase += dt * 4.0
        if self.kind == "pumpkin":
            self.y += self.speed
        elif self.kind == "witch_hat":
            self.y += self.speed * 0.7
            self.x += math.sin(self.phase * 0.5) * 1.4
        elif self.kind == "vampire":
            self.y += self.speed * 1.1
            self.x += math.sin(self.phase * self.zigzag_speed) * self.zigzag_amp
        elif self.kind == "ghost":
            self.y += self.speed * 0.55
            self.x += math.sin(self.phase * 0.6) * 2.2
            self.alpha = min(0.85, self.alpha + dt * 0.6)

    def draw(self, img):
        frames = get_sprite_frames(self.kind)
        if frames:
            idx = int(self.phase / ANIM_PHASE_STEP) % len(frames)
            paste_sprite(img, frames[idx], self.x, self.y, self.size)
            return
        # respaldo: dibujo a mano si faltan los sprites
        if self.kind == "pumpkin":
            draw_pumpkin(img, self.x, self.y, self.size)
        elif self.kind == "vampire":
            draw_bat(img, self.x, self.y, self.size, self.phase * self.zigzag_speed)
        elif self.kind == "ghost":
            draw_ghost(img, self.x, self.y, self.size, self.alpha, self.phase * 0.6)
        elif self.kind == "witch_hat":
            draw_witch_hat(img, self.x, self.y, self.size, self.phase)

    def hit_radius(self, catch_radius_bonus=0):
        return catch_radius_bonus + CATCH_RADIUS + self.size * 0.3

    def touches(self, points, catch_radius_bonus=0):
        r = self.hit_radius(catch_radius_bonus)
        for (px, py) in points:
            if (px - self.x) ** 2 + (py - self.y) ** 2 <= r ** 2:
                return True
        return False


def choose_kind():
    kinds = list(SPAWN_WEIGHTS.keys())
    weights = list(SPAWN_WEIGHTS.values())
    return random.choices(kinds, weights=weights, k=1)[0]


# ----------------------------------------------------------------------------
# Juego
# ----------------------------------------------------------------------------
class Game:
    """Vidas infinitas: tocar un murcielago o un fantasma nunca termina el
    juego, solo asusta (sonido + flash / sacudida) y suma al contador de
    sustos."""

    def __init__(self, base_speed=4.2):
        self.base_speed = base_speed
        self.reset()

    def reset(self):
        self.score = 0
        self.misses = 0
        self.objects = []
        self.bursts = []
        self.frame_count = 0
        self.last_ghost_time = 0.0
        self.powerup_until = 0.0
        self.fright_until = 0.0   # sacudida de camara (fantasma)
        self.flash_until = 0.0    # flash rojo breve (murcielago o fantasma)
        self.green_tint_until = 0.0  # tinte verde de humo de bruja (sombrero)
        self.start_time = time.time()

    def has_powerup(self):
        return time.time() < self.powerup_until

    def current_speed(self):
        return self.base_speed + (self.score // 10) * SPEED_INCREASE_PER_10_POINTS

    def maybe_spawn(self, width):
        if len(self.objects) >= MAX_OBJECTS_ON_SCREEN:
            return
        if self.frame_count % SPAWN_INTERVAL_FRAMES != 0:
            return

        kind = choose_kind()
        if kind == "ghost" and (time.time() - self.last_ghost_time) < GHOST_MIN_INTERVAL_SEC:
            kind = "pumpkin"
        if kind == "ghost":
            self.last_ghost_time = time.time()

        self.objects.append(FallingObject(kind, width, self.current_speed()))

    def update(self, width, height, hand_points, dt):
        """Devuelve una lista de eventos de sonido disparados este frame
        (por ejemplo ['pumpkin'] o ['ghost']), para que quien llame decida
        como reproducirlos."""
        events = []
        self.frame_count += 1
        self.maybe_spawn(width)

        for burst in self.bursts:
            burst.update(dt)
        self.bursts = [b for b in self.bursts if not b.finished()]

        remaining = []
        for obj in self.objects:
            obj.update(dt)

            bonus = POWERUP_CATCH_BONUS if (self.has_powerup() and obj.kind in ("pumpkin", "witch_hat")) else 0
            if obj.touches(hand_points, bonus):
                if obj.kind == "pumpkin":
                    self.score += 1
                    events.append("pumpkin")
                elif obj.kind == "witch_hat":
                    self.score += 3
                    self.powerup_until = time.time() + POWERUP_DURATION_SEC
                    self.bursts.append(Burst(obj.x, obj.y, count=24, smoke_count=9))
                    self.green_tint_until = time.time() + 0.5
                    events.append("witch_hat")
                    events.append("witch_cackle")
                elif obj.kind == "vampire":
                    self.misses += 1
                    self.flash_until = time.time() + 0.18
                    events.append("vampire")
                elif obj.kind == "ghost":
                    self.misses += 1
                    self.flash_until = time.time() + 0.25
                    self.fright_until = time.time() + 0.35
                    events.append("ghost")
                continue

            if obj.y - obj.size > height:
                continue

            remaining.append(obj)
        self.objects = remaining
        return events

    def draw(self, img):
        for obj in self.objects:
            obj.draw(img)
        for burst in self.bursts:
            burst.draw(img)

        draw_text(img, f"Puntaje: {self.score}", (20, 40), scale=1.0, color=(0, 200, 255))
        draw_text(img, f"Sustos: {self.misses}", (20, 75), scale=0.7, color=(180, 180, 180))

        if self.has_powerup():
            remaining = max(0.0, self.powerup_until - time.time())
            draw_text(img, f"MAGIA DEL SOMBRERO: {remaining:0.1f}s", (20, 105), scale=0.75, color=(255, 0, 255))

        if time.time() < self.flash_until:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 160), -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)

        if time.time() < self.green_tint_until:
            remaining = self.green_tint_until - time.time()
            strength = min(0.35, remaining / 0.5 * 0.35)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), COLOR_WITCH_SMOKE, -1)
            cv2.addWeighted(overlay, strength, img, 1 - strength, 0, dst=img)


# ----------------------------------------------------------------------------
# Esqueleto de mano (estilo hueso) dibujado sobre el lienzo de proyeccion
# ----------------------------------------------------------------------------
def draw_bone_skeleton(img, hand_landmarks, connections, width, height):
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]
    for (a, b) in connections:
        p1, p2 = pts[a], pts[b]
        cv2.line(img, p1, p2, SKELETON_BONE_OUTLINE, SKELETON_BONE_THICKNESS + 3, cv2.LINE_AA)
        cv2.line(img, p1, p2, SKELETON_BONE_COLOR, SKELETON_BONE_THICKNESS, cv2.LINE_AA)
        cv2.circle(img, p1, SKELETON_BONE_THICKNESS // 2 + 1, SKELETON_BONE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, p2, SKELETON_BONE_THICKNESS // 2 + 1, SKELETON_BONE_COLOR, -1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, SKELETON_JOINT_RADIUS + 2, SKELETON_BONE_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(img, p, SKELETON_JOINT_RADIUS, SKELETON_BONE_COLOR, -1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Audio: sintesis simple con numpy + reproduccion mezclada con sounddevice
# ----------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 44100

# Calibracion de niveles: la musica de fondo real suena mas fuerte que los
# efectos sintetizados, asi que se mezcla mas bajo para que todo lo demas
# se escuche por encima.
MUSIC_GAIN = 0.30
SFX_GAIN = 1.4
AMBIENT_LAYER_GAIN = 1.2       # aullidos / risa de bruja (poco frecuentes)
SYNTH_AMBIENT_GAIN = 1.0       # zumbido+latido sintetizado, solo si no hay musica real
DEFAULT_VOLUME_PERCENT = 70

BACKGROUND_MUSIC_FILENAME = "background_music.wav"


def load_background_music(target_sample_rate=AUDIO_SAMPLE_RATE):
    """Busca background_music.wav junto a este script y lo carga como un
    arreglo mono float32 (sin librerias externas, solo el modulo 'wave' de
    Python). Si no esta el archivo, o no se puede leer, devuelve None y el
    juego usa su propia ambientacion sintetizada como respaldo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, BACKGROUND_MUSIC_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            print(f"Aviso: {BACKGROUND_MUSIC_FILENAME} tiene un formato no soportado, uso ambiente sintetizado.")
            return None

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        if framerate != target_sample_rate and len(data) > 1:
            duration = len(data) / framerate
            n_target = max(1, int(duration * target_sample_rate))
            x_old = np.linspace(0, duration, len(data), endpoint=False)
            x_new = np.linspace(0, duration, n_target, endpoint=False)
            data = np.interp(x_new, x_old, data)

        return data.astype(np.float32)
    except Exception as exc:
        print(f"Aviso: no se pudo leer {BACKGROUND_MUSIC_FILENAME} ({exc}), uso ambiente sintetizado.")
        return None


def _tone(freq, duration, amp=0.35, sample_rate=AUDIO_SAMPLE_RATE, fade=0.01):
    n = max(1, int(sample_rate * duration))
    t = np.linspace(0, duration, n, endpoint=False)
    s = np.sin(2 * np.pi * freq * t)
    n_fade = max(1, int(sample_rate * fade))
    n_fade = min(n_fade, n // 2) if n // 2 > 0 else 1
    env = np.ones(n, dtype=np.float32)
    env[:n_fade] = np.linspace(0, 1, n_fade)
    env[-n_fade:] = np.linspace(1, 0, n_fade)
    return (s * env * amp).astype(np.float32)


def _sweep(f_start, f_end, duration, amp=0.35, sample_rate=AUDIO_SAMPLE_RATE, fade=0.015):
    n = max(1, int(sample_rate * duration))
    t = np.linspace(0, duration, n, endpoint=False)
    freq_t = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(freq_t) / sample_rate
    s = np.sin(phase)
    n_fade = max(1, min(int(sample_rate * fade), n // 2 if n // 2 > 0 else 1))
    env = np.ones(n, dtype=np.float32)
    env[:n_fade] = np.linspace(0, 1, n_fade)
    env[-n_fade:] = np.linspace(1, 0, n_fade)
    return (s * env * amp).astype(np.float32)


def _noise_burst(duration, amp=0.2, sample_rate=AUDIO_SAMPLE_RATE, fade=0.02):
    n = max(1, int(sample_rate * duration))
    s = (np.random.rand(n).astype(np.float32) * 2 - 1)
    n_fade = max(1, min(int(sample_rate * fade), n // 2 if n // 2 > 0 else 1))
    env = np.ones(n, dtype=np.float32)
    env[:n_fade] = np.linspace(0, 1, n_fade)
    env[-n_fade:] = np.linspace(1, 0, n_fade)
    return s * env * amp


def _concat(*arrays):
    return np.concatenate(arrays) if arrays else np.zeros(1, dtype=np.float32)


def _mix_add(*arrays):
    n = max(len(a) for a in arrays)
    out = np.zeros(n, dtype=np.float32)
    for a in arrays:
        out[:len(a)] += a
    return out


def _ghost_howl(sample_rate=AUDIO_SAMPLE_RATE):
    """Aullido largo de fantasma: el tono sube y despues baja como un arco,
    con vibrato y un poco de aire (ruido suave) para que no suene a silbido puro."""
    dur = 2.4
    n = int(sample_rate * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    freq_curve = 180 + 220 * np.sin(np.pi * t / dur)
    vibrato = 1 + 0.04 * np.sin(2 * np.pi * 6 * t)
    phase = 2 * np.pi * np.cumsum(freq_curve * vibrato) / sample_rate
    s = np.sin(phase) * 0.4
    breath = (np.random.rand(n).astype(np.float32) * 2 - 1) * 0.05
    n_fade = int(sample_rate * 0.15)
    env = np.ones(n, dtype=np.float32)
    env[:n_fade] = np.linspace(0, 1, n_fade)
    env[-n_fade:] = np.linspace(1, 0, n_fade)
    return ((s + breath) * env).astype(np.float32)


def _wolf_howl(sample_rate=AUDIO_SAMPLE_RATE):
    """Aullido de lobo: sube rapido, se sostiene arriba, y despues baja
    lento -- el contorno clasico de un aullido."""
    dur = 2.8
    n = int(sample_rate * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rise = np.clip(t / 0.4, 0, 1)
    decay = np.clip(1 - (t - 0.4) / (dur - 0.4), 0, 1)
    shape = np.where(t < 0.4, rise, decay)
    freq = 220 + 340 * shape
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 4 * t)
    phase = 2 * np.pi * np.cumsum(freq * vibrato) / sample_rate
    s = np.sin(phase) * 0.4
    n_fade = int(sample_rate * 0.1)
    env = np.ones(n, dtype=np.float32)
    env[:n_fade] = np.linspace(0, 1, n_fade)
    env[-n_fade:] = np.linspace(1, 0, n_fade)
    return (s * env).astype(np.float32)


def _witch_cackle(sample_rate=AUDIO_SAMPLE_RATE):
    """Risa de bruja tipo 'ja-ja-ja': una serie de tonos cortos descendentes
    con un poco de textura ruidosa para que suene a risa y no a xilofono."""
    parts = []
    base_freq = 520
    for i in range(5):
        f = base_freq - i * 18
        seg = _tone(f, 0.085, amp=0.28, sample_rate=sample_rate, fade=0.012)
        noise = _noise_burst(0.085, amp=0.07, sample_rate=sample_rate, fade=0.012)
        parts.append((seg + noise[:len(seg)]).astype(np.float32))
        parts.append(np.zeros(int(sample_rate * 0.035), dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _punch(amp=0.4, duration=0.025, sample_rate=AUDIO_SAMPLE_RATE):
    """Un click seco de impacto (ruido con caida muy rapida) para darle
    'pegada' al inicio de un efecto -- lo hace sonar mas exagerado/arcade."""
    n = max(1, int(sample_rate * duration))
    t = np.linspace(0, duration, n, endpoint=False)
    click = (np.random.rand(n).astype(np.float32) * 2 - 1)
    env = np.exp(-t * 90)
    return (click * env * amp).astype(np.float32)


def _heartbeat_loop(sample_rate=AUDIO_SAMPLE_RATE, dur=4.0):
    """Latido grave y lento de fondo -- dos golpes de corazon por ciclo,
    para sumar tension/terror a la ambientacion."""
    n = int(sample_rate * dur)
    sig = np.zeros(n, dtype=np.float32)
    for bt in (0.3, 0.75, 2.3, 2.75):
        idx = int(bt * sample_rate)
        dur_b = int(0.14 * sample_rate)
        if idx + dur_b < n:
            tb = np.linspace(0, 0.14, dur_b, endpoint=False)
            thump = np.sin(2 * np.pi * 50 * tb) * np.exp(-tb * 16)
            sig[idx:idx + dur_b] += thump
    return (sig * 0.5).astype(np.float32)


def build_sound_library(sample_rate=AUDIO_SAMPLE_RATE):
    lib = {}
    # efectos de atrapar/golpear: mas exagerados (mas fuertes, con un click
    # de impacto al inicio, tipo arcade) para que se sientan mas festejados
    lib["pumpkin"] = _mix_add(
        _punch(0.3, 0.02, sample_rate),
        _concat(_tone(523, 0.08, 0.5, sample_rate), _tone(784, 0.06, 0.45, sample_rate),
                _tone(1046, 0.12, 0.45, sample_rate)),
    )
    lib["vampire"] = _mix_add(_sweep(320, 110, 0.20, 0.5, sample_rate), _punch(0.35, 0.02, sample_rate))
    lib["ghost"] = _mix_add(
        _sweep(380, 80, 0.55, 0.5, sample_rate),
        _noise_burst(0.12, 0.35, sample_rate),
        _punch(0.4, 0.02, sample_rate),
    )
    notes = [523, 659, 784, 1046, 1318, 1568]
    arpeggio = _concat(*[_tone(f, 0.09, 0.42, sample_rate) for f in notes])
    lib["witch_hat"] = _mix_add(
        arpeggio, _noise_burst(len(arpeggio) / sample_rate, 0.08, sample_rate), _punch(0.35, 0.02, sample_rate)
    )
    lib["select"] = _tone(700, 0.06, 0.25, sample_rate)
    lib["start"] = _concat(_tone(392, 0.08, 0.3, sample_rate), _tone(523, 0.08, 0.3, sample_rate),
                            _tone(659, 0.12, 0.3, sample_rate))

    # capas de ambiente Halloween: fantasmas aullando, algun lobo de vez en
    # cuando, y risas de bruja -- se disparan solas de tanto en tanto (ver main())
    lib["ghost_howl"] = _ghost_howl(sample_rate)
    lib["wolf_howl"] = _wolf_howl(sample_rate)
    lib["witch_cackle"] = _witch_cackle(sample_rate)

    # ambientacion de fondo: zumbido grave + latido de corazon tenebroso
    dur = 4.0
    n = int(sample_rate * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    drone = (np.sin(2 * np.pi * 55 * t) + np.sin(2 * np.pi * 57 * t)) * 0.5
    tremolo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * t)
    amb = (drone * tremolo * 0.10).astype(np.float32)
    n_fade = int(sample_rate * 0.05)
    amb[:n_fade] *= np.linspace(0, 1, n_fade)
    amb[-n_fade:] *= np.linspace(1, 0, n_fade)
    heartbeat = _heartbeat_loop(sample_rate, dur)
    lib["ambient"] = _mix_add(amb, heartbeat)
    return lib


class AudioEngine:
    """Mezclador simple de audio: permite reproducir varios sonidos cortos
    superpuestos a la vez que un loop ambiental, sin archivos externos.
    Si el dispositivo de audio no esta disponible, se desactiva solo y el
    juego sigue funcionando en silencio."""

    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.enabled = _HAS_SOUNDDEVICE
        self.master_volume = 1.0  # 0.0 a 1.0, ajustable desde el menu de inicio
        self._lock = threading.Lock()
        self._clips = []
        self.stream = None
        if self.enabled:
            try:
                self.stream = sd.OutputStream(
                    channels=1, samplerate=sample_rate, dtype="float32", callback=self._callback
                )
                self.stream.start()
            except Exception as exc:
                print("Audio deshabilitado (no se encontro salida de audio):", exc)
                self.enabled = False
                self.stream = None

    def _callback(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, dtype=np.float32)
        master = self.master_volume
        with self._lock:
            still_active = []
            for clip in self._clips:
                data = clip["data"]
                pos = clip["pos"]
                loop = clip["loop"]
                gain = clip["gain"] * master
                remaining = frames
                out_idx = 0
                while remaining > 0:
                    avail = len(data) - pos
                    take = min(avail, remaining)
                    if take > 0:
                        buf[out_idx:out_idx + take] += data[pos:pos + take] * gain
                    pos += take
                    out_idx += take
                    remaining -= take
                    if pos >= len(data):
                        if loop:
                            pos = 0
                        else:
                            break
                if loop or pos < len(data):
                    clip["pos"] = pos
                    still_active.append(clip)
            self._clips = still_active
        np.clip(buf, -1.0, 1.0, out=buf)
        outdata[:, 0] = buf

    def play(self, data, gain=1.0):
        if not self.enabled or data is None:
            return
        with self._lock:
            self._clips.append({"data": data.astype(np.float32), "pos": 0, "loop": False, "gain": gain})

    def play_loop(self, data, gain=1.0):
        if not self.enabled or data is None:
            return None
        clip = {"data": data.astype(np.float32), "pos": 0, "loop": True, "gain": gain}
        with self._lock:
            self._clips.append(clip)
        return clip

    def stop_loop(self, clip):
        if not self.enabled or clip is None:
            return
        with self._lock:
            if clip in self._clips:
                self._clips.remove(clip)

    def close(self):
        if self.enabled and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Pantalla de inicio: eleccion de velocidad (solo se muestra una vez)
# ----------------------------------------------------------------------------
def _draw_menu_icon(canvas, kind, x, y, size, phase):
    frames = get_sprite_frames(kind)
    if frames:
        idx = int(phase / ANIM_PHASE_STEP) % len(frames)
        paste_sprite(canvas, frames[idx], x, y, size)
    elif kind == "pumpkin":
        draw_pumpkin(canvas, x, y, size)
    elif kind == "witch_hat":
        draw_witch_hat(canvas, x, y, size, phase)
    elif kind == "vampire":
        draw_bat(canvas, x, y, size, phase)


def run_start_menu(window_name, audio, sounds, background_img):
    selected = 1  # "Normal" por defecto
    audio.master_volume = DEFAULT_VOLUME_PERCENT / 100.0
    while True:
        volume_pct = int(round(audio.master_volume * 100))
        canvas = background_img.copy()
        draw_text(canvas, "JUEGO HALLOWEEN", (CAM_WIDTH // 2 - 194, 110),
                  scale=1.3, color=(0, 140, 255), thickness=3)
        t = time.time()
        _draw_menu_icon(canvas, "pumpkin", CAM_WIDTH // 2, 210, 70, t * 4)
        _draw_menu_icon(canvas, "witch_hat", CAM_WIDTH // 2 - 170, 220, 65, t * 2)
        _draw_menu_icon(canvas, "vampire", CAM_WIDTH // 2 + 170, 200, 65, t * 4)

        draw_text(canvas, "Elegi la velocidad de caida:", (CAM_WIDTH // 2 - 230, 330),
                  scale=0.9, color=(255, 255, 255))
        for i, (label, _speed) in enumerate(SPEED_PRESETS):
            y = 390 + i * 65
            is_sel = (i == selected)
            color = (0, 215, 255) if is_sel else (150, 150, 150)
            marker = ">" if is_sel else " "
            draw_text(canvas, f"{marker} {i + 1}) {label}", (CAM_WIDTH // 2 - 140, y), scale=0.95, color=color)

        # control de volumen, con barra visual
        vol_y = 610
        draw_text(canvas, f"Volumen: {volume_pct}%  (- / + para ajustar)",
                  (CAM_WIDTH // 2 - 200, vol_y), scale=0.75, color=(255, 255, 255))
        bar_x0, bar_y0 = CAM_WIDTH // 2 - 150, vol_y + 15
        bar_w, bar_h = 300, 16
        cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (90, 90, 90), -1, cv2.LINE_AA)
        fill_w = int(bar_w * volume_pct / 100)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y0 + bar_h), (0, 165, 255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (200, 200, 200), 1, cv2.LINE_AA)
        if not audio.enabled:
            draw_text(canvas, "(sin salida de audio disponible en esta PC)",
                      (CAM_WIDTH // 2 - 220, vol_y + 55), scale=0.55, color=(150, 150, 150))

        draw_text(canvas, "Presiona 1, 2 o 3 para elegir y empezar (Enter = Normal).",
                  (CAM_WIDTH // 2 - 430, CAM_HEIGHT - 60), scale=0.6, color=(180, 180, 180))
        draw_text(canvas, "Esta pantalla no se muestra durante el juego.",
                  (CAM_WIDTH // 2 - 320, CAM_HEIGHT - 30), scale=0.55, color=(120, 120, 120))

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("-"), ord("_")):
            audio.master_volume = max(0.0, audio.master_volume - 0.1)
        if key in (ord("+"), ord("=")):
            audio.master_volume = min(1.0, audio.master_volume + 0.1)
        if key in (ord("1"), ord("2"), ord("3")):
            selected = int(chr(key)) - 1
            audio.play(sounds.get("start"), gain=SFX_GAIN)
            return SPEED_PRESETS[selected][1]
        if key in (13, 10):  # Enter
            audio.play(sounds.get("start"), gain=SFX_GAIN)
            return SPEED_PRESETS[selected][1]
        if key in (ord("q"), ord("Q"), 27):
            return None


# ----------------------------------------------------------------------------
# Camara
# ----------------------------------------------------------------------------
def open_camera(index=0):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    return cap


def main():
    window_name = "Juego Halloween"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    audio = AudioEngine()
    sounds = build_sound_library() if audio.enabled else {}
    music_data = load_background_music() if audio.enabled else None
    background_img = load_background(CAM_WIDTH, CAM_HEIGHT)

    if music_data is not None:
        ambient_source, ambient_gain = music_data, MUSIC_GAIN
    else:
        ambient_source, ambient_gain = sounds.get("ambient"), SYNTH_AMBIENT_GAIN
    ambient_clip = audio.play_loop(ambient_source, gain=ambient_gain) if audio.enabled else None

    base_speed = run_start_menu(window_name, audio, sounds, background_img)
    if base_speed is None:
        audio.stop_loop(ambient_clip)
        cv2.destroyAllWindows()
        audio.close()
        return

    cap = open_camera(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara. Verifica que este conectada y no este en uso por otra app.")
        audio.stop_loop(ambient_clip)
        cv2.destroyAllWindows()
        audio.close()
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_drawing_utils = mp.solutions.drawing_utils  # solo para HAND_CONNECTIONS, no para el estilo de dibujo

    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    next_ghost_howl = time.time() + random.uniform(12, 25)
    next_wolf_howl = time.time() + random.uniform(30, 55)
    next_witch_cackle = time.time() + random.uniform(20, 40)

    game = Game(base_speed)
    fullscreen = False
    prev_time = time.time()
    hand_was_visible = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: no se pudo leer el frame de la camara.")
                break

            now = time.time()
            dt = max(0.001, min(0.05, now - prev_time))
            prev_time = now

            if audio.enabled:
                if now >= next_ghost_howl:
                    audio.play(sounds.get("ghost_howl"), gain=AMBIENT_LAYER_GAIN)
                    next_ghost_howl = now + random.uniform(18, 35)
                if now >= next_wolf_howl:
                    audio.play(sounds.get("wolf_howl"), gain=AMBIENT_LAYER_GAIN)
                    next_wolf_howl = now + random.uniform(40, 75)
                if now >= next_witch_cackle:
                    audio.play(sounds.get("witch_cackle"), gain=AMBIENT_LAYER_GAIN)
                    next_witch_cackle = now + random.uniform(28, 55)

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            if background_img.shape[:2] != (height, width):
                background_img = load_background(width, height)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            canvas = background_img.copy()

            hand_points = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    draw_bone_skeleton(canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS, width, height)
                    for idx in CATCH_LANDMARKS:
                        lm = hand_landmarks.landmark[idx]
                        hand_points.append((lm.x * width, lm.y * height))

            hand_visible_now = len(hand_points) > 0
            if hand_visible_now and not hand_was_visible and audio.enabled:
                audio.play(sounds.get("ghost"), gain=SFX_GAIN)
            hand_was_visible = hand_visible_now

            events = game.update(width, height, hand_points, dt)
            for ev in events:
                audio.play(sounds.get(ev), gain=SFX_GAIN)

            game.draw(canvas)

            draw_text(canvas, "Q: salir   R: reiniciar puntaje   F: pantalla completa",
                      (20, height - 20), scale=0.6, color=(200, 200, 200))

            if now < game.fright_until:
                shake = 10
                m = np.float32([[1, 0, random.randint(-shake, shake)], [0, 1, random.randint(-shake, shake)]])
                canvas = cv2.warpAffine(canvas, m, (width, height), borderValue=(10, 6, 4))

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                game.reset()
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
    finally:
        audio.stop_loop(ambient_clip)
        audio.close()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
