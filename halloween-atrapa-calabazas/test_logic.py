"""Prueba headless: valida imports, dibujo, logica del juego (vidas
infinitas, sombrero de bruja + particulas), y la sintesis/motor de audio,
sin necesidad de camara, pantalla ni parlantes reales."""
import sys
import time
import numpy as np
import cv2
import mediapipe as mp

sys.path.insert(0, ".")
import game as g

print("Imports OK. cv2:", cv2.__version__, "| mediapipe:", mp.__version__)

frame = np.zeros((720, 1280, 3), dtype=np.uint8)

# 1) Dibujo de cada tipo de objeto no debe tirar excepciones
g.draw_pumpkin(frame, 200, 200, 40)
g.draw_bat(frame, 400, 200, 40, wing_phase=1.2)
g.draw_ghost(frame, 600, 200, 60, alpha=0.8, bob_phase=0.5)
g.draw_witch_hat(frame, 800, 200, 85, wobble=0.4)
g.draw_mini_star(frame, 100, 100, 12, 20, (0, 165, 255))
g.draw_mini_frog(frame, 150, 100, 12)
g.draw_mini_potion(frame, 200, 100, 12, (200, 30, 200))
print("Dibujo de calabaza / murcielago / fantasma / sombrero / particulas: OK")

width, height = 1280, 720

# 2) Atrapar una calabaza suma puntaje
game = g.Game(base_speed=4.2)
obj = g.FallingObject("pumpkin", width, 0)
obj.x, obj.y = 640, 300
game.objects.append(obj)
events = game.update(width, height, [(640, 300)], dt=0.016)
assert game.score == 1, f"Se esperaba score=1, dio {game.score}"
assert events == ["pumpkin"], f"Se esperaba evento ['pumpkin'], dio {events}"
print("Atrapar calabaza: OK (score=1, evento correcto)")

# 3) Tocar un vampirito NO debe terminar el juego (vidas infinitas)
game2 = g.Game(base_speed=4.2)
for _ in range(10):  # muchos golpes seguidos
    vampire = g.FallingObject("vampire", width, 0)
    vampire.x, vampire.y = 640, 300
    game2.objects.append(vampire)
    events2 = game2.update(width, height, [(640, 300)], dt=0.016)
assert game2.misses == 10, f"Se esperaban 10 sustos, dio {game2.misses}"
assert not hasattr(game2, "game_over"), "No deberia existir mas el concepto de game_over"
print("Vidas infinitas: OK (10 sustos con vampiritos, el juego sigue)")

# 4) Tocar el fantasma tambien es infinito, y dispara la sacudida de camara
game3 = g.Game(base_speed=4.2)
ghost = g.FallingObject("ghost", width, 0)
ghost.x, ghost.y = 640, 300
game3.objects.append(ghost)
events3 = game3.update(width, height, [(640, 300)], dt=0.016)
assert game3.misses == 1
assert game3.fright_until > time.time()
assert events3 == ["ghost"]
print("Fantasma: OK (susto infinito + sacudida de camara + evento de sonido)")

# 5) Atrapar el sombrero de bruja activa el power-up y crea una explosion de particulas
game4 = g.Game(base_speed=4.2)
hat = g.FallingObject("witch_hat", width, 0)
hat.x, hat.y = 640, 300
game4.objects.append(hat)
events4 = game4.update(width, height, [(640, 300)], dt=0.016)
assert game4.has_powerup(), "El power-up deberia estar activo"
assert game4.score == 3
assert len(game4.bursts) == 1 and len(game4.bursts[0].particles) == 33, \
    f"Se esperaban 33 particulas (24+9 humo), dio {len(game4.bursts[0].particles)}"
assert game4.green_tint_until > time.time(), "Deberia activarse el tinte verde"
assert events4 == ["witch_hat", "witch_cackle"], f"Se esperaban 2 eventos de sonido, dio {events4}"
print("Sombrero de bruja: OK (power-up activo, score=3, 33 particulas con humo verde, tinte verde, 2 sonidos)")

# 6) Las particulas se actualizan y mueren con el tiempo
burst = game4.bursts[0]
for _ in range(200):
    burst.update(0.02)
assert burst.finished(), "Las particulas deberian haber expirado"
print("Particulas: OK (expiran correctamente)")

# 7) Render de HUD (con power-up y sustos) no debe tirar excepciones
game4.draw(frame)
print("Render de HUD: OK")

# 8) Sintesis de audio: cada sonido generado debe ser un array float32 valido en rango [-1,1]
lib = g.build_sound_library()
for name, data in lib.items():
    assert isinstance(data, np.ndarray), f"{name} no es un ndarray"
    assert data.dtype == np.float32, f"{name} no es float32"
    assert np.all(np.isfinite(data)), f"{name} tiene valores no finitos"
    assert data.max() <= 1.01 and data.min() >= -1.01, f"{name} fuera de rango"
    assert len(data) > 0
for ambient_name in ("ghost_howl", "wolf_howl", "witch_cackle", "ambient"):
    assert ambient_name in lib, f"falta el sonido ambiental {ambient_name}"
print("Sintesis de sonido: OK ->", list(lib.keys()))

# 9) El motor de audio no debe romper el juego aunque no haya dispositivo de sonido
engine = g.AudioEngine()
engine.play(lib["pumpkin"], gain=g.SFX_GAIN)  # no debe tirar excepcion, este container no tiene parlantes reales
clip = engine.play_loop(lib["ambient"], gain=g.SYNTH_AMBIENT_GAIN)
engine.stop_loop(clip)
print(f"AudioEngine: OK (enabled={engine.enabled}, no rompe el juego sin dispositivo de audio)")

# 10) Volumen maestro: se puede ajustar y queda dentro de rango
assert engine.master_volume == 1.0, "el volumen maestro deberia arrancar en 1.0"
engine.master_volume = 0.7
assert abs(engine.master_volume - 0.7) < 1e-6
engine.close()
print("Volumen maestro: OK (ajustable)")

# 11) Musica de fondo real (background_music.wav) si esta presente junto al script
music = g.load_background_music()
if music is not None:
    assert isinstance(music, np.ndarray) and music.dtype == np.float32
    assert np.all(np.isfinite(music))
    assert music.max() <= 1.01 and music.min() >= -1.01
    print(f"Musica de fondo: OK (background_music.wav cargado, {len(music)/g.AUDIO_SAMPLE_RATE:.1f}s)")
else:
    print("Musica de fondo: no hay background_music.wav junto al script (se usaria el ambiente sintetizado)")

print("\nTODAS LAS PRUEBAS PASARON CORRECTAMENTE")
