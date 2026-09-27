# Halloween: Atrapa Calabazas

Juego de camara pensado para proyectar sobre una pared: usa tus manos frente
a la camara (se ve como un **esqueleto de hueso** sobre fondo negro) para
**atrapar calabazas**, **evitar murcielagos** y **al fantasma**, y **atrapar
el sombrero de bruja magico** para conseguir un power-up.

## Requisitos

- Windows 10/11 de 64 bits.
- **Python 3.9 a 3.12** de 64 bits instalado y agregado al PATH
  (https://www.python.org/downloads/ — al instalar, marca la casilla
  "Add python.exe to PATH"). MediaPipe todavia no soporta Python 3.13+.
- Una camara web conectada y libre (que no este siendo usada por Zoom, Teams,
  otra app, etc.).
- Parlantes o salida de audio (opcional): el juego tiene sonido, pero si no
  hay salida de audio disponible sigue funcionando normalmente, en silencio.
- Un proyector o pantalla grande para proyectar en la pared.

## Instalacion y ejecucion

1. Descomprimi esta carpeta en tu computadora.
2. Hace doble clic en **`run.bat`**.
   - La primera vez va a crear un entorno virtual (`venv`) e instalar las
     dependencias (OpenCV, MediaPipe, NumPy, sounddevice). Puede tardar unos
     minutos.
   - Las siguientes veces arranca directo, mucho mas rapido.
3. Se abre primero una **pantalla de inicio** para elegir la velocidad
   (ver mas abajo). Esa pantalla solo aparece una vez, al arrancar.
4. Despues se abre la ventana del juego: fondo negro con el **esqueleto de
   tus manos** en blanco hueso, siguiendo tu movimiento en tiempo real. No
   se muestra el video real de la camara, solo el esqueleto y el juego —
   ideal para proyectar sin mostrar tu habitacion.

## Pantalla de inicio

Al arrancar, elegi la velocidad de caida y el volumen con el teclado:

| Tecla | Accion |
|---|---|
| `1` / `2` / `3` | Velocidad: Lenta / Normal / Rapida (arranca el juego) |
| `Enter` | Empieza en Normal |
| `-` / `+` | Bajar / subir el volumen (se escucha en vivo con la musica de fondo ya sonando) |

Esta pantalla no vuelve a aparecer durante el juego proyectado.

## Controles durante el juego

| Tecla | Accion |
|---|---|
| `Q` o `ESC` | Salir del juego |
| `R` | Reiniciar el puntaje (se puede usar en cualquier momento) |
| `F` | Pantalla completa (util para proyectar en la pared) |

## Como se juega

- 🎃 **Calabaza**: atrapala con la mano para sumar 1 punto (suena un "blip").
- 🧛 **Vampirito**: vuela y hace zigzag. Si lo tocas, te asusta (sonido +
  flash rojo breve) y suma al contador de **sustos**, pero **las vidas son
  infinitas**: nunca se termina el juego por esto.
- 👻 **Fantasma**: aparece cada tanto, flota y es traslucido. Si lo tocas,
  ademas del flash y el sonido, la pantalla se sacude un instante — pero
  tampoco termina el juego.
- 🎩 **Sombrero de bruja magico**: es grande y aparece rara vez. Atraparlo
  da 3 puntos, un power-up de **velocidad** (agranda el area con la que
  atrapas cosas) por 6 segundos, y dispara una animacion de **estrellas,
  sapitos, pociones y humo verde de bruja** brotando del sombrero (con toda
  la pantalla tiñendose de verde un instante), con su propio sonido magico
  MAS una risa de bruja.
- Cada vez que tus manos aparecen frente a la camara, suena un efecto de
  fantasma como saludo.
- Todos los objetos que caen son un 20% mas grandes (de nuevo) sobre su
  tamano anterior — incluido el sombrero esta vez.
- La velocidad de caida aumenta con el puntaje, para que se ponga mas
  dificil (y mas emocionante) con el tiempo.
- Como no hay "game over", el juego esta pensado para jugar sin limite,
  ideal para que se turnen varios chicos frente a la camara en una fiesta.

## Sonido

- **Musica de fondo real**: el juego usa `background_music.wav` (incluido)
  como ambientacion principal, en loop. Si ese archivo no esta, usa
  automaticamente una ambientacion sintetizada de respaldo (zumbido grave +
  latido de corazon).
- **Volumen**: se ajusta en la pantalla de inicio con `-` / `+`, y queda
  aplicado a todo el juego (musica y efectos).
- **Efectos calibrados**: los sonidos de atrapar/golpear estan mezclados
  mas fuerte que la musica de fondo, para que se escuchen bien por encima
  aunque la musica este sonando.
- **Capas que aparecen solas de vez en cuando**, ademas de la musica: un
  fantasma aullando, algun lobo (mas espaciado, de sorpresa) y una risa
  de bruja tipo "ja-ja-ja".
- Todos los efectos de calabaza/vampirito/fantasma/sombrero se generan en
  el momento con matematica simple (sintesis con numpy), sin archivos
  externos.

Si la maquina no tiene salida de sonido disponible, el juego lo detecta
solo y sigue funcionando en silencio, sin errores.

## Fondo de la escena

El juego usa `background.jpg` (incluido en esta carpeta) como fondo fijo.
Se carga una sola vez al arrancar y se reutiliza en cada frame, asi que no
consume rendimiento extra.

- **Si `background.jpg` no esta en la carpeta** (por ejemplo si lo borraste
  o lo moviste), el juego no se rompe: usa automaticamente una escena
  alternativa dibujada por codigo (luna, arboles, cementerio) como
  respaldo.
- **Para cambiar el fondo** por otra imagen tuya: reemplaza el archivo
  `background.jpg` por otra imagen con ese mismo nombre (cualquier tamano
  sirve, el juego la redimensiona sola a 1280x720).

## Personajes

Calabaza, fantasma, sombrero y vampirito son **animaciones 3D reales**
(12 frames cada uno, con transparencia), ubicadas en la carpeta `sprites/`.
Todos un 20% mas grandes, excepto el sombrero que quedo en su tamano
original.

- **Si falta la carpeta `sprites/` o algun archivo**, el juego no se
  rompe: dibuja esa parte a mano (version simplificada hecha en codigo)
  como respaldo.
- **Para cambiar algun personaje**: reemplaza los PNG numerados
  (`00.png` a `11.png`) dentro de la subcarpeta correspondiente
  (`sprites/pumpkin/`, `sprites/ghost/`, `sprites/witch_hat/`,
  `sprites/vampire/`) por tu propia secuencia de imagenes con
  transparencia.


## Nota tecnica: por que MediaPipe y no OpenPose

Se pidio originalmente usar OpenPose para el seguimiento. En la practica,
OpenPose no se instala con `pip install`: hay que compilarlo desde codigo
fuente con CMake y Visual Studio, suele requerir una GPU NVIDIA con
CUDA/cuDNN, y descargar varios GB de modelos por separado — algo inviable
para un simple `run.bat`. Por eso este juego usa **MediaPipe Hands** de
Google, que se instala con un simple `pip install mediapipe`, da 21 puntos
de referencia por mano y corre bien en CPU sin necesidad de GPU.

## Solucion de problemas

- **"No se encontro Python en el PATH"**: reinstala Python marcando la
  casilla "Add python.exe to PATH", o abri una terminal nueva despues de
  instalarlo.
- **Error instalando mediapipe**: verifica que tu Python sea de 64 bits y
  version 3.9 a 3.12 (`python --version` en una terminal).
- **"No se pudo abrir la camara"**: cerra otras apps que puedan estar
  usando la camara (Zoom, Teams, Camara de Windows, OBS, etc.) y volve a
  ejecutar `run.bat`.
- **No se escucha nada**: primero revisa el volumen DENTRO del juego
  (pantalla de inicio, tecla `+`) y el volumen de Windows; tambien que
  haya un dispositivo de salida de audio configurado como predeterminado.
  Si no hay ninguno, el juego sigue funcionando pero sin sonido (no es
  un error).
- **Se detectan mal las manos**: mejora la iluminacion frente a la camara;
  MediaPipe necesita ver bien el contorno de la mano.
- **Para reinstalar desde cero**: borra la carpeta `venv` (dentro de esta
  misma carpeta del juego) y volve a ejecutar `run.bat`.

## Archivos

- `game.py` — codigo del juego.
- `background.jpg` — imagen de fondo de la escena.
- `background_music.wav` — musica de fondo real (loop).
- `sprites/` — animaciones 3D de calabaza, fantasma, sombrero y vampirito
  (12 frames cada uno, con transparencia).
- `requirements.txt` — dependencias de Python.
- `run.bat` — instala dependencias (en un entorno virtual) y ejecuta el juego.
- `test_logic.py` — prueba automatica de la logica del juego (opcional,
  no hace falta para jugar).
