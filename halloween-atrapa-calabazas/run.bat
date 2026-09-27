@echo off
setlocal enabledelayedexpansion

REM ================================================================
REM  Halloween: Atrapa Calabazas
REM  Instala dependencias (en un entorno virtual local) y ejecuta el juego.
REM ================================================================

cd /d "%~dp0"

echo.
echo === Verificando Python ===
where python >nul 2>&1
if errorlevel 1 (
    echo No se encontro Python en el PATH.
    echo Instala Python 3.9 a 3.12 de 64 bits desde https://www.python.org/downloads/
    echo IMPORTANTE: al instalar, marca la casilla "Add python.exe to PATH".
    pause
    exit /b 1
)

echo.
echo === Creando entorno virtual (venv) si no existe ===
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

call "venv\Scripts\activate.bat"

echo.
echo === Instalando/actualizando dependencias ===
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Hubo un error instalando las dependencias. Revisa el mensaje de arriba.
    echo Si el error menciona mediapipe, verifica que estes usando Python 3.9 a 3.12
    echo de 64 bits - mediapipe no soporta todas las versiones de Python.
    pause
    exit /b 1
)

echo.
echo === Iniciando el juego ===
echo Controles: Q para salir, R para reiniciar, F para pantalla completa.
echo.
python game.py

echo.
echo Juego cerrado.
pause
