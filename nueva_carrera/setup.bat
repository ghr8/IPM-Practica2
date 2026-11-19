@echo off
rem Instalacion en Windows: crea el entorno virtual e instala dependencias.
cd /d "%~dp0"
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
python download_model.py
echo.
echo Instalacion completada. Ejecuta: run.bat
pause
exit /b 0
:error
echo ERROR: comprueba que Python 3.10-3.12 esta instalado y en el PATH.
pause
exit /b 1
