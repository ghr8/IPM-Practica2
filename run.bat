@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python race3d.py %*
pause
