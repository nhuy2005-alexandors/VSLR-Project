@echo off
chcp 65001 >nul
title VSLR - Chay Public Cloudflare Tunnel (30 FPS Muot Nhu Localhost)

echo ====================================================================
echo    VSLR: BAT SERVER PUBLIC CLOUDFLARE TUNNEL (30 FPS - 0 LAG)
echo    - Toc do: Ping noi dia 5-15ms, muot 100%% nhu localhost
echo    - TTS: VieNeu-TTS Truc Ly 48kHz
echo    - Tu dong luu: O cung local + Hugging Face Dataset
echo ====================================================================
echo.

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

:: Kiem tra moi truong ao venv
if exist "%CD%\venv\Scripts\python.exe" (
    set "PY_BIN=%CD%\venv\Scripts\python.exe"
) else if exist "python.exe" (
    set "PY_BIN=python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" scripts\start_cloudflare_tunnel.py

echo.
echo Server da dung.
pause
