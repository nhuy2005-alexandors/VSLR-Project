@echo off
chcp 65001 >nul
title VSLR - Chạy Public Cloudflare Tunnel (30 FPS Mượt Như Localhost)

echo ====================================================================
echo    VSLR: BẬT SERVER PUBLIC CLOUDFLARE TUNNEL (30 FPS - 0 LAG)
echo    - Tốc độ: Ping nội địa 5-15ms, mượt 100% như localhost
echo    - TTS: VieNeu-TTS Trúc Ly 48kHz
echo    - Tự động lưu: Ổ cứng local + Hugging Face Dataset (ntbii305/vslr-remote)
echo ====================================================================
echo.

cd /d "%~dp0"

:: Kiểm tra môi trường ảo venv
if exist "%CD%\venv\Scripts\python.exe" (
    set "PY_BIN=%CD%\venv\Scripts\python.exe"
) else if exist "python.exe" (
    set "PY_BIN=python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" scripts\start_cloudflare_tunnel.py

echo.
echo Server đã dừng.
pause
