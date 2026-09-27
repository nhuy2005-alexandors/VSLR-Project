@echo off
chcp 65001 >nul
title VSLR - Day Backend len Hugging Face Space (Tu dong 1-Click)

echo ====================================================================
echo    TU DONG DAY BACKEND VSLR LEN HUGGING FACE SPACE (ZERO GPU FREE)
echo    - Space dich:   ntbii305/vslr-backend
echo    - Dataset dich: ntbii305/vslr-remote
echo ====================================================================
echo.

cd /d "%~dp0"

:: Kiem tra moi truong ao venv
if exist "%CD%\venv\Scripts\python.exe" (
    set "PY_BIN=%CD%\venv\Scripts\python.exe"
) else if exist "python.exe" (
    set "PY_BIN=python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" scripts\deploy_hf_space.py

echo.
pause
