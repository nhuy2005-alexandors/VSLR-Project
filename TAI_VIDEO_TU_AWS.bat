@echo off
chcp 65001 >nul
title VSLR - Tai Toan Bo Video Nguoi Test Tu AWS Ve May

echo ====================================================================
echo    TU DONG TAI TOAN BO VIDEO TEST TU SERVER AWS VE MAY TINH
echo    - Thu muc luu: D:\1. Nguyễn Trường Thọ\NCKH\github Nghị gửi\VSLR-Project\videos
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

"%PY_BIN%" scripts\download_hf_dataset.py --output-dir "videos"

echo.
echo ====================================================================
echo Da dong bo thanh cong! Dang mo thu muc videos...
echo ====================================================================
if exist "videos" (
    start "" "videos"
)
pause
