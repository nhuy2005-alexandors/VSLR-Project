@echo off
chcp 65001 >nul
title VSLR - Tai Video Test tu AWS & Hugging Face ve may

echo ====================================================================
echo    TAI TOAN BO VIDEO TEST TU SERVER AWS & HUGGING FACE VE MAY
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
echo Da tai xong! Dang mo thu muc videos...
echo ====================================================================
if exist "videos" (
    start "" "videos"
)
pause
