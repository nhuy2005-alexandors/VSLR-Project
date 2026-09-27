@echo off
chcp 65001 >nul
title VSLR - Tai Video Test tu Hugging Face Dataset ve may

echo ====================================================================
echo    TAI TOAN BO VIDEO TEST TU HUGGING FACE DATASET VE PHAN TICH
echo    - Dataset: ntbii305/vslr-remote
echo    - Thu muc luu: videos_from_huggingface/
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

"%PY_BIN%" scripts\download_hf_dataset.py --signer "khach"

echo.
echo ====================================================================
echo Da tai xong! Dang mo thu muc chua video...
echo ====================================================================
if exist "videos_from_huggingface\data\khach" (
    start "" "videos_from_huggingface\data\khach"
) else (
    start "" "videos_from_huggingface"
)
pause
