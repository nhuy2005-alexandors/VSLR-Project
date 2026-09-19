@echo off
chcp 65001 >nul
title VSLR - Cai Dat Moi Truong Tu Dong

echo ====================================================================
echo    HE THONG VSLR - CAI DAT MOI TRUONG TU DONG (1-CLICK SETUP)
echo ====================================================================
echo.
cd /d "%~dp0"

echo [1/4] Kiem tra Python tren he thong...
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python! Vui long cai dat Python 3.10 hoac 3.11 tu python.org va nho tich vao "Add Python to PATH".
    pause
    exit /b 1
)
python --version

echo.
echo [2/4] Khoi tao moi truong ao (Virtual Environment: venv)...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo [Canh bao] Khong the tao venv, se dung truc tiep Python he thong.
    ) else (
        echo Da tao moi truong ao venv thanh cong!
    )
) else (
    echo Moi truong ao venv da ton tai san.
)

if exist "venv\Scripts\python.exe" (
    set "PY_BIN=%CD%\venv\Scripts\python.exe"
    set "PIP_BIN=%CD%\venv\Scripts\pip.exe"
) else (
    set "PY_BIN=python"
    set "PIP_BIN=pip"
)

echo.
echo [3/4] Nang cap pip va cai dat cac thu vien can thiet (co the mat vai phut)...
"%PIP_BIN%" install --upgrade pip
"%PIP_BIN%" install -r requirements.txt

echo.
echo [4/4] Cai dat goi vslr che do editable...
if exist "VSLR-Project-pipeline-signer-split\pyproject.toml" (
    "%PIP_BIN%" install -e VSLR-Project-pipeline-signer-split
) else if exist "pyproject.toml" (
    "%PIP_BIN%" install -e .
)

echo.
echo ====================================================================
echo    CAI DAT HOAN TAT! BAN CO THE BAT DAU SU DUNG HE THONG:
echo    - Chay ban Web: Nhap dup chuot vao CHAY_WEB.bat
echo    - Chay Webcam: Nhap dup chuot vao CHAY_CAMERA_WEBCAM.bat
echo    - Chay Camera ngoai (A16): Nhap dup chuot vao CHAY_CAMERA_A16.bat
echo ====================================================================
echo.
pause
