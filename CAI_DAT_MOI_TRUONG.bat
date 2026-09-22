@echo off
chcp 65001 >nul
title VSLR - Cai Dat Moi Truong Tu Dong (1-Click Setup)

echo ====================================================================
echo    HE THONG VSLR - CAI DAT MOI TRUONG TU DONG (1-CLICK SETUP)
echo    Nhan dien 24 cu chi ngon ngu ky hieu Viet Nam (MediaPipe + BiLSTM)
echo ====================================================================
echo.
cd /d "%~dp0"

echo [1/4] Kiem tra Python tren he thong...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [LOI] Khong tim thay Python tren may tinh!
    echo Vui long cai dat Python 3.10 hoac 3.11 tu https://www.python.org/downloads/
    echo LUU Y: Khi cai dat, nho tich vao o "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PY_VER=%%i
echo Tim thay: %PY_VER%

echo.
echo [2/4] Khoi tao moi truong ao (Virtual Environment: venv)...
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    if errorlevel 1 (
        echo [Canh bao] Khong the tao venv, he thong se su dung Python toan cuc.
    ) else (
        echo Da tao thanh cong thu muc moi truong ao venv!
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
echo [3/4] Cai dat cac goi thu vien can thiet (PyTorch, MediaPipe, OpenCV, FastAPI, TTS)...
echo Qua trinh nay co the mat tu 2 den 5 phut tuy toc do mang. Xin vui long cho...
"%PY_BIN%" -m pip install --upgrade pip
"%PY_BIN%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [LOI] Co loi xay ra trong qua trinh cai dat thu vien!
    pause
    exit /b 1
)

echo.
echo [4/4] Cai dat package he thong VSLR o che do editable...
"%PY_BIN%" -m pip install -e .

echo.
echo ====================================================================
echo    CHUC MUNG! CAI DAT MOI TRUONG HOAN TAT THANH CONG!
echo.
echo    De bat dau su dung, thay/co co the mo:
echo    - Ban Giao Dien Web (Khuyen nghi): Nhap dup vao CHAY_WEB.bat
echo    - Ban Cua So Webcam truc tiep:     Nhap dup vao CHAY_WEBCAM.bat
echo ====================================================================
echo.
pause
