@echo off
chcp 65001 >nul
title VSLR - Giao Dien Web Nhan Dien Cu Chi (Realtime Studio)

echo ====================================================================
echo    HE THONG NHAN DIEN CU CHI TIENG VIET THOI GIAN THUC (VSLR)
echo    - Mo hinh: BiLSTM Candidate V3 (24 cu chi tieng Viet)
echo    - Backend: MediaPipe Holistic C++ (Toi uu CPU / GPU 25-30 FPS)
echo    - Bo phat am: VieNeu-TTS / Pyttsx3 (Giong doc tieng Viet)
echo    - Tu dong ghi video: Bat (Luu vao thu muc videos/)
echo    - Dia chi Web: http://localhost:8000
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

"%PY_BIN%" -c "import torch, mediapipe, fastapi" >nul 2>&1
if errorlevel 1 (
    echo [CANH BAO] Chua tim thay thu vien he thong!
    echo Co ve nhu thay/co chua chay cai dat lan dau.
    echo Vui long nhap dup chuot vao file CAI_DAT_MOI_TRUONG.bat de he thong tu cai dat truoc.
    echo.
    pause
    exit /b 1
)

:: Xac dinh thu muc videos
if not exist "videos" (
    mkdir "videos"
)
set "RECORD_DIR=%CD%\videos"

:: Xac dinh thu muc web frontend
set "WEB_DIR=%CD%\vslr-web"

:: Xac dinh model moi nhat
set "MODEL_PATH=artifacts/v3-realtime-test-candidate/gesture_lstm.pt"
if not exist "%MODEL_PATH%" (
    if exist "models/gesture_lstm.pt" (
        set "MODEL_PATH=models/gesture_lstm.pt"
    )
)

echo Dang khoi dong he thong Web tai: http://localhost:8000 ...
echo Xin cho trong giay lat de model va camera duoc nap...
echo.

:: Tu dong mo trinh duyet
start http://localhost:8000

:: Khoi dong server
"%PY_BIN%" -m prototype_3_gestures.web_server --model "%MODEL_PATH%" --camera 0 --allow-uncalibrated --confidence 0.72 --cooldown 1.5 --tts-voice "Trúc Ly" --record-dir "%RECORD_DIR%" --web-dir "%WEB_DIR%" --port 8000

echo.
echo Server da dung.
pause
