@echo off
chcp 65001 >nul
title VSLR - Giao Dien Web Nhan Dien Cu Chi (Realtime Studio)

echo ====================================================================
echo    HE THONG NHAN DIEN CU CHI VSLR - GIAO DIEN WEB THOI GIAN THUC
echo    - Model: Candidate V3 13-epoch (24 cu chi tieng Viet)
echo    - MediaPipe Holistic C++ Backend (Sieu muot 30+ FPS)
echo    - TTS: VieNeu-TTS (Giong doc: Truc Ly - 48kHz tu nhien)
echo    - Tu dong quay video: Bat (Luu vao videos/)
echo    - Ket noi Web: http://localhost:8000
echo ====================================================================
echo.

:: Chuyen den dung thu muc chua ma nguon
if exist "%~dp0VSLR-Project-pipeline-signer-split\src" (
    cd /d "%~dp0VSLR-Project-pipeline-signer-split"
) else (
    cd /d "%~dp0"
)

:: Xac dinh thu muc videos
set "RECORD_DIR=%~dp0videos"
if exist "%~dp0..\videos" (
    set "RECORD_DIR=%~dp0..\videos"
)

:: Xac dinh thu muc web
set "WEB_DIR=%~dp0vslr-web"
if not exist "%WEB_DIR%\index.html" (
    if exist "%~dp0..\vslr-web\index.html" (
        set "WEB_DIR=%~dp0..\vslr-web"
    ) else if exist "%CD%\vslr-web\index.html" (
        set "WEB_DIR=%CD%\vslr-web"
    )
)

echo Dang khoi dong tren thu muc: %CD%
echo Thu muc luu video: %RECORD_DIR%
echo Thu muc giao dien web: %WEB_DIR%
echo.
echo Dang mo trinh duyet tai: http://localhost:8000 ...
start http://localhost:8000
echo.

set "MODEL_PATH=artifacts/v3-realtime-test-candidate/gesture_lstm.pt"
if not exist "%MODEL_PATH%" (
    if exist "models/gesture_lstm.pt" (
        set "MODEL_PATH=models/gesture_lstm.pt"
    )
)

if exist "%CD%\venv\Scripts\python.exe" (
    set "PY_BIN=%CD%\venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0..\venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0..\venv\Scripts\python.exe"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_BIN=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" -m prototype_3_gestures.web_server --model "%MODEL_PATH%" --camera 0 --allow-uncalibrated --confidence 0.72 --cooldown 1.5 --tts-voice "Trúc Ly" --record-dir "%RECORD_DIR%" --web-dir "%WEB_DIR%" --port 8000

pause
