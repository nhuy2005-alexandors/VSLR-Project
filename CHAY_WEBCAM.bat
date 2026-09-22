@echo off
chcp 65001 >nul
title VSLR - Nhan Dien Cu Chi Tieng Viet Qua Webcam

echo ====================================================================
echo    HE THONG NHAN DIEN CU CHI TIENG VIET THOI GIAN THUC (VSLR)
echo    - Mo hinh: BiLSTM Candidate V3 (24 cu chi tieng Viet)
echo    - Thiet bi: HD Webcam (Camera 0)
echo    - Bo phat am: VieNeu-TTS / Pyttsx3 (Doc cau tieng Viet)
echo    - Tu dong ghi video: Bat (Luu vao thu muc videos/)
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

"%PY_BIN%" -c "import torch, mediapipe" >nul 2>&1
if errorlevel 1 (
    echo [CANH BAO] Chua tim thay thu vien he thong!
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

:: Xac dinh model moi nhat
set "MODEL_PATH=artifacts/v3-realtime-test-candidate/gesture_lstm.pt"
if not exist "%MODEL_PATH%" (
    if exist "models/gesture_lstm.pt" (
        set "MODEL_PATH=models/gesture_lstm.pt"
    )
)

echo Cac phim tat khi cua so camera mo:
echo   [S]     : Buoc doc ngay cau hien tai (Speak now)
echo   [C]     : Xoa cau hien tai (Clear)
echo   [SPACE] : Chot moc cu chi (Gesture boundary)
echo   [Q]/ESC : Thoat chuong trinh
echo.
echo Dang ket noi Webcam va nap model... Xin cho trong giay lat...
echo.

"%PY_BIN%" -m prototype_3_gestures.realtime --model "%MODEL_PATH%" --camera 0 --allow-uncalibrated --confidence 0.72 --cooldown 1.5 --tts-voice "Trúc Ly" --record-dir "%RECORD_DIR%"

echo.
echo Chuong trinh da ket thuc.
pause
