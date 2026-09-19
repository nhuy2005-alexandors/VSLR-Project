@echo off
chcp 65001 >nul
title VSLR - Nhan Dien Cu Chi Tieng Viet (Camera A16)

echo ====================================================================
echo    HE THONG NHAN DIEN CU CHI TIENG VIET THOI GIAN THUC (VSLR)
echo    - Model: Candidate V3 13-epoch (24 cu chi tieng Viet)
echo    - Camera: A16 cua Tho (Windows Virtual Camera)
echo    - TTS: VieNeu-TTS (Giong doc: Truc Ly - 48kHz tu nhien)
echo    - Tu dong quay video: Bat (Luu vao videos/)
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

echo Dang khoi dong tren thu muc: %CD%
echo.
echo Cac phim tat khi cua so camera mo:
echo   [S]     : Buoc doc ngay cau hien tai (Speak now)
echo   [C]     : Xoa cau hien tai (Clear)
echo   [SPACE] : Chot moc cu chi (Gesture boundary)
echo   [Q]/ESC : Thoat chuong trinh
echo.
echo Video quay tung cu chi se duoc tu dong luu tai:
echo %RECORD_DIR%
echo.
echo --------------------------------------------------------------------
echo Dang ket noi camera A16 va nap model... Xin cho trong giay lat...
echo --------------------------------------------------------------------
echo.

if exist "%CD%\venv\Scripts\python.exe" (
    set "PY_BIN=%CD%\venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0..\venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0..\venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" -m prototype_3_gestures.realtime --model "artifacts/v3-realtime-test-candidate/gesture_lstm.pt" --camera "A16" --allow-uncalibrated --confidence 0.72 --cooldown 1.5 --tts-voice "Trúc Ly" --record-dir "%RECORD_DIR%"

if %ERRORLEVEL% EQU 0 goto END_APP

echo.
echo ====================================================================
echo [CANH BAO] Khong the ket noi Camera A16 hoac chuong trinh ket thuc bat thuong.
echo Ban co muon thu chuyen sang Camera Laptop mac dinh - HD Webcam - khong?
echo ====================================================================
set /p choice="Nhan Y de mo HD Webcam (hoac Enter de thoat): "
if /i not "%choice%"=="Y" goto END_APP

echo.
echo Dang khoi dong lai voi HD Webcam (Camera 0)...
"%PY_BIN%" -m prototype_3_gestures.realtime --model "artifacts/v3-realtime-test-candidate/gesture_lstm.pt" --camera 0 --allow-uncalibrated --confidence 0.72 --cooldown 1.5 --tts-voice "Trúc Ly" --record-dir "%RECORD_DIR%"

:END_APP
echo.
echo Chuong trinh da ket thuc.
pause
