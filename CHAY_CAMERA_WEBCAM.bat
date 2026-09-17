@echo off
chcp 65001 >nul
title VSLR - Nhan Dien Cu Chi Tieng Viet (HD Webcam)

echo ====================================================================
echo    HE THONG NHAN DIEN CU CHI TIENG VIET THOI GIAN THUC (VSLR)
echo    - Model: Candidate V3 13-epoch (24 cu chi tieng Viet)
echo    - Camera: HD Webcam (Camera Laptop mac dinh)
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

python -m prototype_3_gestures.realtime --model "artifacts/v3-realtime-test-candidate/gesture_lstm.pt" --camera 0 --allow-uncalibrated --confidence 0.72 --cooldown 1.5 --tts-voice "Trúc Ly" --record-dir "%RECORD_DIR%"

echo.
echo Chuong trinh da ket thuc.
pause
