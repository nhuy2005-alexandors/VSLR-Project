@echo off
chcp 65001 >nul
title VSLR - Tải Video Test từ Hugging Face Dataset về máy

echo ====================================================================
echo    TẢI TOÀN BỘ VIDEO TEST TỪ HUGGING FACE DATASET VỀ PHÂN TÍCH
echo    - Dataset: ntbii305/vslr-remote
echo    - Thư mục lưu: videos_from_huggingface/
echo ====================================================================
echo.

cd /d "%~dp0"

:: Kiểm tra môi trường ảo venv
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
echo Đã tải xong! Đang mở thư mục chứa video...
echo ====================================================================
if exist "videos_from_huggingface\data\khach" (
    start "" "videos_from_huggingface\data\khach"
) else (
    start "" "videos_from_huggingface"
)
pause
