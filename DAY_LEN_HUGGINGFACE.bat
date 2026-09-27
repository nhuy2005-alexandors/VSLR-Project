@echo off
chcp 65001 >nul
title VSLR - Đẩy Backend lên Hugging Face Space (Tự động 1-Click)

echo ====================================================================
echo    TỰ ĐỘNG ĐẨY BACKEND VSLR LÊN HUGGING FACE SPACE (ZERO GPU FREE)
echo    - Space đích:   ntbii305/vslr-backend
echo    - Dataset đích: ntbii305/vslr-remote
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

"%PY_BIN%" scripts\deploy_hf_space.py

echo.
pause
