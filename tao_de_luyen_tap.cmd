@echo off
chcp 65001 >nul
title Tool Tu Dong Tao De Luyen Tap Phan Xa

cd /d "%~dp0"

if exist "%~dp0tao_de_luyen_tap.py" (
    python "%~dp0tao_de_luyen_tap.py" %*
    goto :done
)

if exist "%~dp0..\tao_de_luyen_tap.py" (
    python "%~dp0..\tao_de_luyen_tap.py" %*
    goto :done
)

if exist "%~dp0..\..\tao_de_luyen_tap.py" (
    python "%~dp0..\..\tao_de_luyen_tap.py" %*
    goto :done
)

if exist "C:\Users\Acer\Downloads\FE\tao_de_luyen_tap.py" (
    python "C:\Users\Acer\Downloads\FE\tao_de_luyen_tap.py" %*
    goto :done
)

echo [!] Khong tim thay file tao_de_luyen_tap.py! Vui long de file .cmd va .py cung nhau.

:done
echo.
pause
