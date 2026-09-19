@echo off
setlocal
echo ===================================================================
echo   Starting Zephyra: Intelligent Energy ^& Equipment Monitoring
echo   Yukthi 2026 National-Level Hackathon Platform
echo ===================================================================
echo.

set "PY_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_EXE=python3"
    ) else (
        where python >nul 2>&1
        if %errorlevel% equ 0 (
            set "PY_EXE=python"
        ) else (
            if exist "C:\Users\hi\AppData\Local\Programs\Python\Python311\python.exe" (
                set "PY_EXE=C:\Users\hi\AppData\Local\Programs\Python\Python311\python.exe"
            )
        )
    )
)

if "%PY_EXE%"=="" (
    echo [ERROR] Python was not found on your system!
    echo Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

echo [INFO] Using Python runtime: %PY_EXE%
echo [INFO] Launching Streamlit Industrial Monitoring Dashboard...
echo [INFO] Local address: http://localhost:8501
echo.

"%PY_EXE%" -m streamlit run app.py

pause
