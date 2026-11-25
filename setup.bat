@echo off
setlocal enabledelayedexpansion

echo =================================================
echo   ViconMoCap Sync Studio -- Setup
echo =================================================
echo.

:: -- Locate conda ------------------------------------------------------------
set "CONDA_EXE="
for %%P in (
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "%USERPROFILE%\Miniconda3\Scripts\conda.exe"
    "%USERPROFILE%\anaconda3\Scripts\conda.exe"
    "%USERPROFILE%\Anaconda3\Scripts\conda.exe"
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
    "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
    "C:\ProgramData\Anaconda3\Scripts\conda.exe"
    "C:\miniconda3\Scripts\conda.exe"
    "C:\anaconda3\Scripts\conda.exe"
) do (
    if exist %%P (
        set "CONDA_EXE=%%~P"
        goto :conda_found
    )
)
where conda >nul 2>&1
if %errorlevel% equ 0 (
    set "CONDA_EXE=conda"
    goto :conda_found
)
echo [ERROR] conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
pause
exit /b 1

:conda_found
echo [OK]    conda: %CONDA_EXE%

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)
echo [OK]    Node.js found

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN]  FFmpeg not found on PATH. Video normalisation needs it: https://ffmpeg.org/download.html
) else (
    echo [OK]    FFmpeg found
)
echo.

:: -- Two conda environments (one per backend) -----------------------------------
echo [1/4] Calibration backend  (conda env 'calib_app')...
call :make_env calib_app "%~dp0calibration\backend\environment.yml"
if errorlevel 1 goto :env_failed
echo.
echo [2/4] Sync backend         (conda env 'sync_app')...
call :make_env sync_app "%~dp0sync\backend\environment.yml"
if errorlevel 1 goto :env_failed
echo.

:: -- Frontend ------------------------------------------------------------------
echo [3/4] Installing Node.js dependencies (npm ci, from package-lock.json)...
cd "%~dp0frontend"
call npm ci
if %errorlevel% neq 0 (
    echo [ERROR] npm ci failed.
    cd "%~dp0"
    pause
    exit /b 1
)
cd "%~dp0"

:: -- Data folders + conda path -----------------------------------------------
echo [4/4] Creating data folders...
if not exist "%~dp0calibration\exports" mkdir "%~dp0calibration\exports"
if not exist "%~dp0sync\outputs"        mkdir "%~dp0sync\outputs"
echo %CONDA_EXE%> "%~dp0.conda_path"

echo.
echo =================================================
echo   Setup complete. Run start.bat to launch.
echo =================================================
pause

:: -- Helpers -------------------------------------------------------------------
:make_env
:: %1 = environment name, %2 = environment.yml. Creates it, or updates it if it already exists.
call "%CONDA_EXE%" env create -f %2
if not errorlevel 1 exit /b 0
echo [INFO]  Create failed (the environment may already exist); trying an update...
call "%CONDA_EXE%" env update -f %2 --prune
exit /b %errorlevel%

:env_failed
echo.
echo [ERROR] Could not create or update the conda environment (see the messages above).
echo         Setup did not finish. Check the internet connection and run setup.bat again.
pause
exit /b 1
