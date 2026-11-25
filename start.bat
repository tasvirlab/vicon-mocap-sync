@echo off
:: Launches ONE Electron window. The window starts both backends itself:
::   calibration  -> conda env calib_app, port 8765
::   sync         -> conda env sync_app,  port 8766
cd /d "%~dp0frontend"
call npm run dev
