@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -File "%~dp0Launch ECG-PPG App.ps1"
if errorlevel 1 pause