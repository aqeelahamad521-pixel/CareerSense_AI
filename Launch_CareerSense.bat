@echo off
title CareerSense AI Launcher
cd /d "%~dp0"
echo ========================================================
echo Starting CareerSense AI Platform...
echo ========================================================
start http://127.0.0.1:8501
python run_demo.py
pause
