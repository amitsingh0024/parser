@echo off
title Unlimited-OCR-2 Document Parser Studio
echo ========================================================
echo   Starting Baidu Unlimited-OCR-2 Web Studio
echo   Indic Fine-Tuned (Hindi, Sanskrit, English)
echo   Opening: http://localhost:8000
echo ========================================================
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
