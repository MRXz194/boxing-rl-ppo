@echo off
REM Menu demo: 1 người, 2 người AI vs AI, bạn đấu AI, đánh giá nhanh. Chạy setup.bat trước (1 lần).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Chua cai dat. Chay setup.bat truoc.
  pause & exit /b 1
)
.venv\Scripts\python.exe demo.py
