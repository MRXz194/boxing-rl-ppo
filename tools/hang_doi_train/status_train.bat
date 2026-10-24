@echo off
REM Xem tinh trang train: job nao xong / dang chay / con cho.
cd /d "%~dp0..\.."
.venv\Scripts\python.exe tools\hang_doi_train\status_train.py
pause
