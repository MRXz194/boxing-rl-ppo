@echo off
REM Cài đặt dự án Boxing RL trên Windows (1 lần). Mọi thứ nằm trong thư mục .venv\ của dự án,
REM khong cai gi vao Python cua may, khong de lai cache tai ve. Xoa thu muc du an la sach.
REM KHONG dung "chcp 65001" o day: no lam cmd doc lai file .bat sai khi duong dan co dau (vd. "chính").
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [LOI] Chua co Python. Cai Python 3.11-3.13 ban 64-bit tu https://www.python.org/downloads/
  echo        (nho tick "Add python.exe to PATH" khi cai^)
  pause & exit /b 1
)
python -c "import sys, struct; assert struct.calcsize('P') == 8 and (3, 11) <= sys.version_info[:2] <= (3, 13), sys.version" 2>nul
if errorlevel 1 (
  echo [LOI] Can Python 3.11-3.13 ban 64-bit. Python hien tai:
  python --version
  pause & exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo === Tao moi truong ao .venv ...
  python -m venv .venv || (echo [LOI] khong tao duoc .venv & pause & exit /b 1)
)
set PY=.venv\Scripts\python.exe
echo === Cap nhat pip ...
%PY% -m pip install --no-cache-dir -q --upgrade pip
echo === Cai PyTorch ban CPU (khoang 200 MB) ...
%PY% -m pip install --no-cache-dir -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu || (echo [LOI] cai torch that bai & pause & exit /b 1)
echo === Cai cac thu vien con lai ...
%PY% -m pip install --no-cache-dir -q -r requirements.txt || (echo [LOI] cai requirements that bai & pause & exit /b 1)
echo === Kiem tra ...
%PY% smoke_test.py
if errorlevel 1 (echo [LOI] Kiem tra that bai, xem thong bao o tren & pause & exit /b 1)
echo.
echo XONG. Chay demo.bat de xem demo.
pause
