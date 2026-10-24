@echo off
REM Chay (hoac chay tiep) hang doi train o nen, an cua so. Job da xong tu bo qua, job dang do train tiep tu last.pt.
REM Dong cua so nay khong lam dung train. Muon dung: stop_train.bat. Xem tien do: status_train.bat
cd /d "%~dp0..\.."
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$r = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_queue.py' };" ^
  "if ($r) { Write-Host 'Train DANG CHAY roi, khong can chay them.'; exit 0 };" ^
  "New-Item -ItemType Directory -Force 'logs\queue' | Out-Null; Remove-Item 'logs\queue\*.lock' -ErrorAction SilentlyContinue;" ^
  "Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u','tools\hang_doi_trainun_queue.py' -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput 'logs\queue\runner.out' -RedirectStandardError 'logs\queue\runner.err';" ^
  "Write-Host 'DA BAT DAU TRAIN (chay an o nen).'"
pause
