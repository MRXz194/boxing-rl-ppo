@echo off
REM Dung hang doi train (vd. de choi game). Tien do da luu (last.pt ~ moi 100k step) -> start_train.bat se train tiep.
cd /d "%~dp0..\.."
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$r = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_queue.py' };" ^
  "if (-not $r) { Write-Host 'Khong co train nao dang chay.'; exit 0 };" ^
  "foreach ($x in $r) { taskkill /T /F /PID $x.ProcessId | Out-Null };" ^
  "Start-Sleep 2; Remove-Item 'logs\queue\*.lock' -ErrorAction SilentlyContinue;" ^
  "Write-Host 'DA DUNG TRAIN. Chay start_train.bat de train tiep tu checkpoint gan nhat.'"
pause
