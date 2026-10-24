"""In tình trạng hàng đợi train: job nào xong, đang chạy (tiến độ, kết quả eval gần nhất), còn chờ."""
import json
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

TOTAL = {"sp_pool": 6e6, "sp_selfplay": 2e6, "sp_ippo": 2e6, "ppo10M_s2": 1e7, "ppo10M_s0": 1e7,
         "abl_clipreward": 2e6, "abl_stack1": 2e6, "abl_noclip": 2e6, "ppo10M_s1": 1e7}

running = "run_queue.py" in subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
    capture_output=True, text=True).stdout
print("HÀNG ĐỢI TRAIN:", "ĐANG CHẠY" if running else "ĐANG DỪNG (chạy start_train.bat để tiếp tục)")
try:
    st = json.load(open("logs/queue/status.json", encoding="utf-8"))
except FileNotFoundError:
    print("Chưa có hàng đợi nào.")
    sys.exit()
for name, total in TOTAL.items():
    s = st.get(name, {}).get("state", "?")
    log = f"logs/queue/{name}.out"
    step, ev = None, ""
    if os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="replace").read()
        m = re.findall(r"^step\s+([\d,]+)", txt, re.M)
        step = int(m[-1].replace(",", "")) if m else None
        e = re.findall(r"^\[EVAL\].*", txt, re.M)
        ev = e[-1][7:] if e else ""
    if s.startswith("done"):
        icon = "✅ xong"
    elif s == "running" and running:
        icon = "▶ đang chạy"
    elif step:
        icon = "⏸ dừng giữa chừng"
    else:
        icon = "⏳ chờ"
    prog = f"{step / total:5.1%} ({step:,}/{int(total):,})" if step else ""
    print(f"  {icon:18s} {name:15s} {prog:28s} {ev}")
n_eval = sum(1 for k, v in st.items() if v.get("lane") == "eval" or k.startswith(("eval_", "2p_", "behavior", "video", "plots")))
done_eval = sum(1 for k, v in st.items() if (k.startswith(("eval_", "2p_", "behavior", "video", "plots"))) and v.get("state", "").startswith("done"))
print(f"  Đánh giá sau train: {done_eval}/{n_eval} việc xong")
