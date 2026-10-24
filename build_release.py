"""
Đóng gói bàn giao: tạo dist/boxing_rl_<ngày>.zip gồm code, model cuối, kết quả, video, tài liệu.
KHÔNG gồm: .venv (người nhận chạy setup.bat), _build_tools (đồ tải về để build DLL), checkpoint trung gian,
log TensorBoard, log hàng đợi.

  python build_release.py
"""
import glob
import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

FILES = sorted(set(
    glob.glob("*.py") + glob.glob("*.bat") + glob.glob("*.md") + ["requirements.txt"] + glob.glob("docs/**/*", recursive=True)
    + glob.glob("wheels/*.whl") + glob.glob("tools/**/*", recursive=True)
    + glob.glob("results/**/*", recursive=True) + glob.glob("videos/*.mp4")
    # model cuối của mỗi run (bỏ last.pt / ckpt_*.pt / best.pt trùng)
    + glob.glob("models/*/final*.pt") + glob.glob("models/*/config.json")
    # snapshot dùng trong demo / giải Elo
    + glob.glob("models/league6/snapshots/snap_000000000.pt")
    + [p for p in glob.glob("models/league6/snapshots/*.pt") if any(k in p for k in ("snap_0010", "snap_0020"))]
    # log dạng CSV (để vẽ lại biểu đồ), không lấy file TensorBoard
    + glob.glob("logs/*/episodes.csv") + glob.glob("logs/*/evals.csv")
))
FILES = [f for f in FILES if os.path.isfile(f) and "__pycache__" not in f and not f.startswith("dist")
         and not os.path.basename(f).startswith("_")]  # bỏ file tạm (vd. results/_demo_*.json)

os.makedirs("dist", exist_ok=True)
out = os.path.join("dist", f"boxing_rl_{time.strftime('%Y%m%d')}.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(f, os.path.join("boxing_rl", f))
size = os.path.getsize(out) / 1e6
print(f"Đã tạo {out}: {len(FILES)} file, {size:.0f} MB")
print("Người nhận: giải nén → nhấp đúp setup.bat → demo.bat")
