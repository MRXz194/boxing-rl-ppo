"""Đóng wheel multi_agent_ale_py cho Windows: phần Python gốc (không sửa) + ale_c.dll vừa build."""
import base64, hashlib, os, sys, zipfile
# Mặc định đọc từ _build_tools/ (nơi tải mã nguồn và build DLL, xem BUILD.md)
BT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_build_tools")
SRC = os.path.join(BT, "src", "multi_agent_ale_py-0.1.12")
DLL = os.path.join(BT, "build", "src", "libale_c.dll")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wheels")
NAME, VER, TAG = "multi_agent_ale_py", "0.1.12", "py3-none-win_amd64"
files = {}
pkg = os.path.join(SRC, NAME)
for fn in sorted(os.listdir(pkg)):
    p = os.path.join(pkg, fn)
    if fn.endswith(".py"):
        files[f"{NAME}/{fn}"] = open(p, "rb").read()
files[f"{NAME}/ale_c.dll"] = open(DLL, "rb").read()   # trình nạp ctypes của gói tìm "ale_c.dll" trên Windows
lic = open(os.path.join(SRC, "LICENSE.md"), "rb").read()
di = f"{NAME}-{VER}.dist-info"
files[f"{di}/LICENSE.md"] = lic
files[f"{di}/METADATA"] = (f"Metadata-Version: 2.1\nName: {NAME}\nVersion: {VER}\nSummary: Multi-Agent ALE "
    "(Windows build: ale_c.dll compiled with MinGW-w64 GCC, static; zlib replaced by a minimal reader)\n"
    "Home-page: https://github.com/Farama-Foundation/Multi-Agent-ALE\nLicense: GPL-2.0\n"
    "Requires-Python: >=3.9\nRequires-Dist: numpy\n").encode()
files[f"{di}/WHEEL"] = f"Wheel-Version: 1.0\nGenerator: make_wheel.py\nRoot-Is-Purelib: false\nTag: {TAG}\n".encode()
files[f"{di}/top_level.txt"] = f"{NAME}\n".encode()
rec = []
for k, v in files.items():
    h = base64.urlsafe_b64encode(hashlib.sha256(v).digest()).rstrip(b"=").decode()
    rec.append(f"{k},sha256={h},{len(v)}")
rec.append(f"{di}/RECORD,,")
files[f"{di}/RECORD"] = ("\n".join(rec) + "\n").encode()
os.makedirs(OUT, exist_ok=True)
whl = os.path.join(OUT, f"{NAME}-{VER}-{TAG}.whl")
with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
    for k, v in files.items():
        z.writestr(k, v)
print("OK", os.path.abspath(whl), os.path.getsize(whl), "bytes")
