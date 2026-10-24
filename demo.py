"""
Menu demo cho buổi thuyết trình (demo.bat gọi file này). Tự chọn model tốt nhất đang có.

  python demo.py
"""
import glob
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
PY = sys.executable


def first_existing(*paths):
    return next((p for p in paths if os.path.exists(p)), None)


def model_1p():
    return first_existing("models/ppo10M_s0/final.pt", "models/ppo_s0/final.pt")


def model_2p():
    snaps = sorted(glob.glob("models/sp_pool/snapshots/snap_*.pt"))
    return first_existing("models/league6/final.pt", "models/league/final.pt", "models/sp_pool/final.pt") or (snaps[-1].replace("\\", "/") if snaps else None)


def early_2p():
    """Bản đầu của quá trình train 2 người cuối cùng (để so với bản cuối)."""
    for run in ("league6", "league", "sp_pool"):
        snaps = sorted(glob.glob(f"models/{run}/snapshots/snap_*.pt"))
        if snaps:
            return snaps[0].replace("\\", "/")
    return None


def run(*args):
    print("\n>>", " ".join(args), "\n")
    subprocess.call([PY, *args])


# (nhóm, nhãn, hàm chạy, model cần có)
MENU = [
    ("CHẾ ĐỘ 1 NGƯỜI  (bot học bằng cách đấu AI có sẵn của game)", None, None, None),
    (None, "Xem bot đấu AI có sẵn của game (bot cầm võ sĩ TRẮNG)",
     lambda m1, m2: run("play.py", "--agent", "ppo", "--model", m1, "--live"), "1p"),
    (None, "BẠN đấu bot 1 người: bạn cầm võ sĩ ĐEN   [mũi tên di chuyển, SPACE đấm, R ván mới, ESC thoát]",
     lambda m1, m2: run("play_2p.py", "--human", "--human-side", "black", "--white", m1), "1p"),
    (None, "Đánh giá nhanh: bot 1 người đấu AI máy 5 ván",
     lambda m1, m2: run("evaluate.py", "--agent", "ppo", "--model", m1, "--episodes", "5", "--out", "results/_demo_1p.json"), "1p"),
    ("CHẾ ĐỘ 2 NGƯỜI  (cả 2 võ sĩ đều là bot)", None, None, None),
    (None, "Xem 2 BOT ĐÁNH NHAU (cùng 1 mạng cầm cả 2 phe)",
     lambda m1, m2: run("play_2p.py", "--live", "--white", m2, "--black", m2), "2p"),
    (None, "BẠN đấu bot 2 người: bạn cầm võ sĩ TRẮNG   [mũi tên di chuyển, SPACE đấm, R ván mới, ESC thoát]",
     lambda m1, m2: run("play_2p.py", "--human", "--black", m2), "2p"),
    (None, "Bot 1 người (trắng) đấu bot 2 người (đen)",
     lambda m1, m2: run("play_2p.py", "--live", "--white", m1, "--black", m2), "both"),
    (None, "Bản ĐẦU của lần train cuối (trắng) đấu bản CUỐI (đen): thấy rõ bot mạnh lên",
     lambda m1, m2: run("play_2p.py", "--live", "--white", early_2p(), "--black", m2), "2p"),
    (None, "Đánh giá nhanh: bot 2 người đấu random 6 ván (3 cầm trắng, 3 cầm đen)",
     lambda m1, m2: run("evaluate_2p.py", "vs-random", "--model", m2, "--games", "6", "--out", "results/_demo_2p.json"), "2p"),
    ("KHÁC", None, None, None),
    (None, "Kiểm chứng thuật toán PPO tự cài (10 phép thử so với công thức toán)",
     lambda m1, m2: run("verify_algorithm.py"), None),
    (None, "Mở thư mục biểu đồ (results) và video (videos)",
     lambda m1, m2: [os.startfile(os.path.join(ROOT, d)) for d in ("results", "videos")], None),
]


def main():
    m1, m2 = model_1p(), model_2p()
    items = [x for x in MENU if x[1]]
    while True:
        print("\n" + "=" * 78)
        print(" DEMO BOXING RL  (PPO tự cài đặt, không dùng thư viện RL)")
        print(f"   bot 1 người: {m1 or 'CHƯA CÓ'}")
        print(f"   bot 2 người: {m2 or 'CHƯA CÓ'}")
        print("=" * 78)
        n = 0
        for group, label, _, _ in MENU:
            if group:
                print(f"\n [{group}]")
            else:
                n += 1
                print(f"  {n}. {label}")
        print("\n  0. Thoát")
        c = input("\nChọn: ").strip()
        if c in ("0", "q", ""):
            return
        if not c.isdigit() or not 1 <= int(c) <= len(items):
            continue
        _, label, fn, need = items[int(c) - 1]
        if (need in ("1p", "both") and not m1) or (need in ("2p", "both") and not m2):
            print("!! Chưa có model cho mục này. Train trước (xem README).")
            continue
        fn(m1, m2)


if __name__ == "__main__":
    main()
