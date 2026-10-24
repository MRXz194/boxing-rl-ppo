"""
Hàng đợi train + đánh giá chạy tự động trên máy (không cần ai trông).

- 2 làn song song, tất cả chạy trên Windows (không cần Linux/WSL):
    "2p" : Boxing 2 người (self-play)
    "1p" : Boxing 1 người
  Sau khi cả 2 làn xong, làn "eval" chạy toàn bộ phần đánh giá.
- Mỗi việc: nếu file "done" đã có thì bỏ qua. Nếu chưa, chạy với --resume; crash (vd. hết RAM)
  thì đợi 3 phút (và đợi máy đủ bộ nhớ) rồi chạy lại, tối đa 60 lần.
- Giữ máy không sleep trong lúc chạy (SetThreadExecutionState, tự hết khi tiến trình này thoát;
  không đổi cài đặt nào của Windows).
- Trạng thái: logs/queue/status.json; log từng việc: logs/queue/<tên>.out

  Chạy:  .venv\\Scripts\\python.exe run_queue.py            (thường được khởi động ẩn ở nền)
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # thư mục gốc dự án
QDIR = os.path.join(ROOT, "logs", "queue")
WIN_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")


def win(*args):
    """Lệnh chạy 1 script bằng Python của .venv. Một chuỗi có dấu cách sẽ được tách thành nhiều đối số."""
    flat = []
    for x in args:
        flat += x.split(" ") if (" " in x and not os.path.exists(x)) else [x]
    return [WIN_PY, "-u", *flat]


def M(*p):
    return os.path.join(ROOT, "models", *p)


def R(*p):
    return os.path.join(ROOT, "results", *p)


def snap(run: str, target: int, suffix: str = "") -> str:
    """Đường dẫn (tương đối) của snapshot gần mốc target nhất. Gọi lúc việc bắt đầu chạy."""
    import glob
    import re
    files = glob.glob(os.path.join(ROOT, "models", run, "snapshots", f"snap_*{suffix}.pt"))
    if not files:
        return f"models/{run}/MISSING_snap_{target}{suffix}.pt"
    best = min(files, key=lambda f: abs(int(re.search(r"snap_(\d+)", f).group(1)) - target))
    return f"models/{run}/snapshots/{os.path.basename(best)}"


# ----------------------------------------------------------------------------- danh sách việc
JOBS = [
    # ===== làn 2p: Boxing 2 người (self-play) =====
    dict(name="sp_pool", lane="2p", done=M("sp_pool", "final.pt"),
         cmd=win("selfplay_ppo.py", "--mode", "pool", "--timesteps", "6000000", "--run-name", "sp_pool", "--resume")),
    dict(name="sp_selfplay", lane="2p", done=M("sp_selfplay", "final.pt"),
         cmd=win("selfplay_ppo.py", "--mode", "selfplay", "--timesteps", "2000000", "--run-name", "sp_selfplay", "--resume")),
    dict(name="sp_ippo", lane="2p", done=M("sp_ippo", "final_b.pt"),
         cmd=win("selfplay_ppo.py", "--mode", "ippo", "--timesteps", "2000000", "--run-name", "sp_ippo", "--resume")),
    # ===== làn 1p: Boxing 1 người =====
    dict(name="ppo10M_s0", lane="1p", done=M("ppo10M_s0", "final.pt"),
         cmd=win("train_ppo.py", "--timesteps", "10000000", "--seed", "0", "--run-name", "ppo10M_s0", "--resume")),
    dict(name="abl_clipreward", lane="1p", done=M("abl_clipreward", "final.pt"),
         cmd=win("train_ppo.py", "--timesteps", "2000000", "--seed", "0", "--run-name", "abl_clipreward",
                 "--clip-reward", "--resume")),
    dict(name="abl_stack1", lane="1p", done=M("abl_stack1", "final.pt"),
         cmd=win("train_ppo.py", "--timesteps", "2000000", "--seed", "0", "--run-name", "abl_stack1",
                 "--frame-stack", "1", "--resume")),
    dict(name="abl_noclip", lane="1p", done=M("abl_noclip", "final.pt"),
         cmd=win("train_ppo.py", "--timesteps", "2000000", "--seed", "0", "--run-name", "abl_noclip",
                 "--clip-range", "1000", "--resume")),
    dict(name="ppo10M_s1", lane="1p", done=M("ppo10M_s1", "final.pt"),
         cmd=win("train_ppo.py", "--timesteps", "10000000", "--seed", "1", "--run-name", "ppo10M_s1", "--resume")),
    # seed 2 chạy ở làn 2p (làn này xong sớm hơn) để 2 làn cân bằng
    dict(name="ppo10M_s2", lane="2p", done=M("ppo10M_s2", "final.pt"),
         cmd=win("train_ppo.py", "--timesteps", "10000000", "--seed", "2", "--run-name", "ppo10M_s2", "--resume")),
    # ===== BẢN SỬA self-play: league (khởi đầu từ agent 1 người + AI máy + random + pool) =====
    dict(name="league", lane="2p", done=M("league", "final.pt"),
         # v5: TINH CHỈNH agent 1 người (lr nhỏ 5e-5, entropy 0.001 để không "quên") + shaping 3 (giảm dần)
         # đấu tập đối thủ CỐ ĐỊNH: AI máy, agent 1 người (cả 2 phe), random, đứng yên. Không self-play thuần.
         # Lịch sử: v1-v3 (lr 2.5e-4, entropy 0.01 -> quên, tụt điểm), v4 (học từ đầu -> quá chậm). Xem TAI_LIEU_DU_AN.md.
         cmd=win("league_ppo.py", "--init", "models/ppo10M_s0/final.pt", "--timesteps", "3000000",
                 "--learning-rate", "5e-05", "--ent-coef", "0.001",
                 "--n-2p", "6", "--n-cpu", "2", "--p-self", "0", "--p-random", "0.35", "--p-still", "0.2",
                 "--opp-models", "models/ppo10M_s0/final.pt", "--freeze-pool",
                 "--shaping", "3", "--shaping-decay", "0.6", "--run-name", "league", "--resume")),
]

# ===== làn eval: chạy sau khi train xong =====
TRAIN = [j["name"] for j in JOBS]
EVAL_1P = []
for run in ["ppo10M_s0", "ppo10M_s1", "ppo10M_s2", "abl_clipreward", "abl_stack1", "abl_noclip"]:
    for ck in ["final", "best"]:
        EVAL_1P.append(dict(name=f"eval_{run}_{ck}", lane="eval", after=TRAIN, done=R(f"{run}_{ck}_d0.json"),
                            cmd=win("evaluate.py", "--agent", "ppo", "--model", M(run, f"{ck}.pt"), "--episodes", "30")))
for d in (1, 2, 3):
    EVAL_1P.append(dict(name=f"eval_ppo10M_s0_d{d}", lane="eval", after=TRAIN, done=R(f"ppo10M_s0_final_d{d}.json"),
                        cmd=win("evaluate.py", "--agent", "ppo", "--model", M("ppo10M_s0", "final.pt"),
                                "--episodes", "30", "--difficulty", str(d))))
EVAL_1P += [
    dict(name="eval_ppo10M_s0_stoch", lane="eval", after=TRAIN, done=R("ppo10M_s0_final_d0_stoch.json"),
         cmd=win("evaluate.py", "--agent", "ppo", "--model", M("ppo10M_s0", "final.pt"), "--episodes", "30", "--stochastic")),
    dict(name="behavior_ppo10M_s0", lane="eval", after=TRAIN, done=R("behavior_ppo10M_s0.json"),
         cmd=win("analyze_behavior.py", "--model", M("ppo10M_s0", "final.pt"), "--out", R("behavior_ppo10M_s0.json"))),
    dict(name="video_ppo10M_s0", lane="eval", after=TRAIN, done=os.path.join(ROOT, "videos", "ppo10M_s0_final.mp4"),
         cmd=win("play.py", "--agent", "ppo", "--model", M("ppo10M_s0", "final.pt"),
                 "--out", os.path.join(ROOT, "videos", "ppo10M_s0_final.mp4"))),
]
EVAL_2P = [
    dict(name="2p_vsrandom_pool", lane="eval", after=TRAIN, done=R("2p_vs-random_sp_pool_final_vs_random.json"),
         cmd=win("evaluate_2p.py", "vs-random", "--model", "models/sp_pool/final.pt", "--games", "20")),
    dict(name="2p_vsrandom_selfplay", lane="eval", after=TRAIN, done=R("2p_vs-random_sp_selfplay_final_vs_random.json"),
         cmd=win("evaluate_2p.py", "vs-random", "--model", "models/sp_selfplay/final.pt", "--games", "20")),
    dict(name="2p_vsrandom_ippo", lane="eval", after=TRAIN, done=R("2p_vs-random_sp_ippo_final(w+b)_vs_random.json"),
         cmd=win("evaluate_2p.py", "vs-random", "--model", "models/sp_ippo/final_w.pt+models/sp_ippo/final_b.pt", "--games", "20")),
    dict(name="2p_vsrandom_ppo10M", lane="eval", after=TRAIN, done=R("2p_vs-random_ppo10M_s0_final_vs_random.json"),
         cmd=win("evaluate_2p.py", "vs-random", "--model", "models/ppo10M_s0/final.pt", "--games", "20")),
    dict(name="2p_vscpu_pool", lane="eval", after=TRAIN, done=R("2p_vscpu_sp_pool_final_d0.json"),
         cmd=win("evaluate_2p.py", "vs-cpu", "--model", "models/sp_pool/final.pt", "--episodes", "30")),
    dict(name="2p_vscpu_selfplay", lane="eval", after=TRAIN, done=R("2p_vscpu_sp_selfplay_final_d0.json"),
         cmd=win("evaluate_2p.py", "vs-cpu", "--model", "models/sp_selfplay/final.pt", "--episodes", "30")),
    dict(name="2p_league", lane="eval", after=TRAIN, done=R("league_main.json"),
         cmd=lambda: win("evaluate_2p.py league --games 8 --name main --players random models/ppo_s0/final.pt "
                         "models/ppo10M_s0/final.pt "
                         + " ".join(snap("sp_pool", k * 1_000_000) for k in range(1, 6)) +
                         " models/sp_pool/final.pt " + snap("sp_selfplay", 1_000_000) + " models/sp_selfplay/final.pt "
                         + snap("sp_ippo", 1_000_000, "_w") + "+" + snap("sp_ippo", 1_000_000, "_b") +
                         " models/sp_ippo/final_w.pt+models/sp_ippo/final_b.pt")),
    dict(name="2p_video_final_vs_final", lane="eval", after=TRAIN, done=os.path.join(ROOT, "videos", "2p_pool_final_vs_final.mp4"),
         cmd=win("play_2p.py", "--white", "models/sp_pool/final.pt", "--black", "models/sp_pool/final.pt", "--out", "videos/2p_pool_final_vs_final.mp4")),
    dict(name="2p_video_1M_vs_final", lane="eval", after=TRAIN, done=os.path.join(ROOT, "videos", "2p_pool_1M_vs_final.mp4"),
         cmd=lambda: win("play_2p.py --white " + snap("sp_pool", 1_000_000) + " --black models/sp_pool/final.pt "
                         "--out videos/2p_pool_1M_vs_final.mp4")),
    dict(name="2p_video_pool_vs_ppo1p", lane="eval", after=TRAIN, done=os.path.join(ROOT, "videos", "2p_ppo1P_vs_pool.mp4"),
         cmd=win("play_2p.py", "--white", "models/ppo10M_s0/final.pt", "--black", "models/sp_pool/final.pt", "--out", "videos/2p_ppo1P_vs_pool.mp4")),
]
EVAL_LEAGUE = [
    dict(name="L_vsrandom", lane="eval", after=["league"], done=R("2p_vs-random_league_final_vs_random.json"),
         cmd=win("evaluate_2p.py", "vs-random", "--model", "models/league/final.pt", "--games", "30")),
    dict(name="L_vscpu", lane="eval", after=["league"], done=R("2p_vscpu_league_final_d0.json"),
         cmd=win("evaluate_2p.py", "vs-cpu", "--model", "models/league/final.pt", "--episodes", "30")),
    dict(name="L_vs_ppo1p", lane="eval", after=["league"], done=R("2p_match_league_final_vs_ppo10M_s0_final.json"),
         cmd=win("evaluate_2p.py", "match", "--a", "models/league/final.pt", "--b", "models/ppo10M_s0/final.pt", "--games", "20")),
    dict(name="L_vs_sppool", lane="eval", after=["league"], done=R("2p_match_league_final_vs_sp_pool_final.json"),
         cmd=win("evaluate_2p.py", "match", "--a", "models/league/final.pt", "--b", "models/sp_pool/final.pt", "--games", "20")),
    dict(name="L_tournament", lane="eval", after=["league"], done=R("league_final.json"),
         cmd=lambda: win("evaluate_2p.py league --games 8 --name final --players random models/ppo10M_s0/final.pt "
                         "models/sp_pool/final.pt models/sp_selfplay/final.pt "
                         "models/sp_ippo/final_w.pt+models/sp_ippo/final_b.pt "
                         + " ".join(snap("league", k) for k in (0, 1_000_000, 2_000_000)) + " models/league/final.pt")),
    dict(name="L_video_vs_self", lane="eval", after=["league"], done=os.path.join(ROOT, "videos", "2p_league_vs_league.mp4"),
         cmd=win("play_2p.py", "--white", "models/league/final.pt", "--black", "models/league/final.pt", "--out", "videos/2p_league_vs_league.mp4")),
    dict(name="L_video_vs_random", lane="eval", after=["league"], done=os.path.join(ROOT, "videos", "2p_league_vs_random.mp4"),
         cmd=win("play_2p.py", "--white", "models/league/final.pt", "--black", "random", "--out", "videos/2p_league_vs_random.mp4")),
    dict(name="L_video_vs_ppo1p", lane="eval", after=["league"], done=os.path.join(ROOT, "videos", "2p_ppo1P_vs_league.mp4"),
         cmd=win("play_2p.py", "--white", "models/ppo10M_s0/final.pt", "--black", "models/league/final.pt", "--out", "videos/2p_ppo1P_vs_league.mp4")),
    dict(name="L_plots", lane="eval", after=["league"], done=None, cmd=win("plot_results.py")),
]
EVAL_FINAL = [dict(name="plots", lane="eval", after=TRAIN, done=None, cmd=win("plot_results.py"))]
ALL = JOBS + EVAL_1P + EVAL_2P + EVAL_FINAL + EVAL_LEAGUE

# ----------------------------------------------------------------------------- chạy
status: dict[str, dict] = {}
lock = threading.Lock()


def save_status():
    with lock:
        tmp = os.path.join(QDIR, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        os.replace(tmp, os.path.join(QDIR, "status.json"))


def is_done(job) -> bool:
    return job["done"] is not None and os.path.exists(job["done"])


MAX_ATTEMPTS = 60
MIN_FREE_COMMIT_GB = 2.5  # job train ~1.5 GB sau khi sửa OpenBLAS; chừa ~1 GB cho chương trình khác


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_commit_gb() -> float:
    """Bộ nhớ ảo (commit) còn trống của Windows, tính bằng GB. Lỗi 'paging file too small' xảy ra khi số này về 0."""
    m = _MEMSTATUS()
    m.dwLength = ctypes.sizeof(_MEMSTATUS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 1024 ** 3


def pid_alive(pid: int) -> bool:
    """Kiểm tra tiến trình còn chạy không (Windows API, không gửi tín hiệu gì)."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def wait_existing(name: str):
    """Nếu việc này đang chạy dở do một runner trước khởi động (có file lock), đợi nó xong, không chạy trùng."""
    lock = os.path.join(QDIR, f"{name}.lock")
    if not os.path.exists(lock):
        return
    try:
        pid = int(open(lock).read().strip())
    except ValueError:
        return
    while pid_alive(pid):
        status[name] = {"state": "running", "note": f"đang chạy từ runner trước (pid {pid})"}
        save_status()
        time.sleep(60)


def run_job(job) -> bool:
    name = job["name"]
    wait_existing(name)
    if is_done(job):
        status[name] = {"state": "done (có sẵn)"}
        save_status()
        return True
    for dep in job.get("after", []):  # đợi các việc phụ thuộc
        while status.get(dep, {}).get("state", "").split(" ")[0] not in ("done", "failed"):
            time.sleep(30)
    cmd = job["cmd"]() if callable(job["cmd"]) else job["cmd"]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        while avail_commit_gb() < MIN_FREE_COMMIT_GB:  # máy đang thiếu bộ nhớ (chương trình khác) -> đợi
            status[name] = {"state": "đợi bộ nhớ", "attempt": attempt,
                            "free_commit_gb": round(avail_commit_gb(), 1)}
            save_status()
            time.sleep(60)
        status[name] = {"state": "running", "attempt": attempt, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_status()
        with open(os.path.join(QDIR, f"{name}.out"), "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} lần chạy {attempt}: {' '.join(cmd)}\n")
            log.flush()
            # OPENBLAS_NUM_THREADS=1: numpy (OpenBLAS) mặc định giữ bộ đệm cho mỗi luồng CPU, ~500 MB commit
            # MỖI tiến trình; 8 tiến trình game -> ~4 GB lãng phí, gây lỗi "paging file too small".
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "1"}
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            with open(os.path.join(QDIR, f"{name}.lock"), "w") as lf:
                lf.write(str(proc.pid))
            rc = proc.wait()
        if rc == 0 and (job["done"] is None or is_done(job)):
            status[name] = {"state": "done", "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "attempts": attempt}
            save_status()
            return True
        status[name] = {"state": f"crash rc={rc}, đợi chạy lại", "attempt": attempt}
        save_status()
        time.sleep(180)
    status[name] = {"state": "failed", "attempts": MAX_ATTEMPTS}
    save_status()
    return False


def lane_worker(lane: str):
    for job in [j for j in ALL if j["lane"] == lane]:
        run_job(job)


def keep_awake():
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    os.makedirs(QDIR, exist_ok=True)
    pid_file = os.path.join(QDIR, "runner.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    for j in ALL:
        status[j["name"]] = {"state": "done (có sẵn)" if is_done(j) else "pending", "lane": j["lane"]}
    save_status()
    keep_awake()
    threads = [threading.Thread(target=lane_worker, args=(ln,), daemon=True) for ln in ("2p", "1p", "eval")]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        keep_awake()
        time.sleep(60)
    status["_queue"] = {"state": "ALL DONE", "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_status()


if __name__ == "__main__":
    main()
