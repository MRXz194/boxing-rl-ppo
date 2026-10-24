"""
Vẽ biểu đồ cho báo cáo:
  1. results/learning_curve.png : đường học 1 người (mọi run, trung bình ± std giữa các seed ppo10M_*)
  2. results/ablation_curve.png : ablation so với baseline cùng 2M step
  3. results/selfplay_curve.png : self-play 2 người (đấu random, đấu snapshot cũ)
  4. results/comparison.png     : so sánh các agent 1 người từ results/*_d0.json
  (Elo + heatmap của giải đấu 2 người do evaluate_2p.py league vẽ.)

  python plot_results.py            -> lưu vào results/*.png
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình; phải đặt TRƯỚC khi import numpy
import csv
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



def load_csv(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    # train bị ngắt rồi --resume từ checkpoint cũ hơn -> có đoạn bị ghi 2 lần.
    # Khi global_step giảm, bỏ các dòng cũ có step >= điểm resume (giữ dữ liệu của lần chạy sau).
    rows = []
    for r in raw:
        s = float(r["global_step"])
        if rows and s < float(rows[-1]["global_step"]):  # step giảm = chỗ resume
            while rows and float(rows[-1]["global_step"]) >= s:
                rows.pop()
        rows.append(r)
    num = [k for k in raw[0] if all(_isnum(r[k]) for r in rows)] if rows else []
    return {k: np.array([float(r[k]) for r in rows]) for k in num} if rows else None


def _isnum(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def smooth(y, k=20):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def _eval_curve(run):
    ev = load_csv(os.path.join("logs", run, "evals.csv"))
    return None if ev is None or "mean_return" not in ev else ev


def _finish(ax, fig, path, title, ylabel, xlabel="Timesteps (1 step = 4 frame)"):
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Đã lưu", path)


def plot_learning_curves():
    """Phần 1 (1 người): đường học eval argmax (5 ván/lần) cộng reward lúc train làm mượt."""
    runs_1p = sorted(os.path.basename(d) for d in glob.glob("logs/*")
                     if (_eval_curve(os.path.basename(d)) is not None))
    if not runs_1p:
        print("Chưa có log train 1 người trong logs/")
        return
    main = [r for r in runs_1p if not r.startswith("abl_")]
    # (a) các run chính; nếu có >= 2 seed ppo10M_* thì vẽ thêm trung bình ± std giữa các seed
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in main:
        ev = _eval_curve(r)
        ep = load_csv(os.path.join("logs", r, "episodes.csv"))
        line, = ax.plot(ev["global_step"], ev["mean_return"], "o-", ms=3, lw=1, label=f"{r} (eval)")
        if ep is not None and "return" in ep:
            ys = smooth(ep["return"])
            ax.plot(ep["global_step"][len(ep["return"]) - len(ys):], ys, color=line.get_color(), alpha=0.35, lw=1)
    seeds = [r for r in main if r.startswith("ppo10M_s")]
    if len(seeds) >= 2:
        curves = [_eval_curve(r) for r in seeds]
        n = min(len(c["global_step"]) for c in curves)
        x = curves[0]["global_step"][:n]
        Y = np.stack([c["mean_return"][:n] for c in curves])
        ax.plot(x, Y.mean(0), "k-", lw=2.5, label=f"TB {len(seeds)} seed")
        ax.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color="k", alpha=0.12, label="± std giữa các seed")
    ax.axhline(100, color="red", lw=0.8, ls=":", label="KO (100)")
    _finish(ax, fig, "results/learning_curve.png", "Boxing 1 người – PPO tự cài (đường mờ: reward lúc train)",
            "Hiệu điểm mỗi ván (mình − AI máy)")

    # (b) ablation, so với baseline cùng ngân sách 2M step (ppo_s0)
    abl = [r for r in runs_1p if r.startswith("abl_")]
    if abl:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = {"ppo_s0": "baseline (4 frame, không clip reward, PPO clip 0.1)",
                  "abl_clipreward": "clip reward về ±1", "abl_stack1": "chỉ 1 frame (không xếp chồng)",
                  "abl_noclip": "bỏ clip của PPO (ε = 1000)"}
        for r in (["ppo_s0"] if "ppo_s0" in runs_1p else []) + abl:
            ev = _eval_curve(r)
            ax.plot(ev["global_step"], ev["mean_return"], "o-", ms=3, label=labels.get(r, r))
        _finish(ax, fig, "results/ablation_curve.png", "Ablation – cùng ngân sách 2M step, seed 0",
                "Hiệu điểm (eval argmax, 5 ván)")


def plot_selfplay_curves():
    """Phần 2 (2 người): agent đấu random (eval định kỳ) và hiệu điểm khi đấu snapshot cũ."""
    runs = sorted(os.path.basename(d) for d in glob.glob("logs/sp_*")
                  if os.path.exists(os.path.join(d, "evals.csv")))
    if not runs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in runs:
        ev = load_csv(os.path.join("logs", r, "evals.csv"))
        if ev is not None and "vs_random_mean" in ev:
            a1.plot(ev["global_step"] / 1e6, ev["vs_random_mean"], "o-", ms=3, label=r)
        ep = load_csv(os.path.join("logs", r, "episodes.csv"))
        if ep is not None and "learner_side" in ep:
            m = ep["learner_side"] >= 0  # ván đấu snapshot cũ
            if m.sum() > 20:
                ret = np.where(ep["learner_side"][m] == 0, ep["white_return"][m], -ep["white_return"][m])
                ys = smooth(ret, 50)
                a2.plot(ep["global_step"][m][len(ret) - len(ys):] / 1e6, ys, label=r)
    for ax, t in ((a1, "Agent (argmax) đấu random, 4 ván/lần"), (a2, "Agent đấu snapshot cũ của chính nó (TB 50 ván)")):
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Triệu env-step")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    a1.set_ylabel("Hiệu điểm của agent")
    fig.tight_layout()
    fig.savefig("results/selfplay_curve.png", dpi=150)
    plt.close(fig)
    print("Đã lưu results/selfplay_curve.png")


def plot_league_curves():
    """Phần 2 bản cuối (league_ppo.py): hiệu điểm theo từng loại đối thủ + điểm/ván khi tự đánh."""
    runs = [r for r in ("league6", "league", "league_v4") if os.path.exists(os.path.join("logs", r, "episodes.csv"))]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in runs:
        with open(os.path.join("logs", r, "episodes.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lab = {"league6": "v6 (cuối)", "league": "v5", "league_v4": "v4"}.get(r, r)
        for opp, ax, key, title in (("self", axes[1], "points_total", "Tự đánh với chính mình: tổng điểm/ván (cao = có đánh nhau)"),
                                    ("cpu", axes[0], "learner_return", "Hiệu điểm khi đấu AI máy / random")):
            sel = [(float(x["global_step"]), float(x[key])) for x in rows if x["opponent"] == opp and x[key] != "nan"]
            if len(sel) > 10:
                xs, ys = np.array(sel).T
                sm = smooth(ys, 30)
                ax.plot(xs[len(ys) - len(sm):] / 1e6, sm, label=f"{lab} – {'AI máy' if opp == 'cpu' else 'tự đánh'}")
        sel = [(float(x["global_step"]), float(x["learner_return"])) for x in rows if x["opponent"] == "random"]
        if len(sel) > 10:
            xs, ys = np.array(sel).T
            sm = smooth(ys, 30)
            axes[0].plot(xs[len(ys) - len(sm):] / 1e6, sm, ls="--", label=f"{lab} – random")
    for ax, t in zip(axes, ("Hiệu điểm khi đấu AI máy (liền) và random (đứt)", "Tự đánh với chính mình: tổng điểm/ván (cao = có đánh nhau)")):
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Triệu env-step")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/league_curve.png", dpi=150)
    plt.close(fig)
    print("Đã lưu results/league_curve.png")


def plot_comparison():
    files = sorted(glob.glob("results/*.json"))
    if not files:
        print("Chưa có results/*.json (chạy evaluate.py trước)")
        return
    loaded = [(os.path.basename(f)[:-5], json.load(open(f, encoding="utf-8"))) for f in files]
    # chỉ lấy file của evaluate.py, difficulty 0, argmax (các biến thể khác in bảng riêng ở báo cáo)
    loaded = [(n, d) for n, d in loaded if "mean_return" in d and n.endswith("_d0")]
    names = [n for n, _ in loaded]
    data = [d for _, d in loaded]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(11, 0.9 * len(names) + 4), 4.8))
    a1.bar(names, [d["mean_return"] for d in data], yerr=[d["std_return"] for d in data], capsize=4)
    a1.axhline(0, color="gray", lw=0.8)
    a1.set_ylabel("Hiệu điểm TB")
    a1.set_title("Điểm trung bình (± std)")
    a2.bar(names, [100 * d["win_rate"] for d in data], label="thắng")
    a2.bar(names, [100 * d["ko_rate"] for d in data], label="KO", alpha=0.7)
    a2.set_ylim(0, 105)
    a2.set_ylabel("%")
    a2.set_title("Tỉ lệ thắng / KO")
    a2.legend()
    for ax in (a1, a2):
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/comparison.png", dpi=150)
    print("Đã lưu results/comparison.png")

    print(f"\n{'agent':32s} {'điểm TB':>10s} {'thắng':>7s} {'KO':>6s}")
    for n, d in zip(names, data):
        print(f"{n:32s} {d['mean_return']:+6.1f}±{d['std_return']:<4.1f} {d['win_rate']:6.0%} {d['ko_rate']:6.0%}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_learning_curves()
    plot_selfplay_curves()
    plot_league_curves()
    plot_comparison()
