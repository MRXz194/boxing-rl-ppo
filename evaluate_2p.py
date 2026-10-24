"""
Đánh giá Boxing 2 người (chạy trên Windows, Linux, macOS).

  # 1. agent đấu random, cầm trắng một nửa số ván và cầm đen một nửa (kiểm tra cả 2 phe đều mạnh)
  python evaluate_2p.py vs-random --model models/sp_pool/final.pt --games 20
  # 2. agent self-play cầm trắng đấu AI có sẵn của game (chưa từng gặp lúc train)
  python evaluate_2p.py vs-cpu --model models/sp_pool/final.pt --episodes 30
  # 3. giải đấu vòng tròn: Elo (Bradley-Terry) + heatmap tỉ lệ thắng
  python evaluate_2p.py league --players random models/sp_pool/snapshots/snap_*.pt ... --games 8 --name main
  # 4. một cặp đấu bất kỳ
  python evaluate_2p.py match --a models/sp_pool/final.pt --b models/ppo_s0/final.pt --games 20

Người chơi: "random" hoặc đường dẫn .pt. Model của phần 1 (train với AI máy, không che HUD) được nhận ra
qua metadata và chỉ được cho cầm phe trắng với góc nhìn gốc của nó (xem `seat_mask`).
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình; phải đặt TRƯỚC khi import numpy

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

from envs_2p import Boxing1PCustom, make_vec_env_2p

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROUND_FRAMES = 7141


class Player:
    """spec: "random" | "model.pt" | "white.pt+black.pt" (IPPO: mạng riêng cho mỗi phe)."""

    def __init__(self, spec: str, device: str = "cpu", deterministic: bool = True):
        from ppo import load_policy_meta
        self.spec, self.device, self.deterministic = spec, device, deterministic
        if spec == "random":
            self.models, self.meta, self.name = None, {"mask_hud": True}, "random"
        elif "+" in spec:
            w, b = spec.split("+")
            (mw, self.meta), (mb, _) = load_policy_meta(w, device), load_policy_meta(b, device)
            self.models = (mw, mb)
            self.name = short_name(w).replace("_w", "") + "(w+b)"
        else:
            m, self.meta = load_policy_meta(spec, device)
            self.models = (m, m)
            self.name = short_name(spec)
        self.mask_hud = bool(self.meta.get("mask_hud", False))  # model phần 1 không có -> False

    def act(self, obs: np.ndarray, rng: np.random.Generator, sides) -> np.ndarray:
        """obs: (n, 4, 84, 84); sides: phe (0 trắng / 1 đen) của từng obs."""
        n = len(obs)
        if self.models is None:
            return rng.integers(0, 18, n)
        sides = np.broadcast_to(np.asarray(sides), (n,))
        out = np.zeros(n, dtype=np.int64)
        with torch.no_grad():
            for side in (0, 1):
                m = sides == side
                if m.any():
                    a = self.models[side].get_action_and_value(torch.as_tensor(obs[m], device=self.device),
                                                               deterministic=self.deterministic)[0]
                    out[m] = a.cpu().numpy()
        return out


def short_name(path: str) -> str:
    run = os.path.basename(os.path.dirname(path))
    if run == "snapshots":
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("snap_"):
        parts = base.split("_")
        base = f"{int(parts[1]) / 1e6:.2f}M" + (f"_{parts[2]}" if len(parts) > 2 else "")
    return f"{run}/{base}"


def play_match(A: Player, B: Player, games: int = 8, seed: int = 0) -> list[dict]:
    """games ván: một nửa A cầm trắng, một nửa B cầm trắng. Kết quả theo góc nhìn của A."""
    k = games // 2
    a_side = np.array([0] * k + [1] * (games - k))
    # che HUD theo yêu cầu của người ngồi ở mỗi phe; phe đen (lật gương) luôn được che
    masks = [((A if s == 0 else B).mask_hud, True) for s in a_side]
    envs = make_vec_env_2p(games, asynchronous=False, mask_hud=masks)
    obs, _ = envs.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out, done = [None] * games, np.zeros(games, bool)
    idx = np.arange(games)
    while not done.all():
        act = np.zeros((games, 2), dtype=np.int64)
        act[idx, a_side] = A.act(obs[idx, a_side], rng, a_side)
        act[idx, 1 - a_side] = B.act(obs[idx, 1 - a_side], rng, 1 - a_side)
        obs, r, term, trunc, infos = envs.step(act)
        fi = infos.get("final_info", {})
        if "episode" in fi:
            for i in np.flatnonzero(fi["_episode"]):
                if not done[i]:
                    done[i] = True
                    e = {k2: float(v[i]) for k2, v in fi["episode"].items() if not k2.startswith("_")}
                    sign = 1 if a_side[i] == 0 else -1
                    pa, pb = (e["pts_w"], e["pts_b"]) if sign == 1 else (e["pts_b"], e["pts_w"])
                    out[i] = {"a_side": "white" if sign == 1 else "black", "a_return": sign * e["r"],
                              "a_points": pa, "b_points": pb, "frames": int(e["l"]),
                              "ko": int(e["l"]) < ROUND_FRAMES - 50}
    envs.close()
    return out


def summarize(res: list[dict]) -> dict:
    r = np.array([x["a_return"] for x in res])
    by_side = {s: [x["a_return"] for x in res if x["a_side"] == s] for s in ("white", "black")}
    return {"games": len(r), "mean": float(r.mean()), "std": float(r.std()),
            "win": float((r > 0).mean()), "draw": float((r == 0).mean()), "loss": float((r < 0).mean()),
            "ko": float(np.mean([x["ko"] and x["a_return"] > 0 for x in res])),
            "mean_as_white": float(np.mean(by_side["white"])) if by_side["white"] else None,
            "mean_as_black": float(np.mean(by_side["black"])) if by_side["black"] else None}


def bradley_terry_elo(names: list[str], score: np.ndarray, n_games: np.ndarray, anchor: str | None = "random",
                      iters: int = 5000) -> np.ndarray:
    """
    Ước lượng Elo bằng mô hình Bradley-Terry (cực đại likelihood, gradient ascent):
        P(i thắng j) = 1 / (1 + 10^((R_j - R_i)/400))
    score[i,j] = tổng điểm i ghi được trước j (thắng 1, hòa 0.5, thua 0); n_games[i,j] = số ván.
    Thêm prior nhỏ (0.5 ván hòa ảo với mỗi đối thủ) để Elo không ra vô cực khi thắng 100%.
    Neo: anchor (random) = 0 Elo; nếu không có thì trung bình = 1000.
    """
    n = len(names)
    s = score + 0.25 * (n_games > 0)
    g = n_games + 0.5 * (n_games > 0)
    R = np.zeros(n)
    c = np.log(10) / 400
    for _ in range(iters):
        p = 1 / (1 + np.exp(-c * (R[:, None] - R[None, :])))
        grad = (s - g * p).sum(1)
        R += 20 * grad / max(1.0, g.sum(1).max())
        R -= R.mean()
    if anchor in names:
        R -= R[names.index(anchor)]
    else:
        R += 1000
    return R


def run_league(specs: list[str], games: int, name: str, device: str):
    players = [Player(s, device) for s in specs]
    names = [p.name for p in players]
    n = len(players)
    score = np.zeros((n, n))
    ng = np.zeros((n, n))
    mean_diff = np.full((n, n), np.nan)
    pairs = list(itertools.combinations(range(n), 2))
    t0 = time.time()
    for c, (i, j) in enumerate(pairs, 1):
        res = play_match(players[i], players[j], games, seed=50_000 + 97 * i + j)
        for x in res:
            sc = 1.0 if x["a_return"] > 0 else (0.5 if x["a_return"] == 0 else 0.0)
            score[i, j] += sc
            score[j, i] += 1 - sc
            ng[i, j] += 1
            ng[j, i] += 1
        md = float(np.mean([x["a_return"] for x in res]))
        mean_diff[i, j], mean_diff[j, i] = md, -md
        el = time.time() - t0
        print(f"[{c}/{len(pairs)}] {names[i]} vs {names[j]}: hiệu điểm TB {md:+.1f}, "
              f"điểm {score[i, j]:.1f}/{games}  ({el / c * (len(pairs) - c) / 60:.0f} phút còn lại)", flush=True)
    elo = bradley_terry_elo(names, score, ng)
    order = np.argsort(-elo)
    os.makedirs("results", exist_ok=True)
    out = {"players": names, "specs": specs, "elo": elo.tolist(), "score": score.tolist(), "games": ng.tolist(),
           "mean_diff": np.nan_to_num(mean_diff).tolist(), "games_per_pair": games}
    with open(f"results/league_{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nBẢNG XẾP HẠNG ELO (random = 0)")
    for r, i in enumerate(order, 1):
        wr = score[i].sum() / max(ng[i].sum(), 1)
        print(f"{r:2d}. {names[i]:32s} Elo {elo[i]:+7.0f} | tỉ lệ điểm {wr:.0%}")
    plot_league(out, name)


def plot_league(out: dict, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, elo = out["players"], np.array(out["elo"])
    score, ng = np.array(out["score"]), np.array(out["games"])
    wr = np.where(ng > 0, score / np.maximum(ng, 1), np.nan)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n + 3), max(5, 0.5 * n + 2)))
    im = ax.imshow(wr, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n), names, rotation=70, fontsize=7, ha="right")
    ax.set_yticks(range(n), names, fontsize=7)
    for i in range(n):
        for j in range(n):
            if not np.isnan(wr[i, j]):
                ax.text(j, i, f"{wr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Tỉ lệ điểm của hàng (dòng) khi đấu cột (1 = thắng hết)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(f"results/league_{name}_heatmap.png", dpi=150)

    # Elo theo thời gian train cho từng run (snapshot có dạng run/xx.xxM)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    runs = {}
    for nm, e in zip(names, elo):
        if "/" in nm and nm.split("/")[1][:-1].replace(".", "").isdigit() and nm.endswith("M"):
            runs.setdefault(nm.split("/")[0], []).append((float(nm.split("/")[1][:-1]), e))
    for run, pts in runs.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=run)
    for nm, e in zip(names, elo):
        if not (nm.endswith("M")):
            ax.axhline(e, ls=":", lw=1, color="gray")
            ax.text(0, e, f" {nm}", fontsize=7, va="bottom", color="gray")
    ax.set_xlabel("Triệu env-step đã train")
    ax.set_ylabel("Elo (random = 0)")
    ax.set_title("Boxing 2 người – Elo của các snapshot theo thời gian")
    ax.grid(alpha=0.3)
    if runs:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/league_{name}_elo.png", dpi=150)
    print(f"Đã lưu results/league_{name}.json, _heatmap.png, _elo.png")


def vs_cpu(spec: str, episodes: int, difficulty: int, device: str) -> dict:
    """Agent cầm trắng đấu AI có sẵn của game, dùng cùng tiền xử lý tự viết như lúc self-play."""
    import gymnasium as gym
    P = Player(spec, device)
    n = min(5, episodes)
    envs = gym.vector.SyncVectorEnv([lambda: Boxing1PCustom(difficulty, P.mask_hud) for _ in range(n)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = envs.reset(seed=10_000)
    rng = np.random.default_rng(0)
    rets, quota = [], np.array([(episodes + i) // n for i in range(n)])
    cnt = np.zeros(n, int)
    while len(rets) < episodes:
        obs, r, term, trunc, infos = envs.step(P.act(obs, rng, 0))
        fi = infos.get("final_info", {})
        if "episode" in fi:
            for i in np.flatnonzero(fi["_episode"]):
                if cnt[i] < quota[i]:
                    cnt[i] += 1
                    rets.append(float(fi["episode"]["r"][i]))
    envs.close()
    rets = np.array(rets)
    return {"model": spec, "difficulty": difficulty, "episodes": episodes, "mean": float(rets.mean()),
            "std": float(rets.std()), "win": float((rets > 0).mean()), "returns": rets.tolist()}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("vs-random"); s1.add_argument("--model", required=True); s1.add_argument("--games", type=int, default=20)
    s2 = sub.add_parser("vs-cpu"); s2.add_argument("--model", required=True); s2.add_argument("--episodes", type=int, default=30)
    s2.add_argument("--difficulty", type=int, default=0)
    s3 = sub.add_parser("league"); s3.add_argument("--players", nargs="+", required=True)
    s3.add_argument("--games", type=int, default=8); s3.add_argument("--name", default="main")
    s4 = sub.add_parser("match"); s4.add_argument("--a", required=True); s4.add_argument("--b", required=True)
    s4.add_argument("--games", type=int, default=20)
    for s in (s1, s2, s3, s4):
        s.add_argument("--device", default="cpu")
        s.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_num_threads(4)
    os.makedirs("results", exist_ok=True)

    if a.cmd == "league":
        run_league(a.players, a.games, a.name, a.device)
        return
    if a.cmd == "vs-cpu":
        s = vs_cpu(a.model, a.episodes, a.difficulty, a.device)
        print(f"{short_name(a.model)} (trắng) vs AI máy, difficulty {a.difficulty}: "
              f"{s['mean']:+.1f} ± {s['std']:.1f}, thắng {s['win']:.0%} ({a.episodes} ván)")
        out = a.out or f"results/2p_vscpu_{short_name(a.model).replace('/', '_')}_d{a.difficulty}.json"
    else:
        A = Player(a.model if a.cmd == "vs-random" else a.a, a.device)
        B = Player("random" if a.cmd == "vs-random" else a.b, a.device)
        res = play_match(A, B, a.games, seed=60_000)
        s = {"a": A.name, "b": B.name, **summarize(res), "detail": res}
        print(f"{A.name} vs {B.name}: hiệu điểm TB {s['mean']:+.1f} ± {s['std']:.1f} | thắng {s['win']:.0%} "
              f"hòa {s['draw']:.0%} thua {s['loss']:.0%} | cầm trắng {s['mean_as_white']:+.1f}, cầm đen {s['mean_as_black']:+.1f}")
        out = a.out or f"results/2p_{a.cmd}_{A.name.replace('/', '_')}_vs_{B.name.replace('/', '_')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print("Đã lưu", out)


if __name__ == "__main__":
    main()
