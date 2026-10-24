"""
Đánh giá agent trên N ván theo chuẩn v5 (sticky actions 0.25, reward không clip, noop reset).
Seed đánh giá (10000+) tách biệt với seed lúc train.

  python evaluate.py --agent random --episodes 30
  python evaluate.py --agent heuristic --episodes 30
  python evaluate.py --agent ppo --model models/ppo_s0/final.pt --episodes 30
  python evaluate.py --agent ppo --model models/ppo_s0/final.pt --difficulty 3
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình; phải đặt TRƯỚC khi import numpy

import argparse
import json
import os
import time

import numpy as np
import torch

from envs import finished_episodes, make_vec_env

EVAL_SEED_BASE = 10_000
ROUND_FRAMES = 7141  # 1 hiệp đủ 2 phút; ván kết thúc sớm hơn => có KO


def evaluate(agent: str = "ppo", episodes: int = 30, model=None, model_path: str | None = None,
             difficulty: int = 0, n_envs: int = 5, seed: int = EVAL_SEED_BASE,
             deterministic: bool = True, device: str = "cpu", verbose: bool = True,
             frame_stack: int = 4) -> dict:
    n_envs = min(n_envs, episodes)
    if agent == "ppo" and model is None:
        from ppo import load_policy_meta
        model, meta = load_policy_meta(model_path, device)
        frame_stack = meta.get("frame_stack", 4)  # model ablation 1 frame cần env 1 frame
    envs = make_vec_env(n_envs, difficulty, asynchronous=False, frame_stack=frame_stack)  # Sync để đọc RAM
    if agent == "heuristic":
        from heuristic_agent import HeuristicAgent
        heur = HeuristicAgent()
    rng = np.random.default_rng(seed)

    obs, _ = envs.reset(seed=seed)
    quota = np.array([(episodes + i) // n_envs for i in range(n_envs)])  # mỗi env chạy bao nhiêu ván
    done_count = np.zeros(n_envs, dtype=int)
    pts_me, pts_cpu = np.zeros(n_envs), np.zeros(n_envs)
    results, t0 = [], time.time()

    while len(results) < episodes:
        if agent == "random":
            actions = rng.integers(0, envs.single_action_space.n, size=n_envs)
        elif agent == "heuristic":
            actions = np.array([heur.act(e.unwrapped.ale.getRAM()) for e in envs.envs])
        else:
            with torch.no_grad():
                a, _, _, _ = model.get_action_and_value(torch.as_tensor(obs, device=device),
                                                        deterministic=deterministic)
            actions = a.cpu().numpy()

        obs, rewards, term, trunc, infos = envs.step(actions)
        pts_me += np.clip(rewards, 0, None)
        pts_cpu += np.clip(-rewards, 0, None)

        for i, ret, frames in finished_episodes(infos):
            if done_count[i] < quota[i]:
                done_count[i] += 1
                results.append({
                    "return": ret, "player_points": float(pts_me[i]), "cpu_points": float(pts_cpu[i]),
                    "frames": frames, "win": ret > 0, "draw": ret == 0,
                    "ko_win": ret > 0 and frames < ROUND_FRAMES - 50,
                })
                if verbose:
                    r = results[-1]
                    tag = "KO!" if r["ko_win"] else ("THẮNG" if r["win"] else ("hòa" if r["draw"] else "thua"))
                    print(f"  ván {len(results):3d}: ~{pts_me[i]:3.0f} - {pts_cpu[i]:<3.0f}"
                          f" hiệu {ret:+4.0f}  {tag}", flush=True)
            pts_me[i] = pts_cpu[i] = 0
    envs.close()

    rets = np.array([r["return"] for r in results])
    return {
        "agent": agent, "model": model_path, "difficulty": difficulty, "episodes": episodes,
        "deterministic": deterministic,
        "mean_return": float(rets.mean()), "std_return": float(rets.std()),
        "min_return": float(rets.min()), "max_return": float(rets.max()),
        "win_rate": float(np.mean([r["win"] for r in results])),
        "draw_rate": float(np.mean([r["draw"] for r in results])),
        "loss_rate": float(np.mean([not r["win"] and not r["draw"] for r in results])),
        "ko_rate": float(np.mean([r["ko_win"] for r in results])),
        "mean_player_points": float(np.mean([r["player_points"] for r in results])),
        "mean_cpu_points": float(np.mean([r["cpu_points"] for r in results])),
        "eval_seconds": round(time.time() - t0, 1),
        "episodes_detail": results,
    }


def print_summary(s: dict):
    print("=" * 60)
    print(f"Agent: {s['agent']} | difficulty {s['difficulty']} | {s['episodes']} ván"
          f" | {'argmax' if s['deterministic'] else 'lấy mẫu'}")
    print(f"Điểm TB: {s['mean_return']:+.1f} ± {s['std_return']:.1f}  (min {s['min_return']:+.0f}, max {s['max_return']:+.0f})")
    print(f"Thắng {s['win_rate']:.0%} | Hòa {s['draw_rate']:.0%} | Thua {s['loss_rate']:.0%} | KO {s['ko_rate']:.0%}")
    print(f"Điểm mình ~{s['mean_player_points']:.1f} vs máy ~{s['mean_cpu_points']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["random", "heuristic", "ppo"], required=True)
    p.add_argument("--model", help="file .pt (cho ppo)")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--difficulty", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--n-envs", type=int, default=5)
    p.add_argument("--seed", type=int, default=EVAL_SEED_BASE)
    p.add_argument("--stochastic", action="store_true", help="lấy mẫu action thay vì argmax")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="file JSON (mặc định results/<tên>_d<difficulty>.json)")
    a = p.parse_args()
    if a.agent == "ppo" and not a.model:
        p.error("--model là bắt buộc với ppo")

    s = evaluate(a.agent, a.episodes, model_path=a.model, difficulty=a.difficulty, n_envs=a.n_envs,
                 seed=a.seed, deterministic=not a.stochastic, device=a.device)
    print_summary(s)
    os.makedirs("results", exist_ok=True)
    name = a.agent if not a.model else (
        f"{os.path.basename(os.path.dirname(a.model))}_{os.path.splitext(os.path.basename(a.model))[0]}")
    out = a.out or f"results/{name}_d{a.difficulty}{'_stoch' if a.stochastic else ''}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print("Đã lưu", out)
