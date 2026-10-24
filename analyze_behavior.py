"""
Phân tích hành vi agent bằng vị trí trong RAM, để kiểm chứng các nhận xét trong báo cáo:
  - Agent có dồn đối thủ vào dây đài (sát mép sàn) không?
  - Agent đứng cách đối thủ bao xa khi ghi điểm?
  - Agent hay dùng những action nào?

  python analyze_behavior.py --model models/ppo_s0/final.pt --episodes 5
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình; phải đặt TRƯỚC khi import numpy
import argparse
import json
import os

import numpy as np
import torch

from envs import make_vec_env, read_state

ACTION_NAMES = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
                "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE",
                "DOWNLEFTFIRE"]


def run(agent: str, model_path: str | None, episodes: int, seed: int):
    envs = make_vec_env(1, asynchronous=False)
    raw = envs.envs[0]
    model = None
    if agent == "ppo":
        from ppo import load_policy
        model = load_policy(model_path)
    rng = np.random.default_rng(seed)
    obs, _ = envs.reset(seed=seed)
    pos, hits, actions, done = [], [], [], 0
    while done < episodes:
        s = read_state(raw.unwrapped.ale.getRAM())
        if model is not None:
            with torch.no_grad():
                a = model.get_action_and_value(torch.as_tensor(obs), deterministic=True)[0].numpy()
        else:
            a = rng.integers(0, 18, size=1)
        obs, r, term, trunc, _ = envs.step(a)
        pos.append((s["player_x"], s["player_y"], s["enemy_x"], s["enemy_y"]))
        actions.append(int(a[0]))
        if r[0] > 0:
            hits.append((s["player_x"], s["player_y"], s["enemy_x"], s["enemy_y"]))
        done += int(term[0] or trunc[0])
    envs.close()
    return np.array(pos), np.array(hits), np.array(actions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_s0/final.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--out", default="results/behavior.json")
    a = p.parse_args()

    data = {name: run(name, a.model, a.episodes, a.seed) for name in ["random", "ppo"]}
    # biên sàn đấu: lấy từ vị trí min/max quan sát được của cả 2 võ sĩ, mọi agent
    allpos = np.concatenate([d[0] for d in data.values()])
    xs = np.concatenate([allpos[:, 0], allpos[:, 2]])
    ys = np.concatenate([allpos[:, 1], allpos[:, 3]])
    x_lo, x_hi, y_lo, y_hi = xs.min(), xs.max(), ys.min(), ys.max()
    margin = 8
    print(f"Biên quan sát: x [{x_lo}, {x_hi}], y [{y_lo}, {y_hi}]; 'sát dây' = cách biên ≤ {margin}")

    report = {}
    for name, (pos, hits, acts) in data.items():
        ex, ey = pos[:, 2], pos[:, 3]
        near = (ex - x_lo <= margin) | (x_hi - ex <= margin) | (ey - y_lo <= margin) | (y_hi - ey <= margin)
        dx = np.abs(pos[:, 2] - pos[:, 0])
        dy = np.abs(pos[:, 3] - pos[:, 1])
        hx = np.abs(hits[:, 2] - hits[:, 0]) if len(hits) else np.array([np.nan])
        hy = np.abs(hits[:, 3] - hits[:, 1]) if len(hits) else np.array([np.nan])
        hnear = ((hits[:, 2] - x_lo <= margin) | (x_hi - hits[:, 2] <= margin) |
                 (hits[:, 3] - y_lo <= margin) | (y_hi - hits[:, 3] <= margin)) if len(hits) else np.array([False])
        counts = np.bincount(acts, minlength=18) / len(acts)
        top = sorted(range(18), key=lambda i: -counts[i])[:5]
        fire_share = counts[[1] + list(range(10, 18))].sum()
        report[name] = {
            "steps": int(len(pos)), "hit_steps": int(len(hits)),
            "enemy_near_rope_pct": float(near.mean() * 100),
            "enemy_near_rope_when_hit_pct": float(hnear.mean() * 100),
            "mean_dx": float(dx.mean()), "mean_dy": float(dy.mean()),
            "hit_mean_dx": float(np.nanmean(hx)), "hit_mean_dy": float(np.nanmean(hy)),
            "fire_action_pct": float(fire_share * 100),
            "top_actions": {ACTION_NAMES[i]: round(float(counts[i]) * 100, 1) for i in top},
        }
        r = report[name]
        print(f"\n== {name.upper()} ({a.episodes} ván, {r['steps']} step, {r['hit_steps']} step có ghi điểm)")
        print(f"  Đối thủ sát dây: {r['enemy_near_rope_pct']:.0f}% thời gian, "
              f"{r['enemy_near_rope_when_hit_pct']:.0f}% số lần mình ghi điểm")
        print(f"  Khoảng cách TB tới đối thủ: |dx| {r['mean_dx']:.1f}, |dy| {r['mean_dy']:.1f}; "
              f"lúc ghi điểm |dx| {r['hit_mean_dx']:.1f}, |dy| {r['hit_mean_dy']:.1f}")
        print(f"  Tỉ lệ action có đấm: {r['fire_action_pct']:.0f}% | hay dùng nhất: {r['top_actions']}")
    os.makedirs("results", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nĐã lưu", a.out)


if __name__ == "__main__":
    main()
