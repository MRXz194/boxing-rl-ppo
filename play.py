"""
Xem agent đánh Boxing: mở cửa sổ trực tiếp (--live, dùng khi demo) hoặc quay video mp4
(15 fps, đúng tốc độ thật vì mỗi step là 4 frame).

  python play.py --agent ppo --model models/ppo_s0/final.pt --live      # xem trực tiếp
  python play.py --agent ppo --model models/ppo_s0/final.pt             # quay mp4
  python play.py --agent heuristic --live
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình; phải đặt TRƯỚC khi import numpy
import argparse
import os

import cv2
import numpy as np
import torch

from envs import finished_episodes, make_vec_env

SCALE = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["random", "heuristic", "ppo"], required=True)
    p.add_argument("--model")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--seed", type=int, default=None, help="mặc định: --live thì mỗi lần một ván mới; video thì 12345")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out")
    p.add_argument("--live", action="store_true", help="mở cửa sổ xem trực tiếp (60 fps) thay vì quay video")
    a = p.parse_args()
    if a.seed is None:
        import time
        a.seed = int(time.time()) % 100_000 if a.live else 12_345

    envs = make_vec_env(1, a.difficulty, asynchronous=False, render_mode="human" if a.live else "rgb_array")
    raw = envs.envs[0]
    if a.agent == "ppo":
        from ppo import load_policy
        model = load_policy(a.model)
    elif a.agent == "heuristic":
        from heuristic_agent import HeuristicAgent
        heur = HeuristicAgent()

    os.makedirs("videos", exist_ok=True)
    out = a.out or f"videos/{a.agent}_d{a.difficulty}.mp4"
    writer, me, cpu, finished = None, 0.0, 0.0, 0
    obs, _ = envs.reset(seed=a.seed)
    while finished < a.episodes:
        if a.agent == "ppo":
            with torch.no_grad():
                act, _, _, _ = model.get_action_and_value(torch.as_tensor(obs), deterministic=not a.stochastic)
            action = act.numpy()
        elif a.agent == "heuristic":
            action = np.array([heur.act(raw.unwrapped.ale.getRAM())])
        else:
            action = envs.action_space.sample()

        frame = None if a.live else raw.render()  # vẽ trước khi step để không dính frame đầu ván mới khi auto-reset
        obs, r, term, trunc, infos = envs.step(action)  # chế độ live: ALE tự vẽ lên cửa sổ
        me += max(r[0], 0)
        cpu += max(-r[0], 0)
        label = f"{a.agent.upper()}  {me:.0f} - {cpu:.0f}"
        for _, ret, _ in finished_episodes(infos):
            finished += 1
            print(f"Ván {finished}: hiệu {ret:+.0f}")
            me = cpu = 0.0
        if a.live:
            continue

        frame = cv2.resize(frame, (frame.shape[1] * SCALE, frame.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, label, (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    envs.close()
    if writer is not None:
        writer.release()
        print("Đã lưu video:", out)


if __name__ == "__main__":
    main()
