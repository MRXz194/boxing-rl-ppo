"""
Vòng lặp train PPO, dùng thuật toán tự cài trong ppo.py.

  python train_ppo.py --timesteps 50000 --run-name smoke            # chạy thử
  python train_ppo.py --timesteps 2000000 --seed 0                  # ~1 giờ trên CPU máy này
  python train_ppo.py --timesteps 10000000 --seed 0 --resume        # cloud; --resume để train tiếp
  tensorboard --logdir logs

Mỗi vòng lặp (1 "update"):
  (1) Thu rollout: 8 env x 128 bước bằng policy hiện tại.
  (2) Tính advantage bằng GAE.
  (3) Cập nhật mạng: 4 epoch x 4 minibatch với loss PPO-Clip.
  (4) Giảm dần learning rate và clip_range; ghi log; lưu checkpoint; eval định kỳ.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình; phải đặt TRƯỚC khi import numpy

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np

from envs import N_ACTIONS, finished_episodes, make_vec_env

# torch, ppo, evaluate được import BÊN TRONG main(): trên Windows, mỗi tiến trình con chạy game (AsyncVectorEnv,
# kiểu "spawn") đều import lại file này; import torch ở đầu file thì 8 tiến trình con tốn thêm ~2 GB vô ích.


def main():
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from evaluate import evaluate
    from ppo import ActorCritic, PPOConfig, RolloutBuffer, compute_gae, ppo_update, save_policy

    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=10_000_000, help="tổng số step (1 step = 4 frame), cộng dồn qua mọi env")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--device", default="auto")
    p.add_argument("--run-name", default=None, help="mặc định ppo_s<seed>")
    p.add_argument("--checkpoint-freq", type=int, default=500_000)
    p.add_argument("--eval-freq", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=4, help="ablation: số frame xếp chồng")
    p.add_argument("--clip-reward", action="store_true", help="ablation: clip reward về {-1,0,1} khi train")
    p.add_argument("--threads", type=int, default=4, help="số luồng PyTorch trên CPU")
    p.add_argument("--resume", action="store_true", help="train tiếp từ models/<run>/last.pt")
    # hyperparam PPO (mặc định theo RL-Zoo atari)
    d = PPOConfig()
    for k, v in asdict(d).items():
        if k != "n_envs":
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    a = p.parse_args()
    cfg = PPOConfig(**{k: getattr(a, k) for k in asdict(d) if k != "n_envs"}, n_envs=a.n_envs)
    cfg.anneal = bool(cfg.anneal)

    run = a.run_name or f"ppo_s{a.seed}"
    model_dir, log_dir = os.path.join("models", run), os.path.join("logs", run)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    if device.type == "cpu":
        torch.set_num_threads(a.threads)  # đo thực tế: 4 luồng nhanh nhất khi chạy cùng 8 tiến trình game

    # ---- seed để tái lập kết quả
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    envs = make_vec_env(cfg.n_envs, a.difficulty, asynchronous=True, frame_stack=a.frame_stack,
                        clip_reward=a.clip_reward)
    meta = {"frame_stack": a.frame_stack}
    obs_shape = envs.single_observation_space.shape  # (4, 84, 84)
    model = ActorCritic(N_ACTIONS, obs_shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    batch = cfg.n_envs * cfg.n_steps
    num_updates = a.timesteps // batch
    start_update, global_step, best_eval = 1, 0, -float("inf")

    last_path = os.path.join(model_dir, "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_update, global_step, best_eval = ck["update"] + 1, ck["global_step"], ck["best_eval"]
        print(f"Train tiếp từ step {global_step:,}")

    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "timesteps": a.timesteps, "seed": a.seed, "difficulty": a.difficulty,
                   "device": str(device), "frame_stack": a.frame_stack,
                   "clip_reward": a.clip_reward}, f, indent=2)
    writer = SummaryWriter(log_dir)
    ep_file = open(os.path.join(log_dir, "episodes.csv"), "a", newline="")
    ep_csv = csv.writer(ep_file)
    ev_file = open(os.path.join(log_dir, "evals.csv"), "a", newline="")
    ev_csv = csv.writer(ev_file)
    # file mới thì ghi tiêu đề. Không dùng tell(): ở chế độ append trên Windows nó có thể trả về 0
    if os.path.getsize(os.path.join(log_dir, "episodes.csv")) == 0:
        ep_csv.writerow(["global_step", "return", "frames", "wall_time"])
    if os.path.getsize(os.path.join(log_dir, "evals.csv")) == 0:
        ev_csv.writerow(["global_step", "mean_return", "std_return", "win_rate"])

    print(f"Run '{run}' | PPO tự cài | {cfg.n_envs} env | device={device} | "
          f"{num_updates} update x {batch} step = {num_updates * batch:,} step")

    buf = RolloutBuffer(cfg.n_steps, cfg.n_envs, obs_shape)
    next_obs, _ = envs.reset(seed=a.seed + 1000 * start_update)
    next_done = np.zeros(cfg.n_envs, dtype=np.float32)
    recent = []
    t0, step0 = time.time(), global_step
    next_ckpt = (global_step // a.checkpoint_freq + 1) * a.checkpoint_freq
    next_eval = (global_step // a.eval_freq + 1) * a.eval_freq

    for update in range(start_update, num_updates + 1):
        # (4) giảm tuyến tính lr và clip_range: 1.0 ở update đầu -> ~0 ở update cuối
        frac = 1.0 - (update - 1) / num_updates if cfg.anneal else 1.0
        optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate
        clip_range = frac * cfg.clip_range

        # (1) thu rollout
        model.eval()
        for t in range(cfg.n_steps):
            global_step += cfg.n_envs
            buf.obs[t] = next_obs
            buf.dones[t] = next_done
            with torch.no_grad():
                action, logp, _, value = model.get_action_and_value(torch.as_tensor(next_obs, device=device))
            buf.actions[t] = action.cpu().numpy()
            buf.logprobs[t] = logp.cpu().numpy()
            buf.values[t] = value.cpu().numpy()

            next_obs, reward, term, trunc, infos = envs.step(buf.actions[t])
            buf.rewards[t] = reward
            next_done = (term | trunc).astype(np.float32)

            for _, ret, frames in finished_episodes(infos):
                recent = (recent + [ret])[-20:]
                ep_csv.writerow([global_step, ret, frames, round(time.time(), 1)])
                writer.add_scalar("charts/episode_return", ret, global_step)

        # (2) GAE: bootstrap bằng V(s) của obs cuối cùng
        with torch.no_grad():
            last_value = model.get_value(torch.as_tensor(next_obs, device=device)).cpu().numpy()
        compute_gae(buf, last_value, next_done, cfg.gamma, cfg.gae_lambda)

        # (3) cập nhật mạng
        model.train()
        stats = ppo_update(model, optimizer, buf, cfg, clip_range, device)

        # ---- log
        fps = int((global_step - step0) / (time.time() - t0))
        for k, v in stats.items():
            writer.add_scalar(f"losses/{k}", v, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/fps", fps, global_step)
        if update % 10 == 0 or update == num_updates:
            r = f"{np.mean(recent):+6.1f}" if recent else "   n/a"
            print(f"step {global_step:>9,} | reward TB 20 ván {r} | entropy {stats['entropy']:.2f} | "
                  f"kl {stats['approx_kl']:.4f} | clip {stats['clip_frac']:.2f} | "
                  f"EV {stats['explained_variance']:.2f} | {fps} step/s", flush=True)
            ep_file.flush()

        # ---- eval định kỳ (argmax policy), lưu best
        if global_step >= next_eval or update == num_updates:
            next_eval += a.eval_freq
            model.eval()
            s = evaluate("ppo", a.eval_episodes, model=model, difficulty=a.difficulty,
                         seed=20_000 + global_step, device=str(device), verbose=False, frame_stack=a.frame_stack)
            ev_csv.writerow([global_step, s["mean_return"], s["std_return"], s["win_rate"]])
            ev_file.flush()
            writer.add_scalar("eval/mean_return", s["mean_return"], global_step)
            writer.add_scalar("eval/win_rate", s["win_rate"], global_step)
            msg = ""
            if s["mean_return"] > best_eval:
                best_eval = s["mean_return"]
                save_policy(os.path.join(model_dir, "best.pt"), model, {**meta, "global_step": global_step})
                msg = " -> lưu best.pt"
            print(f"[EVAL] step {global_step:,}: {s['mean_return']:+.1f} ± {s['std_return']:.1f}, "
                  f"thắng {s['win_rate']:.0%}{msg}", flush=True)

        # ---- last.pt (để --resume) mỗi ~100k step; ckpt_<step>.pt giữ lại mỗi checkpoint_freq
        if update % max(1, 100_000 // batch) == 0 or global_step >= next_ckpt or update == num_updates:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                     "global_step": global_step, "best_eval": best_eval}
            torch.save(state, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)  # ghi file tạm rồi đổi tên: không hỏng file nếu bị ngắt giữa chừng
            if global_step >= next_ckpt:
                next_ckpt += a.checkpoint_freq
                torch.save(state, os.path.join(model_dir, f"ckpt_{global_step}.pt"))

    save_policy(os.path.join(model_dir, "final.pt"), model, {**meta, "global_step": global_step})
    print(f"Xong {global_step:,} step sau {(time.time() - t0) / 60:.1f} phút. Đã lưu {model_dir}/final.pt và best.pt")
    envs.close()
    writer.close()
    ep_file.close()
    ev_file.close()


if __name__ == "__main__":
    main()
