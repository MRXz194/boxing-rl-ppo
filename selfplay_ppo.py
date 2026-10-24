"""
Self-play PPO cho Boxing 2 người: cả võ sĩ trắng lẫn đen đều do agent điều khiển.
Lõi PPO (GAE, loss PPO-Clip) dùng lại ppo.py; file này viết thêm phần multi-agent.

3 chế độ (--mode):
  ippo     : Independent PPO. 2 mạng riêng (1 trắng, 1 đen), mỗi mạng chỉ học từ phe của mình.
             Đơn giản nhất, nhưng mỗi bên thấy "môi trường" luôn thay đổi vì đối thủ cũng đang học.
  selfplay : 1 mạng chung chơi CẢ HAI phe (nhờ chuẩn hóa góc nhìn trong envs_2p.py).
             Gấp đôi dữ liệu cho một mạng, đối thủ luôn ngang tài ngang sức.
  pool     : selfplay + OPPONENT POOL (Fictitious Self-Play). Cứ --snapshot-freq step lưu một bản sao;
             mỗi ván, với xác suất --pool-prob đối thủ là một bản sao cũ chọn ngẫu nhiên (agent học
             chỉ điều khiển 1 phe); còn lại đánh với chính mình. Tránh "quên" cách thắng các lối chơi
             cũ và tránh vòng lặp kiểu oẳn tù tì. (Ý tưởng như AlphaStar / OpenAI Five, bản đơn giản.)

Dữ liệu: n_envs env x 2 phe = 2*n_envs "luồng" (stream). Luồng s = env*2 + phe.
Mỗi luồng có "chủ": mạng đang học (được dùng để update) hoặc snapshot cũ (bị loại khỏi update qua mặt nạ).

  python selfplay_ppo.py --mode pool --timesteps 6000000 --run-name sp_pool            (Windows, không cần Linux)
  python selfplay_ppo.py --mode pool --timesteps 6000000 --run-name sp_pool --resume   (train tiếp)
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

from envs_2p import make_vec_env_2p

# torch và ppo được nạp BÊN TRONG hàm: trên Windows mỗi tiến trình con chạy game (AsyncVectorEnv "spawn")
# nạp lại file này; để torch ở đầu file thì 8 tiến trình con tốn thêm vài GB bộ nhớ vô ích.

META = {"frame_stack": 4, "mask_hud": True, "perspective": "mirror"}


def quick_eval_vs_random(models: list, device, games: int = 4, seed: int = 0) -> dict:
    """Agent (argmax) đấu random: games/2 ván cầm trắng, games/2 ván cầm đen. Trả về hiệu điểm theo góc nhìn agent."""
    import torch
    k = games // 2
    envs = make_vec_env_2p(2 * k, asynchronous=False)
    obs, _ = envs.reset(seed=seed)
    rng = np.random.default_rng(seed)
    agent_side = np.array([0] * k + [1] * k)
    results, done = [], np.zeros(2 * k, bool)
    while not done.all():
        act = rng.integers(0, 18, (2 * k, 2))
        o = torch.as_tensor(obs[np.arange(2 * k), agent_side], device=device)
        with torch.no_grad():
            for side in (0, 1):
                m = agent_side == side
                if m.any():
                    a = models[side if len(models) == 2 else 0].get_action_and_value(o[m], deterministic=True)[0]
                    act[m, side] = a.cpu().numpy()
        obs, r, term, trunc, infos = envs.step(act)
        fi = infos.get("final_info", {})
        if "episode" in fi:
            for i in np.flatnonzero(fi["_episode"]):
                if not done[i]:
                    done[i] = True
                    rw = float(fi["episode"]["r"][i])
                    results.append(rw if agent_side[i] == 0 else -rw)
    envs.close()
    res = np.array(results)
    return {"mean": float(res.mean()), "std": float(res.std()), "win_rate": float((res > 0).mean())}


def main():
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from ppo import ActorCritic, PPOConfig, RolloutBuffer, compute_gae, load_policy_meta, ppo_update, save_policy

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pool", "selfplay", "ippo"], default="pool")
    p.add_argument("--timesteps", type=int, default=10_000_000, help="số env-step (mỗi env-step = 4 frame, cho tới 2 mẫu)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-prob", type=float, default=0.5)
    p.add_argument("--snapshot-freq", type=int, default=250_000)
    p.add_argument("--eval-freq", type=int, default=500_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-name", default=None)
    p.add_argument("--threads", type=int, default=4, help="số luồng PyTorch trên CPU")
    p.add_argument("--resume", action="store_true")
    d = PPOConfig()
    for k, v in asdict(d).items():
        if k != "n_envs":
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    a = p.parse_args()
    cfg = PPOConfig(**{k: getattr(a, k) for k in asdict(d) if k != "n_envs"}, n_envs=a.n_envs)
    cfg.anneal = bool(cfg.anneal)

    run = a.run_name or f"sp_{a.mode}_s{a.seed}"
    model_dir, log_dir = os.path.join("models", run), os.path.join("logs", run)
    snap_dir = os.path.join(model_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    if device.type == "cpu":
        torch.set_num_threads(a.threads)  # đo thực tế: 4 luồng nhanh nhất khi chạy cùng 8 tiến trình game
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    N = cfg.n_envs
    S = 2 * N  # số luồng dữ liệu
    n_learners = 2 if a.mode == "ippo" else 1
    learners = [ActorCritic().to(device) for _ in range(n_learners)]
    opts = [torch.optim.Adam(m.parameters(), lr=cfg.learning_rate, eps=1e-5) for m in learners]

    batch = N * cfg.n_steps
    num_updates = a.timesteps // batch
    start_update, global_step, pool = 1, 0, []  # pool: danh sách đường dẫn snapshot
    last_path = os.path.join(model_dir, "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        for m, o, sm, so in zip(learners, opts, ck["models"], ck["optimizers"]):
            m.load_state_dict(sm)
            o.load_state_dict(so)
        start_update, global_step, pool = ck["update"] + 1, ck["global_step"], ck["pool"]
        print(f"Train tiếp từ step {global_step:,} (pool {len(pool)} snapshot)")

    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), **vars(a), "device": str(device)}, f, indent=2)
    writer = SummaryWriter(log_dir)
    ep_file = open(os.path.join(log_dir, "episodes.csv"), "a", newline="")
    ep_csv = csv.writer(ep_file)
    ev_file = open(os.path.join(log_dir, "evals.csv"), "a", newline="")
    ev_csv = csv.writer(ev_file)
    # file mới thì ghi tiêu đề. Không dùng tell(): ở chế độ append trên Windows nó có thể trả về 0
    if os.path.getsize(os.path.join(log_dir, "episodes.csv")) == 0:
        ep_csv.writerow(["global_step", "opponent", "learner_side", "white_return", "pts_white", "pts_black", "frames"])
    if os.path.getsize(os.path.join(log_dir, "evals.csv")) == 0:
        ev_csv.writerow(["global_step", "vs_random_mean", "vs_random_std", "vs_random_win_rate"])

    print(f"Run '{run}' | mode={a.mode} | {N} env x 2 phe | device={device} | "
          f"{num_updates} update x {batch} env-step = {num_updates * batch:,} env-step")

    snap_cache: dict[str, ActorCritic] = {}

    def get_snap(path):
        if path not in snap_cache:
            snap_cache[path] = load_policy_meta(path, str(device))[0]
        return snap_cache[path]

    # vai trò của từng env trong ván hiện tại: opp[e] = None (tự đánh với mình) hoặc đường dẫn snapshot;
    # lside[e] = phe do mạng học điều khiển khi đấu snapshot
    opp = [None] * N
    lside = np.zeros(N, dtype=int)

    def assign_role(e):
        if a.mode == "pool" and pool and rng.random() < a.pool_prob:
            opp[e] = pool[int(rng.integers(len(pool)))]
            lside[e] = int(rng.integers(2))
        else:
            opp[e] = None

    for e in range(N):
        assign_role(e)

    def owner_of_streams() -> np.ndarray:
        """owner[s] = chỉ số mạng học điều khiển luồng s, hoặc -1 nếu là snapshot."""
        own = np.zeros(S, dtype=int)
        for e in range(N):
            for side in (0, 1):
                s = e * 2 + side
                if a.mode == "ippo":
                    own[s] = side
                elif opp[e] is not None and side != lside[e]:
                    own[s] = -1
                else:
                    own[s] = 0
        return own

    def learner_of_stream(s):  # mạng dùng để tính value cho luồng s
        return (s % 2) if a.mode == "ippo" else 0

    buf = RolloutBuffer(cfg.n_steps, S, (4, 84, 84))
    owners = np.zeros((cfg.n_steps, S), dtype=int)
    envs = make_vec_env_2p(N, asynchronous=True)
    next_obs, _ = envs.reset(seed=a.seed + 1000 * start_update)
    next_obs = next_obs.reshape(S, 4, 84, 84)
    next_done = np.zeros(S, dtype=np.float32)
    recent_vs_snap = []
    t0, step0 = time.time(), global_step
    next_snap = (global_step // a.snapshot_freq + 1) * a.snapshot_freq
    next_eval = (global_step // a.eval_freq + 1) * a.eval_freq

    for update in range(start_update, num_updates + 1):
        frac = 1.0 - (update - 1) / num_updates if cfg.anneal else 1.0
        for o in opts:
            o.param_groups[0]["lr"] = frac * cfg.learning_rate
        clip_range = frac * cfg.clip_range
        for m in learners:
            m.eval()

        # ---------- (1) thu rollout cho cả 2 phe
        for t in range(cfg.n_steps):
            global_step += N
            own = owner_of_streams()
            owners[t] = own
            buf.obs[t] = next_obs
            buf.dones[t] = next_done
            obs_t = torch.as_tensor(next_obs, device=device)
            actions = np.zeros(S, dtype=np.int64)
            with torch.no_grad():
                for k, m in enumerate(learners):  # mạng học: lấy mẫu action, lưu logp và value
                    idx = np.flatnonzero(np.array([learner_of_stream(s) for s in range(S)]) == k)
                    act, logp, _, val = m.get_action_and_value(obs_t[idx])
                    buf.values[t, idx] = val.cpu().numpy()
                    mine = own[idx] == k
                    buf.logprobs[t, idx] = logp.cpu().numpy()
                    actions[idx[mine]] = act.cpu().numpy()[mine]
                for path in {opp[e] for e in range(N) if opp[e] is not None}:  # snapshot: chỉ lấy action
                    idx = np.array([e * 2 + (1 - lside[e]) for e in range(N) if opp[e] == path])
                    actions[idx] = get_snap(path).get_action_and_value(obs_t[idx])[0].cpu().numpy()
            buf.actions[t] = actions

            obs, r_w, term, trunc, infos = envs.step(actions.reshape(N, 2))
            # zero-sum (đã kiểm chứng mọi bước): thưởng đen = -thưởng trắng. Không lấy infos["r_black"]
            # vì ở bước kết thúc ván, env đã tự reset và info đó thuộc ván mới.
            r_b = -r_w
            buf.rewards[t, 0::2] = r_w
            buf.rewards[t, 1::2] = r_b
            done_env = (term | trunc)
            next_done = np.repeat(done_env, 2).astype(np.float32)
            next_obs = obs.reshape(S, 4, 84, 84)

            fi = infos.get("final_info", {})
            if "episode" in fi:
                for e in np.flatnonzero(fi["_episode"]):
                    ep = {k: float(v[e]) for k, v in fi["episode"].items() if not k.startswith("_")}
                    name = "self" if opp[e] is None else os.path.basename(opp[e])
                    ep_csv.writerow([global_step, name, int(lside[e]) if opp[e] else -1, ep["r"],
                                     ep["pts_w"], ep["pts_b"], int(ep["l"])])
                    if opp[e] is not None:  # hiệu điểm của mạng học khi đấu bản cũ
                        ret = ep["r"] if lside[e] == 0 else -ep["r"]
                        recent_vs_snap = (recent_vs_snap + [ret])[-20:]
                        writer.add_scalar("selfplay/return_vs_snapshot", ret, global_step)
                    writer.add_scalar("selfplay/points_per_game", ep["pts_w"] + ep["pts_b"], global_step)
                    assign_role(e)  # vai trò mới cho ván mới (bắt đầu từ obs kế tiếp)

        # ---------- (2) GAE trên mọi luồng (luồng của snapshot bị loại ở bước update)
        with torch.no_grad():
            obs_t = torch.as_tensor(next_obs, device=device)
            last_value = np.zeros(S, dtype=np.float32)
            for k, m in enumerate(learners):
                idx = np.flatnonzero(np.array([learner_of_stream(s) for s in range(S)]) == k)
                last_value[idx] = m.get_value(obs_t[idx]).cpu().numpy()
        compute_gae(buf, last_value, next_done, cfg.gamma, cfg.gae_lambda)

        # ---------- (3) update từng mạng học trên phần dữ liệu của nó
        stats_all = []
        for k, (m, o) in enumerate(zip(learners, opts)):
            m.train()
            stats_all.append(ppo_update(m, o, buf, cfg, clip_range, device, valid=(owners == k)))

        # ---------- log
        fps = int((global_step - step0) / (time.time() - t0))
        for k, st in enumerate(stats_all):
            for key, v in st.items():
                writer.add_scalar(f"losses_{k}/{key}", v, global_step)
        writer.add_scalar("charts/fps", fps, global_step)
        if update % 10 == 0 or update == num_updates:
            st = stats_all[0]
            vs = f"{np.mean(recent_vs_snap):+6.1f}" if recent_vs_snap else "   n/a"
            print(f"step {global_step:>10,} | vs snapshot (20 ván) {vs} | pool {len(pool):3d} | "
                  f"entropy {st['entropy']:.2f} | kl {st['approx_kl']:.4f} | clip {st['clip_frac']:.2f} | "
                  f"EV {st['explained_variance']:.2f} | mẫu {st['n_samples']} | {fps} env-step/s", flush=True)
            ep_file.flush()

        # ---------- snapshot (pool) + eval nhanh với random
        if global_step >= next_snap:
            next_snap += a.snapshot_freq
            if a.mode == "ippo":
                for k, side in enumerate("wb"):
                    save_policy(os.path.join(snap_dir, f"snap_{global_step:09d}_{side}.pt"), learners[k],
                                {**META, "global_step": global_step})
            else:
                path = os.path.join(snap_dir, f"snap_{global_step:09d}.pt")
                save_policy(path, learners[0], {**META, "global_step": global_step})
                pool.append(path)
        if global_step >= next_eval or update == num_updates:
            next_eval += a.eval_freq
            for m in learners:
                m.eval()
            s = quick_eval_vs_random(learners, device, games=4, seed=40_000 + global_step)
            ev_csv.writerow([global_step, s["mean"], s["std"], s["win_rate"]])
            ev_file.flush()
            writer.add_scalar("eval/vs_random_mean", s["mean"], global_step)
            print(f"[EVAL] step {global_step:,}: vs random {s['mean']:+.1f} ± {s['std']:.1f}, "
                  f"thắng {s['win_rate']:.0%}", flush=True)

        # ---------- checkpoint (last.pt mỗi ~100k env-step)
        if update % max(1, 100_000 // batch) == 0 or update == num_updates:
            state = {"models": [m.state_dict() for m in learners], "optimizers": [o.state_dict() for o in opts],
                     "update": update, "global_step": global_step, "pool": pool}
            torch.save(state, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)

    names = ["final_w.pt", "final_b.pt"] if a.mode == "ippo" else ["final.pt"]
    for m, n in zip(learners, names):
        save_policy(os.path.join(model_dir, n), m, {**META, "global_step": global_step})
    print(f"Xong {global_step:,} env-step sau {(time.time() - t0) / 60:.1f} phút. Đã lưu {model_dir}/{names}")
    envs.close()
    writer.close()
    ep_file.close()
    ev_file.close()


if __name__ == "__main__":
    main()
