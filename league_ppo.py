"""
League training (bản đơn giản, ý tưởng như AlphaStar) cho Boxing 2 người: bản SỬA cho self-play thuần.

VẤN ĐỀ của self-play thuần (selfplay_ppo.py, xem TAI_LIEU_DU_AN.md mục 7.8):
  1 mạng chơi cả 2 phe, bị đấm (−1) "đau" bằng đấm trúng (+1). Từ điểm xuất phát ngẫu nhiên,
  hai bên học cách NÉ NHAU: điểm/ván giảm từ ~20 xuống ~3, policy không mạnh hơn random.

CÁCH SỬA, 3 ý:
  1. KHỞI ĐẦU TỪ AGENT 1 NGƯỜI (--init): mạng đã biết đánh ngay từ đầu, không phải học lại từ con số 0.
  2. AI CÓ SẴN CỦA GAME làm đối thủ cố định trong --n-cpu game: nó luôn lao vào tấn công,
     nên chiến thuật "né" không còn hiệu quả, tín hiệu thưởng luôn dày.
  3. Đối thủ của các game 2 người được chọn ngẫu nhiên MỖI VÁN:
        tự đánh với chính mình (self)  |  snapshot cũ (pool)  |  RANDOM (đứng thụ động, nên phải chủ động áp sát)
     → agent phải thắng được mọi kiểu đối thủ: tấn công (AI máy), thụ động (random), ngang tài (chính nó).

Dữ liệu mỗi bước: n_2p game × 2 phe + n_cpu game × 1 phe = các "luồng". Luồng do đối thủ (snapshot, random)
điều khiển bị loại khỏi update bằng mặt nạ (PPO là on-policy). Lõi PPO dùng lại 100% ppo.py.

  python league_ppo.py --init models/ppo10M_s0/final.pt --timesteps 3000000 --run-name league
  python league_ppo.py ... --resume       (train tiếp)
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình

import argparse
import csv
import json
import random
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np

import ale_py  # đăng ký namespace ALE cho cả tiến trình con (spawn nạp lại file này)

from envs_2p import Boxing1PCustom, info_field, make_vec_env_2p

gym.register_envs(ale_py)

META = {"frame_stack": 4, "mask_hud": True, "perspective": "mirror"}


def make_cpu_envs(n: int, shaping_gamma=None, gym_env: bool = False):
    """n game 1 người (đấu AI máy), cùng tiền xử lý tự viết như game 2 người (agent luôn cầm phe trắng).
    gym_env=True (gỡ lỗi): dùng env của envs.py (AtariPreprocessing của gymnasium) thay vì Boxing1PCustom."""
    if gym_env:
        from envs import make_env
        fns = [make_env() for _ in range(n)]
    else:
        fns = [lambda: Boxing1PCustom(difficulty=0, mask_hud=True, shaping_gamma=shaping_gamma) for _ in range(n)]
    cls = gym.vector.AsyncVectorEnv if n > 1 else gym.vector.SyncVectorEnv
    return cls(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def quick_eval_vs_cpu(model, device, games: int = 4, seed: int = 0) -> dict:
    import torch
    envs = gym.vector.SyncVectorEnv([lambda: Boxing1PCustom(0, True) for _ in range(games)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = envs.reset(seed=seed)
    res, done = [], np.zeros(games, bool)
    while not done.all():
        with torch.no_grad():
            a = model.get_action_and_value(torch.as_tensor(obs, device=device), deterministic=True)[0]
        obs, r, te, tr, infos = envs.step(a.cpu().numpy())
        fi = infos.get("final_info", {})
        if "episode" in fi:
            for i in np.flatnonzero(fi["_episode"]):
                if not done[i]:
                    done[i] = True
                    res.append(float(fi["episode"]["r"][i]))
    envs.close()
    res = np.array(res)
    return {"mean": float(res.mean()), "win_rate": float((res > 0).mean())}


def main():
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from ppo import ActorCritic, PPOConfig, RolloutBuffer, compute_gae, load_policy_meta, ppo_update, save_policy
    from selfplay_ppo import quick_eval_vs_random

    p = argparse.ArgumentParser()
    p.add_argument("--init", default="models/ppo10M_s0/final.pt", help="model khởi đầu (agent 1 người); 'none' = từ đầu")
    p.add_argument("--timesteps", type=int, default=3_000_000, help="tổng env-step (cộng cả game 2 người và game đấu AI máy)")
    p.add_argument("--n-2p", type=int, default=4, help="số game 2 người chạy song song")
    p.add_argument("--n-cpu", type=int, default=4, help="số game đấu AI máy chạy song song")
    p.add_argument("--p-self", type=float, default=0.4, help="xác suất 1 ván 2 người là tự đánh với chính mình")
    p.add_argument("--p-random", type=float, default=0.3, help="xác suất đối thủ là random")
    p.add_argument("--p-still", type=float, default=0.0, help="xác suất đối thủ đứng yên (còn lại: snapshot cũ)")
    p.add_argument("--shaping", type=float, default=0.0,
                   help="hệ số reward shaping thế năng (kéo 2 võ sĩ lại gần nhau, Ng 1999); 0 = tắt. Chỉ dùng lúc train")
    p.add_argument("--shaping-decay", type=float, default=0.6,
                   help="hệ số shaping giảm tuyến tính về 0 khi đi được tỉ lệ này của quá trình train")
    p.add_argument("--passive-penalty", type=float, default=0.0,
                   help="luật chống né tránh (chỉ khi train): mỗi bước 2 võ sĩ đứng xa hơn --passive-dist thì CẢ HAI bị trừ chừng này")
    p.add_argument("--passive-dist", type=float, default=60.0, help="ngưỡng |dx|+|dy| (pixel) coi là né tránh")
    p.add_argument("--opp-models", default="", help="danh sách model đối thủ CỐ ĐỊNH, cách nhau bởi dấu phẩy (thêm vào pool)")
    p.add_argument("--freeze-pool", action="store_true", help="không thêm snapshot của chính mình vào pool (pool = đối thủ cố định)")
    p.add_argument("--gym-cpu-env", action="store_true", help="(gỡ lỗi) game đấu AI máy dùng env của envs.py")
    p.add_argument("--still-opp", action="store_true", help="(gỡ lỗi) đối thủ 'random' đứng yên (NOOP)")
    p.add_argument("--snapshot-freq", type=int, default=250_000)
    p.add_argument("--eval-freq", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--run-name", default="league")
    p.add_argument("--resume", action="store_true")
    d = PPOConfig()
    for k, v in asdict(d).items():
        if k != "n_envs":
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    a = p.parse_args()
    N2, NC = a.n_2p, a.n_cpu
    cfg = PPOConfig(**{k: getattr(a, k) for k in asdict(d) if k != "n_envs"}, n_envs=N2 + NC)
    cfg.anneal = bool(cfg.anneal)
    S = 2 * N2 + NC  # luồng: [game2p_0 trắng, game2p_0 đen, ..., cpu_0, cpu_1, ...]

    model_dir, log_dir = os.path.join("models", a.run_name), os.path.join("logs", a.run_name)
    snap_dir = os.path.join(model_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    model = ActorCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    batch = (N2 + NC) * cfg.n_steps
    num_updates = a.timesteps // batch
    start_update, global_step, pool = 1, 0, []
    last_path = os.path.join(model_dir, "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_update, global_step, pool = ck["update"] + 1, ck["global_step"], ck["pool"]
        print(f"Train tiếp từ step {global_step:,} (pool {len(pool)} snapshot)")
    else:
        if a.init.lower() != "none":
            model.load_state_dict(load_policy_meta(a.init)[0].state_dict())
            print("Khởi đầu từ", a.init)
        path0 = os.path.join(snap_dir, "snap_000000000.pt")  # bản khởi đầu cũng vào pool
        save_policy(path0, model, {**META, "global_step": 0})
        if not a.freeze_pool:
            pool.append(path0)
        pool += [x for x in a.opp_models.split(",") if x]

    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), **vars(a)}, f, indent=2)
    writer = SummaryWriter(log_dir)
    ep_path, ev_path = os.path.join(log_dir, "episodes.csv"), os.path.join(log_dir, "evals.csv")
    ep_file, ev_file = open(ep_path, "a", newline=""), open(ev_path, "a", newline="")
    ep_csv, ev_csv = csv.writer(ep_file), csv.writer(ev_file)
    if os.path.getsize(ep_path) == 0:
        ep_csv.writerow(["global_step", "opponent", "learner_side", "learner_return", "points_total", "frames"])
    if os.path.getsize(ev_path) == 0:
        ev_csv.writerow(["global_step", "vs_random_mean", "vs_random_win_rate", "vs_cpu_mean", "vs_cpu_win_rate"])
    print(f"Run '{a.run_name}' | league | {N2} game 2 người + {NC} game đấu AI máy | "
          f"{num_updates} update x {batch} env-step = {num_updates * batch:,} env-step")

    snap_cache = {}

    def get_snap(path):
        if path not in snap_cache:
            snap_cache[path] = load_policy_meta(path)[0]
        return snap_cache[path]

    # vai trò từng game 2 người trong ván hiện tại
    role = ["self"] * N2            # "self" | "random" | đường dẫn snapshot
    lside = np.zeros(N2, dtype=int)  # phe do mạng học điều khiển (khi đối thủ không phải chính nó)

    def assign_role(e):
        u = rng.random()
        if u < a.p_self:
            role[e] = "self"
        elif u < a.p_self + a.p_random:
            role[e] = "random"
        elif u < a.p_self + a.p_random + a.p_still:
            role[e] = "still"
        else:
            role[e] = pool[int(rng.integers(len(pool)))]
        lside[e] = int(rng.integers(2))

    for e in range(N2):
        assign_role(e)

    def owners() -> np.ndarray:
        own = np.zeros(S, dtype=int)  # 0 = mạng học, -1 = đối thủ (bị loại khỏi update)
        for e in range(N2):
            if role[e] != "self":  # random / still / snapshot: phe đối thủ không được học
                own[e * 2 + (1 - lside[e])] = -1
        return own

    buf = RolloutBuffer(cfg.n_steps, S, (4, 84, 84))
    own_buf = np.zeros((cfg.n_steps, S), dtype=int)
    sg = cfg.gamma if a.shaping > 0 else None
    env2 = make_vec_env_2p(N2, asynchronous=True, shaping_gamma=sg,
                           passive_dist=a.passive_dist if a.passive_penalty > 0 else None)
    envc = make_cpu_envs(NC, shaping_gamma=sg, gym_env=a.gym_cpu_env) if NC > 0 else None
    o2, _ = env2.reset(seed=a.seed + 1000 * start_update)
    oc = envc.reset(seed=a.seed + 7000 + 1000 * start_update)[0] if envc else np.zeros((0, 4, 84, 84), np.uint8)
    next_obs = np.concatenate([o2.reshape(2 * N2, 4, 84, 84), oc])
    next_done = np.zeros(S, dtype=np.float32)
    recent = {"cpu": [], "random": [], "still": [], "snap": [], "self_pts": []}
    t0, step0 = time.time(), global_step
    next_snap = (global_step // a.snapshot_freq + 1) * a.snapshot_freq
    next_eval = (global_step // a.eval_freq + 1) * a.eval_freq

    def log_ep(opp, side, ret, pts, frames):
        ep_csv.writerow([global_step, opp, side, ret, pts, frames])
        key = {"cpu": "cpu", "random": "random", "still": "still", "self": "self_pts"}.get(opp, "snap")
        recent[key] = (recent[key] + [pts if key == "self_pts" else ret])[-20:]

    for update in range(start_update, num_updates + 1):
        frac = 1.0 - (update - 1) / num_updates if cfg.anneal else 1.0
        opt.param_groups[0]["lr"] = frac * cfg.learning_rate
        clip_range = frac * cfg.clip_range
        progress = (update - 1) / num_updates
        shaping_coef = a.shaping * max(0.0, 1.0 - progress / a.shaping_decay) if a.shaping > 0 else 0.0
        model.eval()
        for t in range(cfg.n_steps):
            global_step += N2 + NC
            own = owners()
            own_buf[t] = own
            buf.obs[t] = next_obs
            buf.dones[t] = next_done
            obs_t = torch.as_tensor(next_obs, device=device)
            with torch.no_grad():
                act, logp, _, val = model.get_action_and_value(obs_t)
            actions = act.cpu().numpy()
            buf.logprobs[t] = logp.cpu().numpy()
            buf.values[t] = val.cpu().numpy()
            # đối thủ: random hoặc snapshot (lấy mẫu theo policy của snapshot)
            for e in range(N2):
                if role[e] == "random":
                    actions[e * 2 + (1 - lside[e])] = 0 if a.still_opp else rng.integers(0, 18)
                elif role[e] == "still":
                    actions[e * 2 + (1 - lside[e])] = 0  # NOOP: đứng yên
            for path in {r for r in role if r not in ("self", "random", "still")}:
                idx = np.array([e * 2 + (1 - lside[e]) for e in range(N2) if role[e] == path])
                with torch.no_grad():
                    actions[idx] = get_snap(path).get_action_and_value(obs_t[idx])[0].cpu().numpy()
            buf.actions[t] = actions

            o2, r2, te2, tr2, inf2 = env2.step(actions[:2 * N2].reshape(N2, 2))
            if envc is not None:
                oc, rc, tec, trc, infc = envc.step(actions[2 * N2:])
            else:
                rc, tec, trc, infc = np.zeros(0), np.zeros(0, bool), np.zeros(0, bool), {}
            buf.rewards[t, 0:2 * N2:2] = r2
            buf.rewards[t, 1:2 * N2:2] = -r2  # zero-sum: thưởng đen = −thưởng trắng
            buf.rewards[t, 2 * N2:] = rc
            if a.shaping > 0:  # thưởng phụ: khoảng cách đối xứng nên 2 phe nhận cùng F
                f2 = info_field(inf2, "shaping", N2)
                buf.rewards[t, 0:2 * N2:2] += shaping_coef * f2
                buf.rewards[t, 1:2 * N2:2] += shaping_coef * f2
                if NC > 0:
                    buf.rewards[t, 2 * N2:] += shaping_coef * info_field(infc, "shaping", NC)
            if a.passive_penalty > 0:  # luật chống né tránh: cả 2 phe cùng bị trừ khi đứng xa nhau
                pen = a.passive_penalty * info_field(inf2, "passive", N2)
                buf.rewards[t, 0:2 * N2:2] -= pen
                buf.rewards[t, 1:2 * N2:2] -= pen
            d2 = (te2 | tr2)
            next_done = np.concatenate([np.repeat(d2, 2), (tec | trc)]).astype(np.float32)
            next_obs = np.concatenate([o2.reshape(2 * N2, 4, 84, 84), oc])

            fi = inf2.get("final_info", {})
            if "episode" in fi:
                for e in np.flatnonzero(fi["_episode"]):
                    r_w = float(fi["episode"]["r"][e])
                    pts = float(fi["episode"]["pts_w"][e] + fi["episode"]["pts_b"][e])
                    opp = role[e] if role[e] in ("self", "random", "still") else "snap"
                    ret = r_w if lside[e] == 0 else -r_w
                    log_ep(opp if opp != "snap" else os.path.basename(role[e]), int(lside[e]), ret, pts,
                           int(fi["episode"]["l"][e]))
                    assign_role(e)  # đối thủ mới cho ván mới
            fc = infc.get("final_info", {})
            if "episode" in fc:
                for i in np.flatnonzero(fc["_episode"]):
                    log_ep("cpu", 0, float(fc["episode"]["r"][i]), float("nan"), int(fc["episode"]["l"][i]))

        with torch.no_grad():
            last_value = model.get_value(torch.as_tensor(next_obs, device=device)).cpu().numpy()
        compute_gae(buf, last_value, next_done, cfg.gamma, cfg.gae_lambda)
        model.train()
        st = ppo_update(model, opt, buf, cfg, clip_range, device, valid=(own_buf == 0))

        fps = int((global_step - step0) / (time.time() - t0))
        for k, v in st.items():
            writer.add_scalar(f"losses/{k}", v, global_step)
        for k, v in recent.items():
            if v:
                writer.add_scalar(f"league/{k}", float(np.mean(v)), global_step)
        if update % 10 == 0 or update == num_updates:
            f = lambda k: f"{np.mean(recent[k]):+6.1f}" if recent[k] else "   n/a"
            print(f"step {global_step:>10,} | vs AI máy {f('cpu')} | vs random {f('random')} | vs snapshot {f('snap')} | "
                  f"vs đứng yên {f('still')} | điểm/ván tự đánh {f('self_pts')} | shaping {shaping_coef:.1f} | entropy {st['entropy']:.2f} | kl {st['approx_kl']:.4f} | "
                  f"{fps} env-step/s", flush=True)
            ep_file.flush()

        if global_step >= next_snap:
            next_snap += a.snapshot_freq
            path = os.path.join(snap_dir, f"snap_{global_step:09d}.pt")
            save_policy(path, model, {**META, "global_step": global_step})
            if not a.freeze_pool:
                pool.append(path)
        if global_step >= next_eval or update == num_updates:
            next_eval += a.eval_freq
            model.eval()
            sr = quick_eval_vs_random([model], device, games=4, seed=40_000 + global_step)
            sc = quick_eval_vs_cpu(model, device, games=4, seed=50_000 + global_step)
            ev_csv.writerow([global_step, sr["mean"], sr["win_rate"], sc["mean"], sc["win_rate"]])
            ev_file.flush()
            writer.add_scalar("eval/vs_random", sr["mean"], global_step)
            writer.add_scalar("eval/vs_cpu", sc["mean"], global_step)
            print(f"[EVAL] step {global_step:,}: vs random {sr['mean']:+.1f} (thắng {sr['win_rate']:.0%}) | "
                  f"vs AI máy {sc['mean']:+.1f} (thắng {sc['win_rate']:.0%})", flush=True)
        if update % max(1, 100_000 // batch) == 0 or update == num_updates:
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "update": update,
                     "global_step": global_step, "pool": pool}
            torch.save(state, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)

    save_policy(os.path.join(model_dir, "final.pt"), model, {**META, "global_step": global_step})
    print(f"Xong {global_step:,} env-step sau {(time.time() - t0) / 60:.1f} phút. Đã lưu {model_dir}/final.pt")
    env2.close()
    if envc is not None:
        envc.close()
    writer.close()
    ep_file.close()
    ev_file.close()


if __name__ == "__main__":
    main()
