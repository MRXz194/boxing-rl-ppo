"""
Kiểm tra nhanh sau khi cài đặt (khoảng 1 phút). setup.bat tự chạy file này.

  python smoke_test.py

Kiểm tra: thư viện → env 1 người → env 2 người (zero-sum, lật gương) → PPO update chạy được → nạp model đã train.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import time
import traceback

results = []


def check(name):
    def deco(fn):
        t = time.time()
        try:
            msg = fn() or ""
            results.append((True, name, msg, time.time() - t))
        except Exception as e:  # noqa: BLE001
            results.append((False, name, f"{type(e).__name__}: {e}", time.time() - t))
            traceback.print_exc()
        return fn
    return deco


@check("Thư viện")
def _libs():
    import ale_py, cv2, gymnasium, matplotlib, multi_agent_ale_py, numpy, pygame, torch  # noqa: F401
    return (f"python {sys.version.split()[0]}, torch {torch.__version__}, gymnasium {gymnasium.__version__}, "
            f"ale-py {ale_py.__version__}, numpy {numpy.__version__}")


@check("Env 1 người (đấu AI máy)")
def _env1():
    import numpy as np
    from envs import make_vec_env
    v = make_vec_env(2, asynchronous=False)
    obs, _ = v.reset(seed=0)
    for _ in range(50):
        obs, *_ = v.step(np.random.randint(0, 18, 2))
    v.close()
    assert obs.shape == (2, 4, 84, 84), obs.shape
    return f"obs {obs.shape}"


@check("Env 2 người (cả 2 võ sĩ là agent)")
def _env2():
    import numpy as np
    from envs_2p import Boxing2P, find_rom_dir
    e = Boxing2P()
    o, _ = e.reset(seed=1)
    rng = np.random.default_rng(0)
    zero_sum = True
    for _ in range(400):
        o, r, t, tr, info = e.step(rng.integers(0, 18, 2))
        zero_sum &= abs(r + info["r_black"]) < 1e-9
    assert o.shape == (2, 4, 84, 84) and zero_sum
    return f"obs {o.shape}, zero-sum đúng, ROM {find_rom_dir()}"


@check("PPO update (thuật toán tự cài)")
def _ppo():
    import numpy as np
    import torch
    from ppo import ActorCritic, PPOConfig, RolloutBuffer, compute_gae, ppo_update
    cfg = PPOConfig(n_envs=2, n_steps=16, batch_size=8, n_epochs=1)
    m = ActorCritic()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    b = RolloutBuffer(16, 2, (4, 84, 84))
    b.obs[:] = np.random.randint(0, 255, b.obs.shape)
    b.rewards[:] = np.random.choice([-1, 0, 1], b.rewards.shape)
    compute_gae(b, np.zeros(2, np.float32), np.zeros(2, np.float32))
    s = ppo_update(m, opt, b, cfg, 0.1, torch.device("cpu"))
    return f"policy_loss {s['policy_loss']:+.4f}, entropy {s['entropy']:.2f}"


@check("Model đã train")
def _models():
    import glob
    from ppo import load_policy_meta
    found = [p for p in ["models/ppo10M_s0/final.pt", "models/league6/final.pt", "models/ppo_s0/final.pt"]
             if os.path.exists(p)] 
    for p in found:
        load_policy_meta(p)
    return ", ".join(found) if found else "chưa có model (chạy train trước)"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    print("\n" + "=" * 70)
    for ok, name, msg, dt in results:
        print(f"[{'OK ' if ok else 'LỖI'}] {name:38s} {dt:5.1f}s  {msg}")
    print("=" * 70)
    bad = [r for r in results if not r[0]]
    print("TẤT CẢ ĐỀU ỔN." if not bad else f"CÓ {len(bad)} LỖI, xem chi tiết ở trên.")
    sys.exit(1 if bad else 0)
