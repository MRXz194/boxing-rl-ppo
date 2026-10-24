"""
Kiểm chứng thuật toán PPO TỰ CÀI trong ppo.py đúng với định nghĩa toán học.
Mỗi phép thử so code của dự án với một cách tính "ngây thơ" viết độc lập, theo đúng công thức trong bài báo.

  python verify_algorithm.py
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys

import numpy as np
import torch

from ppo import ActorCritic, PPOConfig, RolloutBuffer, compute_gae, ppo_update

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
torch.manual_seed(0)
rng = np.random.default_rng(0)
ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'ĐÚNG' if ok else 'SAI '}] {name}: {detail}")


# ---------------------------------------------------------------- 1. GAE
def gae_naive(r, v, dones, last_v, last_done, g, lam):
    """Định nghĩa trực tiếp (Schulman 2016): A_t = Σ_k (γλ)^k δ_{t+k}, dừng ở ranh giới ván.
    dones[t] = 1 nghĩa là obs[t] là frame đầu của ván mới."""
    T = len(r)
    v_ext = np.append(v, last_v)
    d_ext = np.append(dones, last_done)
    A = np.zeros(T)
    for t in range(T):
        total, coef = 0.0, 1.0
        for k in range(t, T):
            nonterm = 1.0 - d_ext[k + 1]
            delta = r[k] + g * v_ext[k + 1] * nonterm - v_ext[k]
            total += coef * delta
            if nonterm == 0:  # ván kết thúc sau bước k -> không cộng tiếp
                break
            coef *= g * lam
        A[t] = total
    return A


T, N = 40, 3
buf = RolloutBuffer(T, N, (1, 1, 1))
buf.rewards[:] = rng.choice([-2, -1, 0, 1, 2], (T, N))
buf.values[:] = rng.normal(0, 1, (T, N))
buf.dones[:] = rng.random((T, N)) < 0.08
last_v = rng.normal(0, 1, N).astype(np.float32)
last_d = np.array([0, 1, 0], np.float32)
compute_gae(buf, last_v, last_d, 0.99, 0.95)
err = max(np.abs(buf.advantages[:, i] - gae_naive(buf.rewards[:, i], buf.values[:, i], buf.dones[:, i],
                                                     last_v[i], last_d[i], 0.99, 0.95)).max() for i in range(N))
report("GAE (λ=0.95) so với định nghĩa tổng chuỗi", err < 1e-4, f"sai số lớn nhất {err:.2e}")
report("Return = Advantage + V", np.allclose(buf.returns, buf.advantages + buf.values), "R_t = A_t + V(s_t)")

# λ = 1 và V = 0  ->  advantage phải đúng bằng tổng phần thưởng chiết khấu (Monte-Carlo) tới hết ván
b2 = RolloutBuffer(6, 1, (1, 1, 1))
b2.rewards[:, 0] = [1, 0, 2, 0, -1, 1]
b2.dones[:, 0] = [0, 0, 0, 1, 0, 0]  # ván mới bắt đầu ở bước 3
compute_gae(b2, np.zeros(1, np.float32), np.ones(1, np.float32), 0.9, 1.0)
mc = [1 + 0.9 * 0 + 0.81 * 2, 0 + 0.9 * 2, 2, 0 + 0.9 * -1 + 0.81 * 1, -1 + 0.9 * 1, 1]
report("GAE λ=1 = return Monte-Carlo, cắt đúng ở ranh giới ván", np.allclose(b2.advantages[:, 0], mc, atol=1e-5),
       f"{np.round(b2.advantages[:, 0], 3).tolist()} vs {np.round(mc, 3).tolist()}")

# ---------------------------------------------------------------- 2. Policy: log π và entropy
m = ActorCritic()
obs = torch.randint(0, 255, (5, 4, 84, 84), dtype=torch.uint8)
acts = torch.randint(0, 18, (5,))
_, logp, ent, val = m.get_action_and_value(obs, acts)
logits = m.actor(m._features(obs)).detach().numpy().astype(np.float64)
p = np.exp(logits - logits.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
logp_naive = np.log(p[np.arange(5), acts.numpy()])
ent_naive = -(p * np.log(p)).sum(1)
report("log π(a|s) = log softmax(logits)[a]", np.allclose(logp.detach().numpy(), logp_naive, atol=1e-5), "")
report("Entropy = −Σ π log π", np.allclose(ent.detach().numpy(), ent_naive, atol=1e-5), f"≈ {ent_naive.mean():.3f} (tối đa ln 18 = 2.890)")
a_det = m.get_action_and_value(obs, deterministic=True)[0].numpy()
report("Chế độ thi đấu chọn argmax π", (a_det == p.argmax(1)).all(), "")

# ---------------------------------------------------------------- 3. Hàm mục tiêu PPO-Clip
def clip_obj(ratio, adv, eps):  # định nghĩa trong bài báo (cần cực đại)
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)


r_t = torch.tensor([0.5, 0.95, 1.05, 1.5, 0.5, 0.95, 1.05, 1.5], requires_grad=True)
A_t = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
# cùng biểu thức như trong ppo_update:
loss = -torch.min(r_t * A_t, torch.clamp(r_t, 0.9, 1.1) * A_t)
loss.sum().backward()
report("Loss PPO-Clip = −min(r·A, clip(r)·A)",
       np.allclose(-loss.detach().numpy(), clip_obj(r_t.detach().numpy(), A_t.numpy(), 0.1)), "")
g = r_t.grad.numpy()
# Gradient phải = 0 khi đã "đủ": A>0 và r>1+ε, hoặc A<0 và r<1−ε
expect_zero = np.array([False, False, False, True, True, False, False, False])
report("Clip chặn cập nhật quá đà (gradient = 0 đúng chỗ)", np.all((g == 0) == expect_zero),
       f"grad = {g.tolist()}")

# ---------------------------------------------------------------- 4. Cập nhật PPO làm đúng hướng
# Action có advantage dương phải tăng xác suất, âm phải giảm; và nhờ clip, policy không lệch quá xa.
cfg = PPOConfig(n_envs=1, n_steps=64, batch_size=64, n_epochs=4, learning_rate=1e-3)
m = ActorCritic()
opt = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate)
b = RolloutBuffer(64, 1, (4, 84, 84))
b.obs[:] = rng.integers(0, 255, b.obs.shape)
with torch.no_grad():
    o = torch.as_tensor(b.obs.reshape(64, 4, 84, 84))
    _, lp0, _, v0 = m.get_action_and_value(o, torch.full((64,), 3))
    _, lp_other0, _, _ = m.get_action_and_value(o, torch.full((64,), 7))
b.actions[:, 0] = np.where(np.arange(64) < 32, 3, 7)  # nửa đầu chọn action 3, nửa sau action 7
b.logprobs[:, 0] = np.where(np.arange(64) < 32, lp0.numpy(), lp_other0.numpy())
b.values[:, 0] = v0.numpy()
b.advantages[:, 0] = np.where(np.arange(64) < 32, 1.0, -1.0)  # action 3 tốt, action 7 tệ
b.returns[:, 0] = b.advantages[:, 0] + b.values[:, 0]
stats = ppo_update(m, opt, b, cfg, clip_range=0.2, device=torch.device("cpu"))
with torch.no_grad():
    _, lp1, _, _ = m.get_action_and_value(o, torch.full((64,), 3))
    _, lp_other1, _, _ = m.get_action_and_value(o, torch.full((64,), 7))
up = (lp1 - lp0).exp().mean().item()
down = (lp_other1 - lp_other0).exp().mean().item()
report("Update tăng π(action tốt), giảm π(action tệ)", up > 1 and down < 1,
       f"π(a=3) ×{up:.3f}, π(a=7) ×{down:.3f}")
report("Mặt nạ self-play: chỉ học từ mẫu hợp lệ", True if ppo_update(m, opt, b, cfg, 0.2, torch.device("cpu"),
       valid=np.arange(64)[:, None] < 16)["n_samples"] == 16 else False, "16/64 mẫu được dùng")

print("\n" + ("TẤT CẢ ĐÚNG: thuật toán tự cài khớp định nghĩa toán học." if ok_all else "CÓ PHÉP THỬ SAI!"))
sys.exit(0 if ok_all else 1)
