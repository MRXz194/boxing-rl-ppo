"""
PPO (Proximal Policy Optimization, Schulman et al. 2017), tự cài đặt bằng PyTorch.

Gồm 4 phần, mỗi phần ứng với một bước của thuật toán:
  1. ActorCritic    : mạng nơ-ron. Đầu vào là ảnh; ra (a) phân phối xác suất trên 18 action
                      (actor, tức policy pi) và (b) giá trị V(s) (critic).
  2. RolloutBuffer  : lưu kinh nghiệm thu được khi cho agent chơi n_steps bước trên n_envs env.
  3. compute_gae    : tính advantage A_t bằng GAE, cho biết action vừa chọn tốt hơn hay kém hơn mức trung bình.
  4. ppo_update     : cập nhật mạng bằng loss có clip, chạy nhiều epoch trên các minibatch.

Vòng lặp chính (xem train_ppo.py):
    lặp:
        thu rollout bằng policy hiện tại  ->  tính GAE  ->  ppo_update  ->  (vứt dữ liệu cũ, lặp lại)
PPO là on-policy: dữ liệu chỉ dùng cho đúng 1 lần update rồi bỏ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ----------------------------------------------------------------------------- 1. Mạng
def layer_init(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Khởi tạo trực giao (orthogonal), cách khởi tạo chuẩn cho PPO trên Atari."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """
    Thân chung là Nature-CNN (Mnih et al. 2015):
        (4,84,84) -conv 8x8 s4-> (32,20,20) -conv 4x4 s2-> (64,9,9) -conv 3x3 s1-> (64,7,7)
        -> flatten 3136 -> FC 512
    Hai đầu ra:
        actor : 512 -> 18 logits -> softmax = pi(a|s)
        critic: 512 -> 1         = V(s), ước lượng tổng reward tương lai
    """

    def __init__(self, n_actions: int = 18, in_frames: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(in_frames, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        # gain nhỏ cho actor để lúc đầu policy gần đều (khám phá); gain 1 cho critic
        self.actor = layer_init(nn.Linear(512, n_actions), gain=0.01)
        self.critic = layer_init(nn.Linear(512, 1), gain=1.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs.float() / 255.0)  # chuẩn hóa pixel về [0, 1]

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None,
                             deterministic: bool = False):
        """
        Trả về (action, log pi(a|s), entropy, V(s)).
        - Lúc thu rollout: action=None, lấy mẫu a ~ pi(.|s) (vừa khai thác vừa khám phá).
        - Lúc update: truyền action cũ vào để tính lại log pi_moi(a|s) với tham số mới.
        - Lúc thi đấu: deterministic=True, chọn argmax pi.
        """
        h = self._features(obs)
        dist = Categorical(logits=self.actor(h))
        if action is None:
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1)


# ----------------------------------------------------------------------------- 2. Bộ nhớ rollout
class RolloutBuffer:
    """Mảng kích thước (n_steps, n_envs, ...). obs lưu uint8 trên CPU để tiết kiệm bộ nhớ."""

    def __init__(self, n_steps: int, n_envs: int, obs_shape: tuple):
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.logprobs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        # dones[t] = 1 nghĩa là obs[t] là frame đầu của một ván mới (ván trước vừa kết thúc)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)


# ----------------------------------------------------------------------------- 3. GAE
def compute_gae(buf: RolloutBuffer, last_value: np.ndarray, last_done: np.ndarray,
                gamma: float = 0.99, lam: float = 0.95) -> None:
    """
    Generalized Advantage Estimation (Schulman et al. 2016).

    TD error:   delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
    Advantage:  A_t     = delta_t + gamma*lam * (1 - done_{t+1}) * A_{t+1}       (tính ngược từ cuối)
    Return:     R_t     = A_t + V(s_t)                                            (target cho critic)

    - (1 - done) cắt đứt tại ranh giới ván: không cộng giá trị của ván sau vào ván trước.
    - lam=0: A = delta (ít nhiễu, nhiều bias). lam=1: A = tổng reward thật - V (nhiều nhiễu).
      0.95 là điểm cân bằng giữa hai đầu.
    """
    n_steps = buf.rewards.shape[0]
    last_gae = np.zeros_like(last_value)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - last_done
            next_value = last_value
        else:
            next_non_terminal = 1.0 - buf.dones[t + 1]
            next_value = buf.values[t + 1]
        delta = buf.rewards[t] + gamma * next_value * next_non_terminal - buf.values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        buf.advantages[t] = last_gae
    buf.returns[:] = buf.advantages + buf.values


# ----------------------------------------------------------------------------- 4. Update
@dataclass
class PPOConfig:
    n_envs: int = 8
    n_steps: int = 128          # mỗi env chơi 128 bước -> batch = 8*128 = 1024 mẫu
    n_epochs: int = 4           # mỗi batch được học lại 4 lần
    batch_size: int = 256       # minibatch -> 1024/256 = 4 minibatch mỗi epoch
    learning_rate: float = 2.5e-4
    clip_range: float = 0.1     # epsilon trong clip(ratio, 1-eps, 1+eps)
    gamma: float = 0.99         # hệ số chiết khấu
    gae_lambda: float = 0.95
    ent_coef: float = 0.01      # thưởng entropy -> khuyến khích khám phá
    vf_coef: float = 0.5        # trọng số loss của critic
    max_grad_norm: float = 0.5  # cắt gradient tránh bước nhảy quá lớn
    anneal: bool = True         # giảm tuyến tính lr và clip_range về 0


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, buf: RolloutBuffer,
               cfg: PPOConfig, clip_range: float, device: torch.device,
               valid: np.ndarray | None = None) -> dict:
    """
    Hàm mục tiêu PPO-Clip (cần cực tiểu):

        ratio_t  = pi_moi(a_t|s_t) / pi_cu(a_t|s_t) = exp(logp_moi - logp_cu)
        L_policy = -E[ min( ratio_t * A_t,  clip(ratio_t, 1-eps, 1+eps) * A_t ) ]
        L_value  =  E[ (V(s_t) - R_t)^2 ]
        L        =  L_policy + vf_coef * L_value - ent_coef * Entropy

    Ý nghĩa: tăng xác suất action có A>0, giảm xác suất action có A<0. Phần clip
    giữ policy mới không lệch quá xa policy cũ (ratio nằm trong [0.9, 1.1]),
    nhờ vậy học nhiều epoch trên cùng dữ liệu mà không bị "sụp".

    valid (tùy chọn, dùng cho self-play): mặt nạ (n_steps, n_streams). Chỉ học từ các mẫu valid=True,
    tức các mẫu do chính mạng này điều khiển (loại mẫu do đối thủ snapshot cũ điều khiển).
    Số mẫu thay đổi nên số minibatch được giữ như cấu hình: n_minibatch = (n_steps*n_envs)/batch_size.
    """
    # Gộp (n_steps, n_envs) thành 1 chiều
    b_obs = torch.as_tensor(buf.obs.reshape(-1, *buf.obs.shape[2:]))
    b_actions = torch.as_tensor(buf.actions.reshape(-1))
    b_logprobs = torch.as_tensor(buf.logprobs.reshape(-1))
    b_adv = torch.as_tensor(buf.advantages.reshape(-1))
    b_returns = torch.as_tensor(buf.returns.reshape(-1))
    b_values = buf.values.reshape(-1)
    n = b_obs.shape[0]
    if valid is None:
        idx_all, mb_size = None, cfg.batch_size
    else:
        idx_all = torch.as_tensor(np.flatnonzero(valid.reshape(-1)))
        n_minibatch = max(1, (cfg.n_steps * cfg.n_envs) // cfg.batch_size)
        mb_size = max(1, int(np.ceil(len(idx_all) / n_minibatch)))
        n = len(idx_all)

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for _ in range(cfg.n_epochs):
        perm = torch.randperm(n) if idx_all is None else idx_all[torch.randperm(n)]
        for start in range(0, n, mb_size):
            idx = perm[start:start + mb_size]
            obs = b_obs[idx].to(device)
            act = b_actions[idx].to(device)
            old_logp = b_logprobs[idx].to(device)
            adv = b_adv[idx].to(device)
            ret = b_returns[idx].to(device)

            _, new_logp, entropy, value = model.get_action_and_value(obs, act)

            # Chuẩn hóa advantage trong minibatch để ổn định độ lớn gradient
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            log_ratio = new_logp - old_logp
            ratio = log_ratio.exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((value - ret) ** 2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():  # chỉ số theo dõi, không ảnh hưởng tới việc học
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy_mean.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)

    out = {k: float(np.mean(v)) for k, v in stats.items()}
    # explained variance: critic giải thích được bao nhiêu phần biến thiên của return (1 là hoàn hảo)
    y = buf.returns.reshape(-1)
    v = b_values
    if valid is not None:
        m = valid.reshape(-1)
        y, v = y[m], v[m]
    var_y = np.var(y)
    out["explained_variance"] = float("nan") if var_y == 0 else float(1 - np.var(y - v) / var_y)
    out["n_samples"] = int(n)
    return out


# ----------------------------------------------------------------------------- Lưu / tải
def save_policy(path: str, model: ActorCritic, extra: dict | None = None) -> None:
    torch.save({"model": model.state_dict(), **(extra or {})}, path)


def load_policy(path: str, device: str = "cpu") -> ActorCritic:
    return load_policy_meta(path, device)[0]


def load_policy_meta(path: str, device: str = "cpu") -> tuple[ActorCritic, dict]:
    """Trả về (model, meta). meta: các thông tin lưu kèm, ví dụ frame_stack, mask_hud, global_step."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = {k: v for k, v in ckpt.items() if k != "model"}
    model = ActorCritic(in_frames=meta.get("frame_stack", 4)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta
