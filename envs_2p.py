"""
Boxing 2 người chơi: cả võ sĩ TRẮNG (người chơi 1) và võ sĩ ĐEN (người chơi 2) đều do agent điều khiển.
Lõi giả lập: `multi_agent_ale_py`, bản ALE hỗ trợ 2 người, cũng là lõi của PettingZoo `atari.boxing_v2`.
Trên Windows cài từ wheels/ (bản build sẵn ale_c.dll, xem tools/build_windows_wheel/).
Toàn bộ tiền xử lý dưới đây là TỰ VIẾT (lõi ALE trả về từng frame thô, không skip, không sticky).

Mỗi step của env:
  1. Nhận 2 action (trắng, đen). Action của đen đang ở "góc nhìn lật gương" nên đổi trái <-> phải.
  2. Lặp 4 frame. Mỗi frame, mỗi người chơi có 25% khả năng giữ action cũ (sticky actions, Machado 2018).
  3. Lấy max 2 frame cuối (chống nhấp nháy) -> tô dải đen 8 cột ở mép trái thành màu nền
     -> góc nhìn đen: lật ngang + đổi màu đen <-> trắng -> resize 84x84 -> xếp chồng 4 frame.
  4. Trả về obs (2, 4, 84, 84): obs[0] = góc nhìn trắng, obs[1] = góc nhìn đen.
     reward = điểm của trắng; điểm của đen = -reward (zero-sum, đã kiểm chứng).

CHUẨN HÓA GÓC NHÌN: sau khi lật và đổi màu, võ sĩ đen thấy mình là võ sĩ trắng đứng bên trái,
giống hệt góc nhìn của trắng. Nhờ vậy MỘT mạng chơi được cả 2 phe (self-play chia sẻ tham số).
Giá trị xám (đo trực tiếp): 214 = trắng, 0 = đen, 132 = nền, 150 = dây đài, 41 = đồng hồ/logo.
"""
from __future__ import annotations

import glob
import sys
import os
from collections import deque

import cv2
import gymnasium as gym
import numpy as np

from rom_path import ensure_ale_roms_dir, find_rom_dir

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

FRAME_SKIP = 4
FRAME_STACK = 4
STICKY_PROB = 0.25
NOOP_MAX = 30
GRAY_WHITE, GRAY_BLACK, GRAY_BG = 214, 0, 132
HUD_COLS = 8  # dải đen ở mép trái màn hình Atari (HMOVE), không thuộc sàn đấu

# Đổi trái <-> phải khi soi gương. Thứ tự action ALE:
# 0 NOOP 1 FIRE 2 UP 3 RIGHT 4 LEFT 5 DOWN 6 UPRIGHT 7 UPLEFT 8 DOWNRIGHT 9 DOWNLEFT
# 10 UPFIRE 11 RIGHTFIRE 12 LEFTFIRE 13 DOWNFIRE 14 UPRIGHTFIRE 15 UPLEFTFIRE 16 DOWNRIGHTFIRE 17 DOWNLEFTFIRE
MIRROR_ACTION = np.array([0, 1, 2, 4, 3, 5, 7, 6, 9, 8, 10, 12, 11, 13, 15, 14, 17, 16], dtype=np.int64)


def potential(ram) -> float:
    """
    Thế năng cho reward shaping (Ng, Harada & Russell 1999): Φ(s) = −(|dx| + |dy|) / 100,
    với dx, dy = khoảng cách giữa 2 võ sĩ (đọc từ RAM: x ở byte 32/33, y ở byte 34/35).
    Thưởng phụ F = γΦ(s') − Φ(s): tiến lại gần đối thủ được cộng một chút, lùi xa bị trừ một chút.
    Định lý của Ng 1999: thưởng phụ dạng thế năng KHÔNG làm đổi chiến lược tối ưu, chỉ giúp khám phá.
    Khoảng cách đối xứng nên cả 2 phe nhận cùng F (đều bị "kéo" lại gần nhau).
    """
    return -(abs(int(ram[33]) - int(ram[32])) + abs(int(ram[35]) - int(ram[34]))) / 100.0


def info_field(infos: dict, key: str, n: int) -> np.ndarray:
    """Lấy 1 trường info từ vector env (SAME_STEP): env vừa hết ván thì giá trị nằm trong final_info."""
    out = np.zeros(n, dtype=np.float32)
    if key in infos:
        m = infos.get("_" + key, np.ones(n, bool))
        out[m] = np.asarray(infos[key])[m]
    fi = infos.get("final_info", {})
    if key in fi:
        m = fi.get("_" + key, np.zeros(n, bool))
        out[m] = np.asarray(fi[key])[m]
    return out


def process_frame(f1: np.ndarray, f2: np.ndarray, mirror: bool, mask_hud: bool) -> np.ndarray:
    """2 frame xám 210x160 -> 1 frame 84x84 uint8 (max-pool, che HUD, lật gương nếu cần)."""
    f = np.maximum(f1, f2)
    if mask_hud or mirror:
        f = f.copy()
        f[:, :HUD_COLS] = GRAY_BG
    if mirror:
        f = f[:, ::-1]
        w, b = f == GRAY_WHITE, f == GRAY_BLACK
        f = f.copy()
        f[w], f[b] = GRAY_BLACK, GRAY_WHITE
    return cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)


class Boxing2P(gym.Env):
    """Env Gymnasium bọc Boxing 2 người, để dùng được gym.vector.AsyncVectorEnv."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, sticky: float = STICKY_PROB, mask_hud: tuple = (True, True), render_mode=None,
                 shaping_gamma: float | None = None, passive_dist: float | None = None):
        # Dùng thẳng lõi ALE 2 người (multi_agent_ale_py), khởi tạo đúng trình tự như PettingZoo boxing_v2,
        # nhưng nạp ROM qua đường dẫn ASCII (PettingZoo gọi Path.resolve() làm hỏng đường dẫn ngắn 8.3).
        import multi_agent_ale_py
        multi_agent_ale_py.ALEInterface.setLoggerMode("error")
        self.ale = multi_agent_ale_py.ALEInterface()
        self.ale.setFloat(b"repeat_action_probability", 0.0)  # sticky tự làm ở step(), riêng từng người chơi
        self._rom = os.path.join(find_rom_dir(), "boxing.bin")
        self._load(seed=0)
        self.sticky = sticky
        self.mask_hud = mask_hud
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(0, 255, (2, FRAME_STACK, 84, 84), np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([18, 18])
        self._stacks = [deque(maxlen=FRAME_STACK), deque(maxlen=FRAME_STACK)]
        self._rng = np.random.default_rng()
        self.shaping_gamma = shaping_gamma  # None = tắt reward shaping (mặc định, dùng khi đánh giá)
        # Luật "chống né tránh" (chỉ khi train): info["passive"] = 1 nếu 2 võ sĩ đứng xa nhau hơn passive_dist
        self.passive_dist = passive_dist
        self.frame_hook = None  # hàm(rgb_frame, thống_kê_ván) được gọi sau MỖI frame giả lập (chỉ dùng khi demo)

    def _load(self, seed: int):
        """Đặt seed giả lập, nạp ROM, chọn chế độ 2 người (giống PettingZoo BaseAtariEnv._seed)."""
        self.ale.setInt(b"random_seed", int(seed))
        self.ale.loadROM(self._rom)
        self.ale.setMode(self.ale.getAvailableModes(2)[0])
        assert self.ale.numPlayersActive() == 2, "ROM không ở chế độ 2 người"

    def _screen(self) -> np.ndarray:
        return self.ale.getScreenGrayscale().reshape(210, 160)

    def _obs(self) -> np.ndarray:
        return np.stack([np.stack(self._stacks[0]), np.stack(self._stacks[1])])

    def _push(self, f1, f2, fill: bool = False):
        for side in (0, 1):
            fr = process_frame(f1, f2, mirror=(side == 1), mask_hud=self.mask_hud[side])
            if fill:
                self._stacks[side].clear()
                for _ in range(FRAME_STACK):
                    self._stacks[side].append(fr)
            else:
                self._stacks[side].append(fr)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._load(seed)
        self.ale.reset_game()
        prev = self._screen()
        for _ in range(int(self._rng.integers(0, NOOP_MAX + 1))):  # noop reset ở mức frame
            self.ale.act(np.array([0, 0], dtype=np.int32))
            prev = self._screen()
        cur = self._screen()
        self._push(prev, cur, fill=True)
        self._last = np.zeros(2, dtype=np.int64)  # action thật đang được giữ (cho sticky)
        self._ep = {"r": 0.0, "pts_w": 0.0, "pts_b": 0.0, "l": 0}
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.int64)
        want = np.array([a[0], MIRROR_ACTION[a[1]]])  # đen: đổi từ góc nhìn gương về thật
        phi0 = potential(self.ale.getRAM()) if self.shaping_gamma is not None else 0.0
        r_w = r_b = 0.0
        frames = []
        terminated = False
        for _ in range(FRAME_SKIP):
            keep = self._rng.random(2) < self.sticky
            self._last = np.where(keep, self._last, want)
            rew = self.ale.act(self._last.astype(np.int32))
            r_w += float(rew[0])
            r_b += float(rew[1])
            # trong 1 frame chỉ 1 bên ghi điểm -> thưởng dương của bên nào là điểm bên đó
            self._ep["pts_w"] += max(float(rew[0]), 0.0)
            self._ep["pts_b"] += max(float(rew[1]), 0.0)
            frames.append(self._screen())
            self._ep["l"] += 1
            if self.frame_hook is not None:  # dùng khi xem trực tiếp: vẽ từng frame (60 fps) cho mượt
                self.frame_hook(self.render(), self._ep)
            if self.ale.game_over():
                terminated = True
                break
        f2 = frames[-1]
        f1 = frames[-2] if len(frames) > 1 else f2
        self._push(f1, f2)
        self._ep["r"] += r_w
        info = {"r_black": r_b}
        if self.shaping_gamma is not None:  # Φ(trạng thái cuối ván) = 0 để đúng điều kiện của định lý
            phi1 = 0.0 if terminated else potential(self.ale.getRAM())
            info["shaping"] = self.shaping_gamma * phi1 - phi0
        if self.passive_dist is not None:
            info["passive"] = float(-potential(self.ale.getRAM()) * 100 > self.passive_dist)
        if terminated:
            info["episode"] = dict(self._ep)
        return self._obs(), r_w, terminated, False, info

    def render(self):
        return self.ale.getScreenRGB().reshape(210, 160, 3)

    def close(self):
        self.ale = None


def make_vec_env_2p(n_envs: int, asynchronous: bool = True, mask_hud=(True, True), render_mode=None,
                    shaping_gamma: float | None = None, passive_dist: float | None = None):
    """mask_hud: 1 tuple dùng chung cho mọi env, hoặc danh sách tuple riêng cho từng env."""
    masks = mask_hud if isinstance(mask_hud, list) else [mask_hud] * n_envs
    fns = [lambda m=m: Boxing2P(mask_hud=m, render_mode=render_mode, shaping_gamma=shaping_gamma,
                                passive_dist=passive_dist) for m in masks]
    cls = gym.vector.AsyncVectorEnv if (asynchronous and n_envs > 1) else gym.vector.SyncVectorEnv
    return cls(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


class Boxing1PCustom(gym.Env):
    """
    Boxing 1 người (đấu AI máy, v5 chuẩn) nhưng dùng CÙNG tiền xử lý tự viết như bản 2 người,
    để đánh giá agent self-play khi gặp AI máy (agent luôn là võ sĩ trắng, không lật gương).
    """

    def __init__(self, difficulty: int = 0, mask_hud: bool = True, shaping_gamma: float | None = None):
        import ale_py
        gym.register_envs(ale_py)
        ensure_ale_roms_dir()
        self._env = gym.make("ALE/Boxing-v5", frameskip=1, repeat_action_probability=STICKY_PROB,
                             obs_type="grayscale", difficulty=difficulty)
        self.ale = self._env.unwrapped.ale
        self.mask_hud = mask_hud
        self.shaping_gamma = shaping_gamma
        self.observation_space = gym.spaces.Box(0, 255, (FRAME_STACK, 84, 84), np.uint8)
        self.action_space = gym.spaces.Discrete(18)
        self._stack = deque(maxlen=FRAME_STACK)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o, _ = self._env.reset(seed=seed)
        prev = o
        for _ in range(int(self.np_random.integers(0, NOOP_MAX + 1))):
            prev = o
            o, *_ = self._env.step(0)
        fr = process_frame(prev, o, mirror=False, mask_hud=self.mask_hud)
        self._stack.clear()
        for _ in range(FRAME_STACK):
            self._stack.append(fr)
        self._ep = {"r": 0.0, "l": 0}
        return np.stack(self._stack), {}

    def step(self, action):
        frames, r, terminated, truncated = [], 0.0, False, False
        phi0 = potential(self.ale.getRAM()) if self.shaping_gamma is not None else 0.0
        for _ in range(FRAME_SKIP):
            o, rr, terminated, truncated, _ = self._env.step(int(action))
            r += rr
            frames.append(o)
            self._ep["l"] += 1
            if terminated or truncated:
                break
        f1 = frames[-2] if len(frames) > 1 else frames[-1]
        self._stack.append(process_frame(f1, frames[-1], mirror=False, mask_hud=self.mask_hud))
        self._ep["r"] += r
        info = {"episode": dict(self._ep)} if (terminated or truncated) else {}
        if self.shaping_gamma is not None:
            phi1 = 0.0 if (terminated or truncated) else potential(self.ale.getRAM())
            info["shaping"] = self.shaping_gamma * phi1 - phi0
        return np.stack(self._stack), r, terminated, truncated, info
