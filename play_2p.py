"""
Demo Boxing 2 người trên Windows: xem trực tiếp, người đấu AI, hoặc quay video.

  # 1. Xem 2 AI đánh nhau trong cửa sổ (60 fps)
  python play_2p.py --live --white models/sp_pool/final.pt --black models/sp_pool/final.pt
  # 2. BẠN điều khiển võ sĩ TRẮNG bằng bàn phím, đấu AI cầm võ sĩ đen
  python play_2p.py --human --black models/sp_pool/final.pt
  # 2b. BẠN cầm võ sĩ ĐEN, đấu agent 1 người (agent này học với AI máy, chỉ biết cầm phe trắng)
  python play_2p.py --human --human-side black --white models/ppo10M_s0/final.pt
  # 3. Quay video mp4 (mặc định)
  python play_2p.py --white models/sp_pool/final.pt --black random

Phím khi chơi (--human): mũi tên = di chuyển (giữ 2 phím để đi chéo), SPACE = đấm,
                         R = chơi ván mới, ESC = thoát.
Người chơi: "random", file .pt, hoặc w.pt+b.pt (model IPPO có mạng riêng cho từng phe).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy mặc định giữ ~360 MB bộ đệm/tiến trình; phải đặt TRƯỚC khi import numpy
import argparse

import numpy as np

from envs_2p import MIRROR_ACTION, make_vec_env_2p
from evaluate_2p import Player

SCALE = 3
W, H = 160 * SCALE, 210 * SCALE
PANEL = 64  # dải thông tin phía dưới cửa sổ

# (dx, dy, fire) -> action ALE. dy âm = đi lên
_DIR = {(0, 0): 0, (0, -1): 2, (1, 0): 3, (-1, 0): 4, (0, 1): 5, (1, -1): 6, (-1, -1): 7, (1, 1): 8, (-1, 1): 9}
_DIR_FIRE = {(0, 0): 1, (0, -1): 10, (1, 0): 11, (-1, 0): 12, (0, 1): 13, (1, -1): 14, (-1, -1): 15, (1, 1): 16,
             (-1, 1): 17}


def keyboard_action(pg) -> int:
    k = pg.key.get_pressed()
    dx = int(k[pg.K_RIGHT]) - int(k[pg.K_LEFT])
    dy = int(k[pg.K_DOWN]) - int(k[pg.K_UP])
    return (_DIR_FIRE if k[pg.K_SPACE] else _DIR)[(dx, dy)]


def run_window(a, white: Player | None, black: Player | None):
    """white/black = None nghĩa là phe đó do NGƯỜI điều khiển bằng bàn phím."""
    import pygame as pg
    pg.init()
    screen = pg.display.set_mode((W, H + PANEL))
    title = "Boxing – " + ("BẠN (trắng) vs AI" if white is None else "AI vs BẠN (đen)" if black is None else "AI vs AI")
    pg.display.set_caption(title)
    font = pg.font.SysFont("arial", 18)
    big = pg.font.SysFont("arial", 30, bold=True)
    clock = pg.time.Clock()
    name_w = "BẠN (bàn phím)" if white is None else white.name
    name_b = "BẠN (bàn phím)" if black is None else black.name

    envs = make_vec_env_2p(1, asynchronous=False, mask_hud=(white.mask_hud if white else True, True))
    raw = envs.envs[0]
    rng = np.random.default_rng(a.seed)
    state = {"quit": False, "restart": False}

    def draw(frame, ep, banner=None):
        surf = pg.surfarray.make_surface(np.ascontiguousarray(frame.transpose(1, 0, 2)))
        screen.blit(pg.transform.scale(surf, (W, H)), (0, 0))
        pg.draw.rect(screen, (25, 25, 25), (0, H, W, PANEL))
        screen.blit(font.render(f"TRẮNG: {name_w}", True, (240, 240, 240)), (8, H + 6))
        screen.blit(font.render(f"ĐEN:   {name_b}", True, (170, 170, 170)), (8, H + 34))
        sc = big.render(f"{ep['pts_w']:.0f} - {ep['pts_b']:.0f}", True, (255, 220, 80))
        screen.blit(sc, (W - sc.get_width() - 10, H + 14))
        if banner:
            t = big.render(banner, True, (255, 255, 255), (0, 0, 0))
            screen.blit(t, ((W - t.get_width()) // 2, H // 2 - 20))
        pg.display.flip()

    def on_frame(frame, ep):  # gọi sau mỗi frame giả lập -> 60 fps
        for ev in pg.event.get():
            if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                state["quit"] = True
            if ev.type == pg.KEYDOWN and ev.key == pg.K_r:
                state["restart"] = True
        draw(frame, ep)
        clock.tick(60 * a.speed)

    raw.frame_hook = on_frame
    games = 0
    while not state["quit"]:
        obs, _ = envs.reset(seed=a.seed + games)
        state["restart"] = False
        while not (state["quit"] or state["restart"]):
            act_w = keyboard_action(pg) if white is None else white.act(obs[:, 0], rng, 0)[0]
            # phe đen: env nhận action theo "góc nhìn gương" rồi tự đổi trái<->phải; phím của người là
            # hướng thật trên màn hình nên phải đổi trước để bấm → thì võ sĩ đi sang phải
            act_b = MIRROR_ACTION[keyboard_action(pg)] if black is None else black.act(obs[:, 1], rng, 1)[0]
            obs, r, term, trunc, infos = envs.step(np.array([[act_w, act_b]]))
            fi = infos.get("final_info", {})
            if "episode" in fi:
                ep = {k: float(v[0]) for k, v in fi["episode"].items() if not k.startswith("_")}
                d = ep["pts_w"] - ep["pts_b"]
                ww = "BẠN" if white is None else "TRẮNG"
                wb = "BẠN" if black is None else "ĐEN"
                msg = f"{ww} THẮNG {ep['pts_w']:.0f}-{ep['pts_b']:.0f}" if d > 0 else (
                    f"{wb} THẮNG {ep['pts_b']:.0f}-{ep['pts_w']:.0f}" if d < 0 else f"HÒA {ep['pts_w']:.0f}-{ep['pts_b']:.0f}")
                print(f"Ván {games + 1}: {msg}")
                draw(raw.render(), ep, msg + "  (R: ván mới, ESC: thoát)")
                while not (state["quit"] or state["restart"]):  # đợi người xem bấm phím
                    for ev in pg.event.get():
                        if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                            state["quit"] = True
                        if ev.type == pg.KEYDOWN and ev.key == pg.K_r:
                            state["restart"] = True
                    clock.tick(30)
                break
        games += 1
    envs.close()
    pg.quit()


def record_video(a, W_: Player, B_: Player):
    import cv2
    envs = make_vec_env_2p(1, asynchronous=False, mask_hud=(W_.mask_hud, True))
    raw = envs.envs[0]
    obs, _ = envs.reset(seed=a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs("videos", exist_ok=True)
    out = a.out or f"videos/2p_{W_.name.replace('/', '_')}_vs_{B_.name.replace('/', '_')}.mp4"
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 60, (W, H))

    def on_frame(frame, ep):  # ghi MỌI frame -> video 60 fps đúng tốc độ thật
        f = cv2.cvtColor(cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        cv2.putText(f, f"TRANG {W_.name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(f, f"DEN   {B_.name}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        cv2.putText(f, f"{ep['pts_w']:.0f} - {ep['pts_b']:.0f}", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        writer.write(f)

    raw.frame_hook = on_frame
    while True:
        act = np.array([[W_.act(obs[:, 0], rng, 0)[0], B_.act(obs[:, 1], rng, 1)[0]]])
        obs, r, term, trunc, infos = envs.step(act)
        if term[0] or trunc[0]:
            ep = {k: float(v[0]) for k, v in infos["final_info"]["episode"].items() if not k.startswith("_")}
            break
    writer.release()
    envs.close()
    print(f"Trắng {W_.name} {ep['pts_w']:.0f} - {ep['pts_b']:.0f} Đen {B_.name}. Đã lưu {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--white", default="random", help='"random", file .pt, hoặc w.pt+b.pt (IPPO)')
    p.add_argument("--black", default="random")
    p.add_argument("--live", action="store_true", help="mở cửa sổ xem trực tiếp")
    p.add_argument("--human", action="store_true", help="bạn điều khiển 1 võ sĩ bằng bàn phím")
    p.add_argument("--human-side", choices=["white", "black"], default="white",
                   help="phe của bạn. black: để đấu agent 1 người (agent đó chỉ biết cầm phe trắng)")
    p.add_argument("--speed", type=float, default=1.0, help="tốc độ khi xem (1 = tốc độ thật)")
    p.add_argument("--seed", type=int, default=None,
                   help="mặc định: xem/chơi trực tiếp thì mỗi lần một ván mới; quay video thì cố định 12345 (tái lập được)")
    p.add_argument("--out", default=None, help="file mp4 (chế độ quay video)")
    a = p.parse_args()
    if a.seed is None:
        import time
        a.seed = int(time.time()) % 100_000 if (a.live or a.human) else 12_345

    if a.human:
        if a.human_side == "white":
            run_window(a, None, Player(a.black))
        else:
            run_window(a, Player(a.white), None)
    elif a.live:
        run_window(a, Player(a.white), Player(a.black))
    else:
        record_video(a, Player(a.white), Player(a.black))


if __name__ == "__main__":
    main()
