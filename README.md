# Boxing RL: PPO tự cài đặt chơi Atari Boxing (1 người và 2 người)

Agent học tăng cường **PPO tự cài đặt bằng PyTorch** (không dùng thư viện RL có sẵn) chơi game Atari **Boxing**:
- **Chế độ 1 người:** agent cầm võ sĩ trắng đấu AI có sẵn của game, **thắng 30/30 ván**.
- **Chế độ 2 người:** **cả 2 võ sĩ đều là agent**, tự học bằng cách đánh với chính mình (self-play + opponent pool).

**Chạy hoàn toàn trên Windows** (không cần Linux/WSL, không cần trình biên dịch, không cần GPU).

| Tài liệu | Nội dung |
|---|---|
| **[TAI_LIEU_DU_AN.md](TAI_LIEU_DU_AN.md)** | **Giải thích toàn bộ dự án**: kiến thức nền, PPO, thưởng/phạt khi 2 con cùng muốn thắng, Elo, Q&A khi bảo vệ |
| [THUAT_TOAN.md](THUAT_TOAN.md) | Công thức PPO, GAE, self-play chi tiết; kịch bản demo |
| **[KICH_BAN_BAO_VE.md](KICH_BAN_BAO_VE.md)** | **Kịch bản bảo vệ**: 13 slide, demo từng bước, số liệu cần nhớ, câu hỏi thường gặp |
| [docs/ke_hoach/](docs/ke_hoach/) | Các bản kế hoạch và audit trong quá trình làm |

---

## 1. Cài đặt (1 lần, khoảng 3–5 phút)
**Cần:** Windows 10/11 64-bit, **Python 3.11–3.13 bản 64-bit** (https://www.python.org/downloads/, nhớ tick *Add python.exe to PATH*), có mạng để tải thư viện (~300 MB).

**Nhấp đúp `setup.bat`.** Script sẽ:
1. Tạo môi trường ảo `.venv\` **ngay trong thư mục dự án**: không cài gì vào Python của máy, không để lại cache. Xóa thư mục dự án là sạch.
2. Cài PyTorch bản CPU, các thư viện trong `requirements.txt`, và lõi Atari 2 người cho Windows (`wheels\`, build sẵn).
3. Chạy `smoke_test.py` kiểm tra mọi thứ. Thấy `TẤT CẢ ĐỀU ỔN.` là xong.

Thư mục dự án **đặt tên có dấu cũng được** (đã kiểm tra với `thử bàn giao`).

## 2. Demo
**Nhấp đúp `demo.bat`**, rồi chọn trong menu:
| Mục | Nhóm | Nội dung |
|---|---|---|
| 1 | 1 người | Xem bot đấu AI có sẵn của game (bot cầm võ sĩ trắng) |
| 2 | 1 người | **BẠN đấu bot 1 người**: bạn cầm võ sĩ đen |
| 3 | 1 người | Đánh giá nhanh: bot đấu AI máy 5 ván |
| 4 | 2 người | **Xem 2 BOT ĐÁNH NHAU** (cùng 1 mạng cầm cả 2 phe) |
| 5 | 2 người | **BẠN đấu bot 2 người**: bạn cầm võ sĩ trắng |
| 6 | 2 người | Bot 1 người (trắng) đấu bot 2 người (đen) |
| 7 | 2 người | Bản đầu của lần train cuối đấu bản cuối: thấy rõ bot mạnh lên |
| 8 | 2 người | Đánh giá nhanh: bot 2 người đấu random 6 ván (cả 2 phe) |
| 9 | Khác | Kiểm chứng thuật toán PPO tự cài (10 phép thử) |
| 10 | Khác | Mở thư mục biểu đồ và video |

**Phím khi tự chơi:** mũi tên để di chuyển (giữ 2 phím để đi chéo), SPACE để đấm, R ván mới, ESC thoát.

Chạy tay (sau khi kích hoạt môi trường: `.venv\Scripts\activate`):
```bat
python play.py --agent ppo --model models\ppo10M_s0\final.pt --live
python play_2p.py --live --white models\league6\final.pt --black models\league6\final.pt
python play_2p.py --human --black models\league6\final.pt
python play_2p.py --human --human-side black --white models\ppo10M_s0\final.pt
```

## 3. Đánh giá
```bat
python verify_algorithm.py        &REM chứng minh PPO tự cài khớp định nghĩa toán học (GAE, log π, entropy, loss clip, hướng cập nhật)
python evaluate.py --agent random --episodes 30
python evaluate.py --agent ppo --model models\ppo_s0\final.pt --episodes 30
python evaluate_2p.py vs-random --model models\sp_pool\final.pt --games 20
python evaluate_2p.py vs-cpu    --model models\sp_pool\final.pt --episodes 30
python evaluate_2p.py league --players random models\ppo_s0\final.pt models\sp_pool\final.pt --games 8
python analyze_behavior.py --model models\ppo_s0\final.pt
python plot_results.py
```
`evaluate_2p.py vs-cpu` cho agent self-play (chưa từng gặp AI máy) đấu AI máy. `league` đấu vòng tròn, tính Elo và vẽ heatmap.

## 4. Train lại từ đầu
```bat
python train_ppo.py    --timesteps 2000000 --run-name ppo_s0            &REM 1 người, ~1–2 giờ trên CPU
REM 2 người, bản cuối (v6) = 2 bước:
REM  B1 (v5): tinh chỉnh agent 1 người, đấu tập đối thủ cố định + shaping
python league_ppo.py --init models\ppo10M_s0inal.pt --timesteps 3000000 --learning-rate 5e-05 --ent-coef 0.001 --n-2p 6 --n-cpu 2 --p-self 0 --p-random 0.35 --p-still 0.2 --opp-models models/ppo10M_s0/final.pt --freeze-pool --shaping 3 --shaping-decay 0.6 --run-name league
REM  B2 (v6): + 50% self-play + luật chống né tránh
python league_ppo.py --init models\leagueinal.pt --timesteps 3000000 --learning-rate 5e-05 --ent-coef 0.001 --n-2p 6 --n-cpu 2 --p-self 0.5 --p-random 0.15 --p-still 0.1 --opp-models models/league/final.pt,models/ppo10M_s0/final.pt --passive-penalty 0.02 --passive-dist 40 --run-name league6
REM  (thêm --resume để train tiếp nếu bị ngắt; selfplay_ppo.py = các bản self-play thuần dùng để so sánh)
tensorboard --logdir logs
```
`selfplay_ppo.py --mode` có 3 chế độ: `pool` (chính), `selfplay`, `ippo`; giải thích ở TAI_LIEU_DU_AN.md mục 7.
Tốc độ trên CPU 12 luồng: ~300–550 step/s (1 người), ~150–250 env-step/s (2 người). Máy đang chạy game hoặc trình duyệt nặng thì chậm hơn.

## 5. Kết quả (chi tiết: TAI_LIEU_DU_AN.md mục 6, 7.8, 7.9)
**1 người: agent đấu AI của game** (30 ván mỗi model, chuẩn v5)
| Agent | Điểm TB | Thắng | KO |
|---|---|---|---|
| Random | −0.1 | 43% | 0% |
| Luật viết tay | −2.4 | 10% | 0% |
| PPO tự cài, 2M step | +24.4 | 100% | 0% |
| **PPO tự cài, 10M step (3 seed)** | **+90.7 / +89.9 / +87.7** | **100%** | **100%** |

**2 người: cả 2 võ sĩ là agent** (model cuối `models/league6/final.pt`)
| Phép đo | Kết quả |
|---|---|
| Hai bot tự đánh nhau | **26.8 điểm/ván** (đánh nhau thật) |
| Đấu random (cầm cả 2 phe) | **+7.4, thắng 93%** |
| Đấu AI của game | **+85.6, thắng 100%** |
| Đối đầu agent 1 người / bản v5 | **thắng 90% / 100%** |
| Elo (vòng tròn 7 người chơi) | **hạng 1, +456** |

Self-play thuần **thất bại** (2 bên học cách né nhau). Quá trình chẩn đoán và sửa qua 6 bản được ghi ở TAI_LIEU_DU_AN.md mục 7.8.

## 6. Cấu trúc thư mục
| File / thư mục | Vai trò |
|---|---|
| **`ppo.py`** | **Lõi thuật toán tự cài**: mạng Actor-Critic (CNN), RolloutBuffer, GAE, loss PPO-Clip, update có mặt nạ |
| `train_ppo.py` | Train 1 người |
| `league_ppo.py` | **Train 2 người, bản cuối**: tinh chỉnh + tập đối thủ + self-play + reward shaping + luật chống né tránh |
| `selfplay_ppo.py` | Train 2 người, self-play thuần (pool / selfplay / IPPO), dùng để so sánh |
| `envs.py` / `envs_2p.py` | Môi trường 1 / 2 người + tiền xử lý (2 người: tự viết, có lật gương góc nhìn) |
| `rom_path.py` | Xử lý đường dẫn có dấu cho lõi C++ của ALE |
| `evaluate.py` / `evaluate_2p.py` | Đánh giá 1 / 2 người (vs random, vs AI máy, Elo) |
| `play.py` / `play_2p.py` | Xem trực tiếp, người đấu AI, quay video |
| `analyze_behavior.py` | Phân tích hành vi agent từ RAM |
| `heuristic_agent.py` | Agent luật viết tay (baseline) |
| `plot_results.py` | Vẽ biểu đồ vào `results\` |
| `demo.py`, `demo.bat`, `setup.bat`, `smoke_test.py` | Menu demo, cài đặt, kiểm tra |
| `verify_algorithm.py` | Kiểm chứng từng phần thuật toán tự cài so với công thức toán (10 phép thử) |
| `wheels\` | Lõi Atari 2 người cho Windows (build sẵn; cách build: `tools\build_windows_wheel\BUILD.md`) |
| `models\`, `results\`, `videos\`, `logs\` | Model đã train, kết quả, video, log train |
| `build_release.py` | Đóng gói zip bàn giao |
| `tools/hang_doi_train/` | Công cụ nội bộ: hàng đợi train dài tự resume (không cần cho người dùng) |
| `tools/build_windows_wheel/` | Cách build lõi Atari 2 người cho Windows |

**Các model trong `models\`**
| Thư mục | Là gì |
|---|---|
| **`ppo10M_s0`**, `ppo10M_s1`, `ppo10M_s2` | **Bot 1 người cuối** (3 seed, 10M step, +90 đấu AI máy). Demo dùng `ppo10M_s0` |
| **`league6`** | **Bot 2 người cuối (v6)**, kèm 3 snapshot (0, 1M, 2M step) dùng cho demo và giải Elo |
| `league` | Bản v5 (bước trung gian của bot 2 người) |
| `ppo_s0` | Bot 1 người 2M step (bản đầu tiên) |
| `abl_clipreward`, `abl_stack1`, `abl_noclip` | Các thí nghiệm ablation |
| `sp_pool`, `sp_selfplay`, `sp_ippo` | Self-play thuần (thất bại, giữ lại để so sánh trong báo cáo) |

**Thư viện dùng:** gymnasium + ale-py (môi trường của đề), multi_agent_ale_py (lõi Atari 2 người), PyTorch (tensor, autograd, Adam), numpy, opencv, pygame, matplotlib, tensorboard. **Không dùng thư viện RL nào.**
