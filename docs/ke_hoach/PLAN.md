> **Tài liệu lịch sử.** File này ghi lại kế hoạch/audit **trong quá trình làm**. Một số nội dung đã lỗi thời (ví dụ: train trên cloud, bản Stable-Baselines3, WSL). Trạng thái cuối cùng: xem [README.md](../../README.md) và [TAI_LIEU_DU_AN.md](../../TAI_LIEU_DU_AN.md).

# Kế hoạch: Agent thắng game Atari Boxing (Gymnasium / ALE)

> **Yêu cầu bổ sung (2026-10-20):** Không được dùng thư viện RL "gọi 1–2 dòng là chạy" như Stable-Baselines3, vì phải demo và giải thích thuật toán.
> → **PPO được tự cài đặt từ đầu bằng PyTorch** (`ppo.py`): mạng Actor-Critic, rollout buffer, GAE, loss có clip, vòng update minibatch.
> Thư viện vẫn dùng: Gymnasium + ale-py (môi trường của đề, kèm wrapper tiền xử lý Atari), PyTorch (tensor, autograd, Adam), numpy, matplotlib, tensorboard.
> Bản SB3 cũ chuyển sang `tham_khao_sb3/` (đã xóa khỏi dự án), chỉ để **đối chiếu** xem bản tự code có đạt mức tương đương không. Không dùng để nộp.

## 1. Bài toán (từ đề bài + manual gốc + kiểm tra trực tiếp trên env)

| Mục | Nội dung |
|---|---|
| Env | `ALE/Boxing-v5`. Bạn điều khiển võ sĩ **trắng**, máy điều khiển võ sĩ **đen** |
| Luật (manual Activision) | Mỗi hiệp 2 phút. Đấm thẳng từ xa (jab) được **1 điểm**, đấm móc cự ly gần được **2 điểm**. Đạt **100 điểm là KO**, ván kết thúc ngay |
| Action | `Discrete(18)`: NOOP, FIRE, UP, RIGHT, LEFT, DOWN, 4 hướng chéo, và 8 hướng kèm FIRE |
| Observation | RGB 210×160×3 (mặc định), hoặc `obs_type="ram"` (128 byte), hoặc `"grayscale"` |
| Reward | Mỗi step nhận một trong **{−2, −1, 0, +1, +2}** (đã kiểm chứng). Tổng reward = điểm mình − điểm máy |
| Độ dài ván | Cố định **1786 step** (≈ 7144 frame ≈ 2 phút), trừ khi có KO. Không có mạng (`lives=0`) |
| v5 | `frameskip=4`, `repeat_action_probability=0.25` (sticky actions theo Machado et al. 2018). Docs khuyên dùng v5 |
| Flavor | mode chỉ có `[0]`. difficulty có `[0,1,2,3]` (mặc định 0) |
| **Điều kiện thắng** | **Tổng reward của ván > 0**. Thắng tuyệt đối là KO (+100, ván kết thúc sớm) |

**Đặc điểm của đối thủ máy (theo manual):**
- Mạnh hơn khi đang dẫn điểm.
- Sau phút đầu thì đuối, phản xạ chậm lại.
- Đứng quá sát thì máy càng khó đánh; đứng quá xa thì không đấm tới.
- Mỗi cú trúng đẩy đối thủ lùi lại một chút. Dồn được vào dây đài thì đối thủ khó né.

→ Đây là bài toán **điều khiển tuần tự, đối kháng**: agent phải học khoảng cách, góc và thời điểm ra đòn, cách né đòn. Chọn action ngay lập tức theo khung hình hiện tại là không đủ.

**Thông tin RAM** (dùng cho agent luật và để kiểm tra kết quả):
- `ram[32]` / `ram[33]` = x của mình / của máy; `ram[34]` / `ram[35]` = y của mình / của máy.
- `ram[18]` / `ram[19]` = điểm của mình / của máy, lưu dạng **BCD** (0x37 = 37). Đã đối chiếu: hiệu điểm giải mã từ RAM khớp với tổng reward.
- `ram[17]` = đồng hồ.

### Mốc tham chiếu
| Agent | Điểm | Nguồn |
|---|---|---|
| Ngẫu nhiên (đo trên máy này, 5 ván) | trung bình ≈ +0.6, dao động −1 đến +3 | tự đo |
| Ngẫu nhiên (bài báo) | 0.1 | DNA / Rainbow |
| Người chơi | 4.3–12.1 | tùy bài báo |
| DQN | 71.8 | Nature 2015 |
| PPO | ~95 | DNA paper |
| Rainbow | 99.6 | Hessel 2018 |

Có ván ngẫu nhiên được +12 (ở difficulty 3). Vì vậy **một ván thắng chưa chứng minh được gì**; phải đánh giá trên nhiều ván, nhiều seed.

## 2. Môi trường (đã xong)
- Python toàn máy đã sạch: đã gỡ `torch` và `ale-py` mà mình lỡ cài lúc đầu.
- Mọi thứ nằm trong `.venv/` của thư mục, cài bằng `--no-cache-dir`. Xóa thư mục là sạch.
- Phiên bản: gymnasium 1.4.0, ale-py 0.12.1 (đã có sẵn ROM, không cần AutoROM), torch 2.14 (không cài thư viện RL nào) (bản CPU trên máy này; trên cloud dùng bản CUDA).
- Link đề bài là docs gymnasium **0.28.1**, nhưng API vẫn giống nhau. Bản 1.x chỉ cần thêm `gym.register_envs(ale_py)`. Thông số env v5 giống hệt (đã kiểm tra `spec.kwargs`).

## 3. Các bẫy kỹ thuật đã kiểm tra (đã xử lý trong `envs.py`)
1. **Skip frame 2 lần:** v5 mặc định đã skip 4 frame. Nếu bọc thêm wrapper tiền xử lý Atari (cũng skip 4) thì mỗi step thành **16 frame**: agent phản ứng chậm và không còn so sánh được với kết quả đã công bố.
   → Tạo env với `frameskip=1` và để `AtariPreprocessing` skip 4. Giữ `repeat_action_probability=0.25` ở tầng ALE để vẫn đúng chuẩn v5.
2. **Clip reward:** nhiều code Atari clip reward về ±1, như vậy mất phân biệt đấm móc (2 điểm) với đấm thẳng (1 điểm). → **Không clip**, vì |r| ≤ 2 vốn đã nhỏ.
3. **Auto-reset của vector env:** gymnasium 1.x mặc định reset ở bước *sau* khi ván kết thúc, tạo ra một transition rác. → Dùng `AutoresetMode.SAME_STEP`; thống kê ván cũ lấy từ `info["final_info"]`.
4. `terminal_on_life_loss=False`, vì Boxing không có mạng.
5. Điểm trong RAM lưu dạng BCD, kiểu `uint8` → phải ép sang `int` trước khi trừ, nếu không sẽ tràn số.

## 4. Phương pháp

### Tầng 0 – Ngẫu nhiên (baseline)
### Tầng 1 – Agent luật dựa trên RAM (`heuristic_agent.py`)
- Đứng ngang hàng với đối thủ, giữ khoảng cách vừa tầm, vào tầm thì đấm. Chỉ dùng để so sánh.

### Tầng 2 – PPO tự cài đặt (lời giải chính: `ppo.py` + `train_ppo.py`)
Giải thích chi tiết ở [THUAT_TOAN.md](../../THUAT_TOAN.md). Các phần tự viết:
| Phần | Trong code |
|---|---|
| Mạng Actor-Critic (Nature CNN, 2 đầu ra π và V) | `ActorCritic` |
| Bộ nhớ rollout (n_steps × n_envs) | `RolloutBuffer` |
| Generalized Advantage Estimation | `compute_gae` |
| Loss PPO-Clip + value loss + entropy bonus, minibatch nhiều epoch | `ppo_update` |
| Giảm tuyến tính lr và ε, cắt gradient, chuẩn hóa advantage, khởi tạo orthogonal | `ppo.py`, `train_ppo.py` |
| Vòng lặp thu dữ liệu, log, checkpoint, resume, eval | `train_ppo.py` |

Hyperparam (theo chuẩn PPO Atari của Schulman 2017 / RL-Zoo):

| Tham số | Giá trị |
|---|---|
| n_envs | 8 |
| n_steps | 128 |
| n_epochs | 4 |
| batch_size | 256 |
| learning_rate | 2.5e-4, giảm tuyến tính |
| clip ε | 0.1, giảm tuyến tính |
| γ / λ | 0.99 / 0.95 |
| vf_coef | 0.5 |
| ent_coef | 0.01 |
| max_grad_norm | 0.5 |
| Adam eps | 1e-5 |

### Đối chiếu
Bản SB3 trong `tham_khao_sb3/` (đã xóa khỏi dự án) dùng cùng hyperparam và tiền xử lý. Nó dùng để kiểm tra bản tự code có học được tương đương không: nếu kém hơn hẳn thì có bug.

### Thí nghiệm bổ sung (điểm cộng)
- Train 10M step trên cloud, 3 seed.
- Đánh giá ở difficulty 0 đến 3.
- So sánh action argmax với action lấy mẫu.
- (Tùy chọn) Tự cài thêm DQN để so sánh với PPO.

## 5. Chạy ở đâu
- **Máy này (CPU, ~550 step/s):** các lần train ≤ 3 giờ (2M step ≈ 1 giờ 15 phút).
- **Cloud:** các lần train trên 3 giờ (10M step ≈ 5.5 giờ trên CPU máy này), chạy bằng `cloud_train.ipynb`.
  - Code tự dùng CUDA nếu có GPU.
  - Lưu `last.pt` mỗi 500k step; `--resume` để train tiếp.

## 6. Cấu trúc code
```
chính/
├── ppo.py              # ★ thuật toán PPO tự cài
├── train_ppo.py        # vòng lặp train
├── envs.py             # env + tiền xử lý
├── evaluate.py  play.py  plot_results.py  heuristic_agent.py
├── cloud_train.ipynb   # train trên Colab/Kaggle
├── THUAT_TOAN.md       # giải thích thuật toán cho demo
├── PLAN.md  README.md  requirements.txt
├── models/<run>/       # final.pt, best.pt, last.pt, ckpt_*.pt
├── logs/<run>/         # episodes.csv, evals.csv, TensorBoard
├── results/  videos/
```

## 7. Giao thức đánh giá
- Env **v5 chuẩn** (sticky 0.25, không clip reward, noop reset), difficulty 0, **30 ván**, seed đánh giá riêng (10000+).
- Mặc định chọn action argmax; báo cáo thêm bản lấy mẫu.
- Chỉ số: điểm TB ± std, min/max, tỉ lệ thắng/hòa/thua, tỉ lệ KO (ván kết thúc sớm).
- **Tiêu chí:**
  - Tối thiểu theo đề: thắng (> 0).
  - Mục tiêu của bản 2M step: thắng ≥ 95% số ván.
  - Mục tiêu nâng cao (10M step trên cloud): điểm TB ≥ +80.

## 8. Thứ tự thực hiện
1. ✅ Nghiên cứu bài toán, dựng venv cô lập.
2. ✅ Baseline random và heuristic (30 ván).
3. ✅ Bản SB3 để đối chiếu: 2M step, thắng 30/30 ván.
4. ✅ **Tự cài PPO** (`ppo.py`, `train_ppo.py`, `envs.py`); smoke test và resume chạy đúng.
5. ✅ Train PPO tự cài 2M step trên máy: **thắng 30/30 ván, TB +24.4**.
6. ⏳ Cloud: PPO tự cài 10M step (seed 0–2).
7. ⏳ Báo cáo và demo (THUAT_TOAN.md).

## 9. Kết quả (cập nhật 2026-10-20)

### ★ PPO tự cài (`ppo.py`), 2M step: kết quả chính để nộp
Train trên CPU máy này mất 48 phút cộng 30 phút trước khi bị ngắt. Lần đầu dừng ở 1.46M step vì máy hết RAM do các chương trình khác. Đã kiểm tra: code không rò rỉ bộ nhớ, tiến trình train ổn định ~0.5 GB. Train tiếp bằng `--resume` từ checkpoint 1M.

Đánh giá 30 ván, seed đánh giá riêng, chuẩn v5:

| Agent | Điểm TB ± std | Min / Max | Thắng | KO |
|---|---|---|---|---|
| Random | −0.1 ± 5.6 | −15 / +15 | 43% | 0% |
| Heuristic (RAM) | −2.4 ± 8.2 | −14 / +29 | 10% | 0% |
| PPO tự cài – best.pt (1.4M step) | +25.8 ± 12.5 | +3 / +53 | **100%** | 0% |
| **PPO tự cài – final.pt (2M step)** | **+24.4 ± 9.1** | **+8 / +40** | **100%** | 0% |
| final.pt, action lấy mẫu | +10.2 ± 6.8 | 0 / +27 | 93% | 0% |
| final.pt @ difficulty 1 | +4.5 ± 5.5 | −5 / +20 | 80% | 0% |
| final.pt @ difficulty 2 | +22.9 ± 7.8 | +6 / +36 | 100% | 0% |
| final.pt @ difficulty 3 | +6.0 ± 4.9 | −1 / +17 | 80% | 0% |

→ **Đạt yêu cầu của đề: thắng 30/30 ván.** Tỉ số trung bình mỗi ván ~63 – 38, khoảng gấp đôi mức người chơi.

**Hành vi đã học** (`analyze_behavior.py`, chi tiết ở THUAT_TOAN.md mục 6):
- Số lần ghi điểm gần gấp đôi random.
- Kiểu đánh "đấm chéo rồi lùi": 61% số action là đấm chéo, 39% là lùi.
- **Không** dồn đối thủ vào dây (9% thời gian, random là 22%).
- Cú đấm trúng khi lệch nhau ~15 px theo chiều dọc. Điều này giải thích vì sao agent luật (cố đứng thẳng hàng) thua.

**So với bản SB3 (cùng hyperparam, cùng 2M step):** đường học gần như trùng ở mốc 1M step (+24.0 so với +23.0, cùng thắng 100%). Các chỉ số chẩn đoán cũng khớp (entropy 2.4–2.8, KL ~0.002, EV ≈ 0) → thuật toán cài đúng. Tuy nhiên ở cuối, bản SB3 đạt +41.6 còn bản tự cài +24.4. Mỗi bản mới chạy **1 seed** nên **chưa kết luận được** chênh lệch là do seed hay do khác biệt cài đặt. Những điểm khác nhỏ đã biết:
- SB3 dùng `FireResetEnv` và `MaxAndSkipEnv` của nó; bản tự cài dùng `AtariPreprocessing` của gymnasium.
- Bản tự cài bị ngắt rồi resume, nên RNG của env khác đi.

Cách kiểm chứng: chạy 3 seed × 10M step trên cloud (`cloud_train.ipynb`).

Hình ảnh: `results/learning_curve.png`, `results/comparison.png`, `videos/ppo_s0_final.mp4`.

### Đối chiếu: bản SB3 (code đã xóa khỏi dự án, chỉ giữ số liệu)
**PPO-SB3 2M step** (train trên máy này, CPU, 62 phút). Đánh giá 30 ván, seed riêng cho đánh giá, chuẩn v5:

| Agent | Điểm TB ± std | Min / Max | Thắng | KO |
|---|---|---|---|---|
| Random | +0.3 ± 5.2 | −12 / +12 | 50% | 0% |
| Heuristic (RAM) | −0.1 ± 8.4 | −11 / +27 | 20% | 0% |
| PPO 2M – best_model | +37.7 ± 12.1 | +12 / +63 | **100%** | 0% |
| **PPO 2M – final** | **+41.6 ± 14.2** | **+10 / +75** | **100%** | 3% |
| PPO 2M – final, action ngẫu nhiên (stochastic) | +10.5 ± 8.6 | −1 / +35 | 93% | 0% |

→ **Đã đạt yêu cầu của đề: thắng 30/30 ván.** Ván kém nhất vẫn hơn máy 10 điểm. Điểm TB trên ván là 79 – 38, gấp khoảng 3.5 lần mức người chơi (12.1). Mục tiêu nâng cao (TB ≥ +80, hay KO) cần train 10M step trên cloud.

**Tổng quát hóa sang difficulty khác** (model chỉ train ở difficulty 0):

| difficulty | Điểm TB | Thắng |
|---|---|---|
| 0 (train) | +41.6 | 100% |
| 1 | +9.9 | 83% |
| 2 | +46.6 | 100% |
| 3 | +6.8 | 90% |

Diễn giải (suy luận từ manual, chưa kiểm chứng trực tiếp): `difficulty` của ALE bật/tắt 2 công tắc độ khó, mà theo manual công tắc "a" làm võ sĩ **chậm lại**. Ở d1 và d3 điểm tụt mạnh, khả năng là võ sĩ của mình bị làm chậm. Ở d2 điểm còn cao hơn d0, khả năng là đối thủ máy bị làm chậm. Dù vậy agent vẫn thắng ≥ 83% ở mọi mức.

**Nhận xét cho báo cáo:**
- Đường học (`results/learning_curve.png`): giai đoạn 0–400k gần như random. Policy argmax bị kẹt một action (−41, độ lệch 0) vì entropy còn ≈ ln 18. Từ 600k tăng nhanh, khoảng 1.2M step thì bão hòa ở ~+40 (eval).
- Eval argmax (~+40) cao hơn nhiều so với reward lúc train (~+15) và eval stochastic (+10.5). Lý do: ent_coef=0.01 giữ policy khá ngẫu nhiên, nên khi chơi thật phải dùng argmax.
- Heuristic viết tay còn kém hơn random về tỉ lệ thắng. Đối thủ máy né và phản đòn tốt, nên luật đơn giản không đủ. Đây là luận điểm cho việc dùng RL.

Các file: `results/*.json` (chi tiết từng ván), `results/learning_curve.png`, `results/comparison.png`, `videos/ppo2M_final.mp4` (ván demo thắng 83 – 28).

## 10. Rủi ro
| Rủi ro | Cách xử lý |
|---|---|
| Phiên cloud bị ngắt (Colab ~12 giờ) | Checkpoint định kỳ + `--resume`; lưu thư mục ra Google Drive |
| Thầy bắt buộc dùng gymnasium 0.28.1 | Dùng `gymnasium==0.28.1`, `ale-py==0.8.1`, `stable-baselines3==2.0.0`, `AutoROM`; chỉ phải bỏ dòng `register_envs` |
| Bản tự cài kém hơn bản SB3 | So từng chỉ số (entropy, kl, EV) với log SB3 để tìm bug |
| Agent học lối chơi lạ, khác người chơi | Hợp lệ. Đo bằng `analyze_behavior.py` và ghi nhận trong báo cáo (thực tế: đấm chéo rồi lùi, không dồn dây) |

## Nguồn
- Đề bài: https://gymnasium.farama.org/v0.28.1/environments/atari/boxing/
- Tổng quan Atari env (v5, sticky actions): https://gymnasium.farama.org/v0.28.1/environments/atari/
- Manual Boxing (AtariAge): https://atariage.com/manual_html_page.php?SoftwareLabelID=46
- RL-Zoo hyperparams: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
- DNA (bảng điểm Random/Human/Rainbow/PPO): https://arxiv.org/pdf/2206.10027
- Rainbow / DQN scores: https://arxiv.org/pdf/2111.10247
- Machado et al. 2018, Revisiting the ALE (giao thức đánh giá, sticky actions)
- Schulman et al. 2017 (PPO); Mnih et al. 2015 (DQN)
