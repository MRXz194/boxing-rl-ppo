> **Tài liệu lịch sử.** File này ghi lại kế hoạch/audit **trong quá trình làm**. Một số nội dung đã lỗi thời (ví dụ: train trên cloud, bản Stable-Baselines3, WSL). Trạng thái cuối cùng: xem [README.md](../../README.md) và [TAI_LIEU_DU_AN.md](../../TAI_LIEU_DU_AN.md).

# Kế hoạch điểm cao: Boxing 1 người + Boxing 2 người (cả 2 võ sĩ đều là agent)

> Trạng thái hiện tại (2026-10-20): **Phần 1 (1 người) đã đạt yêu cầu tối thiểu.** PPO tự cài thắng 30/30 ván trước máy (+24.4). Kế hoạch này dùng để nâng lên mức xuất sắc.

## ★ CẬP NHẬT 2026-10-20 (tối): đổi sang train toàn bộ trên máy, kết quả audit, lịch chạy
Theo yêu cầu: **không dùng cloud, mọi thứ train trên máy bạn** (bất kể bao lâu). Các mục 2 và 7 bên dưới (cloud) không còn áp dụng; lịch thật ở cuối mục này.

**Kết quả audit kế hoạch: những gì đã kiểm chứng hoặc sửa trước khi chạy**
| # | Giả định / vấn đề | Kết quả |
|---|---|---|
| 1 | Lật gương góc nhìn phe đen để 1 mạng chơi cả 2 phe | ✅ Đúng. Đo giá trị xám: trắng 214, đen 0, nền 132; sàn đài đối xứng (x 24–135). Ảnh kiểm tra ở `results/audit/` |
| 2 | Dải đen 8 cột ở mép trái (lỗi hiển thị Atari) sẽ thành dải trắng ở mép phải khi lật và đổi màu, khiến mạng biết mình đang chơi phe nào | ⚠️ Tìm ra và đã sửa: tô dải đó bằng màu nền ở cả 2 phe |
| 3 | Thứ tự 18 action của PettingZoo có giống ALE không (để đổi trái↔phải) | ✅ Giống hệt (0–17) |
| 4 | PettingZoo mặc định sticky = 0, không skip frame | ⚠️ Đã tự viết: skip 4, max 2 frame, sticky 0.25 riêng mỗi bên, noop reset |
| 5 | Ở bước kết thúc ván, env tự reset → mất thưởng của đen ở bước cuối (có thể là cú KO), và crash nếu 8 ván kết thúc cùng lúc | ⚠️ Lỗi thật, đã sửa: thưởng đen = −thưởng trắng (zero-sum đã kiểm chứng mọi bước) |
| 6 | Model IPPO có 2 mạng riêng cho 2 phe, khi đánh giá phải dùng đúng mạng theo phe | ⚠️ Đã sửa: cú pháp `w.pt+b.pt` |
| 7 | Tên snapshot không tròn mốc (lưu ở cuối update 1024 step) | ⚠️ Đã sửa: hàng đợi tự tìm snapshot gần mốc nhất |
| 8 | Tốc độ: 2 người chỉ ~60–80 env-step/s lúc test | Nguyên nhân: máy đang bận 66–88% CPU vì chương trình khác (TFT, Discord, trình duyệt). PyTorch nhanh nhất ở 4 luồng → đặt `--threads 4` |
| 9 | Model phần 1 (thắng AI máy +24) gặp đối thủ khác trong env 2 người | 🔎 Chỉ +2.5 khi gặp random → học riêng để khai thác AI máy, chưa đánh tổng quát. Đây là lý do cần self-play |
| 10 | Multi_agent_ale_py không có bản cho Windows | ✅ Chạy qua WSL Ubuntu; venv Linux `.venv-linux/` nằm trong thư mục dự án |

**Hàng đợi tự chạy (`run_queue.py`)**: 2 làn song song. Mỗi việc tự `--resume` và tự chạy lại nếu crash (tối đa 8 lần). Hàng đợi giữ máy không sleep trong lúc chạy và chạy ẩn ở nền, độc lập với cửa sổ chat.
| Làn | Việc (theo thứ tự) | Ngân sách |
|---|---|---|
| WSL (2 người) | `sp_pool` (self-play + pool) → `sp_selfplay` (ablation) → `sp_ippo` (ablation) | 6M → 2M → 2M env-step |
| Windows (1 người) | `ppo10M_s0` → `abl_clipreward`, `abl_stack1`, `abl_noclip` (2M mỗi cái) → `ppo10M_s1` → `ppo10M_s2` | 10M, 3×2M, 10M, 10M step |
| eval (sau khi train xong) | 1P: 30 ván cho mọi model, difficulty 1–3, stochastic, hành vi, video. 2P: vs random, vs AI máy, **giải đấu Elo 13 người chơi**, 3 video | — |

Vì máy dùng chung với các chương trình khác nên khó ước lượng chính xác; dự kiến **~1.5–2 ngày** cho toàn bộ hàng đợi. Theo dõi tại `logs/queue/status.json` và `logs/queue/<việc>.out`.

Vì sao self-play 6M chứ không phải 10M: mỗi env-step cho 2 mẫu (cả 2 phe), nên 6M env-step tương đương 12M mẫu, nhiều hơn phần 1 với 10M. Ablation (selfplay, ippo) được so ở cùng mốc 2M với snapshot 2M của `sp_pool` trong giải đấu Elo.

## 0. Cách hiểu yêu cầu "làm cả 2 con"
Manual gốc của Boxing có **Game 2: 2 người chơi**, tay cầm trái điều khiển võ sĩ trắng, tay cầm phải điều khiển võ sĩ đen. Hiểu theo nghĩa đó: **cả 2 võ sĩ đều do agent RL điều khiển và học bằng cách đánh với nhau (self-play)**, thay vì 1 agent đánh với AI có sẵn của game.

Môi trường cho chế độ này đã có: **PettingZoo `atari.boxing_v2`**, đã cài và chạy thử thành công trên máy (xem mục 3).
- Hai agent: `first_0` (trắng) và `second_0` (đen).
- Thưởng **zero-sum**: trắng +r thì đen −r (đã kiểm chứng ở mọi bước).
- Không có AI của máy; cả 2 bên đều do code mình điều khiển.

⚠️ **Cần hỏi lại thầy cho chắc** (xem mục 9). Nếu thầy có ý khác, chẳng hạn chỉ cần "1 agent chơi được cả 2 phe", thì kế hoạch vẫn dùng được gần hết; chỉ đổi phần đánh giá.

## 1. Tổng quan: 2 phần, 7 giai đoạn
| Phần | Nội dung | Trạng thái |
|---|---|---|
| **1. Một người** (agent vs AI máy) | PPO tự cài trên `ALE/Boxing-v5` | ✅ 2M step, thắng 100%. Nâng điểm bằng train 10M trên cloud |
| **2. Hai người** (agent vs agent) | **Self-play** bằng PPO tự cài trên PettingZoo `boxing_v2` | 🆕 cần làm |

| Giai đoạn | Việc | Chạy ở đâu | Thời gian ước tính |
|---|---|---|---|
| G1 | Phần 1: train 10M step × 3 seed | **Cloud** | ~5.5 giờ mỗi seed nếu bằng CPU 12 luồng |
| G2 | Viết env 2 người + tiền xử lý tự viết | Máy (WSL) | 1 buổi |
| G3 | Viết self-play PPO + opponent pool | Máy (WSL) | 1–2 buổi |
| G4 | Smoke test + train ngắn 1–2M step (≤ 3 giờ) | Máy (WSL) | 1–3 giờ |
| G5 | Train self-play đầy đủ 10M step | **Cloud** | ~6–10 giờ |
| G6 | Đánh giá: Elo, ma trận đối đầu, đánh chéo với AI máy | Máy | 1 buổi |
| G7 | Báo cáo + slide + video + demo trực tiếp | Máy | 1–2 buổi |

## 2. Train gì trên cloud và để làm gì

### Cloud-1: Phần 1, PPO 10M step × 3 seed (`cloud_train.ipynb`, đã có sẵn)
Mục đích:
1. **Tăng điểm** từ +24 lên mức các bài báo công bố (~+90, có KO). Hiện mới train 2M step, bằng 1/5 ngân sách chuẩn (10M step = 40M frame).
2. **Báo cáo độ ổn định**: 3 seed cho ra trung bình ± độ lệch giữa các seed. Báo cáo có nhiều seed sẽ được đánh giá cao hơn hẳn chạy 1 lần.
3. **Trả lời câu hỏi** bản tự cài kém bản SB3 (+24 so với +42) là do seed hay do cài đặt. Nếu 3 seed tự cài đều đạt mức cao thì kết luận được cài đặt đúng.

### Cloud-2: Phần 2, self-play 10M step (notebook mới, viết ở G3)
Mục đích: agent 2 bên cần nhiều dữ liệu để vượt qua giai đoạn "cả hai cùng quơ tay ngẫu nhiên". Mỗi step env cho **2 mẫu** (1 cho mỗi bên), nên hiệu quả mẫu tốt hơn phần 1.

### ⚠️ Chọn cloud nào: quan trọng
Game Atari chạy trên **CPU**; GPU chỉ tăng tốc phần mạng nơ-ron. Với mạng nhỏ thế này, **số nhân CPU mới quyết định tốc độ**:
| Lựa chọn | CPU | Đánh giá |
|---|---|---|
| Colab miễn phí | 2 vCPU | ❌ Có thể **chậm hơn máy bạn** (12 luồng). Phiên bị ngắt sau ~12 giờ |
| Kaggle Notebook (miễn phí) | 4 vCPU + GPU P100/T4, 30 giờ GPU/tuần, phiên 12 giờ | ⚠️ Tạm được |
| Máy thuê theo giờ (vast.ai, runpod, …) | **≥ 16 vCPU** + GPU nhỏ | ✅ Tốt nhất, khoảng vài USD cho cả dự án |
| Máy bạn chạy qua đêm | 12 luồng, ~550 step/s | ✅ Cũng được, nếu không dùng máy và không bật game cùng lúc |

Code đã có `--resume` và lưu `last.pt` mỗi 100k step, nên dù cloud ngắt giữa chừng vẫn train tiếp được.

## 3. Môi trường cho phần 2 (đã kiểm tra)
- `multi_agent_ale_py` (lõi ALE 2 người của PettingZoo) **không có bản cho Windows**; build thử thất bại.
- Có bản dựng sẵn cho **Linux**, nên:
  - **Máy bạn:** dùng **WSL Ubuntu** (đã có sẵn trên máy). Đã tạo venv riêng `.venv-linux/` **ngay trong thư mục dự án** (128 MB, chưa có torch). Xóa thư mục là sạch.
  - **Cloud** (Colab/Kaggle là Linux): `pip install "pettingzoo[atari]"` là chạy.
- ROM lấy từ ale-py đã có sẵn trong `.venv` (`auto_rom_install_path`), không cần tải ROM hay chấp nhận license gì thêm.
- Đã chạy thử: 2 agent, ván dài 7139 frame, thưởng zero-sum, có cả đòn ±1 và ±2, khớp với điểm trong RAM.
- PettingZoo **tắt sticky actions và không skip frame** → mình tự viết lớp tiền xử lý (mục 4.1). Đây cũng là thêm một phần tự cài để trình bày.

## 4. Thiết kế phần 2: self-play PPO (tự cài, không dùng thư viện RL)

### 4.1 Env 2 người + tiền xử lý tự viết (`envs_2p.py`)
- Bọc `boxing_v2.parallel_env` thành một env kiểu Gymnasium để dùng được `AsyncVectorEnv` song song:
  - obs: `(2, 4, 84, 84)`, góc nhìn của trắng và của đen.
  - action: `MultiDiscrete([18, 18])`.
  - reward: thưởng của trắng; thưởng của đen = −reward vì zero-sum.
- Tự viết:
  - Skip 4 frame, lấy max 2 frame cuối.
  - **Sticky actions 0.25 riêng cho từng người chơi.**
  - Ảnh xám 84×84, xếp chồng 4 frame.
  - Noop reset.
- **Chuẩn hóa góc nhìn:** đen nhìn thấy cùng một màn hình với trắng nhưng màu áo khác và hướng ngược. Có 2 cách, sẽ chọn sau khi thử:
  - (a) Lật ngang ảnh cho đen, rồi đổi chỗ action trái/phải tương ứng, để một mạng chung chơi được cả 2 phe "như thể mình là trắng".
  - (b) Thêm 1 kênh ảnh báo phe đang chơi.
  - Cả hai đều phải kiểm chứng bằng mắt (lưu ảnh ra xem).

### 4.2 Thuật toán (`selfplay_ppo.py`), từ dễ đến xuất sắc
| Mức | Phương pháp | Ý nghĩa khi trình bày |
|---|---|---|
| Cơ bản | **IPPO**: 2 mạng PPO độc lập, mỗi mạng một phe, học song song | Multi-agent RL đơn giản nhất. Vấn đề: môi trường "không dừng" vì đối thủ cũng đang học |
| Tốt | **Self-play chia sẻ tham số**: 1 mạng chơi cả 2 phe | Gấp đôi dữ liệu cho một mạng, đối thủ luôn ngang tài |
| **Xuất sắc** | **Self-play + opponent pool** (Fictitious Self-Play): cứ 250k step lưu một snapshot; mỗi ván, đối thủ là bản hiện tại (50%) hoặc một snapshot cũ chọn ngẫu nhiên (50%) | Tránh "quên" cách đánh lối chơi cũ và tránh vòng lặp oẳn tù tì (A thắng B, B thắng C, C thắng A). Đây là ý tưởng của AlphaStar/OpenAI Five ở mức đơn giản |

- Phần lõi PPO (GAE, loss clip) **dùng lại `ppo.py`**. Chỉ viết thêm vòng lặp thu dữ liệu 2 phe và opponent pool.

### 4.3 Đánh giá phần 2 (đây là chỗ ăn điểm)
Không có AI máy làm thước đo, nên dùng:
1. **Đấu với random** ở cả 2 phe: tỉ lệ thắng phải ~100%.
2. **Elo rating** của các snapshot (tự tính, đấu vòng tròn): biểu đồ Elo tăng theo thời gian train, bằng chứng agent mạnh dần lên.
3. **Ma trận đối đầu** snapshot × snapshot (heatmap): cho thấy bản sau thắng bản trước, phát hiện vòng lặp nếu có.
4. **Đánh chéo giữa 2 phần:**
   - Agent self-play (phe trắng) đấu với **AI máy** trên `ALE/Boxing-v5`, xem học từ self-play có thắng được AI máy không dù chưa từng gặp nó.
   - Agent phần 1 (học từ AI máy) đấu với agent self-play.
5. **So sánh IPPO, self-play và self-play + pool**: thí nghiệm có đối chứng rõ ràng.
6. Video 2 agent đánh nhau, kèm `analyze_behavior.py` cho cả 2 phe.

## 5. Nâng cấp phần 1 để lấy điểm (ngoài train 10M)
- 3 seed × 10M step; báo cáo trung bình ± std **giữa các seed** (Cloud-1).
- Ablation có đối chứng, mỗi cái 2M step, chạy được trên máy (~1 giờ 15 mỗi cái):
  - (a) Clip reward và không clip.
  - (b) 1 frame và 4 frame xếp chồng.
  - (c) Bỏ phần clip của PPO (ε rất lớn): cho thấy vì sao clip quan trọng, đúng ý tưởng cốt lõi của PPO.
- (Tùy chọn) Tự cài **DQN** để so sánh value-based với policy-based. Chỉ làm nếu dư thời gian; phần 2 quan trọng hơn.

## 6. Sản phẩm nộp
| Sản phẩm | Nội dung |
|---|---|
| Code | `ppo.py` (lõi), `train_ppo.py`, `envs.py`, `envs_2p.py`, `selfplay_ppo.py`, `evaluate*.py`, `play.py` |
| Báo cáo | Bài toán và MDP → PPO (công thức) → thiết kế 1P → 2P self-play → kết quả và đồ thị → phân tích hành vi → hạn chế |
| Slide | 10–15 slide, theo kịch bản THUAT_TOAN.md mục 7, thêm phần 2 |
| Demo trực tiếp | `play.py --live` (1P), `play_2p.py --live` (2 agent đánh nhau) |
| Video | 1P thắng AI máy; 2P snapshot sớm và snapshot cuối đánh nhau |
| Đồ thị | Đường học 1P (3 seed, có vùng std), Elo 2P, heatmap đối đầu, bảng ablation |

## 7. Thứ tự làm (đề xuất)
1. **Ngay bây giờ (song song):** bắt đầu Cloud-1 (phần 1, 10M × 3 seed) để cloud chạy trong lúc làm phần 2 trên máy.
2. G2 → G3 → G4 trên máy (WSL): env 2 người, self-play, smoke test, train ngắn ≤ 3 giờ.
3. Cloud-2: self-play 10M step (có thể chạy cả 3 biến thể IPPO / self-play / pool nếu có đủ máy).
4. Ablation phần 1 trên máy (mỗi cái ≤ 1 giờ 15).
5. G6 đánh giá → G7 báo cáo, slide, demo.

## 8. Rủi ro
| Rủi ro | Cách xử lý |
|---|---|
| Self-play rơi vào vòng lặp, Elo không tăng | Opponent pool; theo dõi heatmap đối đầu; luôn đo thêm với random |
| Hai agent "thỏa thuận ngầm" không đánh nhau (hòa 0-0) | Tổng thưởng zero-sum nên không có lợi khi hòa; theo dõi số điểm mỗi ván, nếu cần thưởng thêm khi ghi điểm |
| Chuẩn hóa góc nhìn sai (lật ảnh nhưng map action sai) | Kiểm tra bằng ảnh và video trước khi train dài; so sánh với phương án (b) |
| WSL/Linux venv lỗi | Train toàn bộ phần 2 trên cloud Linux |
| Colab chậm vì chỉ có 2 CPU | Dùng Kaggle (4 CPU) hoặc thuê máy ≥ 16 vCPU |

## 9. Cần bạn xác nhận
1. **"Cả 2 con" có đúng là chế độ 2 người (2 agent đánh nhau, self-play) không?** Hay thầy chỉ muốn 1 agent chơi được cả 2 phe?
2. Có được dùng **PettingZoo** (chỉ là môi trường, giống Gymnasium) không? Thường là được, vì nó không phải thư viện RL.
3. Hạn nộp là khi nào, để chia thời gian giữa cloud và máy.
4. Bạn dùng cloud nào (Kaggle / máy thuê / Colab Pro)?
