# Tài liệu dự án: Học tăng cường cho Atari Boxing (1 người và 2 người)

> Tài liệu này giải thích **toàn bộ dự án, từ đầu đến hiện tại**: bài toán, kiến thức nền, thuật toán, cách làm, các quyết định và lý do, các lỗi đã gặp và cách sửa, kết quả.
> **Phần 2 người (mục 7) được viết kỹ nhất**, vì đây là phần dễ bị hỏi nhất: *"2 con cùng muốn thắng thì thưởng/phạt thế nào, có mâu thuẫn không?"*
>
> Tài liệu liên quan:
> - [THUAT_TOAN.md](THUAT_TOAN.md): công thức PPO chi tiết, kịch bản demo.
> - [KICH_BAN_BAO_VE.md](KICH_BAN_BAO_VE.md): kịch bản bảo vệ, demo, số liệu cần nhớ.
> - [docs/ke_hoach/](docs/ke_hoach/): các bản kế hoạch và audit trong quá trình làm (PLAN.md, PLAN_DIEM_CAO.md, PLAN_BAN_GIAO.md).

---

## Mục lục
1. [Tổng quan](#1-tổng-quan)
2. [Hành trình dự án: từ đầu đến giờ](#2-hành-trình-dự-án-từ-đầu-đến-giờ)
3. [Kiến thức nền RL (đủ để hiểu phần còn lại)](#3-kiến-thức-nền-rl)
4. [Môi trường Boxing](#4-môi-trường-boxing)
5. [Thuật toán PPO (tự cài)](#5-thuật-toán-ppo-tự-cài)
6. [Phần 1: agent đấu AI của game](#6-phần-1-agent-đấu-ai-của-game)
7. [**Phần 2: hai agent đấu nhau. Thưởng, phạt và "mâu thuẫn"**](#7-phần-2-hai-agent-đấu-nhau)
8. [Đánh giá khi không có thước đo cố định: Elo](#8-đánh-giá-elo)
9. [Hạ tầng: môi trường cài đặt, hàng đợi train, các sự cố](#9-hạ-tầng-và-sự-cố)
10. [Cấu trúc code](#10-cấu-trúc-code)
11. [Hạn chế và hướng phát triển](#11-hạn-chế-và-hướng-phát-triển)
12. [Câu hỏi thường gặp khi bảo vệ (Q&A)](#12-câu-hỏi-thường-gặp-khi-bảo-vệ)

---

## 1. Tổng quan
| | |
|---|---|
| **Đề bài** | Gymnasium `ALE/Boxing-v5` (docs v0.28.1): làm agent chơi Boxing và thắng |
| **Yêu cầu của thầy** | (1) Không dùng thư viện RL kiểu "gọi 1–2 dòng là chạy" như Stable-Baselines3, vì phải giải thích được thuật toán. (2) Muốn điểm cao thì làm **cả 2 võ sĩ**: chế độ 2 người |
| **Thuật toán** | **PPO** (Proximal Policy Optimization), **tự cài bằng PyTorch** trong `ppo.py` |
| **Phần 1** | 1 agent (võ sĩ trắng) đấu AI có sẵn của game. **Thắng 30/30 ván** (+24.4 điểm TB, bản 2M step) |
| **Phần 2** | Cả trắng lẫn đen đều là agent. Self-play thuần **thất bại** (2 bên né nhau). Bản cuối (v6) là tinh chỉnh agent 1 người, đấu tập đối thủ (AI máy, random, đứng yên, các bản cũ) cộng self-play và **luật chống né tránh**: 2 bot đánh nhau thật (26.8 điểm/ván), thắng random 93%, thắng AI máy 100%, **Elo hạng 1** |
| **Thư viện dùng** | gymnasium + ale-py (môi trường của đề), multi_agent_ale_py (lõi Atari 2 người, cũng là lõi của PettingZoo), PyTorch (tensor, đạo hàm tự động, Adam), numpy, opencv, pygame, matplotlib |

---

## 2. Hành trình dự án: từ đầu đến giờ
Phần này ghi lại *vì sao* dự án có hình dạng như hiện tại. Thầy hỏi "sao em làm thế này" thì trả lời được.

| Giai đoạn | Việc đã làm | Bài học / quyết định |
|---|---|---|
| **1. Nghiên cứu** | Đọc docs đề bài, manual gốc của game (AtariAge), bảng điểm trong các bài báo (random ~0, người chơi 4–12, DQN 72, PPO ~95, Rainbow ~100) | Boxing là game "dễ" với RL. Mục tiêu thắng (> 0) là khả thi |
| **2. Môi trường cô lập** | Lúc đầu lỡ cài `torch`, `ale-py` vào Python của máy. Đã gỡ, chuyển hết vào `.venv/` trong thư mục dự án | Xóa thư mục là sạch máy |
| **3. Bản đầu dùng Stable-Baselines3** | PPO của SB3, 2M step, thắng 30/30 (+41.6) | Phát hiện 2 bẫy: **skip frame 2 lần** (16 frame/step) và **clip reward làm mất cú đấm 2 điểm**. Đã xử lý |
| **4. Yêu cầu "không dùng thư viện RL"** | **Tự cài PPO** (mạng, GAE, loss clip, vòng update). Train 2M step: thắng 30/30 (+24.4). Bản SB3 chuyển sang `tham_khao_sb3/` (đã xóa khỏi dự án) để đối chiếu | Ở mốc 1M, bản tự cài (+24) gần như trùng bản SB3 (+23), nên cài đúng. Chênh lệch ở cuối chưa kết luận được vì mới chạy 1 seed |
| **5. Audit lần 1** | Gỡ hẳn SB3 khỏi venv mà mọi thứ vẫn chạy. Đo lại baseline. **Phát hiện tài liệu viết sai** ("agent học dồn dây"): đo bằng `analyze_behavior.py` thì thấy agent *không* dồn dây mà dùng kiểu **"đấm chéo rồi lùi"** | Mọi nhận xét trong báo cáo phải có số liệu đo được |
| **6. Phần 2 người** | Môi trường Boxing 2 người (lõi của PettingZoo `boxing_v2`). Tự viết tiền xử lý và **phép lật gương góc nhìn** để 1 mạng chơi cả 2 phe. Self-play + opponent pool + IPPO. Đánh giá Elo | Audit tìm và sửa 7 lỗi trước khi train dài (xem PLAN_DIEM_CAO.md) |
| **7. Train dài trên máy** | Hàng đợi tự chạy `tools/hang_doi_train/run_queue.py`: 9 job train và phần đánh giá, tự resume khi crash | Tìm ra lỗi **numpy chiếm ~360 MB bộ nhớ mỗi tiến trình** (OpenBLAS), là nguyên nhân thật của các lần "hết bộ nhớ" |
| **8. Bàn giao** | Chế độ 2 người từng cần WSL (Linux). **Đã tự build lõi Atari 2 người cho Windows**, kiểm chứng giống hệt bản Linux từng bit. Thêm `setup.bat`, `demo.bat` (có chế độ **người đấu AI**), `smoke_test.py`. Thử cài đặt từ đầu trong thư mục tên có dấu và sửa 2 lỗi đường dẫn | Toàn bộ chạy native trên Windows (mục 9.3) |
| **9. Sửa chế độ 2 người** | Self-play thuần thất bại (2 bên né nhau). Thử 6 bản: v1–v3 (khởi đầu từ bot 1 người: bị quên), v4 (học từ đầu: quá chậm), v5 (tinh chỉnh + đối thủ cố định: biết áp sát nhưng 2 bot vẫn né nhau), **v6 (+ self-play + luật chống né tránh): 2 bot đánh nhau, Elo hạng 1** | Chi tiết mục 7.8; kết quả mục 7.9 |
| **10. Hoàn thiện bàn giao** | Demo chia 2 nhóm (1 người / 2 người, có chế độ bạn tự đấu bot ở cả hai); `verify_algorithm.py`; KICH_BAN_BAO_VE.md; dọn thư mục; gói zip đã thử cài sạch | — |

---

## 3. Kiến thức nền RL
**Học tăng cường (Reinforcement Learning):** agent tương tác với môi trường qua vòng lặp:
```
          quan sát s_t                      hành động a_t
môi trường ──────────► agent (mạng nơ-ron) ──────────► môi trường
     ▲                                                     │
     └──────────────── phần thưởng r_t, s_{t+1} ◄──────────┘
```
| Khái niệm | Ý nghĩa | Trong Boxing |
|---|---|---|
| Trạng thái `s` | Những gì agent nhìn thấy | 4 khung hình xám 84×84 |
| Hành động `a` | Lựa chọn của agent | 18 action (8 hướng × có/không đấm, đứng yên, đấm tại chỗ) |
| Phần thưởng `r` | Tín hiệu tốt/xấu sau mỗi bước | +1/+2 khi mình đấm trúng, −1/−2 khi bị đấm |
| Policy `π(a|s)` | "Chiến lược": xác suất chọn mỗi action | Đầu ra actor của mạng |
| Return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` | Tổng phần thưởng tương lai (γ = 0.99) | Điểm mình sẽ ghi được trừ điểm sẽ bị ghi, từ giờ đến hết ván |
| Value `V(s) = E[G_t]` | Tình hình đang tốt hay xấu | Đầu ra critic của mạng |
| Advantage `A_t = G_t − V(s_t)` | Action vừa chọn **tốt hơn hay kém hơn** mức trung bình | A > 0: tăng xác suất action đó; A < 0: giảm |

**Mục tiêu:** tìm π sao cho tổng phần thưởng kỳ vọng lớn nhất. Agent **không được dạy cách đánh**; nó tự thử và tự rút kinh nghiệm từ phần thưởng.

**Quan trọng:** phần thưởng không chỉ có "thắng +, thua −" ở cuối ván. Ở Boxing, **mỗi cú đấm trúng đều có thưởng ngay**, nên tín hiệu học dày và dễ học hơn nhiều so với chỉ thưởng khi kết thúc ván.

---

## 4. Môi trường Boxing
- **Luật** (manual Activision): mỗi hiệp 2 phút. Đấm thẳng từ xa được 1 điểm, đấm móc cự ly gần được 2 điểm. Đạt 100 điểm là KO, ván kết thúc ngay.
- **Phần thưởng mỗi bước** ∈ {−2, −1, 0, +1, +2}. Tổng cả ván = điểm mình − điểm đối thủ. **Thắng = tổng > 0.**
- **Tiền xử lý** (chuẩn Atari, Mnih 2015 / Machado 2018): lặp mỗi action 4 frame, lấy max 2 frame cuối, ảnh xám 84×84, xếp chồng 4 frame. Đầu ván NOOP ngẫu nhiên 0–30 frame. Sticky actions 25%: mỗi frame có 25% khả năng action cũ được giữ lại, để agent không học thuộc lòng được.
- **Các bẫy đã xử lý:** skip frame 2 lần; clip reward làm mất cú 2 điểm; điểm trong RAM lưu dạng BCD; auto-reset của vector env.

---

## 5. Thuật toán PPO (tự cài)
Chi tiết công thức ở [THUAT_TOAN.md](THUAT_TOAN.md) mục 2–5. Tóm tắt:
```
lặp:
  ① cho 8 game chạy song song, mỗi game 128 bước theo policy hiện tại  → 1024 mẫu (s, a, r, log π, V)
  ② tính advantage bằng GAE:   δ_t = r_t + γV(s_{t+1}) − V(s_t);   A_t = δ_t + γλ A_{t+1}
  ③ cập nhật mạng 4 lần trên dữ liệu đó (4 minibatch mỗi lần), loss:
        L = −min(ratio·A, clip(ratio, 0.9, 1.1)·A)  +  0.5·(V − R)²  −  0.01·Entropy
        ratio = π_mới(a|s) / π_cũ(a|s)
  ④ bỏ dữ liệu cũ, quay lại ①
```
Ý tưởng cốt lõi của PPO là **phần clip**: mỗi lần cập nhật không cho policy thay đổi quá 10% so với policy cũ, nên học được nhiều lần trên cùng dữ liệu mà không "sụp".

Code nằm ở `ppo.py`: `ActorCritic` (mạng), `RolloutBuffer`, `compute_gae`, `ppo_update`.

---

## 6. Phần 1: agent đấu AI của game
| Agent (30 ván, chuẩn v5) | Điểm TB | Thắng |
|---|---|---|
| Random | −0.1 | 43% |
| Luật viết tay | −2.4 | 10% |
| **PPO tự cài, 2M step** | **+24.4** | **100%** |
| **PPO tự cài, 10M step × 3 seed** | **+90.7 / +89.9 / +87.7** | **100%, KO 100%** |

**Ablation** (cùng 2M step, seed 0; baseline `ppo_s0` = +24.4):
| Thay đổi | Kết quả | Kết luận |
|---|---|---|
| Bỏ clip của PPO (ε = 1000) | **+11.6, thắng 90%** | Kém nhất: phần clip là ý tưởng cốt lõi của PPO |
| Chỉ dùng 1 frame | +19.5 | Xếp chồng 4 frame có ích |
| Clip reward về ±1 | +35.0 | Không làm kém đi (cao hơn baseline); với 1 seed thì chưa kết luận được là tốt hơn |

Biểu đồ: `results/ablation_curve.png`. Đường học 3 seed × 10M: `results/learning_curve.png`.

**Phát hiện về hành vi** (`analyze_behavior.py`):
- Agent học kiểu "đấm chéo rồi lùi": 61% số action là đấm chéo, 39% là lùi.
- Cú đấm trúng khi hai võ sĩ lệch nhau ~15 px theo chiều dọc. Agent luật cố đứng thẳng hàng nên thua.

**Phát hiện quan trọng dẫn tới phần 2:** agent này thắng AI máy +24, nhưng gặp đối thủ random trong chế độ 2 người thì **chỉ hơn +2.5**. Nó học cách *khai thác riêng* AI của game (ví dụ: AI hay lao vào, nên agent chờ rồi phản đòn), chứ chưa biết đánh tổng quát. Học với một đối thủ cố định thì sẽ overfit vào đối thủ đó.

---

## 7. Phần 2: hai agent đấu nhau

### 7.1 Câu hỏi: "RL là thắng thì thưởng, thua thì phạt. Cả 2 con đều muốn thắng thì thưởng phạt thế nào? Có mâu thuẫn không?"

**Trả lời ngắn:** không có mâu thuẫn trong thuật toán. **Mỗi con có phần thưởng riêng, tính từ góc nhìn của chính nó**, và hai phần thưởng đó luôn trái dấu nhau:
```
r_đen(t) = − r_trắng(t)        (tại mọi bước; đã kiểm chứng trên môi trường)
```
"Mâu thuẫn" giữa hai con là **mâu thuẫn về mục tiêu**, và đó chính là bản chất của một trận đấu. Nó không phải mâu thuẫn trong code: không có một con số phần thưởng chung nào bị kéo về hai hướng. Mỗi con tự tối đa hóa phần thưởng của mình, giống hai kỳ thủ cờ vua mỗi người đều muốn thắng.

Trong lý thuyết trò chơi, dạng này gọi là **trò chơi tổng bằng 0 (zero-sum game)**: tổng phần thưởng hai bên luôn bằng 0, cái một bên được đúng bằng cái bên kia mất.

### 7.2 Ví dụ cụ thể: một cú đấm tạo ra dữ liệu học thế nào
Giả sử ở bước t, **trắng đấm trúng đen một cú đấm thẳng (1 điểm)**:

| | Trắng | Đen |
|---|---|---|
| Phần thưởng bước t | **+1** | **−1** |
| Advantage của action vừa làm (gần đúng) | > 0: "đòn này tốt" | < 0: "đứng chỗ đó, làm thế là tệ" |
| PPO sẽ | **tăng** xác suất trắng làm lại đúng kiểu đó trong tình huống tương tự | **giảm** xác suất đen làm lại kiểu đó (ví dụ đứng yên trong tầm đấm) |

Cùng một sự kiện cho ra **hai bài học khác nhau cho hai bên**: trắng học tấn công, đen học phòng thủ. Hai bài học không triệt tiêu nhau, vì chúng thuộc **hai tình huống khác nhau**: một bên là "mình đang đấm", bên kia là "mình đang bị đấm".

Hệ quả là một cuộc **chạy đua vũ trang** (arms race):
```
trắng học được đòn mới  →  đen bị đấm nhiều, học cách né  →  trắng phải tìm đòn khác  →  …
```
Hai bên kéo nhau mạnh dần lên. Đây là lý do self-play tạo ra agent mạnh: **đối thủ luôn vừa sức**, không quá yếu (học không được gì) cũng không quá mạnh (bị đánh mãi, không có phần thưởng dương để học).

### 7.3 Một mạng chơi cả 2 phe thì sao? Chẳng lẽ nó tự đánh chính mình?
Ở chế độ chính, **trắng và đen dùng chung một mạng**. Câu hỏi tự nhiên: cùng một mạng nhận cả +1 lẫn −1 thì có bị rối không?

**Không bị rối, nhờ chuẩn hóa góc nhìn** (`envs_2p.py`):
```
góc nhìn của đen:  ảnh gốc ─► lật ngang ─► đổi màu đen↔trắng      action: đổi trái↔phải
```
Sau phép biến đổi, **mạng luôn thấy mình là võ sĩ trắng đứng bên trái**, dù thực tế nó đang cầm phe nào. Mỗi mẫu dữ liệu luôn được gắn với phần thưởng **của chính phe đang được điều khiển**. Ví dụ cú đấm ở 7.2 cho mạng 2 mẫu:

| Mẫu | Mạng "thấy" | Phần thưởng | Mạng học được |
|---|---|---|---|
| Từ phe trắng | "Tôi (trắng, bên trái) vừa đấm trúng đối thủ" | +1 | Làm thế này là tốt |
| Từ phe đen (đã lật gương) | "Tôi (trắng, bên trái) vừa bị đối thủ đấm trúng" | −1 | Để bị đấm thế này là tệ |

Hai mẫu này mô tả **hai tình huống khác nhau** (ai đấm, ai bị đấm), nên không mâu thuẫn. Mạng học được cả cách tấn công lẫn cách phòng thủ, từ cùng một trận. Lợi ích: **gấp đôi dữ liệu**, và đối thủ luôn đúng bằng trình độ hiện tại của chính nó.

Ảnh kiểm chứng phép lật gương: `results/audit/view_white.png` và `results/audit/view_black_mirrored.png`.

### 7.4 Nếu hai bên ngang tài thì kết quả hòa. Vậy làm sao biết agent tiến bộ?
Đúng vậy. Khi mạng tự đánh với chính mình, **trung bình mỗi ván sẽ quanh 0**, vì hai bên giống hệt nhau. Điều này là bình thường, **không phải dấu hiệu không học**.

Trong lý thuyết trò chơi, mục tiêu của self-play là tiến tới **cân bằng Nash (minimax)**: một chiến lược mà đối thủ không có cách nào khai thác được. Đã gần cân bằng thì đánh với bản sao của mình tất nhiên sẽ hòa.

Vì vậy tiến bộ được đo bằng **đối thủ cố định** (mục 8):
- Đấu **random**.
- Đấu **các bản cũ của chính nó** (snapshot): bản mới phải thắng bản cũ.
- Đấu **AI của game**: agent chưa từng gặp đối thủ này lúc train.
- **Elo** của các snapshot theo thời gian.

### 7.5 Ba vấn đề thật sự của self-play, và cách xử lý
| Vấn đề | Giải thích | Cách xử lý trong dự án |
|---|---|---|
| **Môi trường không dừng** (non-stationary) | Đối thủ cũng đang học, nên "môi trường" mà mỗi bên thấy thay đổi liên tục; giả định MDP không còn đúng hẳn | PPO cập nhật từng bước nhỏ (clip) và đối thủ thay đổi chậm, nên vẫn ổn định |
| **Vòng lặp kiểu oẳn tù tì** (cycling) | Chiến thuật A thắng B, B thắng C, C lại thắng A; agent cứ xoay vòng mà không mạnh lên, và quên cách đối phó lối chơi cũ | **Opponent pool:** 50% số ván đấu với một snapshot cũ chọn ngẫu nhiên, nên agent phải thắng được cả các lối chơi cũ (*Fictitious Self-Play*) |
| **"Thông đồng"** (hai bên cùng đứng yên, hòa 0–0) | Có thể là một điểm kẹt cục bộ | Zero-sum nên không bền: bên nào phá thế đứng yên và đấm trước thì được lợi ngay. Log ghi tổng điểm mỗi ván để phát hiện |

### 7.6 Ba biến thể được train và so sánh
| Chế độ | Mạng | Đối thủ khi train | Mục đích |
|---|---|---|---|
| `ippo` | 2 mạng riêng: 1 trắng, 1 đen | Mạng kia (cũng đang học) | Cách đơn giản nhất của multi-agent RL (*Independent PPO*) |
| `selfplay` | 1 mạng chung (lật gương) | Chính nó | Gấp đôi dữ liệu, đối thủ luôn ngang tài |
| **`pool`** (chính) | 1 mạng chung | 50% chính nó, 50% snapshot cũ | Chống vòng lặp và chống quên |

### 7.7 Cài đặt kỹ thuật (`selfplay_ppo.py`)
- 8 game × 2 phe = **16 luồng dữ liệu**. Mỗi luồng được ghi lại: quan sát, action, phần thưởng *của phe đó*, done, giá trị V.
- Phần thưởng: `r_trắng` lấy từ game; `r_đen = −r_trắng`.
- Luồng do snapshot cũ điều khiển **không được dùng để cập nhật mạng**, vì PPO chỉ học được từ action do chính policy hiện tại chọn (on-policy). Phần này loại bằng mặt nạ: `ppo_update(..., valid=mask)`.
- GAE tính riêng từng luồng. Khi hết ván thì cắt bằng `(1 − done)`, nên đổi đối thủ giữa các ván không làm sai advantage.
- **Dùng lại 100% lõi PPO của phần 1** (`ppo.py`). Phần 2 chỉ viết thêm: thu dữ liệu cho 2 phe, mặt nạ, quản lý pool.

### 7.8 Thực tế: self-play thuần THẤT BẠI, và quá trình tìm ra cách sửa
Đây là phần "thử, thất bại, chẩn đoán, sửa". Mỗi bước đều có số liệu; log nằm ở `logs/` và model ở `models/league_v1..v4`.

| Bản | Cách làm | Kết quả | Chẩn đoán |
|---|---|---|---|
| `sp_pool`, `sp_selfplay`, `sp_ippo` (6M / 2M / 2M step) | Self-play từ con số 0, có/không pool, IPPO | Đấu random −4 (thắng 10%), đấu AI máy −2. **Điểm/ván khi tự đánh giảm từ 20 xuống 3** | **Hai bên học cách NÉ NHAU.** Bị đấm (−1) đau bằng đấm trúng (+1), nên "đứng xa" là cân bằng an toàn. Pool không cứu được vì các bản cũ cũng thụ động |
| league v1 | Khởi đầu từ agent 1 người + AI máy + random + pool | Đấu random ~+3, không tăng; đấu AI máy tụt 85 → 55 | Ván đấu random quá ít (~14% dữ liệu); entropy tăng dần, agent bị **quên** |
| league v2 | Như v1, 70% ván đấu random | Đấu random giảm xuống −3.8 | Thêm ván random cũng vô ích, vì agent **không bao giờ tự lại gần** đối thủ thụ động, nên không có thưởng mà học |
| *(đo trực tiếp)* | Agent 1 người gặp đối thủ đứng yên | 0–0, đứng cách xa ~80 px suốt ván | Agent 1 người chỉ học cách **đứng chờ** AI máy lao vào |
| league v3 | v2 + **reward shaping thế năng** (hệ số 1) | Không tiến bộ, entropy tăng | Shaping yếu; learning rate và entropy quá cao cho một model đã giỏi |
| *(gỡ lỗi)* | Học TỪ ĐẦU, đối thủ đứng yên, shaping 10 | **+63 sau 100k step** | Pipeline đúng; shaping có tác dụng khi đủ mạnh |
| league v4 | Học từ đầu + shaping 10 + đối thủ đa dạng | Sau 2M: đấu random −6, đấu AI máy +4 | Học từ đầu quá chậm (bản 1 người cũng cần hơn 1M step mới bứt lên) |
| league v5 | **Tinh chỉnh agent 1 người, learning rate 5e-5, entropy 0.001**, shaping 3 giảm dần, tập đối thủ **cố định**: AI máy, agent 1 người (cả 2 phe), random, đứng yên. Không self-play thuần | Đấu AI máy +88 (thắng 100%), đấu đứng yên +50. Nhưng đấu random chỉ +1.3 (30 ván), **tự đánh với chính mình 0.5 điểm/ván** | Biết áp sát đối thủ thụ động, nhưng gặp đối thủ **cũng biết di chuyển** thì 2 bên giữ khoảng cách |
| *(phân tích)* | — | — | Trong self-play đối xứng, đánh hay né đều có kỳ vọng 0. Nếu người chủ động áp sát dễ bị phản đòn thì **né là hợp lý về toán**. Reward shaping dạng thế năng (theo định nghĩa) không đổi được điều này → **phải đổi luật chơi** |
| **league v6 (CUỐI)** | Từ v5 + **50% self-play** + **luật chống né tránh**: mỗi bước 2 võ sĩ đứng xa nhau quá ngưỡng thì CẢ HAI bị trừ 0.02 (như trọng tài phạt võ sĩ thụ động). Ngưỡng ban đầu 60 → agent **lách luật**, lơ lửng ngay dưới 60 nhưng ngoài tầm đấm (reward hacking) → **hạ ngưỡng xuống 40**, tức phải vào tầm đấm | **Tự đánh 26.8 điểm/ván** (đánh nhau thật); đấu random +7.4 (thắng 93%); đấu AI máy +85.6 (thắng 100%); thắng agent 1 người 90%, thắng v5 100%; **Elo hạng 1 (+456)** | Đạt yêu cầu: 2 bot đánh nhau thật và mạnh hơn mọi bản trước |

**Bài học rút ra** (dùng khi trả lời câu hỏi "vì sao self-play khó"):
1. **Self-play có thể hội tụ về cân bằng thụ động:** hai bên ngang nhau mà chủ động tấn công lại rủi ro, nên cả hai cùng né. Lý thuyết zero-sum không bảo đảm cân bằng phải "đẹp".
2. **Khám phá (exploration) là điểm nghẽn:** muốn học "áp sát rồi đấm" thì phải từng áp sát và được thưởng. Reward shaping dạng thế năng giải quyết việc này mà không đổi chiến lược tối ưu (Ng 1999).
3. **Tinh chỉnh khác học từ đầu:** model đã giỏi cần learning rate và hệ số entropy nhỏ, nếu không sẽ bị quên.
4. **Đối thủ cố định cho tín hiệu học ổn định hơn đối thủ đang học;** đây cũng là lý do AlphaStar dùng các "exploiter" cố định trong league.
5. **Muốn agent làm điều gì thì phần thưởng phải nói đúng điều đó.** "Đánh nhau" không phải là chiến lược tối ưu của luật gốc khi 2 bên ngang tài, nên phải thêm luật chống né tránh. Và agent sẽ khai thác mọi kẽ hở của luật (lơ lửng ngay dưới ngưỡng), nên ngưỡng phải đặt đúng bằng tầm đấm.

### 7.9 Kết quả cuối phần 2 người (model `models/league6/final.pt`)
| Phép đo | Kết quả |
|---|---|
| Đấu random, 30 ván (15 cầm trắng, 15 cầm đen) | **+7.4 ± 4.9, thắng 93%** (cầm trắng +8.7, cầm đen +6.0) |
| Đấu AI của game, 30 ván (chế độ 1 người) | **+85.6 ± 6.9, thắng 100%** |
| Hai bản v6 tự đánh nhau, 10 ván | **26.8 điểm/ván** (cả 2 bên); bản v5 chỉ 0.5 |
| Đối đầu agent 1 người (`ppo10M_s0`), 20 ván | **+12.4, thắng 90%** |
| Đối đầu v5, 20 ván | **+26.6, thắng 100%** |
| Đối đầu `sp_pool` (bản self-play thuần, chuyên né tránh) | +0.1, 90% hòa 0–0: `sp_pool` liên tục chạy trốn. **Hạn chế còn lại** |
| **Elo** (vòng tròn 7 người chơi, 8 ván mỗi cặp) | **v6: +456 (hạng 1)**; v6 lúc 2M: +336; `sp_pool`: +142; agent 1 người: +114; v6 lúc 1M: +107; v5: +73; random: 0 |

Hình: `results/league_curve.png` (đường học), `results/league_final2_elo.png`, `results/league_final2_heatmap.png`.
Video: `videos/2p_league6_vs_league6.mp4` (2 bot đánh nhau), `videos/2p_ppo1P_vs_league6.mp4`, `videos/2p_league6_vs_random.mp4`.

---

## 8. Đánh giá Elo
Ở phần 2 không có một đối thủ cố định làm thước đo, nên dùng hệ số **Elo** như trong cờ vua:
- Tất cả agent (random, các snapshot, các biến thể, agent phần 1) **đấu vòng tròn**, mỗi cặp 8 ván, đổi phe sau mỗi nửa.
- Mô hình Bradley–Terry: `P(i thắng j) = 1 / (1 + 10^((R_j − R_i)/400))`. Tìm R bằng cực đại likelihood trên toàn bộ kết quả; neo random = 0.
- Hơn nhau 200 Elo nghĩa là bên mạnh hơn thắng ~76% số ván.
- Kết quả gồm: bảng xếp hạng, **heatmap tỉ lệ thắng** từng cặp, **biểu đồ Elo theo thời gian train**.

Code: `evaluate_2p.py league`.

---

## 9. Hạ tầng và sự cố

### 9.1 Môi trường cài đặt
- Mọi thư viện nằm trong `.venv/` của thư mục dự án; không cài gì vào Python của máy.
- File tải về để build bản Windows nằm trong `_build_tools/`, **xóa được sau khi build**. Công thức build được giữ lại ở `tools/build_windows_wheel/`.
- Cài cho máy khác: nhấp đúp `setup.bat` (tạo `.venv`, cài thư viện, chạy kiểm tra).

### 9.2 Hàng đợi train (`tools/hang_doi_train/`, công cụ nội bộ của máy train)
- Chạy ẩn, độc lập với cửa sổ chat, giữ máy không sleep.
- Mỗi job: tự `--resume` từ checkpoint, tự chạy lại khi crash, đợi khi máy thiếu bộ nhớ.
- Trạng thái: `logs/queue/status.json`.

### 9.3 Vì sao từng có WSL, và cách bỏ
- Thư viện Atari chuẩn (`ale-py`) **không hỗ trợ người chơi thứ 2**.
- Chế độ 2 người chỉ có ở `multi_agent_ale_py` (lõi của PettingZoo), mà gói này chỉ có bản dựng sẵn cho Linux. Nên ban đầu phần 2 người phải chạy trong WSL (Linux trong Windows).
- Như vậy thì không bàn giao được, nên **đã tự build bản Windows**. Phần Python của gói đã hỗ trợ Windows (nạp thư viện qua `ctypes`), chỉ thiếu file `ale_c.dll`:
  - Build từ mã nguồn chính thức bằng GCC portable, liên kết tĩnh. zlib được thay bằng bản tối giản tự viết, vì ALE chỉ dùng zlib để lưu PNG và đọc ROM.
  - Đóng thành `wheels/multi_agent_ale_py-0.1.12-py3-none-win_amd64.whl`; `setup.bat` tự cài.
  - **Kiểm chứng:** cùng seed và cùng chuỗi action, bản Windows và bản Linux cho cùng điểm, cùng số bước, cùng mã băm SHA-256 của toàn bộ hình ảnh, trên 3 seed. Nhờ vậy model đã train một phần trong WSL được train tiếp trên Windows mà không phải làm lại.
  - Code 2 người dùng thẳng lõi `multi_agent_ale_py`, không còn cần PettingZoo.
  - Chi tiết: `tools/build_windows_wheel/BUILD.md`.

### 9.4 Các sự cố đã gặp (bài học)
| Sự cố | Nguyên nhân thật | Sửa |
|---|---|---|
| Train dừng giữa chừng "hết bộ nhớ" | **numpy (OpenBLAS) giữ ~360 MB bộ đệm mỗi tiến trình**; 8 tiến trình game là 3 GB lãng phí, cộng với game và trình duyệt đang mở | `OPENBLAS_NUM_THREADS=1`: bộ nhớ từ 3.3 GB xuống 0.8 GB |
| Self-play chậm (60 env-step/s) | 6 luồng PyTorch tranh CPU với 8 tiến trình game; máy đang chơi game | 4 luồng là nhanh nhất |
| Mất phần thưởng ở bước cuối ván (2P) | Vector env tự reset, `info` thuộc ván mới | `r_đen = −r_trắng` |
| Tài liệu ghi sai "agent dồn dây" | Viết nhận xét khi chưa đo | Đo bằng `analyze_behavior.py`, sửa theo số liệu |
| Thư mục tên có dấu (`thử bàn giao`): 1 người báo *"No mapping for the Unicode character"*, 2 người báo *"ROM not found"* | Lõi C++ của ALE mở file ROM bằng API "ANSI" của Windows; phần lớn chữ tiếng Việt không có trong bảng mã Windows-1252 | `rom_path.py`: dùng đường dẫn ngắn 8.3 (luôn ASCII), không được thì chép ROM ra thư mục tạm ASCII |
| `chcp 65001` trong file `.bat` | Đổi bảng mã giữa chừng khiến cmd đọc lại file `.bat` sai khi đường dẫn có dấu | Bỏ `chcp`; Python tự in tiếng Việt ra console |
| Chạy tay trong cmd: crash khi in tiếng Việt | Console Windows mặc định cp1252 | Mọi script tự chuyển stdout sang UTF-8 |

---

## 10. Cấu trúc code
| File | Vai trò |
|---|---|
| **`ppo.py`** | **Lõi PPO tự cài**: mạng Actor-Critic, RolloutBuffer, GAE, loss PPO-Clip, update có mặt nạ |
| `train_ppo.py` | Train phần 1 (1 người) |
| `selfplay_ppo.py` | Train 2 người bằng self-play thuần (pool / selfplay / ippo): các bản thất bại, giữ lại để so sánh |
| `envs.py` | Env 1 người + tiền xử lý |
| `envs_2p.py` | Env 2 người + tiền xử lý tự viết + lật gương |
| `evaluate.py` / `evaluate_2p.py` | Đánh giá 1 người / 2 người (vs random, vs AI máy, Elo) |
| `play.py` / `play_2p.py` | Xem trực tiếp, **người đấu AI** (`--human`), quay video |
| `analyze_behavior.py` | Phân tích hành vi từ RAM |
| `rom_path.py` | Đường dẫn ROM dạng ASCII (thư mục có dấu) |
| `setup.bat`, `demo.bat`, `demo.py`, `smoke_test.py` | Cài đặt 1 lệnh, menu demo (có chế độ **người đấu AI**), kiểm tra sau cài |
| `wheels/`, `tools/build_windows_wheel/` | Lõi Atari 2 người cho Windows (build sẵn) và cách build lại |
| `heuristic_agent.py` | Agent luật viết tay (baseline) |
| `plot_results.py` | Vẽ biểu đồ |
| `league_ppo.py` | **Train 2 người, bản cuối**: tinh chỉnh + tập đối thủ + self-play + reward shaping + luật chống né tránh |
| `verify_algorithm.py` | Kiểm chứng PPO tự cài so với công thức toán (10 phép thử) |
| `build_release.py` | Đóng gói zip bàn giao |
| `tools/hang_doi_train/` | Hàng đợi train nội bộ (tự resume, `tools/hang_doi_train/start_train.bat` / `stop_train.bat` / `status_train.bat`); không cần cho người dùng |

---

## 11. Hạn chế và hướng phát triển
- **2 người:** gặp bản self-play cũ chuyên né tránh (`sp_pool`) thì 90% số ván hòa 0–0, vì nó liên tục chạy trốn. Có thể thêm loại đối thủ "chạy trốn" vào tập đối thủ.
- **2 người:** luật chống né tránh là thay đổi có chủ đích về mục tiêu. Chưa thử cách khác như thưởng thêm cho việc ghi điểm trước.
- **1 người:** đổi difficulty thì vẫn thắng 100% nhưng điểm giảm: difficulty 1 là +48.5, difficulty 3 là +18.9 (so với +90.7 ở difficulty 0). Giả thuyết: công tắc độ khó làm chậm võ sĩ của mình; chưa kiểm chứng.
- **1 người:** ablation mới chạy 1 seed mỗi cấu hình, nên chưa kết luận được clip reward tốt hơn hay chỉ do nhiễu.
- Chưa thử league đầy đủ như AlphaStar (main agent và exploiter agent tự sinh), cũng chưa thử population-based training.
- Có thể thử DQN (value-based) để so sánh với PPO (policy-based).

---

## 12. Câu hỏi thường gặp khi bảo vệ
**H: Làm sao chứng minh em tự cài thuật toán, không dùng thư viện?**
Đ: (1) Môi trường cài đặt không có gói RL nào: không SB3, RLlib, Tianshou, CleanRL, TorchRL. (2) Code chỉ import PyTorch cho tensor, lớp mạng, Adam và phân phối Categorical; không có hàm nào làm sẵn GAE, loss PPO hay vòng train. (3) Chạy `python verify_algorithm.py`: so GAE, log π, entropy, loss PPO-Clip, gradient bị chặn khi clip, và hướng cập nhật với công thức toán viết độc lập. 10/10 phép thử đúng.

**H: Sao không dùng Stable-Baselines3 cho nhanh?**
Đ: Thầy yêu cầu tự hiểu và giải thích thuật toán. Bản SB3 chỉ dùng để đối chiếu: ở mốc 1M step, bản tự cài cho kết quả gần như trùng (+24 so với +23), nên cài đặt đúng.

**H: Vì sao PPO mà không phải DQN?**
Đ: PPO ổn định, chạy nhiều game song song nên tận dụng CPU tốt, và dùng lại được nguyên vẹn cho self-play. Trên Boxing, các bài báo cho PPO ~95 điểm, DQN ~72.

**H: Vì sao self-play ban đầu thất bại và em sửa thế nào?**
Đ: Hai bên học cách né nhau: điểm/ván giảm từ 20 xuống 3. Trong self-play đối xứng, đánh hay né đều có kỳ vọng 0, mà né thì an toàn hơn. Em sửa qua 6 bản (mục 7.8). Bản cuối: tinh chỉnh agent 1 người với learning rate nhỏ, đấu cả đối thủ cố định (AI máy, random, đứng yên) lẫn chính mình, và thêm luật chống né tránh như trọng tài boxing thật. Kết quả: 2 bot đánh nhau 26.8 điểm/ván, Elo hạng 1.

**H: Thêm luật chống né tránh có phải "ăn gian" phần thưởng không?**
Đ: Đó là thay đổi có chủ đích về mục tiêu bài toán (giống luật boxing thật phạt võ sĩ thụ động), áp dụng **chỉ khi train** và **cho cả 2 bên như nhau**. Mọi số liệu đánh giá chỉ dùng điểm đấm thật của game. Reward shaping dạng thế năng thì khác: nó không đổi chiến lược tối ưu (Ng 1999) nên không phá được thế né nhau; đó là lý do phải đổi luật.

**H: Hai con cùng muốn thắng thì thưởng/phạt có mâu thuẫn không?**
Đ: Không. Mỗi con có phần thưởng riêng theo góc nhìn của mình, `r_đen = −r_trắng` (zero-sum). Một cú đấm dạy bên đấm "làm thêm" và dạy bên bị đấm "tránh đi"; đó là hai tình huống khác nhau. Xem mục 7.1–7.3.

**H: Self-play tự đánh với mình thì hòa hoài, sao biết nó giỏi lên?**
Đ: Đo bằng đối thủ cố định: random, các bản cũ của chính nó, AI của game, và Elo theo thời gian. Mục 7.4 và 8.

**H: Một mạng chơi cả trắng lẫn đen bằng cách nào?**
Đ: Lật ngang ảnh, đổi màu đen↔trắng và đổi action trái↔phải cho phe đen, để mạng luôn "thấy" mình là trắng. Đã kiểm chứng bằng ảnh và bằng việc agent phần 1 cầm phe đen vẫn thắng random.

**H: Agent học được gì?**
Đ: Đo từ RAM: kiểu "đấm chéo rồi lùi", đấm khi lệch đối thủ ~15 px theo chiều dọc, không dồn dây. Mục 6.

**H: Sticky actions để làm gì?**
Đ: Game Atari tất định; không có ngẫu nhiên thì agent có thể học thuộc một chuỗi nút bấm. Với 25% khả năng giữ action cũ, agent buộc phải phản ứng theo hình ảnh (Machado et al. 2018).
