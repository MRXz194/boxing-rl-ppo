# Giải thích thuật toán: PPO chơi Atari Boxing

Giải thích chi tiết các thuật toán, kèm công thức và chỗ tương ứng trong code.
- Mục 1–6: PPO cho chế độ 1 người (`ppo.py`, `train_ppo.py`, `envs.py`).
- Mục 8: chế độ 2 người (`envs_2p.py`, `league_ppo.py`, `evaluate_2p.py`).

Toàn cảnh dự án: [TAI_LIEU_DU_AN.md](TAI_LIEU_DU_AN.md). Kịch bản bảo vệ: [KICH_BAN_BAO_VE.md](KICH_BAN_BAO_VE.md).

---

## 1. Bài toán dưới dạng MDP

Học tăng cường mô hình hóa trò chơi thành một **Markov Decision Process (S, A, P, R, γ)**:

| Thành phần | Trong Boxing |
|---|---|
| Trạng thái `s_t` | 4 khung hình xám 84×84 liên tiếp (`envs.py`) |
| Hành động `a_t` | 1 trong 18 action: đứng yên, đấm, 8 hướng, 8 hướng + đấm |
| Chuyển trạng thái `P` | Game Atari + đối thủ máy + sticky actions (25% khả năng lặp lại action cũ), nên mang tính ngẫu nhiên |
| Phần thưởng `r_t` | +1/+2 khi mình đấm trúng, −1/−2 khi bị đấm trúng |
| γ = 0.99 | Hệ số chiết khấu: reward ở tương lai xa có giá trị thấp hơn |

**Mục tiêu:** tìm policy π(a|s) để cực đại hóa tổng reward kỳ vọng `E[Σ γ^t r_t]`, tức đánh hơn đối thủ càng nhiều điểm càng tốt.

### Vì sao lại là 4 frame?
Nhìn 1 tấm ảnh thì không biết đối thủ đang tiến hay đang lùi. Xếp chồng 4 frame thì mạng thấy được chuyển động, khi đó trạng thái gần thỏa tính chất Markov.

### Tiền xử lý (`envs.py`)
```
ảnh gốc 210×160 RGB ─► lặp action 4 frame, lấy max 2 frame cuối ─► ảnh xám ─► 84×84 ─► xếp chồng 4 frame
```
- Lặp mỗi action 4 frame: agent ra quyết định 15 lần/giây thay vì 60, giúp train nhanh gấp 4 lần.
- Lấy max 2 frame cuối: Atari vẽ một số vật thể xen kẽ giữa các frame (nhấp nháy). Lấy max thì không bị mất vật thể.
- Noop reset: đầu ván làm ngẫu nhiên 0–30 lần NOOP để ván nào cũng khác nhau, agent không thể học thuộc lòng.

---

## 2. Họ thuật toán: Policy Gradient + Actor-Critic

**Ý tưởng policy gradient:** tăng xác suất của những action dẫn tới kết quả tốt và giảm xác suất của những action dẫn tới kết quả xấu.

```
∇J(θ) = E[ ∇ log π_θ(a_t|s_t) · A_t ]
```

`A_t` (**advantage**) đo xem action `a_t` tốt hơn hay kém hơn mức trung bình tại trạng thái `s_t`:
- `A_t > 0`: action tốt, cần tăng `π(a_t|s_t)`.
- `A_t < 0`: action kém, cần giảm `π(a_t|s_t)`.

Muốn tính được `A_t` thì phải biết "mức trung bình" là bao nhiêu. Đó là việc của **critic** `V(s)`, ước lượng tổng reward tương lai kể từ `s`. Vậy mạng có 2 đầu:

```
                ┌─► Actor  : 18 logits ─► softmax ─► π(a|s)     "nên làm gì"
ảnh 4×84×84 ─► CNN ─► 512
                └─► Critic : 1 số      ─► V(s)        "tình hình đang tốt hay xấu"
```
Code: `class ActorCritic` trong `ppo.py`.
- Thân CNN: 3 lớp conv, theo Nature DQN 2015.
- Đầu actor khởi tạo với gain 0.01, nên lúc đầu π gần như đều trên 18 action. Agent sẽ khám phá ngẫu nhiên trước.

---

## 3. GAE: tính advantage (`compute_gae` trong `ppo.py`)

**TD error**, tức mức "bất ngờ" sau một bước:
```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```
Ví dụ: critic dự đoán `V(s_t) = 5`, ta đấm trúng được `r_t = 2`, và critic đánh giá trạng thái sau là `V(s_{t+1}) = 4`. Khi đó `δ = 2 + 0.99·4 − 5 = 0.96 > 0`, tức tốt hơn dự đoán.

**GAE** cộng dồn các δ tương lai, cái càng xa thì trọng số càng giảm:
```
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + …        (tính ngược từ cuối rollout)
R_t = A_t + V(s_t)                                   (target để train critic)
```
- λ = 0: chỉ dùng δ_t, ít nhiễu nhưng bias nhiều (phụ thuộc vào độ đúng của critic).
- λ = 1: dùng tổng reward thật, không bias nhưng rất nhiễu.
- λ = 0.95 cân bằng giữa hai đầu.
- Tại điểm kết thúc ván, nhân thêm `(1 − done)` để không cộng giá trị của ván sau vào ván trước.

---

## 4. PPO: cập nhật an toàn (`ppo_update` trong `ppo.py`)

**Vấn đề:** policy gradient thường dùng mỗi mẫu dữ liệu 1 lần rồi bỏ, rất phí. Nếu học nhiều lần trên cùng dữ liệu, policy có thể thay đổi quá mạnh và "sụp" (đang đánh tốt bỗng đánh tệ hẳn).

**Giải pháp của PPO (Schulman 2017):** đo xem policy mới đã lệch khỏi policy cũ bao nhiêu:
```
ratio_t = π_mới(a_t|s_t) / π_cũ(a_t|s_t)
```
rồi **cắt** ratio vào khoảng `[1 − ε, 1 + ε]` với ε = 0.1:
```
L_policy = −E[ min( ratio_t · A_t ,  clip(ratio_t, 0.9, 1.1) · A_t ) ]
```
| Trường hợp | Điều xảy ra |
|---|---|
| A > 0, ratio < 1.1 | Tăng xác suất action đó bình thường |
| A > 0, ratio ≥ 1.1 | Gradient = 0: đã tăng đủ, không đẩy thêm |
| A < 0, ratio > 0.9 | Giảm xác suất action đó bình thường |
| A < 0, ratio ≤ 0.9 | Gradient = 0: đã giảm đủ |

Nhờ có clip, PPO học **4 epoch** trên cùng một batch mà vẫn ổn định.

**Hàm loss đầy đủ:**
```
L = L_policy  +  0.5 · (V(s) − R)²  −  0.01 · Entropy(π)
     ↑ actor       ↑ critic học dự đoán     ↑ thưởng cho sự "phân vân", giữ cho agent
                     return                   tiếp tục khám phá, không chốt sớm 1 action
```

Các kỹ thuật ổn định khác có trong code:
- Chuẩn hóa advantage trong mỗi minibatch: `(A − mean)/std`.
- Cắt gradient: `clip_grad_norm_(…, 0.5)`.
- Learning rate và ε giảm tuyến tính về 0 trong suốt quá trình train.
- Khởi tạo trọng số trực giao (orthogonal).

---

## 5. Vòng lặp train (`train_ppo.py`)

```
khởi tạo mạng θ ngẫu nhiên
lặp 9765 lần (10M step ÷ 1024; bản thử đầu tiên 2M step = 1953 lần):
    ① THU DỮ LIỆU   8 env song song, mỗi env chơi 128 bước theo π_θ
                    lưu (s, a, log π(a|s), r, done, V(s))  ─► 1024 mẫu
    ② GAE           tính A_t và R_t cho cả 1024 mẫu
    ③ UPDATE        lặp 4 epoch:
                        xáo trộn, chia 4 minibatch × 256
                        với mỗi minibatch: tính L, backprop, Adam step
    ④ vứt dữ liệu cũ (on-policy) và giảm lr, ε
    cứ 200k step: eval 5 ván bằng argmax π, lưu best.pt
```

### Các chỉ số in ra lúc train (giải thích khi demo)
| Chỉ số | Ý nghĩa | Bình thường |
|---|---|---|
| reward TB 20 ván | Hiệu điểm trung bình 20 ván gần nhất (action lấy mẫu) | Từ ~0 tăng dần |
| entropy | Độ "phân vân" của policy, tối đa ln 18 = 2.89 | Giảm dần khi agent tự tin hơn |
| kl | Policy mới lệch khỏi policy cũ bao nhiêu | Nhỏ, cỡ 0.001–0.02 |
| clip | Tỉ lệ mẫu bị cắt ratio | 0.05–0.2 |
| EV | Explained variance: critic dự đoán return tốt đến đâu (1 là hoàn hảo) | Tăng dần |

---

## 6. Vì sao agent học được cách thắng?

Đường học thật (`results/learning_curve.png`, 3 seed × 10M step):
1. **Lúc đầu** (0–800k step): policy gần như đều, agent quơ tay ngẫu nhiên. Thỉnh thoảng đấm trúng (+1) hoặc bị đấm (−1).
2. **Giai đoạn học** (khoảng 0.8–2M step): GAE gán advantage dương cho chuỗi action dẫn tới những lần ghi điểm và advantage âm cho những chuỗi dẫn tới bị đấm. PPO tăng hoặc giảm xác suất các action đó tương ứng. Eval từ âm lên khoảng +25 đến +50, bắt đầu thắng 100% số ván.
3. **Giai đoạn tinh chỉnh** (2M–10M step): điểm tiếp tục tăng tới **+88 đến +92**, mọi ván đều KO đối thủ trước khi hết giờ. Entropy giảm dần, tức agent ngày càng chắc chắn.

### Agent thực sự học được gì? (đo bằng `analyze_behavior.py`, 5 ván, đọc vị trí từ RAM)
| | Random | PPO |
|---|---|---|
| Số step có ghi điểm | 143 | **262** (gần gấp đôi) |
| Đối thủ ở sát dây đài | 22% thời gian | 9% thời gian |
| Khoảng cách lúc đấm trúng (ngang, dọc) | 22, 16 px | 21, 15 px |
| Action hay dùng nhất | đều (~6% mỗi action) | DOWNRIGHTFIRE 33%, LEFT 26%, UPRIGHTFIRE 15%, UPLEFT 13% |

Nhận xét:
- **Agent không dồn đối thủ vào dây** như manual gợi ý; tỉ lệ đối thủ sát dây còn thấp hơn random.
- Agent hội tụ về kiểu **"đấm chéo rồi lùi"**: đấm theo đường chéo (DOWNRIGHT/UPRIGHT + FIRE) xen với lùi sang trái (LEFT/UPLEFT). Kiểu này giống mẹo "nhảy ra vào tầm đấm" trong manual. Số liệu trên chỉ cho thấy *phân bố action*; chưa chứng minh được lối chơi này khiến máy đấm hụt.
- Cú đấm trúng khi hai võ sĩ **lệch nhau ~15 px theo chiều dọc**, không phải khi đứng thẳng hàng. Đây chính là lý do agent luật viết tay (`heuristic_agent.py`, cố đứng thẳng hàng rồi đấm) chỉ thắng 10% số ván: luật đơn giản do người nghĩ ra sai ở chi tiết này. RL tự tìm ra được từ dữ liệu.
- Chỉ có 5 ván và đây là phân tích tương quan, không phải chứng minh nhân quả.

---

## 7. Kịch bản demo
Kịch bản bảo vệ và demo chi tiết (thứ tự trình bày, mục nào trong `demo.bat`, câu hỏi thường gặp) nằm ở **[KICH_BAN_BAO_VE.md](KICH_BAN_BAO_VE.md)**.

---

## 8. Phần 2: Boxing 2 người, cả 2 võ sĩ đều là bot

### 8.1 Bài toán thay đổi thế nào?
- Không còn AI máy. Trắng và đen đều do mạng nơ-ron điều khiển. Môi trường: lõi Atari 2 người `multi_agent_ale_py` (lõi của PettingZoo `boxing_v2`), tiền xử lý tự viết trong `envs_2p.py` (giống hệt phần 1).
- **Phần thưởng là trò chơi tổng bằng 0 (zero-sum):** `r_đen(t) = − r_trắng(t)` tại mọi bước (đã kiểm chứng). Mỗi bot tối đa hóa phần thưởng **của chính nó**, nên không có mâu thuẫn trong thuật toán; mâu thuẫn chỉ nằm ở mục tiêu, đúng bản chất trận đấu.
- Khó khăn: **môi trường không dừng (non-stationary)**. Đối thủ cũng đang học, nên "môi trường" mà mỗi bên thấy thay đổi liên tục.

### 8.2 Một mạng chơi cả 2 phe: lật gương góc nhìn (`envs_2p.py`)
```
góc nhìn của đen:  ảnh gốc ─► che dải đen 8 cột ở mép trái ─► lật ngang ─► đổi màu đen (0) ↔ trắng (214)
action của đen:    đổi trái ↔ phải (RIGHT↔LEFT, UPRIGHT↔UPLEFT, …)
```
Sau phép biến đổi, mạng **luôn thấy mình là võ sĩ trắng đứng bên trái**. Nhờ vậy một mạng chơi được cả 2 phe, và dùng được luôn bot 1 người (vốn chỉ biết cầm trắng). Ảnh kiểm chứng: `results/audit/view_white.png`, `view_black_mirrored.png`.

### 8.3 Thử nghiệm 1: self-play thuần (`selfplay_ppo.py`) THẤT BẠI
| Biến thể | Ý tưởng |
|---|---|
| IPPO | 2 mạng độc lập, mỗi mạng một phe |
| Self-play | 1 mạng chung, tự đánh với chính mình |
| Self-play + opponent pool | 50% đánh với chính mình, 50% đánh với snapshot cũ (Fictitious Self-Play) |

**Kết quả:** cả 3 đều không thắng nổi random. Điểm/ván khi tự đánh **giảm từ 20 xuống 3**: hai bên học cách **né nhau**.

**Vì sao:** trong self-play đối xứng, mọi chiến lược đối xứng đều cho kỳ vọng bằng 0: đánh nhau thì trung bình hòa, né nhau cũng hòa. Nếu người chủ động áp sát lại dễ bị phản đòn, thì **né là lựa chọn an toàn hơn**, và hai bên hội tụ về thế cân bằng thụ động.

### 8.4 Phương pháp cuối: "league" đơn giản (`league_ppo.py`)
Ghép 5 thành phần (ý tưởng như league của AlphaStar, ở dạng đơn giản):

**(1) Khởi đầu từ bot 1 người, tinh chỉnh nhẹ** (`--init`, learning rate 5e-5, hệ số entropy 0.001).
Bot 1 người đã biết đấm (+90 trước AI máy). Tinh chỉnh với learning rate nhỏ để không bị "quên"; các bản đầu dùng learning rate 2.5e-4 đã làm điểm đấu AI máy tụt từ 85 xuống 55.

**(2) Tập đối thủ đa dạng, chọn ngẫu nhiên mỗi ván:**
| Đối thủ | Dạy bot điều gì |
|---|---|
| AI có sẵn của game (`--n-cpu`) | Đối phó đối thủ hay **tấn công**; giữ kỹ năng cũ |
| Random | Thắng đối thủ di chuyển lộn xộn |
| Đứng yên | **Chủ động áp sát** đối thủ thụ động |
| Bản cũ của chính nó và bot 1 người (cố định) | Thắng các lối chơi đã biết |
| Chính nó, đang học (self-play) | Đánh nhau với đối thủ ngang tài |

Phe do đối thủ điều khiển **không được dùng để cập nhật mạng**, vì PPO chỉ học từ action của chính policy hiện tại (on-policy). Phần này loại bằng mặt nạ `ppo_update(..., valid=mask)`.

**(3) Reward shaping dạng thế năng** (Ng, Harada & Russell 1999). Dùng ở bước v5 để dạy bot áp sát:
```
Φ(s) = − (|dx| + |dy|) / 100          (dx, dy = khoảng cách 2 võ sĩ, đọc từ RAM)
F    = γ·Φ(s') − Φ(s)                 tiến lại gần: F > 0, lùi xa: F < 0
r'   = r + c·F                        c giảm tuyến tính về 0 trong 60% đầu quá trình train
```
**Định lý (Ng 1999):** thưởng phụ dạng `γΦ(s') − Φ(s)` **không làm đổi chiến lược tối ưu**, vì tổng có chiết khấu của nó cả ván bằng `−Φ(s₀)`, một hằng số. Nó chỉ giúp bot khám phá nhanh hơn.

**(4) Luật chống né tránh** (`--passive-penalty`). Đây là thành phần quyết định cho "2 bot đánh nhau":
```
mỗi bước:  nếu |dx| + |dy| > 40  thì  r_trắng −= 0.02  và  r_đen −= 0.02
```
- Giống trọng tài boxing thật phạt võ sĩ né tránh. **Chỉ áp dụng lúc train**, cho **cả 2 bên như nhau**; khi đánh giá chỉ tính điểm đấm thật.
- Khác reward shaping: luật này **cố ý đổi mục tiêu**, khiến né nhau trở nên tốn kém (cả ván né mất khoảng 36 điểm), nên phá được thế cân bằng thụ động. Shaping theo định lý ở (3) thì không làm được việc đó.
- **Reward hacking đã gặp:** với ngưỡng 60, bot lơ lửng ngay dưới 60, vừa đủ không bị phạt mà vẫn ngoài tầm đấm (đấm trúng khi |dx| + |dy| ≈ 36–40). Hạ ngưỡng xuống 40 thì bot phải vào tầm đấm thật, và điểm/ván khi tự đánh tăng từ 3 lên 27.

**(5) PPO giữ nguyên:** dùng lại 100% `ppo.py` (GAE, loss clip). Phần 2 người chỉ viết thêm phần thu dữ liệu cho 2 phe, mặt nạ và quản lý đối thủ.

### 8.5 Quy trình train bản cuối (tái tạo được)
| Bước | Cách làm | Thời gian (CPU 12 luồng) | Kết quả |
|---|---|---|---|
| 0 | Bot 1 người: `train_ppo.py --timesteps 10000000` | ~7–9 giờ | +90 đấu AI máy |
| 1 (v5) | `league_ppo.py --init models/ppo10M_s0/final.pt`, đối thủ cố định (AI máy, bot 1 người, random 35%, đứng yên 20%), shaping 3 | 3M env-step, ~1 giờ 45 phút | Biết áp sát đối thủ thụ động; 2 bot tự đánh vẫn né (0.5 điểm/ván) |
| 2 (v6) | `league_ppo.py --init models/league/final.pt`, **50% self-play**, random 15%, đứng yên 10%, bản cũ 25%, **luật chống né tránh 0.02, ngưỡng 40** | 3M env-step, ~1 giờ 45 phút | **2 bot đánh nhau 26.8 điểm/ván, Elo hạng 1** |

Lệnh đầy đủ: README.md mục 4.

Tham số chung: 6 game 2 người và 2 game đấu AI máy song song, n_steps 128, 4 epoch, minibatch tự chia 4, γ 0.99, λ 0.95, clip 0.1 (giảm dần), learning rate 5e-5 (giảm dần), hệ số entropy 0.001.

### 8.6 Đánh giá (`evaluate_2p.py`)
1. **Đấu random:** một nửa số ván cầm trắng, một nửa cầm đen. Thắng ở cả 2 phe chứng tỏ phép lật gương đúng.
2. **Đấu AI của game**, dùng cùng tiền xử lý.
3. **Đối đầu trực tiếp** với các bot khác.
4. **Elo (Bradley–Terry):** đấu vòng tròn, `P(i thắng j) = 1 / (1 + 10^((R_j − R_i)/400))`. Tìm R bằng cực đại likelihood, neo random = 0. Hơn 200 Elo nghĩa là thắng khoảng 76% số ván.
5. **Tự đánh với chính mình:** đo tổng điểm/ván để biết 2 bot có thật sự đánh nhau không.

### 8.7 Kết quả bản cuối (`models/league6/final.pt`)
| Phép đo | Kết quả |
|---|---|
| 2 bot tự đánh nhau | **26.8 điểm/ván** (bản trước: 0.5) |
| Đấu random, 30 ván | **+7.4, thắng 93%** |
| Đấu AI của game, 30 ván | **+85.6, thắng 100%** |
| Đối đầu bot 1 người / bản v5 | **thắng 90% / 100%** |
| Elo | **hạng 1 (+456)**: tăng đều 1M (+107) → 2M (+336) → cuối (+456) |

Toàn bộ quá trình thử, thất bại và chẩn đoán qua 6 bản: [TAI_LIEU_DU_AN.md](TAI_LIEU_DU_AN.md) mục 7.8.

## Tài liệu tham khảo
- Heinrich & Silver 2016. *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (Fictitious Self-Play).
- Vinyals et al. 2019. *Grandmaster level in StarCraft II using multi-agent RL* (AlphaStar, league training).
- de Witt et al. 2020. *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?* (IPPO).
- Terry et al. 2021. *PettingZoo: Gym for Multi-Agent Reinforcement Learning*. NeurIPS.
- Schulman et al. 2017. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- Schulman et al. 2016. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438
- Mnih et al. 2015. *Human-level control through deep reinforcement learning*. Nature
- Machado et al. 2018. *Revisiting the Arcade Learning Environment*. JAIR
- Huang et al. 2022. *The 37 Implementation Details of Proximal Policy Optimization*. ICLR Blog Track
- Ng, Harada & Russell 1999. *Policy invariance under reward transformations: Theory and application to reward shaping*. ICML.
