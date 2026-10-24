# Kịch bản bảo vệ: Boxing RL (PPO tự cài, 1 người + 2 người)

Tài liệu dùng trực tiếp cho buổi bảo vệ:
1. Chuẩn bị trước buổi bảo vệ
2. Trình tự trình bày (khoảng 15 phút, 13 slide)
3. Demo trực tiếp (khoảng 5 phút)
4. Số liệu cần nhớ
5. Câu hỏi thường gặp

Giải thích chi tiết từng phần: [THUAT_TOAN.md](THUAT_TOAN.md) (thuật toán, công thức) và [TAI_LIEU_DU_AN.md](TAI_LIEU_DU_AN.md) (toàn bộ dự án).

---

## 1. Chuẩn bị (làm trước 1 ngày)
- [ ] Đã chạy `setup.bat`, thấy `TẤT CẢ ĐỀU ỔN.`
- [ ] Mở `demo.bat`, thử nhanh mục 1, 4, 5 (ESC để thoát mỗi mục).
- [ ] **Tắt game, trình duyệt nặng và Discord** khi demo: chạy game Atari cần CPU, máy bận thì hình bị giật.
- [ ] Mở sẵn các file sau để chiếu khi máy demo trục trặc:
  - Video: `videos/2p_league6_vs_league6.mp4` (2 bot đánh nhau), `videos/ppo10M_s0_final.mp4` (bot 1 người thắng AI máy).
  - Biểu đồ: `results/learning_curve.png`, `results/league_curve.png`, `results/league_final2_elo.png`, `results/league_final2_heatmap.png`.
- [ ] Đọc lại mục 4 (số liệu) và mục 5 (câu hỏi).

---

## 2. Trình tự trình bày (gợi ý 13 slide)
| # | Slide | Nói gì | Hình / số liệu |
|---|---|---|---|
| 1 | Bài toán | Gymnasium `ALE/Boxing-v5`: làm bot thắng. Yêu cầu: **không dùng thư viện RL**, làm **cả 2 võ sĩ** | Ảnh game |
| 2 | MDP & tiền xử lý | Trạng thái là 4 ảnh xám 84×84; 18 action; thưởng ±1 (đấm thẳng) / ±2 (đấm móc); sticky actions 25% | THUAT_TOAN mục 1 |
| 3 | PPO | Actor-Critic → GAE → loss clip `min(r·A, clip(r)·A)`; **tự cài trong `ppo.py`** | THUAT_TOAN mục 2–5 |
| 4 | Chứng minh tự cài đúng | `verify_algorithm.py`: 10/10 phép thử khớp công thức toán; `.venv` không có gói RL nào | Chạy trực tiếp (demo mục 9) |
| 5 | Train 1 người | 8 game song song, 10M step × 3 seed; đường học; 3 thí nghiệm ablation (bỏ clip của PPO là kém nhất) | `results/learning_curve.png`, `results/ablation_curve.png` |
| 6 | Kết quả 1 người | **+88 đến +92, thắng 100%, KO 100%** (ngang bài báo ~95) | Bảng mục 4 |
| 7 | Bot 1 người học được gì | "Đấm chéo rồi lùi"; **gặp đối thủ đứng yên thì 0–0**: nó chỉ biết đứng chờ AI máy lao vào (overfit) | `analyze_behavior.py` |
| 8 | Sang 2 người | Phần thưởng zero-sum `r_đen = −r_trắng`; **lật gương góc nhìn** để 1 mạng chơi cả 2 phe | `results/audit/view_*.png` |
| 9 | Self-play thuần thất bại | 2 bên **né nhau**: điểm/ván giảm 20 → 3. Vì sao: đánh hay né đều có kỳ vọng 0, mà né an toàn hơn | `results/selfplay_curve.png` |
| 10 | Hành trình sửa (6 bản) | Quên → không biết áp sát → shaping → tinh chỉnh → **luật chống né tránh** → reward hacking ở ngưỡng 60 → hạ xuống 40 | Bảng TAI_LIEU 7.8 |
| 11 | Phương pháp cuối | Tinh chỉnh (learning rate 5e-5) + tập đối thủ (AI máy, random, đứng yên, bản cũ, chính nó) + shaping thế năng (Ng 1999) + luật chống né tránh | THUAT_TOAN mục 8.4 |
| 12 | Kết quả 2 người | 2 bot tự đánh **26.8 điểm/ván**; thắng random 93%, AI máy 100%, bot 1 người 90%; **Elo hạng 1** | `results/league_curve.png`, `results/league_final2_elo.png` |
| 13 | Hạn chế & hướng phát triển | Gặp bản né tránh `sp_pool` thì hòa; chưa thử league đầy đủ như AlphaStar; có thể thử DQN | TAI_LIEU mục 11 |

---

## 3. Demo trực tiếp (`demo.bat`)
| Thứ tự | Mục menu | Mục đích | Thời lượng |
|---|---|---|---|
| 1 | **1**: Xem bot đấu AI có sẵn của game | Bot 1 người thắng áp đảo, thường KO trước khi hết giờ | 1 phút (ESC để dừng) |
| 2 | **4**: Xem 2 BOT ĐÁNH NHAU | Yêu cầu chính của đề: cả 2 võ sĩ là bot và đánh nhau thật | 1 phút |
| 3 | **5**: BẠN đấu bot 2 người | Mời thầy hoặc bạn trong lớp chơi thử: mũi tên di chuyển, SPACE đấm | 1–2 phút |
| 4 | **2**: BẠN đấu bot 1 người (bạn cầm đen) | Cho thấy bot 1 người chỉ biết đứng chờ: người chơi tiến lại đánh là thắng → lý do phải làm phần 2 | 1 phút |
| 5 | **7**: Bản đầu vs bản cuối | Bot sau khi train mạnh hơn rõ ràng | 1 phút |
| 6 | **9**: Kiểm chứng thuật toán | 10 phép thử "ĐÚNG": bằng chứng tự cài | 10 giây |

Nếu máy demo có vấn đề: mở các video trong `videos/`.

---

## 4. Số liệu cần nhớ
**1 người** (30 ván, chuẩn v5)
| | Điểm TB | Thắng | KO |
|---|---|---|---|
| Random | −0.1 | 43% | 0% |
| Luật viết tay | −2.4 | 10% | 0% |
| PPO 2M step | +24.4 | 100% | 0% |
| **PPO 10M step (3 seed)** | **+90.7 / +89.9 / +87.7** | **100%** | **100%** |

Ablation (cùng 2M step, seed 0, so với baseline +24.4):
| Thay đổi | Kết quả | Kết luận |
|---|---|---|
| Bỏ clip của PPO (ε = 1000) | **+11.6, thắng 90%** (thấp nhất) | Phần clip, ý tưởng cốt lõi của PPO, quan trọng nhất |
| Chỉ dùng 1 frame | +19.5 | Xếp chồng 4 frame có ích (thấy được chuyển động) |
| Clip reward về ±1 | +35.0 | **Không làm kém đi**, thậm chí cao hơn baseline. Với 1 seed thì chưa kết luận được là tốt hơn |

**2 người** (bản cuối v6)
| Phép đo | Kết quả |
|---|---|
| 2 bot tự đánh | 26.8 điểm/ván |
| Đấu random | +7.4, thắng 93% |
| Đấu AI máy | +85.6, thắng 100% |
| Đấu bot 1 người | thắng 90% |
| Elo | hạng 1, +456 |

---

## 5. Câu hỏi thường gặp (trả lời ngắn; chi tiết ở TAI_LIEU_DU_AN.md mục 12)
**Thuật toán**
- **PPO là gì, khác policy gradient thường ở đâu?** Có thêm phần clip `clip(π_mới/π_cũ, 0.9, 1.1)`, giới hạn mỗi lần cập nhật để học được nhiều epoch trên cùng dữ liệu mà không "sụp".
- **GAE để làm gì?** Ước lượng advantage, tức action tốt hơn trung bình bao nhiêu. λ = 0.95 cân bằng giữa bias (λ = 0) và variance (λ = 1).
- **Actor và critic khác nhau thế nào?** Actor quyết định nên làm gì (π); critic đánh giá tình hình tốt hay xấu (V). Chung thân CNN, 2 đầu ra.
- **Vì sao 4 frame?** Một ảnh không cho biết hướng di chuyển. Ablation (2M step): chỉ dùng 1 frame thì +19.5, baseline 4 frame là +24.4.
- **Sticky actions để làm gì?** Để bot không học thuộc một chuỗi nút bấm (Machado 2018).
- **Sao biết em tự cài, không dùng thư viện?** Không có gói RL nào trong `.venv`; demo mục 9 chạy `verify_algorithm.py`.

**Phần 2 người**
- **2 con cùng muốn thắng thì thưởng phạt có mâu thuẫn không?** Không. Mỗi con có phần thưởng riêng, `r_đen = −r_trắng`. Một cú đấm dạy bên đấm "làm thêm", dạy bên bị đấm "tránh đi"; đó là 2 tình huống khác nhau.
- **1 mạng chơi cả 2 phe bằng cách nào?** Lật gương ảnh, đổi màu và đổi trái↔phải cho phe đen, để mạng luôn thấy mình là võ sĩ trắng.
- **Vì sao self-play thất bại?** Hai bên hội tụ về thế né nhau, vì đánh hay né đều có kỳ vọng 0, mà né thì an toàn hơn.
- **Luật chống né tránh có phải ăn gian không?** Không. Nó là thay đổi có chủ đích về mục tiêu (như trọng tài boxing), áp cho cả 2 bên và chỉ lúc train; đánh giá chỉ dùng điểm thật.
- **Reward shaping có làm sai bài toán không?** Dạng thế năng thì không: định lý Ng 1999 bảo đảm không đổi chiến lược tối ưu.
- **Reward hacking là gì, em gặp chưa?** Có. Với ngưỡng 60, bot lơ lửng ngay dưới ngưỡng mà vẫn ngoài tầm đấm. Hạ xuống 40 (đúng tầm đấm) thì hết.
- **Elo tính thế nào?** Mô hình Bradley–Terry, cực đại likelihood trên kết quả đấu vòng tròn, neo random = 0.
- **Sao không train với AI máy cho cả 2 phe?** AI có sẵn của game chỉ cầm phe đen. Hơn nữa, bot chỉ học với AI máy sẽ overfit: +90 trước AI máy nhưng gặp đối thủ đứng yên thì 0–0.

**Kỹ thuật**
- **Chạy trên Windows bằng cách nào khi thư viện 2 người chỉ có bản Linux?** Em tự build `ale_c.dll` từ mã nguồn chính thức, kiểm chứng giống hệt bản Linux từng bit (`tools/build_windows_wheel/BUILD.md`).
- **Train mất bao lâu?** Trên CPU 12 luồng: 1 người 10M step khoảng 7–9 giờ; 2 người (2 bước × 3M) khoảng 3.5 giờ.
