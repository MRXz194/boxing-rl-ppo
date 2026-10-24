> **Tài liệu lịch sử.** File này ghi lại kế hoạch/audit **trong quá trình làm**. Một số nội dung đã lỗi thời (ví dụ: train trên cloud, bản Stable-Baselines3, WSL). Trạng thái cuối cùng: xem [README.md](../../README.md) và [TAI_LIEU_DU_AN.md](../../TAI_LIEU_DU_AN.md).

# TRẠNG THÁI CUỐI (2026-10-23, 10:30): HOÀN THIỆN
| Hạng mục | Trạng thái |
|---|---|
| 1 người: agent thắng AI máy | ✅ 3 seed × 10M step: **+88 đến +92, thắng 100%, KO 100%** |
| 2 người: 2 bot đánh nhau | ✅ v6 (`models/league6/final.pt`): **tự đánh 26.8 điểm/ván**, thắng random 93%, thắng AI máy 100%, thắng agent 1 người 90%, **Elo hạng 1** |
| Không dùng thư viện RL | ✅ không có gói RL trong `.venv`; `verify_algorithm.py` 10/10 phép thử đúng |
| Chạy hoàn toàn trên Windows | ✅ không cần WSL; đã thử cài sạch trong thư mục tên có dấu |
| Demo | ✅ `demo.bat`: 1 người, 2 bot đánh nhau, bạn đấu AI (cầm trắng hoặc đen), so sánh bản đầu và bản cuối |
| Tài liệu | ✅ README.md, TAI_LIEU_DU_AN.md (mục 7.8 ghi toàn bộ quá trình sửa self-play), THUAT_TOAN.md |
| **Gói bàn giao** | ✅ `dist/boxing_rl_20261023.zip` (163 MB). Đã thử: giải nén → `setup.bat` → mọi thứ chạy |
| Hạn chế còn lại | Gặp bản self-play cũ chuyên né tránh (`sp_pool`) thì hòa 0–0 (nó chạy trốn) |

---

# Audit bàn giao và kế hoạch hoàn thiện (2026-10-20, 22:15; cập nhật 23:00)

> **CẬP NHẬT 23:00: đã bỏ hoàn toàn WSL/Linux.**
> - Lõi Atari 2 người đã được build cho Windows, kiểm chứng giống hệt bản Linux từng bit.
> - Train 2 người đã chuyển sang Windows và train tiếp từ checkpoint, không phải làm lại.
> - Đã có `setup.bat`, `demo.bat` (gồm AI vs AI và **người đấu AI**), `smoke_test.py`.
> - Thử cài đặt sạch trong thư mục tên có dấu (`thử bàn giao`): tìm và sửa thêm 2 lỗi đường dẫn; mọi bước chạy được.
> - **Còn lại:** đợi train xong, cập nhật kết quả, đóng gói zip bàn giao.

## 1. Trả lời thẳng: dự án đã hoàn thiện chưa?
**Chưa.** Phần thuật toán thì xong, nhưng **chưa bàn giao được cho khách** vì 3 lý do:
1. **Chế độ 2 người phụ thuộc WSL (Linux)**. Khách dùng Windows sẽ phải cài WSL và Ubuntu (cần quyền admin, khởi động lại máy) rồi chạy lệnh Linux. Đây là rào cản không chấp nhận được.
2. **Demo chưa có chế độ 2 người trực tiếp**, chỉ có video. Lý do cũng là WSL: Windows 10 không hiện được cửa sổ từ WSL.
3. Train dài vẫn đang chạy. Tài liệu còn chỗ cũ (cloud, thiếu phần 2 người), chưa có script cài đặt 1 lệnh, chưa có gói bàn giao sạch.

## 2. Vì sao lại có WSL, và cách bỏ nó
- `ale-py` (thư viện Atari chuẩn, kể cả bản mới nhất 0.12) **không cho điều khiển người chơi thứ 2**. Đã kiểm tra changelog của ALE: không có bản nào hỗ trợ multiplayer.
- Chế độ 2 người chỉ có trong `multi_agent_ale_py`, lõi Atari của PettingZoo. Gói này **chỉ phát hành bản dựng sẵn cho Linux và macOS**; build trên Windows thì thất bại vì máy không có trình biên dịch C++ 64-bit.
- **Đã kiểm tra: phần Python của gói đã hỗ trợ Windows sẵn.** Nó nạp thư viện C++ qua `ctypes` và trên Windows sẽ tìm file `ale_c.dll`. Gói chỉ thiếu đúng file DLL đó.

**Giải pháp:** build `ale_c.dll` **một lần** trên máy này từ mã nguồn chính thức, đóng thành gói `wheels/multi_agent_ale_py-0.1.12-py3-none-win_amd64.whl` để sẵn trong dự án.
- Khách chỉ cần `pip install`, không cần WSL, không cần trình biên dịch.
- Liên kết tĩnh, nên DLL chạy độc lập, không cần cài thêm runtime nào.
- Vì gói dùng `ctypes` nên chạy được với **mọi bản Python 64-bit**, không phụ thuộc phiên bản.
- **Cần tải:** mã nguồn `multi_agent_ale_py-0.1.12.tar.gz` (0.6 MB, PyPI) và bộ biên dịch portable WinLibs GCC 64-bit (276 MB, GitHub `brechtsanders/winlibs_mingw`). Cả hai giải nén vào thư mục tạm, build xong xóa được, không cài gì vào máy.
- **Kiểm chứng sau khi build:** cùng seed và cùng chuỗi action thì bản Windows và bản Linux phải cho **giống hệt** điểm và màn hình. Như vậy model train trong WSL (đang chạy) dùng được nguyên vẹn trên Windows.

## 3. Danh sách audit đầy đủ
| # | Hạng mục | Hiện trạng | Việc cần làm |
|---|---|---|---|
| 1 | 2 người trên Windows | ✅ xong: wheel build sẵn, giống hệt bản Linux | Build `ale_c.dll`, đóng wheel (mục 2) |
| 2 | Demo 2 người trực tiếp | ✅ xong: `play_2p.py --live`, 60 fps | `play_2p.py --live`: cửa sổ pygame 60 fps, AI vs AI |
| 3 | Demo **người đấu AI** | ✅ xong: `play_2p.py --human` | `play_2p.py --human`: bạn điều khiển võ sĩ trắng bằng bàn phím (mũi tên + Space) đấu AI. Rất ấn tượng khi lên lớp |
| 4 | Cài đặt cho khách | ✅ xong: `setup.bat`, khoảng 3 phút | `setup.bat`: tạo `.venv`, cài torch CPU, requirements và wheel 2 người. 1 lệnh |
| 5 | Chạy demo | ✅ xong: `demo.bat` (8 mục) | `demo.bat`: menu chọn 1P xem AI, 2P AI vs AI, 2P người vs AI, video |
| 6 | Kiểm tra cài đặt | ✅ xong: `smoke_test.py` | `smoke_test.py`: kiểm tra env 1P/2P, nạp model, chạy thử 1 ván, khoảng 1 phút |
| 7 | `requirements.txt` | ✅ xong: ghim phiên bản; PettingZoo không còn cần | Ghim đúng các bản đã test; thêm pettingzoo và wheel 2 người |
| 8 | Train 2 người trên Windows | ✅ xong: đang train trên Windows | Kiểm tra `selfplay_ppo.py` chạy được trên Windows bằng wheel mới |
| 9 | `run_queue.py` | ✅ đã bỏ WSL; là công cụ nội bộ | Là công cụ nội bộ để train; ghi chú rõ, không đưa vào gói bàn giao |
| 10 | Tìm ROM, thư mục có dấu | ✅ xong: `rom_path.py` (sửa cho cả 1P và 2P) | Tìm ROM qua `ale_py` đã cài, không phụ thuộc đường dẫn |
| 11 | Tài liệu | ✅ README mới, TAI_LIEU_DU_AN.md | Viết lại README (cài, demo, train, kết quả); cập nhật THUAT_TOAN và PLAN |
| 12 | `cloud_train.ipynb` | ⚠️ lỗi thời (chỉ 1 người, theo yêu cầu cũ) | Bỏ khỏi gói bàn giao |
| 13 | `tham_khao_sb3/` (code SB3) | ✅ đã xóa (2026-10-20) | Không đưa vào gói bàn giao |
| 14 | Gói bàn giao | ❌ thư mục lẫn venv, log, checkpoint (≈2.6 GB) | `build_release.py`: tạo zip sạch gồm code, model cuối, kết quả, video, tài liệu |
| 15 | Kết quả cuối | ⏳ train đang chạy (máy bận game nên chậm) | Đợi hàng đợi xong; cập nhật bảng kết quả, Elo, video |
| 16 | Tốc độ train | ⚠️ 76 env-step/s (2P), 150 step/s (1P) khi máy chơi game | Không cần sửa code. Train sẽ nhanh gấp 2–3 lần khi máy rảnh |

## 4. Thứ tự làm
1. **(Cần bạn đồng ý tải 2 file)** Build `ale_c.dll` → wheel Windows → kiểm chứng giống hệt bản Linux.
2. Sửa `envs_2p.py` (tìm ROM), thêm `--live` / `--human` vào `play_2p.py`, test trên Windows.
3. `setup.bat`, `demo.bat`, `smoke_test.py`, ghim `requirements.txt`.
4. Thử cài **trong thư mục mới, tạo venv từ đầu như khách** để chắc chắn cài 1 lệnh là chạy được.
5. Viết lại README; cập nhật THUAT_TOAN và PLAN.
6. Khi train xong: đánh giá, vẽ biểu đồ, video, cập nhật kết quả → `build_release.py` → file zip bàn giao.

Hàng đợi train hiện tại vẫn để chạy tiếp trong WSL. Model sinh ra là file `.pt` thuần nên dùng được trên Windows. Lần build DLL sẽ không ảnh hưởng tới nó.
