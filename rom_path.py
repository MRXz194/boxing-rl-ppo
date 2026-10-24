"""
Tìm thư mục ROM Atari (đi kèm ale-py) theo đường dẫn CHỈ GỒM KÝ TỰ ASCII.

Vì sao cần: lõi C++ của ALE mở file ROM bằng API "ANSI" của Windows. Đường dẫn có chữ tiếng Việt
nằm ngoài bảng mã Windows-1252 (ví dụ "thử", "bàn") sẽ không mở được:
  - ale-py (1 người): "RuntimeError: No mapping for the Unicode character exists in the target multi-byte code page"
  - multi_agent_ale_py (2 người): "ROM file ... not found"
Cách xử lý: dùng tên ngắn 8.3 của Windows (luôn là ASCII), nếu không có thì chép ROM ra thư mục tạm có tên ASCII.
"""
from __future__ import annotations

import os


def _ascii_path(path: str) -> str | None:
    """Trên Windows, trả về đường dẫn ngắn 8.3 (chỉ gồm ký tự ASCII) nếu ổ đĩa hỗ trợ."""
    if os.name != "nt":
        return path
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024) and buf.value.isascii():
        return buf.value
    return None


def find_rom_dir() -> str:
    """Thư mục chứa các file ROM .bin của ale-py, theo dạng đường dẫn ASCII."""
    import ale_py
    rom_dir = os.path.join(os.path.dirname(ale_py.__file__), "roms")
    if not os.path.exists(os.path.join(rom_dir, "boxing.bin")):
        raise FileNotFoundError(f"Không tìm thấy boxing.bin trong {rom_dir}")
    if rom_dir.isascii():
        return rom_dir
    short = _ascii_path(rom_dir)
    if short:
        return short
    import shutil
    import tempfile
    for base in (tempfile.gettempdir(), os.environ.get("PUBLIC", ""), "C:\\"):
        if base and not base.isascii():
            base = _ascii_path(base)
        if base and base.isascii() and os.path.isdir(base):
            dst = os.path.join(base, "boxing_rl_roms")
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(rom_dir):  # ale-py kiểm tra md5 theo bảng; chép đủ ROM + bảng md5
                src = os.path.join(rom_dir, fn)
                if os.path.isfile(src) and not os.path.exists(os.path.join(dst, fn)):
                    shutil.copy2(src, os.path.join(dst, fn))
            return dst
    raise RuntimeError("Không tạo được đường dẫn ASCII cho ROM; hãy đặt dự án ở thư mục không dấu")


def ensure_ale_roms_dir():
    """Cho ale-py (1 người) đọc ROM qua đường dẫn ASCII bằng biến ALE_ROMS_DIR (chỉ khi cần)."""
    if "ALE_ROMS_DIR" in os.environ:
        return
    import ale_py
    rom_dir = os.path.join(os.path.dirname(ale_py.__file__), "roms")
    if not rom_dir.isascii():
        os.environ["ALE_ROMS_DIR"] = find_rom_dir()  # tiến trình con (AsyncVectorEnv) thừa hưởng biến này
