# Build wheel `multi_agent_ale_py` cho Windows

> **Người dùng KHÔNG cần làm phần này.** Wheel đã build sẵn ở `wheels/multi_agent_ale_py-0.1.12-py3-none-win_amd64.whl` và `setup.bat` tự cài.
> Tài liệu này chỉ để build lại khi cần, ví dụ khi có phiên bản mới.

## Vì sao phải tự build
- `ale-py` (thư viện Atari chuẩn) **không hỗ trợ người chơi thứ 2**.
- `multi_agent_ale_py` (lõi Atari 2 người của PettingZoo) chỉ phát hành bản dựng sẵn cho Linux và macOS.
- Phần Python của gói đã hỗ trợ Windows: `ale_python_interface.py` dùng `ctypes` và trên Windows sẽ nạp `ale_c.dll`. **Chỉ thiếu đúng file DLL đó.** Vì dùng ctypes nên wheel chạy được với mọi bản Python 64-bit (tag `py3-none-win_amd64`).

## Kiểm chứng đã làm
Bản Windows và bản Linux chính thức cho kết quả **giống hệt từng bit**: cùng seed và cùng chuỗi action thì cùng số bước, cùng điểm, cùng mã băm SHA-256 của toàn bộ hình ảnh, trên 3 seed 1, 7, 123.

## Các bước (khoảng 10 phút, mọi thứ nằm trong `_build_tools/`, xóa được sau khi build)
1. Tải về `_build_tools/`:
   - Mã nguồn: `multi_agent_ale_py-0.1.12.tar.gz` (0.6 MB) từ PyPI.
   - Bộ biên dịch portable: WinLibs GCC 64-bit, UCRT (`winlibs-x86_64-posix-seh-gcc-16.2.0-mingw-w64ucrt-14.0.0-r3.zip`, 276 MB) từ github.com/brechtsanders/winlibs_mingw. Gói này có sẵn `cmake` và `ninja`.
2. Giải nén: mã nguồn vào `_build_tools/src/`, GCC vào `_build_tools/mingw64/`.
3. zlib: bộ GCC không kèm zlib, mà ALE dùng zlib ở đúng 2 chỗ: lưu ảnh PNG (`crc32`, `compress`) và đọc file ROM (`gzopen`/`gzread`/`gzclose`). Thư mục `zlib_min/` ở đây là bản tối giản tự viết:
   - `crc32` theo chuẩn (giá trị kiểm tra `CBF43926`).
   - `compress` tạo luồng zlib hợp lệ kiểu "stored", tức không nén; zlib thật của Python giải nén ra đúng dữ liệu.
   - `gz*` đọc file ROM không nén. ROM của ale-py đều là `.bin` thường; gặp file gzip thật thì báo lỗi.
   ```bash
   export PATH="$PWD/_build_tools/mingw64/bin:$PATH"
   cp -r tools/build_windows_wheel/zlib_min _build_tools/
   cd _build_tools/zlib_min && gcc -O2 -c zlib_min.c -Iinclude -o zlib_min.o && ar rcs libz.a zlib_min.o && cd ../..
   ```
4. Build DLL, liên kết tĩnh: DLL chỉ phụ thuộc `KERNEL32` và UCRT, cả hai có sẵn trong Windows 10/11.
   ```bash
   cd _build_tools
   cmake -S src/multi_agent_ale_py-0.1.12 -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_C_LIB=ON -DBUILD_CPP_LIB=OFF -DUSE_SDL=OFF \
     -DZLIB_INCLUDE_DIR="$PWD/zlib_min/include" -DZLIB_LIBRARY="$PWD/zlib_min/libz.a" \
     -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_SHARED_LINKER_FLAGS="-static -static-libgcc -static-libstdc++"
   cmake --build build -j 6        # -> build/src/libale_c.dll (~5 MB)
   ```
5. Đóng wheel: `python tools/build_windows_wheel/make_wheel.py`, kết quả nằm ở `wheels/`.
6. Có thể xóa `_build_tools/` (~1.3 GB).

## Giấy phép
ALE và Multi-Agent-ALE theo giấy phép GPL-2.0. Wheel này gồm mã Python gốc (không sửa) cùng DLL build từ mã nguồn chính thức. Mã nguồn có tại https://github.com/Farama-Foundation/Multi-Agent-ALE và trong file sdist trên PyPI.
