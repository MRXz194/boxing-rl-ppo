/* zlib tối giản cho việc build ale_c.dll trên Windows (dự án Boxing RL).
 * ALE dùng zlib ở 2 chỗ: ScreenExporter (lưu PNG: crc32, compress) và OSystem::openROM (gzopen/gzread/gzclose).
 * gz*: đọc file ROM KHÔNG nén (ROM của ale-py đều là .bin thường); gặp file gzip thật thì gzopen trả NULL.
 * compress() ở đây tạo luồng zlib HỢP LỆ dạng "stored" (không nén): PNG vẫn đúng chuẩn, chỉ to hơn.
 * Không ảnh hưởng tới giả lập game. */
#ifndef ZLIB_MIN_H
#define ZLIB_MIN_H
#define ZLIB_VERSION "1.3.1-min"
#define Z_OK 0
#define Z_BUF_ERROR (-5)
typedef unsigned char Bytef;
typedef unsigned int uInt;
typedef unsigned long uLong;
typedef unsigned long uLongf;
#ifdef __cplusplus
extern "C" {
#endif
typedef struct gzFile_s *gzFile;
uLong crc32(uLong crc, const Bytef *buf, uInt len);
gzFile gzopen(const char *path, const char *mode);
int gzread(gzFile file, void *buf, unsigned len);
int gzclose(gzFile file);
int compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen);
#ifdef __cplusplus
}
#endif
#endif
