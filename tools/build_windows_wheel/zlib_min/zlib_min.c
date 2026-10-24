#include "zlib.h"
#include <stdio.h>
uLong crc32(uLong crc, const Bytef *buf, uInt len) {
    static unsigned long table[256];
    static int ready = 0;
    if (!ready) {
        for (unsigned long n = 0; n < 256; n++) {
            unsigned long c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320UL ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        ready = 1;
    }
    if (!buf) return 0;
    crc = crc ^ 0xFFFFFFFFUL;
    while (len--) crc = table[(crc ^ *buf++) & 0xFF] ^ (crc >> 8);
    return (crc ^ 0xFFFFFFFFUL) & 0xFFFFFFFFUL;
}
int compress(Bytef *dest, uLongf *destLen, const Bytef *src, uLong n) {
    uLong blocks = n / 65535 + 1, need = 2 + n + 5 * blocks + 4, o = 0, i = 0;
    unsigned long a = 1, b = 0;
    if (*destLen < need) return Z_BUF_ERROR;
    dest[o++] = 0x78; dest[o++] = 0x01;               /* header zlib: deflate, cửa sổ 32K */
    do {
        uLong len = (n - i > 65535) ? 65535 : n - i;
        dest[o++] = (i + len >= n) ? 1 : 0;            /* BFINAL, BTYPE=00 (stored) */
        dest[o++] = len & 0xFF; dest[o++] = len >> 8;
        dest[o++] = ~len & 0xFF; dest[o++] = (~len >> 8) & 0xFF;
        for (uLong k = 0; k < len; k++, i++) {
            dest[o++] = src[i];
            a = (a + src[i]) % 65521; b = (b + a) % 65521;
        }
    } while (i < n);
    unsigned long ad = (b << 16) | a;                  /* adler32, big-endian */
    dest[o++] = ad >> 24; dest[o++] = ad >> 16; dest[o++] = ad >> 8; dest[o++] = ad;
    *destLen = o;
    return Z_OK;
}

/* gz*: chỉ đọc file không nén. File gzip thật (magic 1f 8b) bị từ chối để không đọc ra dữ liệu sai. */
gzFile gzopen(const char *path, const char *mode) {
    FILE *f = fopen(path, mode);
    if (!f) return 0;
    int c1 = fgetc(f), c2 = fgetc(f);
    if (c1 == 0x1f && c2 == 0x8b) {
        fprintf(stderr, "zlib_min: file nen gzip khong duoc ho tro: %s\n", path);
        fclose(f);
        return 0;
    }
    rewind(f);
    return (gzFile)f;
}
int gzread(gzFile file, void *buf, unsigned len) { return (int)fread(buf, 1, len, (FILE *)file); }
int gzclose(gzFile file) { return fclose((FILE *)file); }
