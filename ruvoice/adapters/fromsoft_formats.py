"""Форматы FromSoftware: архивы BHD5/BDT (Dvdbnd5), сжатие DCX, контейнер BND4, тексты FMG.

Чистый Python: RSA-расшифровка заголовков — обычное возведение в степень (pow), AES-128-ECB для
отдельных файлов — pyaes, Oodle Kraken — через oo2core_*.dll из папки игры (ctypes).
Форматы описаны по SoulsFormats (JKAnderson, MIT) и UXM (Nordgaren/JKAnderson).
"""
from __future__ import annotations

import base64
import ctypes
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Хэш пути (BHD5): /sound/vm10_enu.fsb → uint32 (DS3/Sekiro) или uint64 (Elden Ring и новее)


def path_hash(path: str, wide: bool = False) -> int:
    s = path.strip().replace("\\", "/").lower()
    if not s.startswith("/"):
        s = "/" + s
    h = 0
    if wide:
        for c in s:
            h = (h * 0x85 + ord(c)) & 0xFFFFFFFFFFFFFFFF
    else:
        for c in s:
            h = (h * 37 + ord(c)) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# RSA: открытый ключ PKCS#1 ("RSA PUBLIC KEY") → (n, e); расшифровка блоками 256 → 255 байт


def _asn1_items(data: bytes, pos: int = 0):
    """Простейший разбор DER: последовательность (tag, value)."""
    while pos < len(data):
        tag = data[pos]
        length = data[pos + 1]
        pos += 2
        if length & 0x80:
            n = length & 0x7F
            length = int.from_bytes(data[pos:pos + n], "big")
            pos += n
        yield tag, data[pos:pos + length]
        pos += length


def parse_rsa_public_key(pem: str) -> tuple[int, int]:
    body = "".join(l.strip() for l in pem.strip().splitlines() if not l.startswith("-----"))
    der = base64.b64decode(body)
    items = list(_asn1_items(der))
    if not items or items[0][0] != 0x30:
        raise ValueError("Не PKCS#1 ключ")
    ints = [int.from_bytes(v, "big") for t, v in _asn1_items(items[0][1]) if t == 0x02]
    if len(ints) < 2:
        raise ValueError("В ключе нет modulus/exponent")
    return ints[0], ints[1]


def rsa_decrypt_blocks(data: bytes, pem: str) -> bytes:
    """Как CryptographyUtility.DecryptRsa из BinderTool/UXM: каждый блок 256 байт → 255 байт."""
    n, e = parse_rsa_public_key(pem)
    in_size = (n.bit_length() + 7) // 8
    out_size = in_size - 1
    out = bytearray()
    for pos in range(0, len(data), in_size):
        block = data[pos:pos + in_size]
        if len(block) < in_size:
            block = block + b"\0" * (in_size - len(block))
        m = pow(int.from_bytes(block, "big"), e, n)
        out += m.to_bytes(out_size, "big")
    return bytes(out)


# ---------------------------------------------------------------------------
# BHD5 / BDT


@dataclass
class BHDFile:
    hash: int
    offset: int
    padded_size: int
    unpadded_size: int = -1
    aes_key: bytes | None = None
    aes_ranges: list[tuple[int, int]] = field(default_factory=list)
    path: str | None = None


class BHD5:
    """Заголовок архива. game: 'ds3' (DS3/Sekiro, 32-битные хэши) или 'er' (64-битные хэши)."""

    def __init__(self, data: bytes, wide: bool):
        if data[:4] != b"BHD5":
            raise ValueError("После расшифровки нет сигнатуры BHD5 — неверный ключ?")
        self.wide = wide
        self.files: list[BHDFile] = []
        bucket_count, buckets_off = struct.unpack_from("<ii", data, 0x10)
        for b in range(bucket_count):
            count, off = struct.unpack_from("<ii", data, buckets_off + b * 8)
            pos = off
            for _ in range(count):
                if wide:
                    h, padded, unpadded, foff, sha_off, aes_off = struct.unpack_from("<QiiqQQ", data, pos)
                    pos += 40
                else:
                    h, padded, foff, sha_off, aes_off, unpadded = struct.unpack_from("<IiqQQq", data, pos)
                    pos += 40
                f = BHDFile(h, foff, padded, unpadded)
                if aes_off:
                    f.aes_key = data[aes_off:aes_off + 16]
                    rc = struct.unpack_from("<i", data, aes_off + 16)[0]
                    f.aes_ranges = [struct.unpack_from("<qq", data, aes_off + 20 + i * 16) for i in range(rc)]
                self.files.append(f)
        self.by_hash = {f.hash: f for f in self.files}

    def apply_dictionary(self, paths: list[str]):
        for p in paths:
            f = self.by_hash.get(path_hash(p, self.wide))
            if f is not None:
                f.path = p

    def find(self, path: str) -> BHDFile | None:
        return self.by_hash.get(path_hash(path, self.wide))


def _aes_ecb_decrypt(key: bytes, buf: bytearray, start: int, end: int):
    import pyaes
    aes = pyaes.AES(key)
    for pos in range(start, end - 15, 16):
        buf[pos:pos + 16] = bytes(aes.decrypt(buf[pos:pos + 16]))


class Archive:
    """Пара Data1.bhd/Data1.bdt с расшифрованным заголовком."""

    def __init__(self, bhd_path: Path, pem: str | None, wide: bool = False):
        self.bhd_path = Path(bhd_path)
        self.bdt_path = self.bhd_path.with_suffix(".bdt")
        raw = self.bhd_path.read_bytes()
        if raw[:4] != b"BHD5":
            if not pem:
                raise ValueError(f"{self.bhd_path.name} зашифрован, а ключа нет")
            raw = rsa_decrypt_blocks(raw, pem)
        self.header = BHD5(raw, wide)
        self._fh = None

    @property
    def files(self) -> list[BHDFile]:
        return self.header.files

    def apply_dictionary(self, paths: list[str]):
        self.header.apply_dictionary(paths)

    def find(self, path: str) -> BHDFile | None:
        return self.header.find(path)

    def read(self, f: BHDFile) -> bytes:
        if self._fh is None:
            self._fh = open(self.bdt_path, "rb")
        self._fh.seek(f.offset)
        buf = bytearray(self._fh.read(f.padded_size))
        if f.aes_key:
            for start, end in f.aes_ranges:
                if start == -1 or end == -1 or start == end:
                    continue
                _aes_ecb_decrypt(f.aes_key, buf, int(start), min(int(end), len(buf)))
        if f.unpadded_size and f.unpadded_size > 0:
            del buf[f.unpadded_size:]
        return bytes(buf)

    def read_path(self, path: str) -> bytes | None:
        f = self.find(path)
        return self.read(f) if f else None

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# DCX


class OodleMissing(RuntimeError):
    pass


_oodle = None


def load_oodle(game_dir: Path):
    """oo2core_6/8/9_win64.dll из папки игры."""
    global _oodle
    if _oodle is not None:
        return _oodle
    for name in ("oo2core_6_win64.dll", "oo2core_8_win64.dll", "oo2core_9_win64.dll"):
        p = Path(game_dir) / name
        if p.exists():
            dll = ctypes.WinDLL(str(p))
            fn = dll.OodleLZ_Decompress
            fn.restype = ctypes.c_int64
            fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
            _oodle = fn
            return fn
    raise OodleMissing(f"В папке игры нет oo2core_*_win64.dll — не распаковать DCX (Oodle Kraken): {game_dir}")


def oodle_decompress(comp: bytes, out_size: int, game_dir: Path) -> bytes:
    fn = load_oodle(game_dir)
    out = ctypes.create_string_buffer(out_size + 64)
    n = fn(comp, len(comp), out, out_size, 1, 0, 0, None, 0, None, None, None, 0, 3)
    if n != out_size:
        raise ValueError(f"OodleLZ_Decompress вернул {n}, ожидалось {out_size}")
    return out.raw[:out_size]


def is_dcx(data: bytes) -> bool:
    return data[:4] == b"DCX\0" or data[:4] == b"DCP\0"


def dcx_decompress(data: bytes, game_dir: Path | None = None) -> bytes:
    """DCX (DFLT/KRAK/ZSTD) → данные. Для KRAK нужен game_dir с oo2core."""
    if data[:4] == b"DCP\0":
        fmt = data[4:8]
        if fmt != b"DFLT":
            raise NotImplementedError(f"DCP {fmt!r}")
        # DCP DFLT: 'DCP\0' 'DFLT' 0x20 ... 'DCS\0' sizes 'DCA\0' size, затем zlib
        p = data.find(b"DCS\0")
        unc, comp = struct.unpack_from(">II", data, p + 4)
        a = data.find(b"DCA\0", p)
        dca_size = struct.unpack_from(">I", data, a + 4)[0]
        return zlib.decompress(data[a + dca_size:a + dca_size + comp])
    if data[:4] != b"DCX\0":
        raise ValueError("Не DCX")
    if data[0x18:0x1C] != b"DCS\0":
        raise ValueError("DCX: нет DCS")
    unc_size, comp_size = struct.unpack_from(">II", data, 0x1C)
    if data[0x24:0x28] != b"DCP\0":
        raise ValueError("DCX: нет DCP")
    fmt = data[0x28:0x2C]
    dca = data.find(b"DCA\0", 0x2C)
    if dca < 0:
        raise ValueError("DCX: нет DCA")
    dca_size = struct.unpack_from(">I", data, dca + 4)[0]
    start = dca + dca_size
    comp = data[start:start + comp_size]
    if fmt == b"DFLT":
        return zlib.decompress(comp)
    if fmt == b"KRAK":
        if game_dir is None:
            raise OodleMissing("DCX KRAK: не указана папка игры с oo2core_*.dll")
        return oodle_decompress(comp, unc_size, game_dir)
    if fmt == b"ZSTD":
        try:
            import zstandard  # noqa
        except ImportError:
            raise NotImplementedError("DCX ZSTD: нужен пакет zstandard")
        return zstandard.ZstdDecompressor().decompress(comp, max_output_size=unc_size)
    if fmt == b"EDGE":
        return _dcx_edge(data, dca, dca_size, comp_size)
    raise NotImplementedError(f"DCX {fmt!r}")


def _dcx_edge(data: bytes, dca: int, dca_size: int, comp_size: int) -> bytes:
    # DCA: 'DCA\0' size 'EgdT' ... chunkCount, затем таблица (offset,size,last)
    p = dca + 8
    assert data[p:p + 4] == b"EgdT"
    egdt_size, chunk_count = struct.unpack_from(">II", data, p + 0x14)
    table = p + 0x24
    out = bytearray()
    for i in range(chunk_count):
        _, off, size, last = struct.unpack_from(">IIII", data, table + i * 16)
        chunk = data[dca + dca_size + off:dca + dca_size + off + size]
        out += zlib.decompress(chunk, -15) if last == 0 else zlib.decompress(chunk, -15)
    return bytes(out)


# ---------------------------------------------------------------------------
# BND4


@dataclass
class BNDEntry:
    id: int
    name: str | None
    data: bytes


def _reverse_bits(b: int) -> int:
    return int(f"{b:08b}"[::-1], 2)


def bnd4_read(data: bytes, game_dir: Path | None = None) -> list[BNDEntry]:
    if data[:4] != b"BND4":
        raise ValueError("Не BND4")
    big_endian = data[9] != 0
    bit_big_endian = data[10] == 0
    e = ">" if big_endian else "<"
    file_count = struct.unpack_from(e + "i", data, 0x0C)[0]
    unicode = data[0x30] != 0
    raw_fmt = data[0x31]
    reverse = bit_big_endian or ((raw_fmt & 1) != 0 and (raw_fmt & 0x80) == 0)
    fmt = raw_fmt if reverse else _reverse_bits(raw_fmt)
    has_ids = fmt & 0b10
    has_names = fmt & 0b1100
    long_offsets = fmt & 0b10000
    compression = fmt & 0b100000
    names1_only = fmt == 0b100
    pos = 0x40
    entries = []
    for _ in range(file_count):
        flags = data[pos] if bit_big_endian else _reverse_bits(data[pos])
        p = pos + 8
        comp_size = struct.unpack_from(e + "q", data, p)[0]
        p += 8
        if compression:
            p += 8  # uncompressed size
        if long_offsets:
            off = struct.unpack_from(e + "q", data, p)[0]
            p += 8
        else:
            off = struct.unpack_from(e + "I", data, p)[0]
            p += 4
        fid = -1
        if has_ids:
            fid = struct.unpack_from(e + "i", data, p)[0]
            p += 4
        name = None
        if has_names:
            name_off = struct.unpack_from(e + "I", data, p)[0]
            p += 4
            if unicode:
                end = data.find(b"\0\0", name_off)
                while end != -1 and (end - name_off) % 2:
                    end = data.find(b"\0\0", end + 1)
                name = data[name_off:end].decode("utf-16-le" if not big_endian else "utf-16-be")
            else:
                end = data.find(b"\0", name_off)
                name = data[name_off:end].decode("shift_jis", "replace")
        if names1_only:
            p += 8
        pos = p
        blob = data[off:off + comp_size]
        if flags & 1 and is_dcx(blob):
            blob = dcx_decompress(blob, game_dir)
        entries.append(BNDEntry(fid, name, blob))
    return entries


# ---------------------------------------------------------------------------
# FMG


def fmg_read(data: bytes) -> dict[int, str]:
    if len(data) < 0x20 or data[0] != 0:
        raise ValueError("Не FMG")
    big = data[1] != 0
    version = data[2]
    e = ">" if big else "<"
    wide = version == 2
    group_count, string_count = struct.unpack_from(e + "ii", data, 0x0C)
    if wide:
        offsets_off = struct.unpack_from(e + "q", data, 0x18)[0]
        groups_off = 0x28
        gsize = 16
    else:
        offsets_off = struct.unpack_from(e + "i", data, 0x14)[0]
        groups_off = 0x1C
        gsize = 12
    enc = "utf-16-be" if big else "utf-16-le"
    out: dict[int, str] = {}
    for g in range(group_count):
        idx, first, last = struct.unpack_from(e + "iii", data, groups_off + g * gsize)
        for j in range(last - first + 1):
            if wide:
                so = struct.unpack_from(e + "q", data, offsets_off + (idx + j) * 8)[0]
            else:
                so = struct.unpack_from(e + "i", data, offsets_off + (idx + j) * 4)[0]
            if so == 0:
                continue
            end = so
            while end + 1 < len(data) and not (data[end] == 0 and data[end + 1] == 0):
                end += 2
            out[first + j] = data[so:end].decode(enc, "replace")
    return out


# ---------------------------------------------------------------------------
# Словарь путей UXM


def load_dictionary(path: Path) -> list[str]:
    lines = []
    for line in Path(path).read_text("utf-8", "replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines
