"""Чтение архивов Unreal Engine .pak (PakFile версии 8–11) на чистом Python.

Поддерживается сжатие Zlib/Gzip/Oodle (Oodle — через oo2core_*_win64.dll из tools\\oodle или папки игры)
и шифрование AES-256 индекса и данных (ключ даёт пользователь). Футер Black Myth: Wukong с двумя
лишними байтами после версии разбирается автоматически.
"""
import base64
import ctypes
import gzip
import os
import struct
import threading
import zlib
from pathlib import Path

from .. import tools

PAK_MAGIC = 0x5A6F12E1
MIN_INT32 = -0x80000000


# --- AES ---

def parse_aes_key(text: str | None) -> bytes | None:
    """Ключ из hex (с 0x или без) или base64 → 32 байта. Пустая строка → None."""
    if not text:
        return None
    s = text.strip().strip('"').strip()
    if not s:
        return None
    hex_s = s[2:] if s.lower().startswith("0x") else s
    try:
        if len(hex_s) == 64:
            return bytes.fromhex(hex_s)
    except ValueError:
        pass
    try:
        raw = base64.b64decode(s, validate=True)
        if len(raw) == 32:
            return raw
    except Exception:
        pass
    raise ValueError("AES-ключ должен быть 64 шестнадцатеричных символа (можно с 0x) или base64 длиной 32 байта")


def aes_decrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES
    n = len(data) & ~15
    return AES.new(key, AES.MODE_ECB).decrypt(data[:n]) + data[n:]


# --- Oodle ---

class Oodle:
    _lock = threading.Lock()
    _lib = None
    _path: Path | None = None

    @classmethod
    def find(cls, extra_dirs: list[Path] = ()) -> Path | None:
        cands = sorted((tools.TOOLS_DIR / "oodle").glob("oo2core_*_win64.dll"), reverse=True)
        for d in extra_dirs:
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.lower().startswith("oo2core_") and f.lower().endswith("_win64.dll"):
                        cands.append(Path(root) / f)
                if root.count(os.sep) - str(d).count(os.sep) >= 3:
                    dirs.clear()
        return cands[0] if cands else None

    @classmethod
    def load(cls, extra_dirs: list[Path] = ()):
        with cls._lock:
            if cls._lib is None:
                path = cls.find(extra_dirs)
                if not path:
                    raise RuntimeError("Нет библиотеки Oodle (tools\\oodle\\oo2core_9_win64.dll) — без неё архивы "
                                       "Unreal, сжатые Oodle, не распаковать")
                lib = ctypes.CDLL(str(path))
                fn = lib.OodleLZ_Decompress
                fn.restype = ctypes.c_int64
                fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
                cls._lib, cls._path = lib, path
            return cls._lib

    @classmethod
    def decompress(cls, data: bytes, out_size: int, extra_dirs: list[Path] = ()) -> bytes:
        lib = cls.load(extra_dirs)
        out = ctypes.create_string_buffer(out_size + 64)
        n = lib.OodleLZ_Decompress(data, len(data), out, out_size, 1, 0, 0, None, 0, None, None, None, 0, 3)
        if n != out_size:
            raise ValueError(f"Oodle: распаковано {n} байт вместо {out_size}")
        return out.raw[:out_size]


def decompress(method: str, data: bytes, out_size: int, extra_dirs: list[Path] = ()) -> bytes:
    m = method.lower()
    if m == "zlib":
        return zlib.decompress(data)
    if m == "gzip":
        return gzip.decompress(data)
    if m == "oodle":
        return Oodle.decompress(data, out_size, extra_dirs)
    if m == "lz4":
        try:
            import lz4.block
        except ImportError:
            raise RuntimeError("Архив сжат LZ4 — библиотека lz4 не установлена")
        return lz4.block.decompress(data, uncompressed_size=out_size)
    if m == "zstd":
        try:
            import zstandard
        except ImportError:
            raise RuntimeError("Архив сжат Zstd — библиотека zstandard не установлена")
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=out_size)
    raise RuntimeError(f"Неизвестный метод сжатия в архиве Unreal: {method}")


# --- чтение структур ---

class Reader:
    __slots__ = ("b", "pos")

    def __init__(self, b: bytes, pos: int = 0):
        self.b, self.pos = b, pos

    def u8(self):
        v = self.b[self.pos]
        self.pos += 1
        return v

    def i32(self):
        v = struct.unpack_from("<i", self.b, self.pos)[0]
        self.pos += 4
        return v

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.pos)[0]
        self.pos += 4
        return v

    def i64(self):
        v = struct.unpack_from("<q", self.b, self.pos)[0]
        self.pos += 8
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.pos)[0]
        self.pos += 8
        return v

    def bytes(self, n: int) -> bytes:
        v = self.b[self.pos:self.pos + n]
        self.pos += n
        return v

    def fstring(self) -> str:
        n = self.i32()
        if n == 0:
            return ""
        if n < 0:
            raw = self.bytes(-n * 2)
            return raw.decode("utf-16-le", "replace").rstrip("\0")
        raw = self.bytes(n)
        return raw.decode("utf-8", "replace").rstrip("\0")


class PakEntry:
    __slots__ = ("offset", "size", "uncompressed_size", "method", "blocks", "encrypted", "block_size", "header_size")

    def __init__(self):
        self.offset = self.size = self.uncompressed_size = 0
        self.method = 0
        self.blocks: list[tuple[int, int]] = []
        self.encrypted = False
        self.block_size = 0
        self.header_size = 53


class PakFile:
    """Один .pak: files — путь внутри архива (относительно точки монтирования, без '../') → PakEntry."""

    def __init__(self, path: Path, key: bytes | None = None, extra_dll_dirs: list[Path] = ()):
        self.path = Path(path)
        self.key = key
        self.extra_dll_dirs = list(extra_dll_dirs)
        self.size = self.path.stat().st_size
        self.files: dict[str, PakEntry] = {}
        self.mount_point = ""
        self.version = 0
        self.encrypted_index = False
        self.methods: list[str] = []
        self.entry_shift = 0
        self._lock = threading.Lock()
        self._fh = None
        self._read_footer()
        self._read_index()

    # --- футер ---

    def _read_footer(self):
        with open(self.path, "rb") as f:
            f.seek(max(0, self.size - 4096))
            tail = f.read()
        i = tail.rfind(struct.pack("<I", PAK_MAGIC))
        if i < 17:
            raise ValueError("Не похоже на .pak Unreal Engine (нет сигнатуры)")
        self.encrypted_index = tail[i - 1] == 1
        self.version = struct.unpack_from("<I", tail, i + 4)[0]
        if not 8 <= self.version <= 11:
            raise ValueError(f"Версия архива .pak {self.version} не поддерживается (нужны 8–11)")
        n_methods = 4 if self.version == 8 else 5
        expected_after_version = 8 + 8 + 20 + (1 if self.version == 9 else 0) + 32 * n_methods
        extra = len(tail) - (i + 8) - expected_after_version  # Black Myth: Wukong — 2 лишних байта
        if not 0 <= extra <= 16:
            extra = 0
        # У Wukong заголовок записи в области данных тоже на 1 байт длиннее, чем считает индекс
        self.entry_shift = 1 if extra == 2 else 0
        r = Reader(tail, i + 8 + extra)
        self.index_offset, self.index_size = r.i64(), r.i64()
        r.bytes(20)
        if self.version == 9:
            r.u8()
        self.methods = []
        for _ in range(n_methods):
            name = r.bytes(32).split(b"\0")[0].decode("ascii", "replace")
            self.methods.append(name)
        if not 0 < self.index_offset < self.size or self.index_offset + self.index_size > self.size:
            raise ValueError("Повреждённый футер .pak (индекс за пределами файла)")

    # --- индекс ---

    def _read_raw(self, offset: int, size: int, encrypted: bool) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        if encrypted:
            if not self.key:
                raise NeedsKey("Индекс архива зашифрован — нужен AES-ключ игры")
            data = aes_decrypt(data, self.key)
        return data

    def _read_index(self):
        try:
            self._read_index_inner()
        except (struct.error, IndexError, UnicodeDecodeError, MemoryError, OverflowError) as e:
            if self.encrypted_index:
                raise NeedsKey("AES-ключ не подходит к этому архиву") from e
            raise ValueError(f"Повреждённый индекс .pak: {e}") from e

    def _read_index_inner(self):
        data = self._read_raw(self.index_offset, self.index_size, self.encrypted_index)
        r = Reader(data)
        n = struct.unpack_from("<i", data, 0)[0]
        if not -4096 < n < 4096:
            raise IndexError("мусор вместо точки монтирования")
        self.mount_point = r.fstring()
        count = r.i32()
        if not 0 <= count < 50_000_000:
            raise IndexError("мусор вместо числа файлов")
        if self.version >= 10:
            r.u64()  # PathHashSeed
            if r.u32():  # bHasPathHashIndex
                r.i64(), r.i64(), r.bytes(20)
            has_full = r.u32()
            if not has_full:
                raise ValueError("В архиве нет полного индекса имён файлов")
            dir_off, dir_size = r.i64(), r.i64()
            r.bytes(20)
            n_enc = r.i32()
            encoded = r.bytes(n_enc)
            n_files = r.i32()
            plain = [self._read_entry(r) for _ in range(n_files)]
            dr = Reader(self._read_raw(dir_off, dir_size, self.encrypted_index))
            for _ in range(dr.i32()):
                dname = dr.fstring()
                for _ in range(dr.i32()):
                    fname = dr.fstring()
                    loc = dr.i32()
                    if loc == MIN_INT32:
                        continue
                    entry = self._decode_entry(encoded, loc) if loc >= 0 else plain[-loc - 1]
                    self.files[self._norm(dname + fname)] = entry
        else:
            for _ in range(count):
                name = r.fstring()
                self.files[self._norm(name)] = self._read_entry(r)

    def _norm(self, name: str) -> str:
        p = (self.mount_point + name).replace("\\", "/")
        while "//" in p:
            p = p.replace("//", "/")
        while p.startswith("../"):
            p = p[3:]
        return p.lstrip("/")

    def _read_entry(self, r: Reader) -> PakEntry:
        e = PakEntry()
        e.offset, e.size, e.uncompressed_size = r.i64(), r.i64(), r.i64()
        e.method = r.u32()
        r.bytes(20)
        header = 53
        if e.method != 0:
            n = r.i32()
            e.blocks = [(r.i64(), r.i64()) for _ in range(n)]
            header += 4 + 16 * n
        e.encrypted = bool(r.u8() & 1)
        e.block_size = r.u32()
        e.header_size = header
        return e

    @staticmethod
    def _decode_entry(buf: bytes, pos: int) -> PakEntry:
        e = PakEntry()
        r = Reader(buf, pos)
        bits = r.u32()
        e.block_size = r.u32() if (bits & 0x3F) == 0x3F else (bits & 0x3F) << 11
        e.method = (bits >> 23) & 0x3F
        e.offset = r.u32() if bits & (1 << 31) else r.i64()
        e.uncompressed_size = r.u32() if bits & (1 << 30) else r.i64()
        if e.method != 0:
            e.size = r.u32() if bits & (1 << 29) else r.i64()
        else:
            e.size = e.uncompressed_size
        e.encrypted = bool(bits & (1 << 22))
        n_blocks = (bits >> 6) & 0xFFFF
        header = 53 + (4 + 16 * n_blocks if e.method != 0 else 0)
        e.header_size = header
        if n_blocks == 1 and not e.encrypted:
            e.blocks = [(header, header + e.size)]
        elif n_blocks:
            cur = header
            for _ in range(n_blocks):
                bs = r.u32()
                e.blocks.append((cur, cur + bs))
                cur += (bs + 15) & ~15 if e.encrypted else bs
        return e

    # --- данные ---

    def _handle(self):
        if self._fh is None:
            self._fh = open(self.path, "rb")
        return self._fh

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def method_name(self, idx: int) -> str:
        return self.methods[idx - 1] if 0 < idx <= len(self.methods) else f"#{idx}"

    def read(self, name: str) -> bytes:
        e = self.files[name]
        return self.read_entry(e)

    def read_entry(self, e: PakEntry) -> bytes:
        if e.encrypted and not self.key:
            raise NeedsKey("Файлы в архиве зашифрованы — нужен AES-ключ игры")
        with self._lock:
            f = self._handle()
            shift = self.entry_shift
            if e.method == 0:
                f.seek(e.offset + e.header_size + shift)
                n = (e.size + 15) & ~15 if e.encrypted else e.size
                data = f.read(n)
                if e.encrypted:
                    data = aes_decrypt(data, self.key)
                return data[:e.size]
            method = self.method_name(e.method)
            out = bytearray()
            remaining = e.uncompressed_size
            for start, end in e.blocks:
                f.seek(e.offset + start + shift)
                n = end - start
                data = f.read((n + 15) & ~15 if e.encrypted else n)
                if e.encrypted:
                    data = aes_decrypt(data, self.key)[:n]
                want = min(e.block_size or remaining, remaining)
                out += decompress(method, data, want, self.extra_dll_dirs)[:want]
                remaining -= want
            return bytes(out)

    def __contains__(self, name: str) -> bool:
        return name in self.files

    def __len__(self):
        return len(self.files)


class NeedsKey(RuntimeError):
    """Архив зашифрован, а ключа нет (или он неверный)."""
