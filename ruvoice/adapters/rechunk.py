"""Архивы RE Engine (Capcom: Resident Evil 2/3/7/Village, Devil May Cry 5, Monster Hunter Rise, Street Fighter 6…):
re_chunk_000.pak и патчи re_chunk_000.pak.patch_NNN.pak.

Формат KPKA: «KPKA», u8 версия, u8 подверсия, u16 флаги (0x8 — оглавление зашифровано: такие архивы не читаем),
u32 число файлов, u32 отпечаток; дальше оглавление и данные. Запись оглавления версии 4 (48 байт): u32 хеш пути
в нижнем регистре, u32 хеш в верхнем, u64 смещение, u64 размер сжатый, u64 размер исходный, u64 атрибуты
(младшие 4 бита — сжатие: 0 нет, 1 deflate, 2 zstd), u32 и u32 служебные. Версия 2 (24 байта): u64 смещение,
u64 размер, u32 и u32 хеши. Имён файлов в архиве нет — только хеши, поэтому нужные файлы находим по содержимому.

Игра читает архивы по порядку: основной, затем патчи 001, 002…; файл из более позднего патча заменяет
одноимённый (тот же хеш). Установка озвучки — свой патч в первом свободном номере (или вместо пустой
заглушки, которые оставляют менеджеры модов): игровые архивы не меняются, удаление возвращает как было."""
from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MAIN = "re_chunk_000.pak"
PATCH_RE = re.compile(r"^re_chunk_000\.pak\.patch_(\d{3})\.pak$", re.I)


@dataclass
class Entry:
    pak: Path
    hash_lo: int
    hash_hi: int
    offset: int
    csize: int
    usize: int
    attr: int
    extra: tuple = (0, 0)

    @property
    def key(self) -> tuple[int, int]:
        return self.hash_lo, self.hash_hi

    @property
    def compression(self) -> int:
        return self.attr & 0xF


def _read_toc(pak: Path) -> tuple[list[Entry], int, int]:
    with open(pak, "rb") as f:
        hdr = f.read(16)
        if len(hdr) < 16 or hdr[:4] != b"KPKA":
            return [], 0, 0
        major, minor, flags, count = hdr[4], hdr[5], struct.unpack_from("<H", hdr, 6)[0], struct.unpack_from("<I", hdr, 8)[0]
        if flags & 0x8:
            raise PermissionError("оглавление архива зашифровано")
        if major == 4:
            toc = f.read(48 * count)
            out = []
            for i in range(count):
                lo, hi, off, cs, us, attr, a, b = struct.unpack_from("<IIQQQQII", toc, 48 * i)
                out.append(Entry(pak, lo, hi, off, cs, us, attr, (a, b)))
            return out, major, flags
        if major == 2:
            f.seek(16 - 4)  # у версии 2 заголовок 12 байт
            toc = f.read(24 * count)
            out = []
            for i in range(count):
                off, us, lo, hi = struct.unpack_from("<QQII", toc, 24 * i)
                out.append(Entry(pak, lo, hi, off, us, us, 0))
            return out, major, flags
        raise ValueError(f"версия архива {major}.{minor} не поддерживается")


def pak_chain(root: Path) -> list[Path]:
    """Основной архив и патчи по порядку (только настоящие архивы, без пустых заглушек)."""
    main = root / MAIN
    if not main.is_file():
        return []
    patches = []
    for p in root.iterdir():
        m = PATCH_RE.match(p.name)
        if m and p.is_file() and p.stat().st_size > 16:
            patches.append((int(m.group(1)), p))
    return [main] + [p for _, p in sorted(patches)]


def detect(root: Path) -> str | None:
    """None — подходит; иначе причина, почему нет."""
    main = root / MAIN
    if not main.is_file():
        return "нет re_chunk_000.pak"
    with open(main, "rb") as f:
        hdr = f.read(16)
    if hdr[:4] != b"KPKA":
        return "re_chunk_000.pak — не архив RE Engine"
    if struct.unpack_from("<H", hdr, 6)[0] & 0x8:
        return "оглавление архивов зашифровано (так у новых игр Capcom: RE4, SF6, Dragon's Dogma 2, MH Wilds)"
    if hdr[4] not in (2, 4):
        return f"версия архива {hdr[4]}.{hdr[5]} пока не поддерживается"
    return None


class Archive:
    """Все архивы игры как одна файловая система по хешам (поздний патч перекрывает ранний)."""

    def __init__(self, root: Path, exclude: set[str] | None = None):
        self.root = root
        self.files: dict[tuple[int, int], Entry] = {}
        self.major = 4
        for pak in pak_chain(root):
            if exclude and pak.name.lower() in exclude:
                continue
            entries, major, _ = _read_toc(pak)
            if pak.name.lower() == MAIN:
                self.major = major
            for e in entries:
                self.files[e.key] = e
        self._handles: dict[Path, object] = {}
        self._zstd = None

    def _fh(self, pak: Path):
        fh = self._handles.get(pak)
        if fh is None:
            fh = self._handles[pak] = open(pak, "rb")
        return fh

    def read(self, e: Entry, limit: int | None = None) -> bytes:
        f = self._fh(e.pak)
        f.seek(e.offset)
        c = e.compression
        if c == 0:
            return f.read(e.csize if limit is None else min(limit, e.csize))
        raw = f.read(e.csize)
        if c == 1:
            return zlib.decompress(raw, -15)
        if c == 2:
            if self._zstd is None:
                import zstandard
                self._zstd = zstandard.ZstdDecompressor()
            return self._zstd.decompress(raw, max_output_size=e.usize)
        raise ValueError(f"неизвестное сжатие {c}")

    def head(self, e: Entry, n: int = 16) -> bytes:
        """Начало файла без распаковки целиком: сжатые — распаковываем только первый блок."""
        if e.compression == 0:
            return self.read(e, n)
        f = self._fh(e.pak)
        f.seek(e.offset)
        raw = f.read(min(e.csize, 160 << 10))
        try:
            if e.compression == 1:
                return zlib.decompressobj(-15).decompress(raw, n)
            if e.compression == 2:
                import io
                if self._zstd is None:
                    import zstandard
                    self._zstd = zstandard.ZstdDecompressor()
                with self._zstd.stream_reader(io.BytesIO(raw)) as r:
                    return r.read(n)
        except Exception:
            return b""
        return b""

    def find(self, test, max_size: int = 512 << 20):
        """Файлы, чьё начало (16 байт) прошло проверку test(head) → [(запись, данные)]."""
        for e in list(self.files.values()):
            if e.usize > max_size:
                continue
            h = self.head(e)
            if h and test(h):
                try:
                    yield e, self.read(e)
                except Exception:
                    continue

    def close(self):
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


# --- свой патч ---

def patch_slot(root: Path, own: Path | None = None) -> Path:
    """Куда класть свой патч: первый номер после последнего настоящего патча (пустая заглушка тоже годится)."""
    last = 0
    for p in root.iterdir():
        m = PATCH_RE.match(p.name)
        if m and p.is_file() and p.stat().st_size > 16 and (own is None or p != own):
            last = max(last, int(m.group(1)))
    return root / f"re_chunk_000.pak.patch_{last + 1:03d}.pak"


def write_patch(path: Path, files: list[tuple[Entry, bytes]], major: int = 4):
    """Патч-архив KPKA: файлы без сжатия, с хешами путей исходных записей (так игра заменит именно их)."""
    if major != 4:
        raise ValueError("сборка патча для архивов версии 2 пока не сделана")
    count = len(files)
    toc_size = 48 * count
    offset = 16 + toc_size
    toc = bytearray()
    for e, data in files:
        toc += struct.pack("<IIQQQQII", e.hash_lo, e.hash_hi, offset, len(data), len(data), 0, *e.extra)
        offset += len(data)
    tmp = path.with_suffix(".ruvoice.tmp")
    with open(tmp, "wb") as f:
        f.write(b"KPKA" + bytes([4, 0]) + struct.pack("<HII", 0, count, 0))
        f.write(toc)
        for _, data in files:
            f.write(data)
    tmp.replace(path)
