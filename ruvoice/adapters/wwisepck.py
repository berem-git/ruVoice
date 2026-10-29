"""Пакеты Wwise (.pck, сигнатура AKPK) — общий формат озвучки в играх на Wwise, независимо от движка.

Устройство (как у Audiokinetic File Package и в wwiser/vgmstream):
  "AKPK", u32 размер заголовка (без первых 8 байт), u32 версия (1), u32 размер таблицы языков,
  u32 размер таблицы банков, u32 размер таблицы потоков, [u32 размер таблицы внешних файлов — в новых версиях];
  таблица языков: u32 число, пары (u32 смещение имени от начала таблицы, u32 id), затем имена
  (UTF-16 или ASCII); таблицы банков/потоков: u32 число, записи (id, u32 размер блока, u32 размер файла,
  u32 номер начального блока, u32 id языка); у внешних файлов id 64-битный.
  Данные файла лежат с позиции «номер блока × размер блока» от начала пакета.

Озвучка — это файлы с языком (не «sfx»): у каждого языка свой набор .wem. Замена реплики: новые данные
дописываются в конец пакета (выравнивание по блоку), запись в таблице указывает на них — остальной пакет
не трогается, поэтому старые версии Wwise читают его так же."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class PckEntry:
    table: str            # "banks" | "streams" | "externals"
    id: int
    block: int
    size: int
    start_block: int
    lang: int
    pos: int              # где в пакете лежит сама запись (для правки)

    @property
    def offset(self) -> int:
        return self.start_block * max(1, self.block)


@dataclass
class Pck:
    data: bytes
    languages: dict[int, str] = field(default_factory=dict)
    entries: list[PckEntry] = field(default_factory=list)
    wide_ids: bool = False

    @classmethod
    def parse(cls, data: bytes) -> "Pck":
        if data[:4] != b"AKPK":
            raise ValueError("не пакет Wwise (нет AKPK)")
        hdr_size, version = struct.unpack_from("<II", data, 4)
        sizes = list(struct.unpack_from("<4I", data, 12))
        # есть ли таблица внешних файлов: сумма размеров таблиц должна совпасть с размером заголовка
        ext_size = struct.unpack_from("<I", data, 28)[0]
        if 4 + 16 + sum(sizes) + 4 + ext_size == hdr_size:
            sizes.append(ext_size)
            pos = 32
        else:
            pos = 28
        p = cls(data)
        lang_size, banks_size, stm_size = sizes[0], sizes[1], sizes[2]
        p._parse_langs(data[pos:pos + lang_size])
        pos += lang_size
        p._parse_table(data, pos, banks_size, "banks", False)
        pos += banks_size
        # в части игр (2019+) id потоков 64-битные; признак — размер записи
        p._parse_table(data, pos, stm_size, "streams", None)
        pos += stm_size
        if len(sizes) > 3:
            p._parse_table(data, pos, sizes[3], "externals", True)
        return p

    def _parse_langs(self, t: bytes):
        if len(t) < 4:
            return
        n = struct.unpack_from("<I", t, 0)[0]
        pairs = [struct.unpack_from("<II", t, 4 + 8 * i) for i in range(n)]
        for off, lid in pairs:
            # имя: UTF-16, если второй байт нулевой, иначе однобайтное
            wide = off + 1 < len(t) and t[off + 1] == 0
            if wide:
                j = off
                while j + 1 < len(t) and t[j:j + 2] != b"\x00\x00":
                    j += 2
                name = t[off:j].decode("utf-16-le", "replace")
            else:
                j = t.find(b"\x00", off)
                name = t[off:j if j >= 0 else len(t)].decode("latin1", "replace")
            self.languages[lid] = name.lower()

    def _parse_table(self, data: bytes, pos: int, size: int, table: str, wide: bool | None):
        if size < 4:
            return
        n = struct.unpack_from("<I", data, pos)[0]
        if n == 0:
            return
        rec = (size - 4) // n
        wide = rec == 24 if wide is None else wide
        if table == "streams":
            self.wide_ids = wide
        for i in range(n):
            q = pos + 4 + rec * i
            if wide:
                eid, block, fsize, start, lang = struct.unpack_from("<QIIII", data, q)
            else:
                eid, block, fsize, start, lang = struct.unpack_from("<IIIII", data, q)
            self.entries.append(PckEntry(table, eid, block, fsize, start, lang, q))

    def read(self, e: PckEntry) -> bytes:
        return self.data[e.offset:e.offset + e.size]

    def lang_name(self, lid: int) -> str:
        return self.languages.get(lid, str(lid))

    def voice_langs(self) -> set[str]:
        """Языки, у которых есть файлы (кроме sfx)."""
        return {self.lang_name(e.lang) for e in self.entries if self.lang_name(e.lang) != "sfx"}

    def rebuild(self, replace: dict[tuple[str, int, int], bytes]) -> bytes:
        """Новый пакет: replace — {(таблица, id, язык): новые данные}. Новые данные дописываются в конец
        (с выравниванием по размеру блока записи), записи указывают на них; размеры обновляются."""
        out = bytearray(self.data)
        for e in self.entries:
            new = replace.get((e.table, e.id, e.lang))
            if new is None:
                continue
            blk = max(1, e.block)
            pad = (-len(out)) % blk
            out += b"\x00" * pad
            start = len(out) // blk
            out += new
            off = 8 if e.table == "externals" or (e.table == "streams" and self.wide_ids) else 4
            struct.pack_into("<II", out, e.pos + off + 4, len(new), start)
        return bytes(out)


def is_english(name: str) -> bool:
    n = (name or "").lower()
    return n.startswith("english") or n in ("en", "en-us", "en_us", "eng", "us")


def is_russian(name: str) -> bool:
    n = (name or "").lower()
    return n.startswith("russian") or n in ("ru", "ru-ru", "ru_ru", "rus")
