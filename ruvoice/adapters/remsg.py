"""Субтитры и тексты RE Engine (.msg, сигнатура GMSG на смещении 4): в каждом файле все языки сразу.

Заголовок: u32 версия, «GMSG», u64 размер заголовка, u32 число записей, u32 число атрибутов, u32 число языков,
u32 выравнивание, u64 начало данных (строки, зашифрованы), u64 служебное, u64 таблица языков, u64 атрибуты,
u64 имена атрибутов, затем u64 смещения записей. Запись: GUID (16), u32, u32 хеш имени, u64 смещение имени,
u64 атрибуты, u64 смещение строки для каждого языка. Строки — UTF-16, блок данных зашифрован побайтово:
b[i] = c[i] ^ c[i-1] ^ KEY[i % 16] (так же читают RE_MSG и mhrice).
Языки — перечисление движка: 0 японский, 1 английский, 6 русский (полный список — LANGS)."""
from __future__ import annotations

import struct

KEY = bytes([0xCF, 0xCE, 0xFB, 0xF8, 0xEC, 0x0A, 0x33, 0x66, 0x93, 0xA9, 0x1D, 0x93, 0x50, 0x39, 0x5F, 0x09])
EN, RU = 1, 6
LANGS = {0: "ja", 1: "en", 2: "fr", 3: "it", 4: "de", 5: "es", 6: "ru", 7: "pl", 8: "nl", 9: "pt", 10: "pt-br",
         11: "ko", 12: "zh-hant", 13: "zh-hans", 14: "fi", 15: "sv", 16: "da", 17: "no", 18: "cs", 19: "hu",
         20: "sk", 21: "ar", 22: "tr", 23: "bg", 24: "el", 25: "ro", 26: "th", 27: "uk", 28: "vi", 29: "id",
         30: "fiction", 31: "hi", 32: "es-419"}


def is_msg(head: bytes) -> bool:
    return head[4:8] == b"GMSG"


def _decrypt(b: bytes) -> bytes:
    import numpy as np
    a = np.frombuffer(b, dtype=np.uint8)
    prev = np.concatenate([np.zeros(1, np.uint8), a[:-1]])
    key = np.resize(np.frombuffer(KEY, dtype=np.uint8), len(a))
    return (a ^ prev ^ key).tobytes()


def _wstr(d: bytes, off: int) -> str:
    if off <= 0 or off >= len(d):
        return ""
    j = off
    while j + 1 < len(d) and d[j:j + 2] != b"\x00\x00":
        j += 2
    return d[off:j].decode("utf-16-le", "replace")


def parse(d: bytes) -> list[dict]:
    """[{"key": имя записи, "texts": {id языка: строка}}]."""
    ver, magic, hsz, n_ent, n_attr, n_lang, _pad = struct.unpack_from("<I4sQIIII", d, 0)
    if magic != b"GMSG":
        raise ValueError("не файл субтитров RE Engine")
    data_off, _unk, lang_off, _attr, _attr_names = struct.unpack_from("<QQQQQ", d, 0x20)
    ent_offs = struct.unpack_from(f"<{n_ent}Q", d, 0x48)
    langs = struct.unpack_from(f"<{n_lang}I", d, lang_off)
    dec = d[:data_off] + _decrypt(d[data_off:])
    rows = []
    for eo in ent_offs:
        name_off = struct.unpack_from("<Q", d, eo + 24)[0]
        cont = struct.unpack_from(f"<{n_lang}Q", d, eo + 40)
        rows.append({"key": _wstr(dec, name_off), "texts": {langs[k]: _wstr(dec, cont[k]) for k in range(n_lang)}})
    return rows
