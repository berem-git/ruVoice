"""Разбор файлов локализации Unreal Engine .locres (версии 1 Compact, 2 Optimized, 3 Optimized CityHash64).

Результат — {namespace: {key: text}}; удобнее плоский словарь "namespace/key" → text (см. flat()).
"""
import struct

LOCRES_MAGIC = bytes.fromhex("0e147475674a03fc4a15909dc3377f1b")


def _fstring(b: bytes, pos: int) -> tuple[str, int]:
    n = struct.unpack_from("<i", b, pos)[0]
    pos += 4
    if n == 0:
        return "", pos
    if n < 0:
        raw = b[pos:pos + (-n) * 2]
        return raw.decode("utf-16-le", "replace").rstrip("\0"), pos + (-n) * 2
    raw = b[pos:pos + n]
    return raw.decode("utf-8", "replace").rstrip("\0"), pos + n


def parse(data: bytes) -> dict[str, dict[str, str]]:
    if data[:16] != LOCRES_MAGIC:
        return _parse_legacy(data)
    version = data[16]
    pos = 17
    strings_offset = struct.unpack_from("<q", data, pos)[0]
    pos += 8
    optimized = version >= 2
    if optimized:
        pos += 4  # общее число записей
    # Таблица строк
    strings: list[str] = []
    sp = strings_offset
    count = struct.unpack_from("<i", data, sp)[0]
    sp += 4
    for _ in range(count):
        s, sp = _fstring(data, sp)
        if optimized:
            sp += 4  # RefCount
        strings.append(s)
    result: dict[str, dict[str, str]] = {}
    n_ns = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    for _ in range(n_ns):
        if optimized:
            pos += 4  # хеш пространства имён
        ns, pos = _fstring(data, pos)
        n_keys = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        bucket = result.setdefault(ns, {})
        for _ in range(n_keys):
            if optimized:
                pos += 4  # хеш ключа
            key, pos = _fstring(data, pos)
            pos += 4  # хеш исходной строки
            idx = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            if 0 <= idx < len(strings):
                bucket[key] = strings[idx]
    return result


def _parse_legacy(data: bytes) -> dict[str, dict[str, str]]:
    """Самый старый формат (UE 4.0–4.13): без таблицы строк."""
    result: dict[str, dict[str, str]] = {}
    pos = 0
    n_ns = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    for _ in range(n_ns):
        ns, pos = _fstring(data, pos)
        n_keys = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        bucket = result.setdefault(ns, {})
        for _ in range(n_keys):
            key, pos = _fstring(data, pos)
            pos += 4
            text, pos = _fstring(data, pos)
            bucket[key] = text
    return result


def flat(tables: dict[str, dict[str, str]]) -> dict[str, str]:
    """"namespace/key" → text (пустое пространство имён — просто "key")."""
    out = {}
    for ns, keys in tables.items():
        for k, v in keys.items():
            out[f"{ns}/{k}" if ns else k] = v
    return out
