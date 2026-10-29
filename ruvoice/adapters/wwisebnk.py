"""Минимальный разбор банков Wwise (.bnk): какие .wem (SourceID) использует банк и его события.

Нужен, чтобы связать файл NNNN.wem с именем события/банка (а по нему — с персонажем и репликой).
Полный формат HIRC зависит от версии Wwise; здесь читаются только устойчивые части:
Sound (тип 2) → SourceID, Event (тип 4) → список Action, Action (тип 3) → цель, DIDX/DATA — встроенные wem.
"""
import struct


def fnv1_32(name: str) -> int:
    """Хеш Wwise для имён событий/банков (FNV-1 32 бит от имени в нижнем регистре)."""
    h = 2166136261
    for b in name.lower().encode("utf-8"):
        h = ((h * 16777619) & 0xFFFFFFFF) ^ b
    return h


class Bank:
    def __init__(self, data: bytes):
        self.version = 0
        self.bank_id = 0
        self.sounds: dict[int, int] = {}      # id объекта Sound → SourceID (имя .wem)
        self.events: dict[int, list[int]] = {}  # id события → id действий
        self.actions: dict[int, tuple[int, int]] = {}  # id действия → (тип, цель)
        self.embedded: dict[int, bytes] = {}  # SourceID → данные wem, встроенные в банк (DIDX+DATA)
        self.objects: list[tuple[int, int]] = []  # (тип, id) всех объектов HIRC
        self._parse(data)

    def _parse(self, data: bytes):
        pos, didx, blob = 0, [], b""
        while pos + 8 <= len(data):
            tag, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
            body = data[pos + 8:pos + 8 + size]
            if tag == b"BKHD" and len(body) >= 8:
                self.version, self.bank_id = struct.unpack_from("<II", body, 0)
            elif tag == b"DIDX":
                didx = [struct.unpack_from("<III", body, i) for i in range(0, len(body) - 11, 12)]
            elif tag == b"DATA":
                blob = body
            elif tag == b"HIRC":
                self._parse_hirc(body)
            pos += 8 + size
        for sid, off, sz in didx:
            self.embedded[sid] = blob[off:off + sz]

    def _parse_hirc(self, body: bytes):
        if len(body) < 4:
            return
        n = struct.unpack_from("<I", body, 0)[0]
        p = 4
        for _ in range(n):
            if p + 9 > len(body):
                break
            typ = body[p]
            size = struct.unpack_from("<I", body, p + 1)[0]
            oid = struct.unpack_from("<I", body, p + 5)[0]
            obj = body[p + 9:p + 5 + size]
            self.objects.append((typ, oid))
            try:
                if typ == 2 and len(obj) >= 9:
                    # AkBankSourceData: PluginID u32, StreamType (u8 в новых версиях, u32 в старых), SourceID u32
                    if self.version <= 89:
                        src = struct.unpack_from("<I", obj, 8)[0]
                    else:
                        src = struct.unpack_from("<I", obj, 5)[0]
                    self.sounds[oid] = src
                elif typ == 3 and len(obj) >= 6:
                    atype, target = struct.unpack_from("<HI", obj, 0)
                    self.actions[oid] = (atype, target)
                elif typ == 4:
                    if self.version <= 122:
                        cnt = struct.unpack_from("<I", obj, 0)[0]
                        q = 4
                    else:
                        cnt, q = _varint(obj, 0)
                    self.events[oid] = [struct.unpack_from("<I", obj, q + 4 * i)[0] for i in range(cnt)
                                        if q + 4 * i + 4 <= len(obj)]
            except struct.error:
                pass
            p += 5 + size

    @property
    def source_ids(self) -> set[int]:
        return set(self.sounds.values()) | set(self.embedded)


def bank_language(data: bytes) -> int:
    """id языка банка (FNV-1 от имени: fnv1_32("english(us)"), «sfx» — общий звук)."""
    return struct.unpack_from("<I", data, 16)[0] if data[:4] == b"BKHD" and len(data) >= 20 else 0


def _chunks(data: bytes) -> list[tuple[bytes, int, int]]:
    out, pos = [], 0
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        out.append((tag, pos + 8, size))
        pos += 8 + size
    return out


def codec_sources(data: bytes) -> list[tuple[int, int, int, int]]:
    """Звуковые объекты банка (тип 2) с кодеком (не плагины-генераторы вроде тишины):
    [(SourceID, тип хранения, размер в памяти, позиция поля «тип хранения» в банке)].
    Тип хранения (Wwise 2016+): 0 — в памяти (DIDX/DATA), 1 — начало в памяти + поток, 2 — только поток."""
    out = []
    version = struct.unpack_from("<I", data, 8)[0] if data[:4] == b"BKHD" else 0
    if version <= 89:
        return out  # старый формат полей — не правим
    for tag, start, size in _chunks(data):
        if tag != b"HIRC":
            continue
        n = struct.unpack_from("<I", data, start)[0]
        p = start + 4
        for _ in range(n):
            if p + 9 > start + size:
                break
            typ, osz = data[p], struct.unpack_from("<I", data, p + 1)[0]
            o = p + 9  # после типа (1), размера (4) и id объекта (4)
            if typ == 2 and o + 14 <= p + 5 + osz:
                plugin = struct.unpack_from("<I", data, o)[0]
                if plugin & 0xF == 1:  # кодек (Vorbis, PCM, ADPCM, Opus…), а не генератор
                    st = data[o + 4]
                    sid, msz = struct.unpack_from("<II", data, o + 5)
                    out.append((sid, st, msz, o + 4))
            p += 5 + osz
    return out


def patch_bank(data: bytes, new_media: dict[int, bytes], stream_only: set[int] | None = None) -> bytes:
    """Банк с заменённой встроенной озвучкой.
    new_media — {SourceID: новые данные .wem}: заменяются в DIDX/DATA (смещения пересчитываются, выравнивание
    по 16 байт, как у Wwise), у звуковых объектов с этим источником «в памяти» обновляется размер;
    stream_only — источники, которые теперь целиком идут из потока: их «начало в памяти» (prefetch) осталось бы
    от старой записи, поэтому тип хранения меняется на «только поток» (2), размер в памяти — 0."""
    stream_only = stream_only or set()
    chunks = _chunks(data)
    didx_c = next(((s, n) for t, s, n in chunks if t == b"DIDX"), None)
    data_c = next(((s, n) for t, s, n in chunks if t == b"DATA"), None)
    out = bytearray()
    sizes: dict[int, int] = {}
    if didx_c and data_c and any(
            struct.unpack_from("<I", data, didx_c[0] + k)[0] in new_media for k in range(0, didx_c[1] - 11, 12)):
        entries = [struct.unpack_from("<III", data, didx_c[0] + k) for k in range(0, didx_c[1] - 11, 12)]
        blob = bytearray()
        new_didx = bytearray()
        for sid, off, sz in entries:
            media = new_media.get(sid)
            if media is None:
                media = data[data_c[0] + off:data_c[0] + off + sz]
            else:
                sizes[sid] = len(media)
            blob += b"\x00" * ((-len(blob)) % 16)
            new_didx += struct.pack("<III", sid, len(blob), len(media))
            blob += media
        for tag, start, size in chunks:
            body = data[start:start + size]
            if tag == b"DIDX":
                body = bytes(new_didx)
            elif tag == b"DATA":
                body = bytes(blob)
            out += tag + struct.pack("<I", len(body)) + body
    else:
        out = bytearray(data)
    # размеры «в памяти» и prefetch — во всех звуковых объектах, ссылающихся на эти источники
    for sid, st, msz, pos in codec_sources(bytes(out)):
        if sid in stream_only and st == 1:
            out[pos] = 2
            struct.pack_into("<I", out, pos + 5, 0)
        elif st == 0 and sid in new_media and msz:
            struct.pack_into("<I", out, pos + 5, len(new_media[sid]))
    return bytes(out)


def _varint(b: bytes, pos: int) -> tuple[int, int]:
    v, shift = 0, 0
    while pos < len(b):
        c = b[pos]
        pos += 1
        v |= (c & 0x7F) << shift
        if not c & 0x80:
            break
        shift += 7
    return v, pos
