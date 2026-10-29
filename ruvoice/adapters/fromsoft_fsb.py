"""FMOD FSB5: разбор, сборка, шифрование банков (как в Sekiro/DS3) и кодирование Vorbis для FMOD.

Формат по vgmstream (fsb5.c) и python-fsb5. Vorbis в FSB5 — «голые» пакеты libvorbis с 16-битным
префиксом длины; заголовок setup не хранится, а задаётся CRC32 (FMOD держит таблицу готовых setup-пакетов).
libvorbis детерминирован: ffmpeg -c:a libvorbis даёт те же setup-пакеты, что и FMOD, поэтому CRC совпадают
(проверено: 101 из 101 значений -q:a для 48 кГц моно есть в таблице FMOD).
"""
from __future__ import annotations

import struct
import subprocess
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import tools

RATES = {4000: 0, 8000: 1, 11000: 2, 11025: 3, 16000: 4, 22050: 5, 24000: 6, 32000: 7, 44100: 8, 48000: 9,
         96000: 10}
RATES_REV = {v: k for k, v in RATES.items()}
CHANNELS = {1: 0, 2: 1, 6: 2, 8: 3}
CHANNELS_REV = {v: k for k, v in CHANNELS.items()}
CODEC_PCM16 = 2
CODEC_VORBIS = 15
EXTRA_CHANNELS, EXTRA_RATE, EXTRA_VORBIS = 1, 2, 0x0B
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_REV = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))
_REV_NP = np.frombuffer(_REV, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Шифрование FSB (режим «standard»: обратный порядок бит + XOR ключом)


def is_plain_fsb(data: bytes) -> bool:
    return data[:3] == b"FSB"


def fsb_decrypt(data: bytes, key: bytes) -> bytes:
    a = np.frombuffer(data, dtype=np.uint8)
    k = np.resize(np.frombuffer(key, dtype=np.uint8), len(a))
    return (_REV_NP[a] ^ k).tobytes()


def fsb_encrypt(data: bytes, key: bytes) -> bytes:
    a = np.frombuffer(data, dtype=np.uint8)
    k = np.resize(np.frombuffer(key, dtype=np.uint8), len(a))
    return _REV_NP[a ^ k].tobytes()


def recover_key(enc: bytes, max_len: int = 64) -> bytes | None:
    """Ключ шифрования FSB по самому банку, без знания игры. Шифр — разворот бит и XOR повторяющимся ключом,
    а в банке много нулей (выравнивание, пустые поля заголовков): для каждой длины ключа L в каждой позиции
    берём самый частый байт (это байт ключа), первые 4 байта уточняем по сигнатуре «FSB5». Ключ верный, если
    банк после расшифровки разбирается. Проверено на Sekiro: найден тот же ключ, что публикуют моддеры."""
    if is_plain_fsb(enc):
        return None
    a = _REV_NP[np.frombuffer(enc[:1 << 20], dtype=np.uint8)]
    for length in range(1, max_len + 1):
        key = bytes(int(np.bincount(a[r::length], minlength=256).argmax()) for r in range(length))
        fixed = bytearray(key)
        for i, ch in enumerate(b"FSB5"):
            fixed[i % length] = a[i] ^ ch
        for k in (bytes(fixed), key):
            if fsb_decrypt(enc[:64], k)[:4] != b"FSB5":
                continue
            try:
                FSB5.parse(fsb_decrypt(enc, k))
                return k
            except Exception:
                continue
    return None


def fsb_open(data: bytes, key: bytes | None) -> bytes:
    """Расшифровать при необходимости; проверка сигнатуры."""
    if is_plain_fsb(data):
        return data
    if not key:
        raise ValueError("FSB зашифрован, ключ неизвестен")
    dec = fsb_decrypt(data, key)
    if not is_plain_fsb(dec):
        raise ValueError("FSB не расшифровался этим ключом")
    return dec


# ---------------------------------------------------------------------------
# FSB5


@dataclass
class Sample:
    name: str
    rate: int
    channels: int
    num_samples: int
    extras: list[tuple[int, bytes]] = field(default_factory=list)
    data: bytes = b""

    @property
    def vorbis_setup_id(self) -> int | None:
        for t, b in self.extras:
            if t == EXTRA_VORBIS and len(b) >= 4:
                return struct.unpack_from("<I", b, 0)[0]
        return None


class FSB5:
    def __init__(self):
        self.version = 1
        self.codec = CODEC_VORBIS
        self.flags = 0
        self.unk1c = 0  # у банков Sekiro здесь 1 (копируем из оригинала)
        self.tail = b"\0" * 24  # 0x24..0x3C: хэш банка + неизвестное (копируем из оригинала)
        self.samples: list[Sample] = []

    @classmethod
    def parse(cls, data: bytes) -> "FSB5":
        if data[:4] != b"FSB5":
            raise ValueError("Не FSB5")
        self = cls()
        self.version, n, shs, nts, sds, self.codec = struct.unpack_from("<6I", data, 4)
        self.unk1c = struct.unpack_from("<I", data, 0x1C)[0]
        if self.version == 1:
            self.flags = struct.unpack_from("<I", data, 0x20)[0]
            base = 0x3C
            self.tail = data[0x24:0x3C]
        else:
            base = 0x40
            self.tail = data[0x28:0x40]
        pos = base
        raw = []
        for i in range(n):
            mode = struct.unpack_from("<Q", data, pos)[0]
            ns = (mode >> 34) & 0x3FFFFFFF
            doff = ((mode >> 7) & 0x07FFFFFF) << 5
            ch = CHANNELS_REV[(mode >> 5) & 3]
            rate = RATES_REV.get((mode >> 1) & 0xF, 0)
            p = pos + 8
            extras = []
            if mode & 1:
                while True:
                    ef = struct.unpack_from("<I", data, p)[0]
                    et, es, ec = (ef >> 25) & 0x7F, (ef >> 1) & 0xFFFFFF, ef & 1
                    blob = data[p + 4:p + 4 + es]
                    extras.append((et, blob))
                    if et == EXTRA_CHANNELS:
                        ch = blob[0]
                    elif et == EXTRA_RATE:
                        rate = struct.unpack_from("<i", blob, 0)[0]
                    p += 4 + es
                    if not ec:
                        break
            raw.append((doff, Sample("", rate, ch, ns, extras)))
            pos = p
        names_start = base + shs
        data_start = names_start + nts
        if nts:
            for i, (_, s) in enumerate(raw):
                o = struct.unpack_from("<I", data, names_start + 4 * i)[0]
                e = data.index(b"\0", names_start + o)
                s.name = data[names_start + o:e].decode("utf-8", "replace")
        for i, (doff, s) in enumerate(raw):
            end = raw[i + 1][0] if i + 1 < len(raw) else sds
            s.data = data[data_start + doff:data_start + end]
            self.samples.append(s)
        return self

    def build(self) -> bytes:
        # имена
        names = bytearray()
        name_offsets = []
        for s in self.samples:
            name_offsets.append(4 * len(self.samples) + len(names))
            names += s.name.encode("utf-8") + b"\0"
        name_table = b"".join(struct.pack("<I", o) for o in name_offsets) + bytes(names)
        # данные (каждый поток выровнен на 32)
        blobs, hdrs, off = [], bytearray(), 0
        for s in self.samples:
            d = bytes(s.data)
            pad = (-len(d)) % 32
            d += b"\0" * pad
            extras = [(t, b) for t, b in s.extras]
            if s.rate not in RATES and not any(t == EXTRA_RATE for t, _ in extras):
                extras.insert(0, (EXTRA_RATE, struct.pack("<i", s.rate)))
            if s.channels not in CHANNELS and not any(t == EXTRA_CHANNELS for t, _ in extras):
                extras.insert(0, (EXTRA_CHANNELS, bytes([s.channels])))
            mode = (s.num_samples & 0x3FFFFFFF) << 34
            mode |= ((off >> 5) & 0x07FFFFFF) << 7
            mode |= CHANNELS.get(s.channels, 0) << 5
            mode |= RATES.get(s.rate, 0) << 1
            mode |= 1 if extras else 0
            hdrs += struct.pack("<Q", mode)
            for i, (t, b) in enumerate(extras):
                hdrs += struct.pack("<I", (t << 25) | (len(b) << 1) | (1 if i + 1 < len(extras) else 0)) + b
            blobs.append(d)
            off += len(d)
        base = 0x3C if self.version == 1 else 0x40
        # дополняем таблицу имён, чтобы данные начинались с границы 32 байт
        total_hdr = base + len(hdrs) + len(name_table)
        name_table += b"\0" * ((-total_hdr) % 32)
        data = b"".join(blobs)
        out = bytearray(b"FSB5")
        out += struct.pack("<6I", self.version, len(self.samples), len(hdrs), len(name_table), len(data), self.codec)
        out += struct.pack("<I", self.unk1c)
        if self.version == 1:
            out += struct.pack("<I", self.flags) + self.tail[:24].ljust(24, b"\0")
        else:
            out += b"\0" * 8 + self.tail[:24].ljust(24, b"\0")
        out += hdrs + name_table + data
        return bytes(out)


# ---------------------------------------------------------------------------
# Vorbis для FSB через ffmpeg (libvorbis)


def _ogg_packets(data: bytes) -> tuple[list[bytes], int]:
    """Пакеты Ogg и granule position последней страницы (= число сэмплов)."""
    pos, packets, cur, granule = 0, [], b"", 0
    while pos + 27 <= len(data):
        if data[pos:pos + 4] != b"OggS":
            raise ValueError("Ogg: сбой синхронизации")
        g = struct.unpack_from("<q", data, pos + 6)[0]
        if g >= 0:
            granule = g
        nseg = data[pos + 26]
        segs = data[pos + 27:pos + 27 + nseg]
        p = pos + 27 + nseg
        for s in segs:
            cur += data[p:p + s]
            p += s
            if s < 255:
                packets.append(cur)
                cur = b""
        pos = p
    return packets, granule


def ffmpeg_has_libvorbis() -> bool:
    ff = tools.ffmpeg()
    if not ff:
        return False
    try:
        r = subprocess.run([str(ff), "-hide_banner", "-encoders"], capture_output=True, text=True,
                           creationflags=NO_WINDOW)
        return " libvorbis " in r.stdout
    except Exception:
        return False


def encode_vorbis_fsb(wav: Path, rate: int, channels: int, quality: float, work: Path) -> tuple[bytes, int, int]:
    """WAV → (данные потока FSB, CRC32 setup-пакета, число сэмплов)."""
    ff = tools.ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg не найден")
    work.mkdir(parents=True, exist_ok=True)
    ogg = work / (Path(wav).stem + ".ogg")
    subprocess.run([str(ff), "-y", "-loglevel", "error", "-i", str(wav), "-ar", str(rate), "-ac", str(channels),
                    "-c:a", "libvorbis", "-q:a", f"{quality:.1f}", "-f", "ogg", str(ogg)],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    packets, granule = _ogg_packets(ogg.read_bytes())
    ogg.unlink(missing_ok=True)
    if len(packets) < 4 or packets[2][:7] != b"\x05vorbis":
        raise ValueError("Ogg Vorbis: нет setup-пакета")
    setup_crc = zlib.crc32(packets[2]) & 0xFFFFFFFF
    out = bytearray()
    for p in packets[3:]:
        if len(p) >= 0xFFFF:
            raise ValueError("Vorbis-пакет длиннее 64К")
        out += struct.pack("<H", len(p)) + p
    return bytes(out), setup_crc, int(granule)


def vorbis_extra(setup_crc: int, seek: list[tuple[int, int]] | None = None) -> tuple[int, bytes]:
    seek = seek or []
    blob = struct.pack("<II", setup_crc, len(seek) * 8) + b"".join(struct.pack("<iI", s, o) for s, o in seek)
    return EXTRA_VORBIS, blob


def pcm16_data(wav: Path, rate: int, channels: int) -> tuple[bytes, int]:
    import soundfile as sf
    from .. import audio
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if channels == 1:
        data = data.mean(axis=1, keepdims=True)
    elif data.shape[1] != channels:
        data = np.repeat(data[:, :1], channels, axis=1)
    if sr != rate:
        data = np.stack([audio.resample(data[:, c], sr, rate) for c in range(data.shape[1])], axis=1)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    return pcm.tobytes(), pcm.shape[0]


def known_setup_ids(path: Path) -> set[int]:
    if not Path(path).exists():
        return set()
    return {int(x, 16) for x in Path(path).read_text().replace("\n", ",").split(",") if x.strip()}
