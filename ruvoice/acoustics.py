"""Акустика оригинала на русской речи: хвост эха и фон помещения.

Запечённую реверберацию нельзя снять с одной смешанной записи точно, но хвост после последнего
звука речи выдаёт время спада (наклон огибающей в дБ/с → RT60) и уровень шума комнаты (тихие
кадры). По времени спада строится импульс помещения — ранние отражения плюс экспоненциально
затухающий, слегка затемнённый шум — и русская речь смешивается с ним (доля «wet» растёт с
длиной хвоста). Фон подкладывается под готовую реплику, чтобы паузы не звучали «мёртво».

Режимы (настройка effects_mode): auto/unknown — по хвосту каждой реплики, порог спада 0,35 с;
baked — то же с порогом 0,15 с; engine/dry — ничего не накладывать."""
from __future__ import annotations

import numpy as np

from . import audio

VERSION = 1
MIN_RT60 = {"auto": 0.35, "unknown": 0.35, "baked": 0.15}   # короче — естественное затухание голоса, не помещение
MAX_RT60 = 2.0
MAX_WET = 0.4
MIN_TAIL_FRAMES = 10   # хвост над шумом не короче 100 мс
MIN_TAIL_RANGE = 15.0  # и спадает не меньше чем на столько дБ
FRAME = 0.01          # кадр огибающей, с
SPEECH_BAND_DB = 25   # речь — кадры громче пика минус столько


def wants_room(mode: str | None) -> bool:
    return (mode or "auto") in MIN_RT60


def rms_db(x: np.ndarray) -> float:
    return audio.rms_db(np.asarray(x, dtype=np.float32))


def _envelope_db(x: np.ndarray, sr: int) -> tuple[np.ndarray | None, int]:
    f = max(1, int(sr * FRAME))
    n = len(x) // f
    if n < 4:
        return None, f
    rms = np.sqrt(np.mean(x[: n * f].astype(np.float32).reshape(n, f) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms), f


def analyse_room(orig: np.ndarray, sr: int, mode: str = "auto") -> dict | None:
    """Помещение по хвосту оригинала: {"rt60": с, "wet": 0..1, "noise_db": дБFS} или None, если сухо."""
    x = np.asarray(orig, dtype=np.float32)
    db, f = _envelope_db(x, sr)
    if db is None:
        return None
    peak = float(db.max())
    noise = float(np.percentile(db, 2))  # самые тихие кадры — шум комнаты или тишина
    if peak - noise < 20:  # нет динамики — ни речи, ни хвоста
        return None
    if noise < -90:  # цифровая тишина: файл обрезан или сведён с затуханием — хвост не отличить от монтажа
        return None
    speech = np.where(db > peak - SPEECH_BAND_DB)[0]
    end = int(speech[-1]) + 1
    tail = db[end:end + int(3.0 / FRAME)]
    if end + len(tail) >= len(db) - 1:
        tail = tail[:-2]  # последние кадры файла — обрезка или затухание, не хвост
    keep = np.where(tail > noise + 6)[0]
    if keep.size < MIN_TAIL_FRAMES:
        return None
    k = int(keep[-1]) + 1  # хвост — пока он над шумом
    t = np.arange(k) * FRAME
    slope = float(np.polyfit(t, tail[:k], 1)[0])  # дБ/с
    if slope >= -5 or (tail[0] - tail[k - 1]) < MIN_TAIL_RANGE:
        return None
    rt60 = float(np.clip(-60.0 / slope, 0.1, MAX_RT60))
    if rt60 < MIN_RT60.get(mode or "auto", 0.25):
        return None
    wet = float(np.clip(0.12 + 0.25 * (rt60 - 0.2), 0.1, MAX_WET))
    return {"rt60": round(rt60, 2), "wet": round(wet, 2), "noise_db": round(noise, 1)}


def _lowpass(x: np.ndarray, sr: int, fc: float) -> np.ndarray:
    a = 1 - np.exp(-2 * np.pi * fc / sr)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):  # однополюсный ФНЧ; импульс короткий, цикл дешёвый
        acc += a * (v - acc)
        y[i] = acc
    return y


def room_impulse(rt60: float, sr: int, seed: int = 0) -> np.ndarray:
    """Импульс помещения: ранние отражения + затухающий затемнённый шум (−60 дБ к rt60), энергия 1."""
    n = int(min(max(rt60 * 1.2, 0.15), 3.0) * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    late = rng.standard_normal(n).astype(np.float32) * np.exp(-6.908 * t / max(rt60, 0.05)).astype(np.float32)
    late = _lowpass(late, sr, 5000.0)
    h = late * 0.35
    for k, (ms, g) in enumerate(((8, .5), (13, .4), (21, .3), (29, .25), (37, .2), (45, .15))):
        i = int(ms / 1000 * sr)
        if i < n:
            h[i] += g * (1 if k % 2 == 0 else -1)
    h[0] = 0.0  # прямой звук даёт сухая дорожка
    return (h / (np.sqrt(np.sum(h ** 2)) + 1e-9)).astype(np.float32)


def apply_room(gen: np.ndarray, sr: int, room: dict, impulse: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Смешать русскую речь с помещением; длина не меняется (хвост в пределах реплики, конец сглажен)."""
    from . import performance
    x = np.asarray(gen, dtype=np.float32)
    if not x.size:
        return x
    h = impulse if impulse is not None else room_impulse(float(room["rt60"]), sr, seed)
    out = performance.mix_stems(x, sr, impulse=h, wet=float(room["wet"]))[: len(x)]
    fade = min(int(sr * 0.01), len(out) // 2)
    if fade:
        out[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return out.astype(np.float32)


def noise_bed(x: np.ndarray, sr: int, noise_db: float, seed: int = 0) -> np.ndarray:
    """Подложить фон комнаты заданного уровня (дБFS) под реплику."""
    x = np.asarray(x, dtype=np.float32)
    if not x.size or noise_db is None:
        return x
    rng = np.random.default_rng(seed)
    bed = _lowpass(rng.standard_normal(len(x)).astype(np.float32), sr, 2000.0)
    bed *= 10 ** (float(noise_db) / 20) / (np.sqrt(np.mean(bed ** 2)) + 1e-9)
    return (x + bed).astype(np.float32)


def useful_noise(noise_db: float | None) -> bool:
    """Фон стоит подкладывать, когда он слышен, но это ещё фон, а не шум записи."""
    return noise_db is not None and -60 < noise_db < -30
