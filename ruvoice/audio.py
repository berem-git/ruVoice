"""Звук: чтение любых игровых форматов, подгонка русской реплики под оригинал, запись в формат игры."""
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import tools

NO_WINDOW = subprocess.CREATE_NO_WINDOW
VGMSTREAM_EXT = {".wem", ".bnk", ".fsb", ".acb", ".awb", ".hca", ".adx", ".xwb", ".bank"}


def _run(args: list, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, creationflags=NO_WINDOW, check=True, **kw)


def decode_to_wav(src: Path, dst: Path, sr: int | None = None, mono: bool = False):
    """Любой формат → PCM WAV. Игровые форматы (Wwise и др.) — через vgmstream, остальное — ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if src.suffix.lower() in VGMSTREAM_EXT:
        tmp = dst.with_suffix(".vgm.wav")
        _run([tools.vgmstream(), "-o", tmp, src])
        src_for_ffmpeg = tmp
    else:
        tmp, src_for_ffmpeg = None, src
    args = [tools.ffmpeg(), "-y", "-loglevel", "error", "-i", src_for_ffmpeg, "-map", "0:a:0", "-c:a", "pcm_s16le"]
    if sr:
        args += ["-ar", sr]
    if mono:
        args += ["-ac", 1]
    _run(args + [dst])
    if tmp:
        tmp.unlink(missing_ok=True)


def read(path: Path) -> tuple[np.ndarray, int]:
    """Моно float32 и частота."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def write(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_16"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(data, -1, 1), sr, subtype=subtype)


def duration(path: Path) -> float:
    return sf.info(str(path)).duration


def trim_silence(data: np.ndarray, sr: int, threshold_db: float = -45, pad: float = 0.05) -> np.ndarray:
    if data.size == 0:
        return data
    frame = max(1, int(sr * 0.01))
    n = len(data) // frame
    if n == 0:
        return data
    rms = np.sqrt(np.mean(data[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms) > threshold_db)[0]
    if loud.size == 0:
        return data
    start = max(0, loud[0] * frame - int(pad * sr))
    end = min(len(data), (loud[-1] + 1) * frame + int(pad * sr))
    return data[start:end]


def rms_db(data: np.ndarray) -> float:
    active = data[np.abs(data) > 1e-4]
    if active.size == 0:
        return -90.0
    return float(20 * np.log10(np.sqrt(np.mean(active**2)) + 1e-12))


def resample(data: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return data
    import soxr
    return soxr.resample(data, sr_from, sr_to, quality="HQ").astype(np.float32)


def time_stretch(data: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Ускорить речь без изменения тона (ffmpeg atempo). speed > 1 — быстрее."""
    if abs(speed - 1) < 0.01:
        return data
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        write(a, data, sr, "FLOAT")
        _run([tools.ffmpeg(), "-y", "-loglevel", "error", "-i", a, "-filter:a", f"atempo={speed:.4f}", b])
        out, _ = read(b)
    return out


def fit_to_original(gen: np.ndarray, gen_sr: int, orig: np.ndarray, orig_sr: int,
                    max_speedup: float = 1.25, tolerance: float = 1.08) -> tuple[np.ndarray, dict]:
    """Подогнать русскую реплику под оригинал: частота, громкость, длина (ускорение до max_speedup)."""
    gen = trim_silence(gen, gen_sr)
    gen = resample(gen, gen_sr, orig_sr)
    orig_trimmed = trim_silence(orig, orig_sr)
    target = max(len(orig_trimmed) / orig_sr, 0.3)
    length = len(gen) / orig_sr
    speed = 1.0
    if length > target * tolerance:
        speed = min(length / (target * tolerance), max_speedup)
        gen = time_stretch(gen, orig_sr, speed)
    gain = rms_db(orig_trimmed) - rms_db(gen)
    gen = gen * (10 ** (np.clip(gain, -20, 20) / 20))
    peak = np.max(np.abs(gen)) if gen.size else 0
    if peak > 0.98:
        gen = gen * (0.98 / peak)
    return gen.astype(np.float32), {"speed": round(speed, 3), "ratio": round(len(gen) / orig_sr / target, 3)}


# --- WAV-чанки: перенос данных липсинка Source (VDAT/PHONEME) из оригинала ---

def read_riff_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    return read_riff_chunks_from_bytes(Path(path).read_bytes()) if Path(path).exists() else []


def read_riff_chunks_from_bytes(raw: bytes) -> list[tuple[bytes, bytes]]:
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return []
    chunks, pos = [], 12
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        chunks.append((cid, raw[pos + 8:pos + 8 + size]))
        pos += 8 + size + (size & 1)
    return chunks


def append_chunks(path: Path, extra: list[tuple[bytes, bytes]]):
    if not extra:
        return
    raw = bytearray(Path(path).read_bytes())
    for cid, data in extra:
        raw += cid + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")
    raw[4:8] = struct.pack("<I", len(raw) - 8)
    Path(path).write_bytes(bytes(raw))


# --- Запись в формат оригинального файла ---

def encode_like(wav: Path, original_ext: str, out: Path, sr: int | None = None, channels: int | None = None):
    """WAV → формат оригинала (.wav/.ogg/.mp3/.flac/.opus). Для .wem — encode_wem_batch."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = original_ext.lower()
    args = [tools.ffmpeg(), "-y", "-loglevel", "error", "-i", wav]
    if sr:
        args += ["-ar", sr]
    if channels:
        args += ["-ac", channels]
    codec = {".wav": ["-c:a", "pcm_s16le"], ".ogg": ["-c:a", "libvorbis", "-q:a", "6"],
             ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"], ".flac": ["-c:a", "flac"],
             ".opus": ["-c:a", "libopus", "-b:a", "96k"]}.get(ext)
    if codec is None:
        raise ValueError(f"Формат {ext} не поддерживается для записи")
    fmt = ["-f", {".wav": "wav", ".ogg": "ogg", ".mp3": "mp3", ".flac": "flac", ".opus": "ogg"}[ext]]
    _run(args + codec + fmt + [out])


def encode_wem_batch(pairs: list[tuple[Path, Path]], conversion: str = "Vorbis Quality High"):
    """WAV → .wem пачкой через Wwise. conversion — набор настроек проекта: «Vorbis Quality High» (как в
    большинстве игр) или «PCM As Input» — без сжатия, его читает любая версия Wwise в игре."""
    if not pairs:
        return
    console = tools.wwise_console()
    if not console:
        raise RuntimeError("Wwise не найден в tools\\Wwise — он нужен для игр на Wwise")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        items = []
        for i, (wav, _) in enumerate(pairs):
            name = f"{i:06d}.wav"
            (td / name).write_bytes(Path(wav).read_bytes())
            items.append(f'<Source Path="{name}" Conversion="{conversion}"/>')
        (td / "list.wsources").write_text(
            f'<?xml version="1.0" encoding="UTF-8"?>\n<ExternalSourcesList SchemaVersion="1" Root="{td}">\n'
            + "\n".join(items) + "\n</ExternalSourcesList>\n", encoding="utf-8")
        env = {**os.environ, "WWISEROOT": str(Path(console).parents[4])}
        _run([console, "convert-external-source", tools.WWISE_PROJECT, "--platform", "Windows",
              "--source-file", td / "list.wsources", "--output", td / "out"], env=env)
        for i, (_, dst) in enumerate(pairs):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes((td / "out" / "Windows" / f"{i:06d}.wem").read_bytes())


def speech_features(data: np.ndarray, sr: int) -> dict:
    """Темп, паузы и динамика речи по огибающей (без текста — работает для любого языка):
    rate — слогов в секунду (пики огибающей за время звучания), pause_ratio — доля пауз внутри фразы,
    dyn_db — размах громкости (разброс уровня громких кадров), dur — длительность без тишины по краям."""
    data = trim_silence(data, sr)
    frame = max(1, int(sr * 0.02))
    n = len(data) // frame
    if n < 5:
        return {"rate": 0.0, "pause_ratio": 0.0, "dyn_db": 0.0, "dur": round(len(data) / sr, 2)}
    rms = np.sqrt(np.mean(data[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    peak_db = float(np.percentile(db, 95))
    voiced = db > peak_db - 25
    pause_ratio = 1.0 - float(np.mean(voiced))
    loud = db[voiced]
    dyn_db = float(np.std(loud)) * 2 if loud.size > 2 else 0.0
    # сглаженная огибающая: локальные максимумы ≈ ядра слогов
    k = np.ones(5) / 5
    env = np.convolve(rms, k, mode="same")
    peaks = 0
    for i in range(1, n - 1):
        if voiced[i] and env[i] > env[i - 1] and env[i] >= env[i + 1] and env[i] > 0.2 * np.max(env):
            peaks += 1
    voiced_sec = max(float(np.sum(voiced)) * frame / sr, 0.1)
    return {"rate": round(peaks / voiced_sec, 2), "pause_ratio": round(pause_ratio, 2),
            "dyn_db": round(dyn_db, 1), "dur": round(n * frame / sr, 2)}


def f0_median(data: np.ndarray, sr: int, max_sec: float = 40.0) -> float:
    """Медианная высота голоса (Гц) по звучащим кадрам (librosa.pyin — устойчив к октавным ошибкам);
    0, если оценить не удалось."""
    try:
        import librosa
        w = resample(trim_silence(data, sr), sr, 16000)[: int(max_sec * 16000)]
        if len(w) < 8000:
            return 0.0
        f0, voiced, _ = librosa.pyin(w.astype(np.float32), fmin=60, fmax=500, sr=16000, frame_length=1024)
        f0 = f0[voiced & np.isfinite(f0)]
        return float(np.median(f0)) if f0.size > 10 else 0.0
    except Exception:
        return 0.0


def splice_edges(orig: np.ndarray, sr: int, gen: np.ndarray, lead: float, trail: float, fade: float = 0.012) -> np.ndarray:
    """Собрать реплику: [вступление оригинала до lead] + русская речь + [концовка оригинала от trail],
    с короткими кроссфейдами, чтобы не щёлкало."""
    a = orig[: int(lead * sr)].astype(np.float32)
    b = orig[int(trail * sr):].astype(np.float32)
    n = int(fade * sr)
    out = [a]
    g = gen.astype(np.float32).copy()
    if a.size and n and g.size > n:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        g[:n] *= ramp
        out[-1] = np.concatenate([a[:-n], a[-n:] * (1 - ramp)]) if a.size > n else a
    out.append(g)
    if b.size and n and g.size > n:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        out[-1] = np.concatenate([g[:-n], g[-n:] * (1 - ramp)])
        out.append(np.concatenate([b[:n] * ramp, b[n:]]) if b.size > n else b)
    return np.concatenate(out)


def sound_islands(data: np.ndarray, sr: int, rel_db: float = -38.0, gap: float = 0.08, min_len: float = 0.05) -> list[tuple[float, float]]:
    """Участки записи, где что-то звучит (речь, смех, глоток, вдох), — [(начало, конец)] в секундах.
    Порог — относительно самого громкого места; паузы короче gap склеиваются."""
    frame = max(1, int(sr * 0.01))
    n = len(data) // frame
    if n < 3:
        return []
    rms = np.sqrt(np.mean(data[: n * frame].astype(np.float32).reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    on = db > max(float(np.max(db)) + rel_db, -60.0)
    out, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start * 0.01, i * 0.01])
            start = None
    if start is not None:
        out.append([start * 0.01, n * 0.01])
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(round(a, 2), round(b, 2)) for a, b in merged if b - a >= min_len]


def speech_onset(data: np.ndarray, sr: int, start: float, end: float, rel_db: float = -28.0, min_ms: int = 60) -> float | None:
    """Где в отрезке [start, end] начинается речь: первый момент, где громкость держится выше порога
    (относительно пика отрезка) хотя бы min_ms. None, если не нашли."""
    seg = data[int(start * sr): int(end * sr)]
    frame = max(1, int(sr * 0.01))
    n = len(seg) // frame
    if n < 3:
        return None
    rms = np.sqrt(np.mean(seg[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = float(np.max(db)) + rel_db
    need = max(1, min_ms // 10)
    run = 0
    for i, v in enumerate(db):
        run = run + 1 if v > thr else 0
        if run >= need:
            return start + (i - need + 1) * 0.01
    return None


def speech_offset(data: np.ndarray, sr: int, start: float, end: float, rel_db: float = -28.0, min_ms: int = 60) -> float | None:
    """Где в отрезке [start, end] заканчивается речь (последний громкий участок)."""
    seg = data[int(start * sr): int(end * sr)]
    frame = max(1, int(sr * 0.01))
    n = len(seg) // frame
    if n < 3:
        return None
    rms = np.sqrt(np.mean(seg[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = float(np.max(db)) + rel_db
    need = max(1, min_ms // 10)
    run = 0
    for i in range(n - 1, -1, -1):
        run = run + 1 if db[i] > thr else 0
        if run >= need:
            return start + (i + need) * 0.01
    return None


def f0_stats(data: np.ndarray, sr: int, max_sec: float = 20.0) -> tuple[float, float]:
    """(медианная высота голоса в Гц, размах интонации в полутонах — стандартное отклонение log-f0).
    Размах показывает, насколько «живая» подача: крик и волнение — широкий, ровная речь — узкий."""
    try:
        import librosa
        w = resample(trim_silence(data, sr), sr, 16000)[: int(max_sec * 16000)]
        if len(w) < 8000:
            return 0.0, 0.0
        f0, voiced, _ = librosa.pyin(w.astype(np.float32), fmin=60, fmax=500, sr=16000, frame_length=1024)
        f0 = f0[voiced & np.isfinite(f0)]
        if f0.size <= 10:
            return 0.0, 0.0
        st = 12 * np.log2(f0)
        return float(np.median(f0)), float(np.std(st))
    except Exception:
        return 0.0, 0.0
