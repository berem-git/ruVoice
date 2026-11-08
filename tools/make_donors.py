"""Сборка донорского банка (ruvoice/donors): русские голоса-образцы для ступени 1 конвейера v2.

Запускает разработчик, один раз, на видеокарте (из папки программы):
    runtime\\python\\python.exe tools\\make_donors.py --out ruvoice\\donors --design --styles
    runtime\\python\\python.exe tools\\make_donors.py --out ruvoice\\donors --wavs <папка с .wav и .txt> --styles

--design — голоса создаёт Qwen3-TTS VoiceDesign по текстовому описанию (16 описаний ниже: пол × высота ×
  возраст), текст — basevoices.TEXT_SHORT; клипы 6–10 с, 24 кГц, моно, FLAC.
--wavs  — человеческие записи: каждому <имя>.wav нужен <имя>.txt с расшифровкой; пол — по высоте голоса.
--styles — к каждому донору подачи angry и sad тем же голосом: Qwen-RU клонирует спокойный клип (ICL)
  и читает эмоциональную фразу с пунктуацией «!»/«…» (temperature 0.9 / 0.7).
Клипы никогда не делаются из английских записей персонажей — иначе акцент вернётся в банк."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

SR = 24000
MIN_SEC, MAX_SEC = 6.0, 10.0
FEMALE_F0 = 165.0
DESIGNS = [  # (id, пол, возраст, описание голоса для VoiceDesign)
    ("m_low_adult", "male", "adult", "A deep, low, calm adult male voice, warm and unhurried."),
    ("m_low_old", "male", "old", "A low, slightly hoarse elderly male voice, slow and weighty."),
    ("m_low_young", "male", "young", "A low young male voice, soft and relaxed."),
    ("m_mid_young", "male", "young", "A clear mid-range young male voice, lively and friendly."),
    ("m_mid_adult", "male", "adult", "A neutral mid-range adult male voice, confident and even."),
    ("m_mid_old", "male", "old", "A mid-range elderly male voice, tired and measured."),
    ("m_high_young", "male", "young", "A high, light young male voice, energetic, almost boyish."),
    ("m_high_adult", "male", "adult", "A high-pitched adult male voice, tense and nervous."),
    ("f_low_adult", "female", "adult", "A low, velvety adult female voice, calm and composed."),
    ("f_low_old", "female", "old", "A low elderly female voice, slow and kind."),
    ("f_low_young", "female", "young", "A low young female voice, soft and husky."),
    ("f_mid_young", "female", "young", "A clear mid-range young female voice, bright and friendly."),
    ("f_mid_adult", "female", "adult", "A neutral mid-range adult female voice, confident and even."),
    ("f_mid_old", "female", "old", "A mid-range elderly female voice, strict and dry."),
    ("f_high_young", "female", "young", "A high, light young female voice, playful and quick."),
    ("f_high_adult", "female", "adult", "A high adult female voice, anxious and breathy."),
]
STYLE_TEXTS = {
    "angry": "Хватит! Убирайся отсюда немедленно, пока не стало хуже!",
    "sad": "Всё кончено… Я не знаю, как теперь жить дальше, правда не знаю…",
}
STYLE_TEMPERATURE = {"angry": 0.9, "sad": 0.7}


def cut(wav: np.ndarray, sr: int) -> np.ndarray:
    """Обрезать тишину и уложить клип в MIN_SEC–MAX_SEC, заканчивая на паузе."""
    from ruvoice import audio
    wav = audio.trim_silence(wav, sr)
    if len(wav) <= MAX_SEC * sr:
        return wav
    islands = audio.sound_islands(wav, sr)
    ends = [b for _, b in islands if MIN_SEC <= b <= MAX_SEC]
    end = max(ends) if ends else MAX_SEC
    out = wav[: int(end * sr)].copy()
    fade = min(len(out), int(sr * 0.03))
    out[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return out


def write_flac(path: Path, wav: np.ndarray, sr: int) -> float:
    """FLAC 24 кГц моно; возвращает медианную высоту голоса."""
    import soundfile as sf
    from ruvoice import audio
    wav = audio.resample(np.asarray(wav, dtype=np.float32), sr, SR)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1, 1), SR, format="FLAC")
    return audio.f0_median(wav, SR)


def design(out: Path, count: int) -> list[dict]:
    """Голоса по описанию: VoiceDesign читает TEXT_SHORT; пол проверяется по высоте, иначе другое зерно."""
    import torch
    from ruvoice import accel, basevoices, models, tts
    accel.activate()
    if not models.path("tts-design"):
        print("скачиваю Qwen3-TTS VoiceDesign…", flush=True)
        models.download("tts-design")
    tts._load("tts-design")
    rows = []
    for i, (did, gender, age, instruct) in enumerate(DESIGNS[:count]):
        for attempt in range(4):
            torch.manual_seed(1000 * i + 17 * attempt)
            wavs, sr = tts._model.generate_voice_design(text=[basevoices.TEXT_SHORT], language=["Russian"],
                                                       instruct=[instruct], max_new_tokens=600)
            wav = cut(np.asarray(wavs[0], dtype=np.float32).reshape(-1), sr)
            f0 = write_flac(out / f"{did}_calm.flac", wav, sr)
            ok = f0 and ((gender == "female") == (f0 >= FEMALE_F0)) and len(wav) / sr >= MIN_SEC
            print(f"{did}: попытка {attempt + 1}, {len(wav) / sr:.1f} с, {f0:.0f} Гц{' ✓' if ok else ''}", flush=True)
            if ok:
                break
        rows.append({"id": did, "gender": gender, "age": age, "f0": round(float(f0), 1),
                     "source": "Qwen3-TTS VoiceDesign по описанию", "license": "Apache-2.0",
                     "clips": {"calm": {"file": f"{did}_calm.flac", "text": basevoices.TEXT_SHORT}}})
    tts.unload()
    return rows


def from_wavs(out: Path, folder: Path, count: int) -> list[dict]:
    """Человеческие записи <имя>.wav + <имя>.txt; пол — по высоте голоса."""
    from ruvoice import audio
    rows = []
    for src in sorted(folder.glob("*.wav"))[:count]:
        text_file = src.with_suffix(".txt")
        if not text_file.is_file():
            print(f"{src.name}: нет расшифровки .txt — пропуск", flush=True)
            continue
        wav, sr = audio.read(src)
        wav = cut(wav, sr)
        if len(wav) / sr < MIN_SEC:
            print(f"{src.name}: короче {MIN_SEC:.0f} с — пропуск", flush=True)
            continue
        did = "".join(ch if ch.isalnum() else "_" for ch in src.stem.lower())[:24]
        f0 = write_flac(out / f"{did}_calm.flac", wav, sr)
        rows.append({"id": did, "gender": "female" if f0 >= FEMALE_F0 else "male", "age": "adult", "f0": round(float(f0), 1),
                     "source": f"запись {src.name}", "license": "укажите лицензию записи",
                     "clips": {"calm": {"file": f"{did}_calm.flac", "text": text_file.read_text("utf-8").strip()}}})
        print(f"{did}: {len(wav) / sr:.1f} с, {f0:.0f} Гц", flush=True)
    return rows


def add_styles(out: Path, rows: list[dict]):
    """Подачи angry/sad тем же голосом: Qwen-RU клонирует спокойный клип (ICL) и читает эмоциональную фразу."""
    from ruvoice import accel, tts
    accel.activate()
    for i, row in enumerate(rows):
        calm = row["clips"]["calm"]
        ref = {"audio": str(out / calm["file"]), "text": calm["text"], "language": "ru"}
        for j, (style, text) in enumerate(STYLE_TEXTS.items()):
            (wav, sr), = tts.synthesize("ru", [text], [ref], [12.0], {"temperature": STYLE_TEMPERATURE[style],
                                                                     "subtalker_temperature": 0.5}, seed=5000 + 10 * i + j)
            write_flac(out / f"{row['id']}_{style}.flac", cut(wav, sr), sr)
            row["clips"][style] = {"file": f"{row['id']}_{style}.flac", "text": text}
            print(f"{row['id']}: {style} ✓", flush=True)
    tts.unload()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--design", action="store_true")
    src.add_argument("--wavs", type=Path)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--styles", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = design(args.out, args.count) if args.design else from_wavs(args.out, args.wavs, args.count)
    if args.styles:
        add_styles(args.out, rows)
    manifest = {"version": 1, "created_at": time.time(), "sample_rate": SR, "donors": rows}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    print(f"готово: доноров {len(rows)}, {sum(len(r['clips']) for r in rows)} клипов → {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
