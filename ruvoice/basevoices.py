"""Базовые русские голоса (Silero TTS v4, 5 дикторов — носители русского).

Зачем: модели клонирования голоса переносят в русскую речь английскую фонетику образца — получается
«американец говорит по-русски». В режиме «без акцента» русская речь сначала строится с образца
русского диктора (чистое произношение), а затем Seed-VC переносит на неё тембр персонажа.
Базовый голос для персонажа подбирается автоматически — ближайший по эмбеддингу диктора."""
import os
import re
import threading
import urllib.request

import numpy as np

from . import audio
from .paths import DATA_DIR, PROJECT_DIR

MODEL = PROJECT_DIR / "models" / "silero-tts-ru" / "v4_ru.pt"
URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
SR = 48000
SPEAKERS = {"aidar": "мужской", "eugene": "мужской", "baya": "женский", "kseniya": "женский", "xenia": "женский"}
CLIPS_DIR = DATA_DIR / "base_voices"
# Нейтральный текст для образцов (свой, без имён и цифр): длинный — для Qwen, короткий — для F5/Chatterbox/XTTS
TEXT_LONG = ("Мы вышли на дорогу ранним утром, когда над рекой ещё держался туман. Впереди было несколько часов пути, "
             "и никто не торопился. Иногда я останавливался, чтобы послушать лес. Там, за поворотом, стояла старая "
             "мельница, и возле неё всегда пахло мокрой травой и дымом. Мы говорили о пустяках, смеялись и спорили, "
             "куда свернуть у развилки. К полудню стало тепло, и мы сели отдохнуть у воды.")
TEXT_SHORT = ("Мы вышли на дорогу ранним утром, когда над рекой ещё держался туман. Впереди было несколько часов "
              "пути, и никто не торопился.")
_TRANSLIT = {"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "дж",
             "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т",
             "u": "у", "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з"}
_model = None
_lock = threading.Lock()


def available() -> bool:
    return MODEL.exists()


def download():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    part = MODEL.with_suffix(".part")
    urllib.request.urlretrieve(URL, part)
    os.replace(part, MODEL)


def _load():
    global _model
    with _lock:
        if _model is None:
            if not available():
                download()
            import torch
            imp = torch.package.PackageImporter(str(MODEL))
            _model = imp.load_pickle("tts_models", "model")
            _model.to(torch.device("cpu"))
        return _model


def clean_text(text: str) -> str:
    """Silero читает только кириллицу: латиницу транслитерируем, лишние символы убираем."""
    out = []
    for ch in text:
        low = ch.lower()
        if low in _TRANSLIT:
            t = _TRANSLIT[low]
            out.append(t.upper() if ch.isupper() else t)
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub(r"[^А-Яа-яЁё0-9+ .,!?;:\-—…()«»\"']", " ", s)  # «+» — ударение (accent.py)
    return re.sub(r"\s+", " ", s).strip() or "Да."


def synth(text: str, speaker: str) -> tuple[np.ndarray, int]:
    model = _load()
    with _lock:
        wav = model.apply_tts(text=clean_text(text), speaker=speaker, sample_rate=SR, put_accent=True, put_yo=True)
    return np.asarray(wav, dtype=np.float32).reshape(-1), SR


def reference_clip(speaker: str, long: bool) -> dict:
    """Образец базового голоса для клонирования: {"audio", "text"}; создаётся один раз."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    path = CLIPS_DIR / f"{speaker}_{'long' if long else 'short'}.wav"
    text = TEXT_LONG if long else TEXT_SHORT
    if not path.exists():
        wav, sr = synth(text, speaker)
        audio.write(path, audio.resample(wav, sr, 24000), 24000)
    return {"audio": str(path), "text": text}


def embeddings() -> dict[str, np.ndarray]:
    """Эмбеддинги дикторов (для подбора ближайшего к персонажу), кэш в data\\base_voices."""
    from . import judge
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    cache = CLIPS_DIR / "embeddings.npz"
    if cache.exists():
        data = np.load(cache)
        if set(data.files) == set(SPEAKERS):
            return {k: data[k] for k in SPEAKERS}
    embs = {spk: judge.speaker_embed([reference_clip(spk, True)["audio"]]) for spk in SPEAKERS}
    np.savez(cache, **embs)
    return embs


def pick_for(profile_emb: np.ndarray | None) -> str:
    """Базовый голос, ближайший к голосу персонажа (по умолчанию — xenia)."""
    if profile_emb is None:
        return "xenia"
    embs = embeddings()
    return max(embs, key=lambda k: float(np.dot(embs[k], profile_emb)))
