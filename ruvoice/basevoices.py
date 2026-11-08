"""Базовые русские голоса (Silero TTS v5.6, 5 дикторов — носители русского).

Зачем: модели клонирования голоса переносят в русскую речь английскую фонетику образца — получается
«американец говорит по-русски». Silero — источник чистого русского произношения (ступень 1 конвейера v2),
тембр персонажа на него переносит RVC. Диктор персонажу подбирается по полу и отпечатку голоса.
Лицензия моделей Silero — CC BY-NC 4.0 (некоммерческая), поэтому бэкенд включается только явным выбором."""
import re
import threading

import numpy as np

from . import audio, models
from .paths import DATA_DIR

MODEL = models.local_dir("silero-ru") / "v5_6_ru.pt"
SR = 48000
SPEAKERS = {"aidar": "мужской", "eugene": "мужской", "baya": "женский", "kseniya": "женский", "xenia": "женский"}
_GENDER = {"male": "мужской", "female": "женский"}
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


def _load():
    global _model
    with _lock:
        if _model is None:
            if not available():
                models.download("silero-ru")
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


def pick_for(profile_emb: np.ndarray | None, gender: str | None = None) -> str:
    """Диктор, ближайший к голосу персонажа; gender («male»/«female») ограничивает выбор полом.
    Без отпечатка — aidar для мужского голоса, иначе xenia."""
    pool = [k for k, g in SPEAKERS.items() if gender not in _GENDER or g == _GENDER[gender]]
    if profile_emb is None:
        return "aidar" if gender == "male" else "xenia"
    embs = embeddings()
    return max(pool, key=lambda k: float(np.dot(embs[k], profile_emb)))
