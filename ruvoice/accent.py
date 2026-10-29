"""Ударения в русском тексте перед озвучкой (библиотека ruaccent: словарь + нейросеть для омографов
вроде «за́мок/замо́к», восстановление «ё»). Без ударений нейросети речи ставят их наугад —
а это и есть половина «акцента».

Формат: Silero понимает «+» перед ударной гласной (ма+ма); CustomVoice ударения не понимает — ему только «ё».
Модели ruaccent (~150 МБ) докачиваются при первом вызове в models\\ruaccent."""
import re
import threading

from .paths import PROJECT_DIR

WORKDIR = PROJECT_DIR / "models" / "ruaccent"
ACUTE = "́"
_lock = threading.Lock()
_engine = None
_failed = False


def _load():
    global _engine, _failed
    if _engine is not None or _failed:
        return _engine
    try:
        from ruaccent import RUAccent
        from . import accel
        WORKDIR.mkdir(parents=True, exist_ok=True)
        e = RUAccent()
        e.load(omograph_model_size="turbo3.1", use_dictionary=True, workdir=str(WORKDIR),
               device="CUDA" if accel.device() == "cuda" else "CPU")
        _engine = e
    except Exception as ex:
        print(f"ударения: ruaccent недоступен ({ex}) — текст уходит без ударений")
        _failed = True
    return _engine


def available() -> bool:
    return _load() is not None


def plus(text: str) -> str:
    """Текст с «+» перед ударной гласной (как у Silero). Односложные слова без ударения."""
    if not text or not re.search(r"[А-Яа-яЁё]", text):
        return text
    with _lock:
        e = _load()
        if e is None:
            return text
        try:
            return e.process_all(text)
        except Exception as ex:
            print(f"ударения: {ex}")
            return text


def mark(text: str, engine: str) -> str:
    """Подготовить текст для движка: silero понимает «+» перед ударной гласной; CustomVoice знаки ударения
    ломают (проверено: «+» читает как «плюс», U+0301 рвёт слова, CER 0.17–0.86 против 0.00), поэтому ему
    достаётся только восстановленная «ё» — её модель читает верно."""
    t = plus(text)
    if engine == "silero":
        return t
    if engine == "acute":  # русские дообучения Qwen3-TTS (ru-stress): U+0301 после ударной гласной
        return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", lambda m: m.group(1) + ACUTE, t)
    return t.replace("+", "")


def strip(text: str) -> str:
    """Убрать знаки ударения (для сравнения распознанного текста с заданным)."""
    return (text or "").replace(ACUTE, "").replace("+", "")
