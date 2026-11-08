"""Реестр речевых бэкендов конвейера v2 и среда RVC.

Ступень 1 (источник русской речи): `ru` — Qwen3-TTS с русским дообучением (клон с русского донора или,
для эпизодических, с английской записи), `custom` — Qwen CustomVoice (готовый диктор + инструкция),
`silero` — Silero v5.6 (носители русского, без клонирования), `omnivoice` — OmniVoice (клон с донора,
теги смеха и вздохов); `voxcpm` — только быстрый путь (клон с английской записи).
Ступень 2 (тембр персонажа) — RVC (Applio) в отдельном процессе-работнике runtime\\engines\\rvc."""
from pathlib import Path

from ..paths import PROJECT_DIR

ENGINES_DIR = PROJECT_DIR / "runtime" / "engines"
MODELS_DIR = PROJECT_DIR / "models"

BACKENDS = {
    "ru": {
        "name": "Qwen3-TTS, русское дообучение (голос по образцу)", "short": "Qwen-RU",
        "desc": "Русская речь с ударениями: голос снимается с донорского русского образца (без акцента) "
                "или с записи персонажа (быстрый путь).",
        "env": None, "model_keys": ["tts-ru"], "files": [],
        "capabilities": {"stress": "acute", "style": "punct", "nonverbal_tags": False, "clone": True,
                         "cross_lingual": True, "commercial": True},
    },
    "custom": {
        "name": "Qwen3-TTS CustomVoice (готовый диктор + инструкция)", "short": "CustomVoice",
        "desc": "Готовые дикторы с инструкцией под эмоцию; тембр персонажа добавляет перенос тембра.",
        "env": None, "model_keys": ["tts-custom"], "files": [],
        "capabilities": {"stress": "none", "style": "instruct", "nonverbal_tags": False, "clone": False,
                         "cross_lingual": False, "commercial": True},
    },
    "voxcpm": {
        "name": "VoxCPM2: клон голоса с записи персонажа", "short": "VoxCPM2",
        "desc": "Клонирование с английской записи и описание подачи — только для эпизодических персонажей.",
        "env": "voxcpm", "model_keys": ["tts-voxcpm"], "files": [ENGINES_DIR / "voxcpm" / "voxcpm" / "__init__.py"],
        "capabilities": {"stress": "none", "style": "instruct", "nonverbal_tags": False, "clone": True,
                         "cross_lingual": True, "commercial": True},
    },
    "silero": {
        "name": "Silero TTS v5.6 (русские дикторы)", "short": "Silero",
        "desc": "Носители русского, чистое произношение, без клонирования и эмоций. Лицензия CC BY-NC — "
                "только по явному выбору.",
        "env": None, "model_keys": ["silero-ru"], "files": [],
        "capabilities": {"stress": "plus", "style": "punct", "nonverbal_tags": False, "clone": False,
                         "cross_lingual": False, "commercial": False},
    },
    "omnivoice": {
        "name": "OmniVoice (клон с донора, теги смеха и вздохов)", "short": "OmniVoice",
        "desc": "Клонирование с русского донора или с записи персонажа; понимает [laughter], [sigh] в тексте.",
        "env": "omnivoice", "model_keys": ["omnivoice"], "files": [ENGINES_DIR / "omnivoice" / "omnivoice" / "__init__.py"],
        "capabilities": {"stress": "none", "style": "tags", "nonverbal_tags": True, "clone": True,
                         "cross_lingual": True, "commercial": True},
    },
}
TTS_ENGINES = BACKENDS  # прежнее имя реестра
SOURCE_KEYS = ("ru", "custom", "silero", "omnivoice")   # кто умеет говорить по-русски от русского образца
CLONE_KEYS = ("ru", "voxcpm", "omnivoice")              # кто клонирует с английской записи (быстрый путь)

RVC_ENGINE = {
    "name": "RVC (перенос тембра персонажа)",
    "desc": "Обучается на записях персонажа и переносит его тембр на русскую речь, не меняя произношение.",
    "env": "rvc",
    "models": [ENGINES_DIR / "rvc" / "applio" / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0G40k.pth",
               ENGINES_DIR / "rvc" / "applio" / "rvc" / "models" / "predictors" / "rmvpe.pt",
               ENGINES_DIR / "rvc" / "applio" / "rvc" / "models" / "embedders" / "contentvec" / "pytorch_model.bin"],
}


def _env_ready(env: str | None) -> bool:
    return env is None or (ENGINES_DIR / env).is_dir()


def ready(key: str) -> bool:
    """Скачаны ли модели бэкенда и стоит ли его среда."""
    from .. import models
    spec = BACKENDS[key]
    if not _env_ready(spec["env"]):
        return False
    if not all(Path(f).exists() for f in spec["files"]):
        return False
    return all(models.path(k) is not None for k in spec["model_keys"])


def rvc_available() -> bool:
    return _env_ready(RVC_ENGINE["env"]) and all(Path(p).exists() for p in RVC_ENGINE["models"])


def available_tts() -> list[str]:
    return [k for k in BACKENDS if ready(k)]


def sources() -> list[str]:
    return [k for k in SOURCE_KEYS if ready(k)]


def cloners() -> list[str]:
    return [k for k in CLONE_KEYS if ready(k)]


def describe() -> dict:
    return {"tts": {k: {"name": s["name"], "short": s["short"], "desc": s["desc"], "env": s["env"], "ready": ready(k),
                       "capabilities": s["capabilities"]} for k, s in BACKENDS.items()},
            "rvc": {"name": RVC_ENGINE["name"], "desc": RVC_ENGINE["desc"], "ready": rvc_available()}}
