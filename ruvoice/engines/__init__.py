"""Голосовые движки одного конвейера: речь читает русское дообучение Qwen3-TTS по отпечатку голоса персонажа
(в основном процессе; запасной — CustomVoice), тембр персонажа добавляет обученная RVC-модель (Applio) в отдельном процессе-работнике
runtime\\engines\\rvc со своими версиями библиотек (см. worker.py / client.py)."""
from pathlib import Path

from ..paths import PROJECT_DIR

ENGINES_DIR = PROJECT_DIR / "runtime" / "engines"
MODELS_DIR = PROJECT_DIR / "models"

TTS_ENGINES = {
    "voxcpm": {
        "name": "VoxCPM2: русский голос и подача по оригиналу", "short": "VoxCPM2",
        "desc": "Клонирование голоса и описание эмоции, темпа и исполнения. Обучение под каждую игру не требуется.",
        "env": "voxcpm", "models": [MODELS_DIR / "VoxCPM2" / "config.json",
            MODELS_DIR / "VoxCPM2" / "model.safetensors", MODELS_DIR / "VoxCPM2" / "audiovae.pth",
            ENGINES_DIR / "voxcpm" / "voxcpm" / "__init__.py"],
        "capabilities": {"clone": True, "style": True, "stress_marks": False, "requires_training": False},
    },
    "ru": {
        "name": "Qwen3-TTS, русское дообучение (голос по записям персонажа)", "short": "Qwen-RU",
        "desc": "Основной способ: русская речь с ударениями, голос снимается прямо с записей персонажа (отпечаток), "
                "под эмоцию подбирается образец из его же реплик.",
        "env": None, "models": [MODELS_DIR / "qwen3-tts-12hz-1.7b-ru-stress-cf" / "config.json"],
        "capabilities": {"clone": True, "style": False, "stress_marks": True, "requires_training": False},
    },
    "custom": {
        "name": "Qwen3-TTS CustomVoice (эмоция по инструкции)", "short": "CustomVoice",
        "desc": "Запасной способ: готовые дикторы с инструкцией под эмоцию; тембр добавляет обученный RVC-голос.",
        "env": None, "models": [MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice" / "config.json"],
        "capabilities": {"clone": False, "style": True, "stress_marks": False, "requires_training": False},
    },
}

RVC_ENGINE = {
    "name": "RVC (обучение голоса персонажа)",
    "desc": "Обучается на записях персонажа и переносит его тембр на русскую речь, не меняя произношение.",
    "env": "rvc",
    "models": [ENGINES_DIR / "rvc" / "applio" / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0G40k.pth",
               ENGINES_DIR / "rvc" / "applio" / "rvc" / "models" / "predictors" / "rmvpe.pt",
               ENGINES_DIR / "rvc" / "applio" / "rvc" / "models" / "embedders" / "contentvec" / "pytorch_model.bin"],
}


def _ready(spec: dict) -> bool:
    if spec.get("env") and not (ENGINES_DIR / spec["env"]).is_dir():
        return False
    return all(Path(p).exists() for p in spec["models"])


def rvc_available() -> bool:
    return _ready(RVC_ENGINE)


def available_tts() -> list[str]:
    return [k for k, s in TTS_ENGINES.items() if _ready(s)]


def describe() -> dict:
    return {"tts": {k: {"name": s["name"], "short": s["short"], "desc": s["desc"], "ready": _ready(s),
                       "capabilities": s.get("capabilities", {})}
                    for k, s in TTS_ENGINES.items()},
            "rvc": {"name": RVC_ENGINE["name"], "desc": RVC_ENGINE["desc"], "ready": rvc_available()}}
