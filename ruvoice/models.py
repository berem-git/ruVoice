"""Нейросетевые модели в models\\<имя репозитория>: обычные файлы без символических ссылок (Windows их
не любит, и папку программы можно спокойно копировать на другой компьютер). Докачка — из Hugging Face.

Наборы (tiers) подбираются под железо: видеопамять и оперативную память."""
import os
import shutil
import threading
from pathlib import Path

from .paths import PROJECT_DIR

MODELS_DIR = PROJECT_DIR / "models"
GB = 2**30
STD = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.yaml", "vocab*", "speech_tokenizer/*"]


def _m(name, repo, gb, role, file=None, patterns=None, required=None, url=None):
    """required — файлы, по которым модель считается скачанной (по умолчанию все не-glob из patterns);
    url — прямая ссылка на один файл (не Hugging Face), тогда repo=None и папка — models\\<ключ>."""
    return {"name": name, "repo": repo, "file": file, "gb": gb, "role": role, "patterns": patterns or STD,
            "required": required, "url": url}


# роль: tts — речь, llm — перевод, asr — распознавание, judge — контролёр качества и подбор голосов
ALL = {
    "tts-voxcpm": _m("VoxCPM2: голос по образцу и управляемая подача", "openbmb/VoxCPM2", 4.7, "tts",
                     patterns=["*.json", "*.safetensors", "*.pth", "tokenization_voxcpm2.py"],
                     required=["config.json", "model.safetensors", "audiovae.pth", "tokenizer.json"]),
    "tts-ru": _m("Русская речь голосом персонажа (Qwen3-TTS 1.7B, русское дообучение с ударениями)",
                 "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf", 4.3, "tts"),
    "tts-custom": _m("Запасная речь: Qwen3-TTS 1.7B CustomVoice (готовые дикторы + инструкция)", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                     4.3, "tts"),
    "silero-ru": _m("Silero TTS v5.6 — русский диктор (некоммерческая лицензия CC BY-NC)", None, 0.07, "tts",
                    url="https://models.silero.ai/models/tts/ru/v5_6_ru.pt", file="v5_6_ru.pt"),
    "omnivoice": _m("OmniVoice — русский донор и теги смеха/вздохов", "k2-fsa/OmniVoice", 2.5, "tts"),
    "llm-32b-q6": _m("Режиссёр и перевод, студия (Qwen3 32B, Q6 — почти без потерь)", "Qwen/Qwen3-32B-GGUF", 25.0, "llm",
                     file="Qwen3-32B-Q6_K.gguf"),
    "llm-32b": _m("Режиссёр и перевод, ультра (Qwen3 32B)", "Qwen/Qwen3-32B-GGUF", 18.4, "llm", file="Qwen3-32B-Q4_K_M.gguf"),
    "llm-14b": _m("Режиссёр и перевод (Qwen3 14B)", "Qwen/Qwen3-14B-GGUF", 8.4, "llm", file="Qwen3-14B-Q4_K_M.gguf"),
    "judge-emotion8": _m("Эмоция речи, 8 классов (wav2vec2 XLSR)", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                         1.3, "judge", patterns=["*.json", "pytorch_model.bin", "model.safetensors"],
                         required=["config.json"]),
    "asr": _m("Распознавание речи (Whisper large-v3-turbo)", "openai/whisper-large-v3-turbo", 1.6, "asr"),
    "llm-8b": _m("Перевод, качество (Qwen3 8B)", "Qwen/Qwen3-8B-GGUF", 4.7, "llm", file="Qwen3-8B-Q4_K_M.gguf"),
    "llm-4b": _m("Перевод, быстрый (Qwen3 4B)", "Qwen/Qwen3-4B-GGUF", 2.4, "llm", file="Qwen3-4B-Q4_K_M.gguf"),
    # контролёр качества и подбор голосов
    "judge-ecapa": _m("Отпечаток голоса ECAPA (стабильность голоса, кто говорит)", "speechbrain/spkrec-ecapa-voxceleb", 0.09,
                      "judge", patterns=["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
                                         "classifier.ckpt", "label_encoder.txt"],
                      required=["hyperparams.yaml", "embedding_model.ckpt"]),
    "judge-pron": _m("Произношение и акцент (русский wav2vec2: слышит, как слова звучат на самом деле)",
                     "jonatasgrosman/wav2vec2-large-xlsr-53-russian", 1.2, "judge",
                     patterns=["config.json", "preprocessor_config.json", "special_tokens_map.json", "vocab.json",
                               "tokenizer_config.json", "pytorch_model.bin"],
                     required=["config.json", "vocab.json", "pytorch_model.bin"]),
    "judge-ve": _m("Отпечаток голоса (похожесть, подбор диктора)", "ResembleAI/chatterbox", 0.01, "judge",
                   patterns=["ve.pt"], required=["ve.pt"]),
    "judge-events": _m("Контролёр: смех, кашель, крик (AudioSet)", "MIT/ast-finetuned-audioset-10-10-0.4593", 0.4, "judge",
                       patterns=["*.json", "model.safetensors"], required=["model.safetensors", "config.json"]),
}
JUDGES = ["asr", "judge-emotion8", "judge-events", "judge-ve", "judge-ecapa", "judge-pron"]
TIERS = {
    "studio": {
        "name": "Студия", "models": ["tts-ru", "llm-32b-q6"] + JUDGES,
        "needs": "Видеокарта с 32 ГБ видеопамяти и больше (RTX 5090, профессиональные 48 ГБ)",
        "desc": "Самый сильный режиссёр подачи, перевод и разбор «кто говорит» — Qwen3 32B почти без потерь качества; "
                "5 вариантов каждой реплики на выбор контролёру.",
        "vram": 30,
    },
    "ultra": {
        "name": "Ультра", "models": ["tts-ru", "llm-32b"] + JUDGES,
        "needs": "Видеокарта с 24 ГБ видеопамяти и больше (RTX 3090/4090)",
        "desc": "Режиссёр подачи и перевод моделью вдвое больше (Qwen3 32B) — точнее интонации и имена; "
                "5 вариантов каждой реплики на выбор контролёру.",
        "vram": 22,
    },
    "max": {
        "name": "Максимум", "models": ["tts-ru", "llm-14b"] + JUDGES,
        "needs": "Видеокарта с 14 ГБ видеопамяти и больше",
        "desc": "Русская речь голосом персонажа (по его записям), режиссёр подачи и перевод (Qwen3 14B), контролёр качества.",
        "vram": 14,
    },
    "quality": {
        "name": "Качество", "models": ["tts-ru", "llm-8b", "asr", "judge-emotion8", "judge-events", "judge-ve", "judge-ecapa", "judge-pron"],
        "needs": "Видеокарта с 8 ГБ видеопамяти и больше",
        "desc": "Та же речь, перевод моделью поменьше. ~2 с на реплику.",
    },
    "balanced": {
        "name": "Баланс", "models": ["tts-ru", "llm-4b", "asr", "judge-events", "judge-ve", "judge-ecapa", "judge-pron"],
        "needs": "Видеокарта с 6–8 ГБ видеопамяти",
        "desc": "Та же речь, лёгкий перевод, контролёр без оценки эмоций. ~2 с на реплику.",
    },
}
ADDONS: dict = {}  # дополнений нет: один конвейер. RVC (runtime\engines\rvc), Silero и ruaccent (models\) ставятся сами.
_download = {"running": False, "key": None, "queue": [], "message": "", "error": None}
_lock = threading.Lock()


def local_dir(key: str) -> Path:
    repo = ALL[key]["repo"]
    return MODELS_DIR / (repo.split("/")[1] if repo else key)


def path(key: str) -> Path | None:
    """Путь к скачанной модели (папка или файл .gguf) или None."""
    m = ALL[key]
    d = local_dir(key)
    if m["file"]:
        p = d / m["file"]
        return p if p.is_file() else None
    if m["patterns"] is STD:
        return d if any(d.glob("*.safetensors")) and (d / "config.json").exists() else None
    required = m["required"] or [pat for pat in m["patterns"] if "*" not in pat]
    if required:
        return d if all((d / r).exists() for r in required) else None
    return d if d.is_dir() and any(d.iterdir()) else None


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0


def recommend(accel_info: dict | None) -> dict:
    """Какой набор подходит под это железо и чего из него не хватает."""
    a = accel_info or {}
    vram = a.get("vram_gb") or 0
    ram = a.get("ram_gb") or 0
    gpu = a.get("device_name") or (a.get("gpus") or [None])[0]
    warnings = []
    if a.get("device") == "cuda" and vram >= TIERS["studio"]["vram"]:
        tier, why = "studio", f"{gpu}, {vram:g} ГБ видеопамяти"
    elif a.get("device") == "cuda" and vram >= TIERS["ultra"]["vram"]:
        tier, why = "ultra", f"{gpu}, {vram:g} ГБ видеопамяти"
    elif a.get("device") == "cuda" and vram >= 14:
        tier, why = "max", f"{gpu}, {vram:g} ГБ видеопамяти"
    elif a.get("device") == "cuda" and vram >= 8:
        tier, why = "quality", f"{gpu}, {vram:g} ГБ видеопамяти"
    elif a.get("device") == "cuda" and vram >= 6:
        tier, why = "balanced", f"{gpu}, {vram:g} ГБ видеопамяти — для больших моделей мало"
    else:
        # Видеокарты для нейросетей нет — озвучка невозможна; набор не предлагаем
        tier = None
        why = "видеокарта для нейросетей не найдена" if a.get("device") != "cuda" else f"{gpu}, {vram:g} ГБ видеопамяти — мало"
        if a.get("vendor") == "amd" and "rocm" not in (a.get("installed") or []):
            warnings.append("Видеокарта AMD: установите «ускорение AMD» (ROCm) — поддерживаются Radeon RX 9000, часть RX 7000 "
                            "и Ryzen AI 300/400/Max. На других AMD озвучка не работает.")
        else:
            warnings.append("Для озвучки нужна видеокарта NVIDIA с 6 ГБ видеопамяти и больше или поддерживаемая AMD.")
        return {"tier": None, "name": None, "why": why, "models": [], "missing": [], "missing_optional": [],
                "missing_gb": 0, "warnings": warnings, "no_gpu": True}
    keys = TIERS[tier]["models"]
    # Языковая модель (режиссёр подачи, перевод, имена голосов) работу не блокирует: без неё озвучка идёт
    # по официальным субтитрам без указаний режиссёра. Если уже есть модель поменьше — работаем на ней
    missing = [k for k in keys if not path(k) and ALL[k]["role"] != "llm"]
    optional = [k for k in keys if not path(k) and ALL[k]["role"] == "llm"]
    if optional:
        have = next((ALL[k]["name"] for k in ("llm-32b", "llm-14b", "llm-8b", "llm-4b") if k not in optional and path(k)), None)
        warnings.append(f"Языковая модель этого набора ({ALL[optional[0]]['name']}, {ALL[optional[0]]['gb']} ГБ) не скачана — "
                        + (f"пока работает {have}" if have else "без неё нет режиссёра подачи и перевода"))
    return {"tier": tier, "name": TIERS[tier]["name"], "why": why, "models": keys, "missing": missing,
            "missing_optional": optional,
            "missing_gb": round(sum(ALL[k]["gb"] for k in missing), 1), "warnings": warnings}


def _group_view(spec: dict, models: dict) -> dict:
    return {**spec, "total_gb": round(sum(ALL[k]["gb"] for k in spec["models"]), 1),
            "installed": all(models[k]["present"] for k in spec["models"]),
            "missing_gb": round(sum(ALL[k]["gb"] for k in spec["models"] if not models[k]["present"]), 1)}


def status(accel_info: dict | None = None) -> dict:
    models = {}
    for k, m in ALL.items():
        present = path(k) is not None
        entry = {"name": m["name"], "present": present, "size_gb": m["gb"], "role": m["role"]}
        if _download["key"] == k and _download["running"]:
            entry["progress"] = min(0.99, _dir_bytes(local_dir(k)) / (m["gb"] * GB))
        models[k] = entry
    return {"models": models,
            "tiers": {t: _group_view(v, models) for t, v in TIERS.items()},
            "addons": {t: _group_view(v, models) for t, v in ADDONS.items()},
            "download": dict(_download),
            "recommendation": recommend(accel_info) if accel_info and accel_info.get("variant") else None,
            "disk_free_gb": round(shutil.disk_usage(MODELS_DIR.parent).free / GB, 1)}


def download(key: str):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    m = ALL[key]
    d = local_dir(key)
    if m.get("url"):  # один файл по прямой ссылке (Silero): качаем в .part, потом переименовываем
        import urllib.request
        d.mkdir(parents=True, exist_ok=True)
        dest = d / m["file"]
        part = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(m["url"], str(part))
        os.replace(part, dest)
        return
    from huggingface_hub import hf_hub_download, snapshot_download
    if m["file"]:
        hf_hub_download(m["repo"], m["file"], local_dir=d)
    else:
        snapshot_download(m["repo"], local_dir=d, allow_patterns=m["patterns"])


def download_async(keys: list[str]):
    """Поставить модели в очередь скачивания (уже скачанные пропускаются)."""
    with _lock:
        for k in keys:
            if k in ALL and not path(k) and k not in _download["queue"] and _download["key"] != k:
                _download["queue"].append(k)
        if _download["running"]:
            return
        _download.update(running=True, error=None)

    def run():
        try:
            while True:
                with _lock:
                    if not _download["queue"]:
                        break
                    key = _download["queue"].pop(0)
                    _download["key"] = key
                _download["message"] = f"Скачивание: {ALL[key]['name']} (~{ALL[key]['gb']} ГБ)"
                download(key)
            _download["message"] = "Готово"
        except Exception as e:
            _download.update(error=str(e), message="Ошибка скачивания", queue=[])
        finally:
            _download.update(running=False, key=None)
    threading.Thread(target=run, daemon=True).start()


def delete(key: str):
    m = ALL[key]
    d = local_dir(key)
    if m["file"]:
        (d / m["file"]).unlink(missing_ok=True)
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    else:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(d / ".cache", ignore_errors=True)
