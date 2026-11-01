"""Синтез русской речи голосом персонажа — один конвейер:
  • речь: русское дообучение Qwen3-TTS (ударения, голос по отпечатку с записей персонажа; образец под эмоцию);
    запасной — CustomVoice (готовые дикторы + инструкция);
  • по желанию тембр: обученная RVC-модель персонажа (engines/worker.py, отдельный процесс)."""
import contextlib
import io
import tempfile
import threading
from pathlib import Path

import numpy as np

from . import accel, audio, models
from .engines import available_tts, client, rvc_available

TOKENS_PER_SEC = 12

_model = None
_model_key = None
_prompts: dict[tuple, object] = {}
_lock = threading.RLock()


def _load(key: str):
    """Одна модель Qwen3-TTS в памяти: tts-ru (основная) или tts-custom (запасная)."""
    global _model, _model_key
    with _lock:
        if _model is not None and _model_key == key:
            return _model
        unload()
        accel.activate()
        with contextlib.redirect_stdout(io.StringIO()):  # qwen_tts при импорте ругается на SoX/flash-attn
            from qwen_tts import Qwen3TTSModel
        local = key if key not in models.ALL else models.path(key)  # ключ модели или путь к чекпойнту персонажа
        if not local or not Path(local).exists():
            raise RuntimeError(f"Модель речи {models.ALL[key]['name'] if key in models.ALL else key} не найдена")
        dev = accel.device()
        _model = Qwen3TTSModel.from_pretrained(str(local), device_map="cuda:0" if dev == "cuda" else "cpu",
                                               dtype=accel.dtype(), attn_implementation="sdpa")
        _model_key = key
        return _model


def load_ru():
    """Русское дообучение Qwen3-TTS: голос по образцу персонажа (отпечаток), ударения U+0301 — основной способ."""
    return _load("tts-ru")


def load_custom():
    """Qwen3-TTS CustomVoice: готовые дикторы + инструкция («сердито», «шёпотом»…) — запасной способ."""
    return _load("tts-custom")


def unload():
    """Выгрузить модель речи и остановить процесс RVC (освободить видеопамять)."""
    global _model, _model_key
    with _lock:
        _model, _model_key = None, None
        _prompts.clear()
        _attached.clear()
    client.stop()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def custom_speakers() -> list[str]:
    m = load_custom()
    spks = m.model.get_supported_speakers() if hasattr(m.model, "get_supported_speakers") else []
    return sorted(spks)


def _custom_generate(texts, refs, max_seconds, params) -> list[tuple[np.ndarray, int]]:
    load_custom()
    speakers = [r.get("speaker") or params.get("speaker") or custom_speakers()[0] for r in refs]
    instruct = params.get("instruct") or ""
    with _lock:
        limit = int(max(max_seconds) * TOKENS_PER_SEC) + 24
        wavs, sr = _model.generate_custom_voice(text=texts, speaker=speakers, language=["Russian"] * len(texts),
                                                instruct=[instruct] * len(texts), max_new_tokens=limit,
                                                temperature=float(params.get("temperature", 0.8)),
                                                repetition_penalty=1.05)
    return [(np.asarray(w, dtype=np.float32).reshape(-1), sr) for w in wavs]


def _ru_prompt(ref_audio: str, ref_text: str | None):
    """Отпечаток голоса по образцу (x-vector). С ref_text модель ещё и подражает манере образца (ICL) —
    на английских записях это возвращает акцент, поэтому по умолчанию только отпечаток."""
    key = (ref_audio, ref_text)
    with _lock:
        if key not in _prompts:
            # Prompts may contain GPU tensors. Keep only the active reference in VRAM.
            _prompts.clear()
            _prompts[key] = _model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text or None,
                                                             x_vector_only_mode=not ref_text)[0]
        return _prompts[key]


_attached: set[str] = set()


def _use_voice(ref: dict | None):
    """Включить дообученный под персонажа адаптер (ref["voice"], ref["name"]) или чистую модель."""
    from . import ttsft
    talker = _model.model.talker
    name = (ref or {}).get("name") if (ref or {}).get("voice") else None
    if name and name not in _attached:
        ttsft.attach(talker, ref["voice"], name)
        _attached.add(name)
    if _attached:
        ttsft.activate(talker, name)


def _full_generate(texts, ref, max_seconds, params) -> list[tuple[np.ndarray, int]]:
    """Полностью дообученная под персонажа модель (ref["ckpt"], диктор ref["speaker"])."""
    _load(ref["ckpt"])
    with _lock:
        limit = int(max(max_seconds) * TOKENS_PER_SEC) + 24
        wavs, sr = _model.generate_custom_voice(text=texts, speaker=[ref["speaker"]] * len(texts),
                                                language=["Russian"] * len(texts), max_new_tokens=limit,
                                                temperature=float(params.get("temperature", 0.8)), repetition_penalty=1.05)
    return [(np.asarray(w, dtype=np.float32).reshape(-1), sr) for w in wavs]


def _ru_generate(texts, refs, max_seconds, params) -> list[tuple[np.ndarray, int]]:
    if refs and refs[0].get("ckpt"):
        return _full_generate(texts, refs[0], max_seconds, params)
    load_ru()
    icl = bool(params.get("icl"))
    _use_voice(refs[0] if refs else None)  # реплики одного вызова — одного персонажа
    prompts = [_ru_prompt(r["audio"], r.get("text") if icl or (r.get("identity") and r.get("language") == "ru") else None) for r in refs]
    # Дообученный адаптер учился на раскладке без явного языка (language=auto: 3 служебных токена вместо 4) —
    # с токеном «Russian» он ломает речь; язык модель и так берёт из русского текста
    lang = params.get("language") or ("auto" if any(r.get("voice") for r in refs) else "Russian")
    with _lock:
        limit = int(max(max_seconds) * TOKENS_PER_SEC) + 24
        # подвыборка кодов тембра (sub-talker) по умолчанию 0.9 — главный источник «плавающего» голоса между
        # репликами; интонацию задаёт основной поток, поэтому тембр можно держать строже
        extra = {k: params[k] for k in ("subtalker_temperature", "top_k", "top_p", "subtalker_top_k") if k in params}
        wavs, sr = _model.generate_voice_clone(text=texts, language=[lang] * len(texts), voice_clone_prompt=prompts,
                                               max_new_tokens=limit, temperature=float(params.get("temperature", 0.8)),
                                               repetition_penalty=1.05, **extra)
    return [(np.asarray(w, dtype=np.float32).reshape(-1), sr) for w in wavs]


def synthesize(engine: str, texts: list[str], refs: list[dict], max_seconds: list[float],
               params: dict | None = None, seed: int | None = None) -> list[tuple[np.ndarray, int]]:
    """refs[i] = {"speaker": диктор движка}. Ударения расставляются автоматически (accent.py).
    Возвращает [(wav, sr)]."""
    from . import accent
    params = params or {}
    if engine not in available_tts():
        raise RuntimeError(f"Движок речи {engine} не установлен")
    if seed is not None:
        import torch
        torch.manual_seed(seed)
    if engine == "voxcpm":
        if _model is not None:
            unload()
        outputs = []
        for text, ref, seconds in zip(texts, refs, max_seconds):
            if not ref or not ref.get("audio"):
                raise ValueError("VoxCPM2 нужен чистый образец голоса персонажа")
            with tempfile.TemporaryDirectory() as td:
                dst = Path(td) / "generated.wav"
                client.get("voxcpm").request({"op": "generate", "text": accent.mark(text, "custom"),
                    "ref_audio": ref["audio"], "out": str(dst), "params": params,
                    "russian_anchor_text": ref.get("text") if ref.get("identity") and ref.get("language") == "ru" else None,
                    "max_seconds": seconds, "seed": seed})
                outputs.append(audio.read(dst))
        return outputs
    if engine == "ru":  # refs[i] = {"audio": образец голоса персонажа, "text": его расшифровка (для ICL)}
        texts = [accent.mark(t, "acute") for t in texts]
        return _ru_generate(texts, refs, max_seconds, params)
    if engine == "custom":
        texts = [accent.mark(t, "custom") for t in texts]
        return _custom_generate(texts, refs, max_seconds, params)
    raise RuntimeError(f"Неизвестный движок речи {engine}")


def convert_timbre(wav: np.ndarray, sr: int, voice: dict | None, params: dict | None = None) -> tuple[np.ndarray, int]:
    """Перекрасить реплику в тембр персонажа обученным голосом {"method": "rvc", "pth", "index"}.
    Без обученного голоса речь остаётся голосом диктора, подобранного под персонажа."""
    if not voice or not voice.get("pth"):
        return wav, sr
    if not rvc_available():
        raise RuntimeError("RVC не установлен")
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "src.wav", Path(td) / "dst.wav"
        audio.write(src, wav, sr, "FLOAT")
        client.get("rvc").request({"op": "convert", "source": str(src), "out": str(dst), "pth": voice["pth"],
                                   "index": voice.get("index"), "params": params or {}})
        return audio.read(dst)


def train_voice_rvc(dataset_dir: str, out_dir: str, model_name: str, epochs: int, on_progress=None,
                    batch_size: int = 8) -> dict:
    """Обучить RVC-модель голоса персонажа; возвращает {"pth", "index"}."""
    if not rvc_available():
        raise RuntimeError("RVC не установлен")
    r = client.get("rvc").request({"op": "train", "dataset_dir": dataset_dir, "out_dir": out_dir,
                                   "model_name": model_name, "epochs": epochs, "batch_size": batch_size},
                                  timeout=3600 * 3, on_progress=on_progress)
    return {"pth": r["pth"], "index": r.get("index")}
