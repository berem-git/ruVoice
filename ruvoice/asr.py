"""Распознавание речи (Whisper): текст оригинальных реплик, когда у игры нет субтитров."""
import threading

from . import accel, models

_pipe = None
_lock = threading.Lock()


def _input(path: str) -> dict:
    # Supplying samples makes the portable app independent of ffmpeg in the system PATH.
    from . import audio
    w, sr = audio.read(path)
    return {"raw": audio.resample(w, sr, 16000), "sampling_rate": 16000}


def _load():
    global _pipe
    if _pipe is None:
        accel.activate()
        from transformers import pipeline
        local = models.path("asr")
        if not local:
            raise RuntimeError("Модель распознавания речи не скачана — откройте «Система и ускорение»")
        _pipe = pipeline("automatic-speech-recognition", model=str(local), dtype=accel.dtype(),
                         device=accel.device())
    return _pipe


def transcribe(paths: list[str], language: str | None = "english", batch_size: int = 8) -> list[str]:
    """Короткие записи (до 30 с) идут пачкой; длинные Whisper режет на куски и обрабатывает по одной —
    в одной пачке их смешивать нельзя (ошибка «elements of the batch contain different keys»)."""
    import soundfile as sf
    paths = [str(p) for p in paths]
    short, long_ = [], []
    for i, p in enumerate(paths):
        try:
            d = sf.info(p).duration
        except Exception:
            d = 0.0
        (long_ if d > 29.5 else short).append(i)
    out = [""] * len(paths)
    with _lock:
        pipe = _load()
        kwargs = {"task": "transcribe"}
        if language:
            kwargs["language"] = language
        if short:
            res = pipe([_input(paths[i]) for i in short], batch_size=batch_size, generate_kwargs=kwargs)
            for i, r in zip(short, res):
                out[i] = r["text"].strip()
        for i in long_:
            r = pipe(_input(paths[i]), chunk_length_s=30, batch_size=batch_size, generate_kwargs=kwargs)
            out[i] = r["text"].strip()
    return out


def words(path: str, language: str | None = "english") -> list[dict]:
    """Word boundaries for source-event preservation. Never infer them from file names."""
    kwargs = {"task": "transcribe"}
    if language:
        kwargs["language"] = language
    with _lock:
        result = _load()(_input(path), return_timestamps="word", generate_kwargs=kwargs)
    out = []
    for chunk in result.get("chunks", []):
        a, b = chunk.get("timestamp") or (None, None)
        if a is not None and b is not None and b > a:
            out.append({"start": float(a), "end": float(b), "text": chunk.get("text", "").strip()})
    return out


def language_prob(paths: list, lang: str = "ru") -> list[float]:
    """Насколько запись «звучит» на языке lang: вероятность языка по Whisper (первый шаг декодера, токены
    языков). Речь с акцентом получает заметно меньшую вероятность русского, хотя слова распознаются верно —
    так контролёр слышит акцент. Принимает пути или (wav, sr)."""
    import numpy as np
    import torch
    from . import audio
    out = []
    with _lock:
        pipe = _load()
        model, fe, tok = pipe.model, pipe.feature_extractor, pipe.tokenizer
        lang_ids = [tok.convert_tokens_to_ids(t) for t in tok.additional_special_tokens
                    if len(t) in (6, 7) and t.startswith("<|") and t.endswith("|>") and t[2:-2].isalpha() and t[2:-2].islower()]
        target = tok.convert_tokens_to_ids(f"<|{lang}|>")
        sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        for p in paths:
            w, sr = p if isinstance(p, tuple) else audio.read(p)
            w = audio.resample(np.asarray(w, dtype=np.float32), sr, 16000)[: 16000 * 30]
            feats = fe(w, sampling_rate=16000, return_tensors="pt").input_features.to(model.device, model.dtype)
            with torch.no_grad():
                enc = model.model.encoder(feats)
                logits = model(encoder_outputs=enc, decoder_input_ids=torch.tensor([[sot]], device=model.device)).logits[0, -1]
            probs = torch.softmax(logits[lang_ids].float(), dim=-1)
            out.append(float(probs[lang_ids.index(target)]) if target in lang_ids else 0.0)
    return out


def text_nll(items: list, texts: list[str], lang: str = "ru") -> list[float]:
    """Произношение: насколько Whisper «верит», что в записи звучит именно этот текст по-русски — средняя
    потеря на токен при принудительном декодировании заданного текста. Акцент, смазанные слова и чужая
    фонетика повышают её, даже когда слова распознаются верно. items — пути или (wav, sr)."""
    import numpy as np
    import torch
    from . import audio
    out = []
    with _lock:
        pipe = _load()
        model, fe, tok = pipe.model, pipe.feature_extractor, pipe.tokenizer
        prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", f"<|{lang}|>", "<|transcribe|>", "<|notimestamps|>"])
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        for p, text in zip(items, texts):
            w, sr = p if isinstance(p, tuple) else audio.read(p)
            w = audio.resample(np.asarray(w, dtype=np.float32), sr, 16000)[: 16000 * 30]
            feats = fe(w, sampling_rate=16000, return_tensors="pt").input_features.to(model.device, model.dtype)
            ids = tok(" " + text.strip(), add_special_tokens=False).input_ids + [eot]
            dec = torch.tensor([prefix + ids[:-1]], device=model.device)
            with torch.no_grad():
                logits = model(input_features=feats, decoder_input_ids=dec).logits[0, len(prefix) - 1:].float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor(ids, device=model.device)
            out.append(float(-lp[torch.arange(len(ids)), tgt].mean()))
    return out


def unload():
    global _pipe
    with _lock:
        _pipe = None
    _free()


def _free():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
