"""Произношение и акцент: русский распознаватель wav2vec2 (CTC, без словаря и языковой модели) записывает,
как слова звучат на самом деле — «эдо вам» вместо «это вам», «хранитзя», «рамотный». Whisper такие огрехи
молча исправляет по смыслу, поэтому акцент он не слышит (проверено: и вероятность языка, и уверенность в
тексте у чистой и акцентной речи одинаковые). Здесь сравниваем услышанное с заданным текстом по буквам и
называем конкретные слова, которые прозвучали не так — это уходит в план доработки и режиссёру.

Лишние звуки по краям (вклеенный смех, вдох) не считаются ошибкой: вставки до первого и после последнего
совпавшего слова отбрасываются."""
import difflib
import re
import threading

import numpy as np

from . import accel, audio

KEY = "judge-pron"
ACCENT_CER = 0.08     # выше — произношение заметно расходится с текстом (калибровка: чистая речь ~0.02, акцент 0.04–0.26)
STRONG_CER = 0.15

_lock = threading.Lock()
_model = None
_proc = None


def ready() -> bool:
    from . import models
    return models.path(KEY) is not None


def _load():
    global _model, _proc
    if _model is None:
        from . import models
        local = models.path(KEY)
        if not local:
            raise RuntimeError("Модель проверки произношения не скачана")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        _proc = Wav2Vec2Processor.from_pretrained(str(local))
        _model = Wav2Vec2ForCTC.from_pretrained(str(local), torch_dtype=accel.dtype()).to(accel.device()).eval()
    return _model, _proc


def unload():
    global _model, _proc
    with _lock:
        _model = _proc = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _words(t: str) -> list[str]:
    t = (t or "").lower().replace("ё", "е").replace("́", "")
    return [w for w in re.sub(r"[^а-я ]+", " ", t).split() if w]


def _lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)]


def compare(text: str, heard: str) -> dict:
    """Сравнить заданный текст с услышанным: доля ошибок по буквам (без пробелов — «первое сточника» и
    «первоисточника» не ошибка) и список слов, прозвучавших не так: [(как надо, как слышно)]."""
    want, got = _words(text), _words(heard)
    sm = difflib.SequenceMatcher(None, want, got, autojunk=False)
    ops = sm.get_opcodes()
    # вставки по краям — смех, вдох, междометия из оригинала: не ошибка
    while ops and ops[0][0] == "insert":
        ops = ops[1:]
    while ops and ops[-1][0] == "insert":
        ops = ops[:-1]
    core = []
    if ops:
        core = got[ops[0][3]:ops[-1][4]]
    a, b = "".join(want), "".join(core)
    c = _lev(a, b) / max(1, len(a))
    bad = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        w, h = " ".join(want[i1:i2]), " ".join(got[j1:j2])
        if w.replace(" ", "") == h.replace(" ", ""):
            continue  # слово просто разбито пробелом
        if tag == "insert" and len(h) <= 2:
            continue
        bad.append((w or "—", h or "—"))
    return {"cer": round(c, 3), "heard": " ".join(got), "words": bad[:6]}


def check(items: list, texts: list[str]) -> list[dict]:
    """items — пути или (wav, sr); для каждой записи — compare() с текстом."""
    import torch
    out = []
    with _lock:
        model, proc = _load()
        for p, text in zip(items, texts):
            w, sr = p if isinstance(p, tuple) else audio.read(p)
            w = audio.resample(np.asarray(w, dtype=np.float32), sr, 16000)[: 16000 * 30]
            if len(w) < 1600:
                out.append({"cer": 1.0, "heard": "", "words": []})
                continue
            inp = proc(w, sampling_rate=16000, return_tensors="pt").input_values.to(model.device, model.dtype)
            with torch.no_grad():
                ids = model(inp).logits.argmax(-1)
            out.append(compare(text, proc.batch_decode(ids)[0]))
    return out


def describe(r: dict | None) -> str:
    """«это → эдо, ароматный → рамотный» для причин и для указания режиссёру."""
    if not r or not r.get("words"):
        return ""
    return ", ".join(f"«{w}» звучит как «{h}»" if h != "—" else f"«{w}» пропало" for w, h in r["words"][:4])
