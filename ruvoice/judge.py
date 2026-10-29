"""Контролёр качества: вторая «пара ушей» после озвучки.

Для каждой русской реплики проверяет:
  • текст   — Whisper распознаёт русскую речь, сравниваем с заданным текстом (доля ошибок, CER);
  • голос   — похожесть на персонажа по эмбеддингу диктора (VoiceEncoder, как у Resemble);
  • эмоция  — классификатор эмоций на оригинале и на русском: совпадает ли настроение;
  • события — в оригинале смех, кашель, вздох, крик, шёпот, плач? TTS их не воспроизводит — предупреждаем;
  • длина/клиппинг — сильно длиннее или короче оригинала, перегруз.
Итог — оценка 0..100, вердикт ok / rework и список причин по-русски."""
import pathlib
import re
import threading

import numpy as np

from . import accel, audio
from .paths import PROJECT_DIR

VE_WEIGHTS = PROJECT_DIR / "models" / "chatterbox" / "ve.pt"
EMOTION8_KEY = "judge-emotion8"                            # 8 классов эмоций (wav2vec2 XLSR)
EVENTS_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SPEECH_LABELS = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
                 "Conversation", "Narration, monologue", "Speech synthesizer", "Shout", "Yell", "Whispering",
                 "Singing", "Child singing", "Chant"}
# Единые имена эмоций для обеих моделей
EMOTION_NORM = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad",
                "angry": "angry", "calm": "calm", "disgust": "disgust", "fearful": "fearful", "happy": "happy",
                "neutral": "neutral", "surprised": "surprised"}
EVENT_LABELS = {
    "Laughter": "смех", "Giggle": "смешок", "Chuckle, chortle": "усмешка", "Snicker": "хихиканье",
    "Cough": "кашель", "Throat clearing": "прочистка горла", "Sigh": "вздох", "Gasp": "вздох (резкий)",
    "Scream": "крик", "Shout": "выкрик", "Yell": "крик", "Whispering": "шёпот", "Crying, sobbing": "плач",
    "Whimper": "всхлип", "Groan": "стон", "Grunt": "кряхтение", "Breathing": "дыхание", "Pant": "одышка",
    "Sneeze": "чихание", "Burping, eructation": "отрыжка", "Hiccup": "икота", "Humming": "мычание",
    "Singing": "пение", "Screaming": "крик",
}
EMOTION_RU = {"neutral": "нейтрально", "happy": "радость", "angry": "злость", "sad": "грусть", "calm": "спокойно",
              "fearful": "страх", "surprised": "удивление", "disgust": "отвращение"}

_lock = threading.Lock()
_ve = None
_emotion = None
_events = None
_ecapa = None
ECAPA_KEY = "judge-ecapa"   # ECAPA-TDNN (speechbrain): различает голоса заметно лучше VoiceEncoder


def ecapa_ready() -> bool:
    from . import models
    return models.path(ECAPA_KEY) is not None


def _load_ecapa():
    """ECAPA-TDNN из models\\spkrec-ecapa-voxceleb (без сети: источник — локальная папка, файлы копируются)."""
    global _ecapa
    if _ecapa is None:
        from . import models
        local = models.path(ECAPA_KEY)
        if not local:
            raise RuntimeError("Модель отпечатка голоса ECAPA не скачана")
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        _ecapa = EncoderClassifier.from_hparams(source=str(local), savedir=str(local),
                                                run_opts={"device": "cuda" if accel.device() == "cuda" else "cpu"},
                                                local_strategy=LocalStrategy.COPY)
    return _ecapa


def ecapa_embed(paths: list, max_sec: float = 15.0) -> np.ndarray:
    """Нормированные отпечатки голоса ECAPA для файлов (строка на файл). Принимает и (wav, sr)."""
    import torch
    out = []
    with _lock:
        enc = _load_ecapa()
        for p in paths:
            w, sr = p if isinstance(p, tuple) else audio.read(p)
            w = audio.resample(np.asarray(w, dtype=np.float32), sr, 16000)[: int(16000 * max_sec)]
            if len(w) < 1600:
                w = np.pad(w, (0, 1600 - len(w)))
            with torch.no_grad():
                e = enc.encode_batch(torch.from_numpy(w).unsqueeze(0).to(enc.device)).squeeze().float().cpu().numpy()
            out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.stack(out)


def _load_ve():
    global _ve
    if _ve is None:
        import torch
        from .thirdparty.voice_encoder import VoiceEncoder
        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(VE_WEIGHTS, map_location="cpu", weights_only=True))
        _ve = ve.to(accel.device()).eval()
    return _ve


def _pipeline(task, model):
    from . import hfpatch
    hfpatch.install()
    from transformers import pipeline
    return pipeline(task, model=model, device=0 if accel.device() == "cuda" else -1)


class _EmotionHead:
    """Классификатор ehcalabres обучен со своей головой (classifier.dense → tanh → classifier.output, среднее по
    кадрам), а стандартный Wav2Vec2ForSequenceClassification ждёт projector/classifier — и молча создавал
    случайную голову (в журнале: «newly initialized»). Собираем модель вручную и грузим веса как есть."""

    def __init__(self, local: str):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Model
        cfg = AutoConfig.from_pretrained(local)
        self.labels = [cfg.id2label[i] if i in cfg.id2label else cfg.id2label[str(i)] for i in range(len(cfg.id2label))]
        self.fe = AutoFeatureExtractor.from_pretrained(local)
        self.body = Wav2Vec2Model(cfg)
        h = cfg.hidden_size
        self.dense = torch.nn.Linear(h, h)
        self.out = torch.nn.Linear(h, len(self.labels))
        sd = load_file(str(pathlib.Path(local) / "model.safetensors"))
        body_sd = {k[len("wav2vec2."):]: v for k, v in sd.items() if k.startswith("wav2vec2.")}
        missing, unexpected = self.body.load_state_dict(body_sd, strict=False)
        bad = [k for k in missing if not k.startswith("masked_spec_embed")]
        if bad or unexpected:
            raise RuntimeError(f"веса эмоций не совпали: {bad[:3]} {unexpected[:3]}")
        self.dense.load_state_dict({"weight": sd["classifier.dense.weight"], "bias": sd["classifier.dense.bias"]})
        self.out.load_state_dict({"weight": sd["classifier.output.weight"], "bias": sd["classifier.output.bias"]})
        self.dev = accel.device()
        for m in (self.body, self.dense, self.out):
            m.to(self.dev).eval()

    def __call__(self, clip: dict, top_k: int = 8):
        import torch
        inp = self.fe(clip["raw"], sampling_rate=clip["sampling_rate"], return_tensors="pt")
        with torch.no_grad():
            hs = self.body(inp["input_values"].to(self.dev)).last_hidden_state.mean(1)
            logits = self.out(torch.tanh(self.dense(hs)))
            probs = torch.softmax(logits, -1)[0].cpu()
        order = torch.argsort(probs, descending=True)[:top_k]
        return [{"label": self.labels[int(i)], "score": float(probs[i])} for i in order]


def _load_emotion():
    global _emotion
    if _emotion is None:
        from . import models
        local = models.path(EMOTION8_KEY)
        if not local:
            raise RuntimeError("Модель эмоций не скачана")
        _emotion = _EmotionHead(str(local))
    return _emotion


def _load_events():
    global _events
    if _events is None:
        _events = _pipeline("audio-classification", EVENTS_MODEL)
    return _events


def unload():
    global _ve, _emotion, _events, _ecapa
    with _lock:
        _ve = _emotion = _events = _ecapa = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --- отдельные измерения ---

def _norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9 ]+", "", re.sub(r"\s+", " ", s)).strip()


def cer(target: str, heard: str) -> float:
    a, b = _norm(target), _norm(heard)
    if not a:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / len(a)


def speaker_embed(paths: list[str]) -> np.ndarray:
    """Эмбеддинг диктора по нескольким файлам (усреднённый, нормированный)."""
    import torch
    from .thirdparty.voice_encoder import VoiceEncoder
    wavs = []
    for p in paths:
        w, sr = audio.read(p)
        wavs.append(audio.resample(w, sr, 16000))
    with _lock, torch.no_grad():
        embs = _load_ve().embeds_from_wavs(wavs, sample_rate=16000, as_spk=False)
    return VoiceEncoder.utt_to_spk_embed(embs)


def voice_similarity(path: str, profile: np.ndarray) -> float:
    e = speaker_embed([path])
    return float(np.dot(e, profile))


def _clip16k(path: str, max_sec: float = 20.0):
    w, sr = audio.read(path)
    w = audio.resample(w, sr, 16000)[: int(max_sec * 16000)]
    return {"raw": w.astype(np.float32), "sampling_rate": 16000}


def emotion(path: str) -> tuple[str, float]:
    """Эмоция речи (единые имена: neutral, happy, angry, sad, calm, fearful, surprised, disgust) и уверенность."""
    with _lock:
        out = _load_emotion()(_clip16k(path), top_k=8)
    best = out[0]
    return EMOTION_NORM.get(str(best["label"]).lower(), str(best["label"]).lower()), float(best["score"])


def speech_scores(paths: list[str], batch_size: int = 8) -> list[tuple[float, list[str]]]:
    """Для каждого файла: уверенность, что это речь (0..1), и найденные неречевые события (смех, кашель…)."""
    with _lock:
        outs = _load_events()([_clip16k(p, 10.0) for p in paths], top_k=15, batch_size=batch_size)
    res = []
    for out in outs:
        speech = max((float(o["score"]) for o in out if o["label"] in SPEECH_LABELS), default=0.0)
        ev = [EVENT_LABELS[o["label"]] for o in out if o["label"] in EVENT_LABELS and o["score"] >= 0.25]
        res.append((speech, ev))
    return res


EDGES_VERSION = 2
EXTRA_SOUNDS = {"Gargling": "глоток", "Chewing, mastication": "жевание", "Sniff": "шмыганье", "Snort": "фырканье",
                "Wail, moan": "стон", "Wheeze": "хрип", "Snoring": "храп", "Liquid": "глоток", "Slosh": "глоток",
                "Hum": "мычание"}


def _edge_sound(seg: np.ndarray) -> tuple[float, str]:
    """Кусок до/после слов: насколько это речь (0..1) и что это за звук."""
    if len(seg) < 8000:  # классификатору нужно хоть полсекунды — дополняем тишиной
        pad = (8000 - len(seg)) // 2
        seg = np.pad(seg, (pad, 8000 - len(seg) - pad))
    with _lock:
        out = _load_events()({"raw": seg.astype(np.float32), "sampling_rate": 16000}, top_k=8)
    speech = max((float(o["score"]) for o in out if o["label"] in SPEECH_LABELS), default=0.0)
    name = next((EVENT_LABELS.get(o["label"]) or EXTRA_SOUNDS.get(o["label"]) for o in out
                 if o["label"] in EVENT_LABELS or o["label"] in EXTRA_SOUNDS), None)
    return speech, name or "звук"


def classify_segment(wav: np.ndarray, sr: int) -> list[dict]:
    """Calibrated labels for a candidate word-free source interval."""
    w = audio.resample(wav, sr, 16000).astype(np.float32)
    if len(w) < 8000:
        w = np.pad(w, (0, 8000 - len(w)))
    with _lock:
        return _load_events()({"raw": w, "sampling_rate": 16000}, top_k=15)


def event_edges(path: str) -> dict:
    """Звуки до и после слов: смех, вздох, кашель, глоток, вдох, причмокивание — всё, что в оригинале звучит,
    но не является речью. Эти куски берутся из оригинала как есть, русская речь ставится между ними.

    Запись режется по громкости на «островки» звука (тишина в начале не мешает); крайние островки по очереди
    проверяет классификатор звуков: пока это не речь — кусок остаётся из оригинала. Классы не важны — глоток
    («Gargling») или безымянный звук сохраняются так же, как смех."""
    w, sr = audio.read(path)
    w16 = audio.resample(w, sr, 16000).astype(np.float32)
    dur = len(w16) / 16000
    none = {"lead": 0.0, "trail": round(dur, 2), "dur": round(dur, 2), "events": [], "v": EDGES_VERSION}
    if dur < 0.6:
        return none
    isl = audio.sound_islands(w16, 16000)
    if len(isl) < 2:
        return none
    cut = lambda a, b: w16[int(a * 16000): int(b * 16000)]
    lead, lead_ev, k = 0.0, [], 0
    while k < len(isl) - 1:
        a, b = isl[k]
        if b - a > 2.5:
            break
        speech, name = _edge_sound(cut(a, b))
        if speech >= 0.35:
            break
        lead = b
        lead_ev.append(name)
        k += 1
    trail, trail_ev, m = dur, [], len(isl) - 1
    while m > k:
        a, b = isl[m]
        if b - a > 2.5:
            break
        speech, name = _edge_sound(cut(a, b))
        if speech >= 0.35:
            break
        trail = a
        trail_ev.insert(0, name)
        m -= 1
    if trail - lead < 0.3:  # вся реплика — не речь
        return none
    return {"lead": round(lead, 2), "trail": round(trail, 2), "dur": round(dur, 2),
            "events": lead_ev + trail_ev, "v": EDGES_VERSION}


def events(path: str, threshold: float = 0.25) -> list[tuple[str, float]]:
    with _lock:
        out = _load_events()(_clip16k(path), top_k=12)
    found = [(EVENT_LABELS[o["label"]], float(o["score"])) for o in out
             if o["label"] in EVENT_LABELS and o["score"] >= threshold]
    return found


# --- итоговая оценка ---

def voice_type(f0: float) -> str:
    """Тип голоса по высоте: ребёнок / женский / мужской / низкий мужской."""
    if not f0:
        return "неизвестно"
    if f0 >= 260:
        return "детский"
    if f0 >= 165:
        return "женский"
    if f0 >= 115:
        return "мужской"
    return "низкий мужской"


DRIFT_MARGIN = 0.12      # голос реплики дальше от «русского голоса» персонажа, чем обычно, на столько — дрейф


def judge_line(gen_path: str, ru_text: str, orig_path: str, profile: np.ndarray | None,
               heard: str | None = None, orig_events: list | None = None, orig_emotion=None,
               full: bool = True, profile_f0: float | None = None, consistency: float | None = None,
               cons_floor: float | None = None, pron: dict | None = None, gen_stats: tuple | None = None,
               orig_stats: tuple | None = None, events_kept: list | None = None) -> dict:
    """Собрать все проверки для одной реплики. heard — уже распознанный текст (если есть);
    profile_f0 — обычная высота голоса персонажа (Гц): русская реплика не должна уходить в другой тип голоса;
    consistency/cons_floor — близость голоса к остальным русским репликам персонажа (ECAPA) и нижняя граница;
    pron — произношение (pron.check: что слышит распознаватель без словаря); gen_stats/orig_stats — (высота, размах интонации);
    events_kept — смех/вздохи, уже вклеенные из оригинала (за них не штрафуем)."""
    reasons, score = [], 100.0
    result = {}
    problems = []
    if consistency is not None:
        result["consistency"] = round(float(consistency), 3)
        if cons_floor is not None and consistency < cons_floor:
            reasons.append(f"голос отличается от остальных реплик персонажа (сходство {consistency:.2f}, обычно от {cons_floor:.2f})")
            score -= 25
            problems.append("drift")
    if pron is not None:
        from . import pron as _pron
        result["pron_cer"], result["pron_heard"] = pron["cer"], pron["heard"]
        result["pron_words"] = [list(x) for x in pron.get("words") or []]
        if pron["cer"] > _pron.ACCENT_CER and pron.get("words"):
            reasons.append("произношение с акцентом/нечёткое: " + _pron.describe(pron))
            score -= 22 if pron["cer"] > _pron.STRONG_CER else 12
            problems.append("accent")
    if gen_stats and orig_stats and gen_stats[1] and orig_stats[1]:
        result["range_gen"], result["range_orig"] = round(gen_stats[1], 1), round(orig_stats[1], 1)
        if orig_stats[1] - gen_stats[1] > 2.0 and orig_stats[1] > 2.5:
            reasons.append(f"подача ровнее оригинала: интонация {gen_stats[1]:.1f} пт против {orig_stats[1]:.1f}")
            score -= 12
            problems.append("flat")
        elif gen_stats[1] - orig_stats[1] > 3.0:
            reasons.append(f"подача ярче оригинала: интонация {gen_stats[1]:.1f} пт против {orig_stats[1]:.1f}")
            score -= 8
            problems.append("over")
    if profile_f0:
        if gen_stats and gen_stats[0]:
            f0 = gen_stats[0]
        else:
            g, gsr = audio.read(gen_path)
            f0 = audio.f0_median(g, gsr)
        if f0:
            semis = 12 * float(np.log2(f0 / profile_f0))
            result["f0"] = round(f0)
            result["f0_shift"] = round(semis, 1)
            if abs(semis) > 5:
                reasons.append(f"высота голоса не та: {voice_type(f0)} ({f0:.0f} Гц) вместо {voice_type(profile_f0)} "
                               f"({profile_f0:.0f} Гц)")
                score -= 35
            elif abs(semis) > 3:
                reasons.append(f"голос выше/ниже обычного для персонажа на {abs(semis):.0f} полутона")
                score -= 12

    if heard is not None:
        c = cer(ru_text, heard)
        result["cer"] = round(c, 3)
        result["heard"] = heard
        if c > 0.35:
            reasons.append(f"текст искажён: распознано «{heard[:60]}»")
            score -= 45
        elif c > 0.15:
            reasons.append("часть слов произнесена неразборчиво")
            score -= 20

    if profile is not None:
        sim = voice_similarity(gen_path, profile)
        result["voice_sim"] = round(sim, 3)
        if sim < 0.70:
            reasons.append("голос мало похож на персонажа")
            score -= 30
        elif sim < 0.78:
            reasons.append("голос похож лишь отчасти")
            score -= 12

    gen, gsr = audio.read(gen_path)
    orig, osr = audio.read(orig_path)
    g_len = len(audio.trim_silence(gen, gsr)) / gsr
    o_len = max(len(audio.trim_silence(orig, osr)) / osr, 0.3)
    ratio = g_len / o_len
    result["ratio"] = round(ratio, 2)
    if ratio > 1.35:
        reasons.append(f"русская фраза длиннее оригинала в {ratio:.1f} раза — стоит сократить текст")
        score -= 15
    elif ratio < 0.5:
        reasons.append("русская фраза подозрительно короткая — возможно, часть текста пропала")
        score -= 20
    peak = float(np.max(np.abs(gen))) if gen.size else 0.0
    if peak >= 0.999:
        reasons.append("перегруз (клиппинг) в звуке")
        score -= 10
    if g_len < 0.25:
        reasons.append("почти пустая запись")
        score -= 50

    if full:
        if orig_emotion is None:
            orig_emotion = emotion(orig_path)
        gen_emotion = emotion(gen_path)
        result["emotion_orig"] = EMOTION_RU.get(orig_emotion[0], orig_emotion[0])
        result["emotion_gen"] = EMOTION_RU.get(gen_emotion[0], gen_emotion[0])
        calm = {"neutral", "calm"}
        if (orig_emotion[0] != gen_emotion[0] and orig_emotion[1] > 0.55 and orig_emotion[0] not in calm
                and not (orig_emotion[0] in calm and gen_emotion[0] in calm)):
            reasons.append(f"эмоция не совпала: в оригинале {result['emotion_orig']}, в русском {result['emotion_gen']}")
            score -= 15
        if orig_events is None:
            orig_events = events(orig_path)
        orig_events = list(orig_events)
        result["events_orig"] = [e for e, _ in orig_events]
        left = [e for e, _ in orig_events if e not in set(events_kept or [])]
        if left and not events_kept:
            reasons.append("в оригинале есть " + ", ".join(left[:3])
                           + " посреди реплики — нейросеть это не воспроизводит, проверьте на слух")
            score -= 8

    result["codes"] = problems
    result["score"] = int(max(0, min(100, score)))
    result["verdict"] = "ok" if result["score"] >= 70 else "rework"
    result["reasons"] = reasons
    result["problems"] = _problems(result)
    return result


# Код проблемы → что с ней делать (показывается в студии и управляет доработкой)
ACTIONS = {
    "words": "повторить синтез; длинную фразу разбить на две",
    "fast": "не ускорять: сократить русский текст или разрешить фразе быть длиннее",
    "short": "проверить текст — часть могла пропасть; повторить синтез",
    "voice": "проверить образец/дообученную модель персонажа (профиль мог захватить чужие реплики)",
    "emotion": "сменить образец под эмоцию или описание подачи",
    "events": "в оригинале смех/кашель/крик — нейросеть их не воспроизводит, оценить на слух",
    "drift": "переозвучить ближе к остальным репликам персонажа: ниже случайность, больше вариантов, выбор по сходству голоса",
    "accent": "переозвучить: больше вариантов, выбор по чистоте русского произношения; режиссёру — какие слова прозвучали не так",
    "flat": "сыграть живее: режиссёр усиливает пунктуацию, выше выразительность",
    "over": "сыграть спокойнее: ровнее пунктуация, ниже выразительность",
    "pitch": "голос ушёл в другой тип (ребёнок/женщина/мужчина) — повторить синтез; для дообученной модели проверить её контроль",
    "clip": "перегруз — снизить громкость подгонки",
    "empty": "пустая запись — повторить синтез",
}


def _problems(r: dict) -> list[dict]:
    out = [{"code": c, "action": ACTIONS[c]} for c in (r.get("codes") or r.pop("_extra_problems", [])) if c in ACTIONS]
    if r.get("cer") is not None and r["cer"] > 0.15:
        out.append({"code": "words", "action": ACTIONS["words"]})
    if r.get("ratio") is not None and r["ratio"] > 1.35:
        out.append({"code": "fast", "action": ACTIONS["fast"]})
    if r.get("ratio") is not None and r["ratio"] < 0.5:
        out.append({"code": "short", "action": ACTIONS["short"]})
    if r.get("voice_sim") is not None and r["voice_sim"] < 0.78:
        out.append({"code": "voice", "action": ACTIONS["voice"]})
    if r.get("f0_shift") is not None and abs(r["f0_shift"]) > 5:
        out.append({"code": "pitch", "action": ACTIONS["pitch"]})
    if any("эмоция не совпала" in x for x in r.get("reasons", [])):
        out.append({"code": "emotion", "action": ACTIONS["emotion"]})
    if r.get("events_orig"):
        out.append({"code": "events", "action": ACTIONS["events"]})
    if any("клиппинг" in x for x in r.get("reasons", [])):
        out.append({"code": "clip", "action": ACTIONS["clip"]})
    if any("пустая" in x for x in r.get("reasons", [])):
        out.append({"code": "empty", "action": ACTIONS["empty"]})
    return out
