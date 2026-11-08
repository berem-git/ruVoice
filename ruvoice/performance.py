"""Source-directed dubbing, independent of game and synthesis backend.

Only confidently isolated non-speech is copied. Overlapping English speech is never
mixed underneath a translation. Baked reverb cannot be inferred reliably from a
single mixed recording: it needs an explicit room impulse response or a clean stem.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from . import audio

VERSION = 1
EVENTS = {"Laughter", "Giggle", "Chuckle, chortle", "Cough", "Sneeze", "Sigh", "Gasp",
          "Breathing", "Sniff", "Throat clearing", "Crying, sobbing", "Groan", "Grunt"}


def fingerprint(path: Path) -> str:
    p = Path(path)
    s = p.stat()
    return hashlib.sha256(f"{p.resolve()}:{s.st_size}:{s.st_mtime_ns}:{VERSION}".encode()).hexdigest()


def analyse(path: Path, words: list[dict], classify=None, effects: str = "engine") -> dict:
    """Word-aligned source gaps are eligible for keeping; uncertain gaps stay untouched."""
    w, sr = audio.read(path)
    dur = len(w) / sr
    words = sorted([{"start": max(0., float(x["start"])), "end": min(dur, float(x["end"])),
                     "text": x.get("text", "")} for x in words
                    if x.get("start") is not None and x.get("end") is not None
                    and 0 <= float(x["start"]) < float(x["end"]) <= dur + .1], key=lambda x: x["start"])
    # Merge overlapping word regions; never expose a spoken word as a non-speech gap.
    covered = []
    for x in words:
        a, b = max(0., x["start"] - .12), min(dur, x["end"] + .12)
        if covered and a <= covered[-1][1]:
            covered[-1][1] = max(b, covered[-1][1])
        else:
            covered.append([a, b])
    gaps, cursor = [], 0.
    for a, b in covered:
        if a - cursor >= .22:
            gaps.append((cursor, a))
        cursor = b
    if dur - cursor >= .22:
        gaps.append((cursor, dur))
    events, pauses, warnings = [], [], []
    if not words:
        warnings.append("Нет надёжных границ слов: оригинальный звук не подмешивается")
        gaps = []
    for a, b in gaps:
        seg = w[int(a * sr):int(b * sr)]
        if audio.rms_db(seg) < -45:
            pauses.append({"start": a, "end": b})
            continue
        if classify is None or b - a > 6:
            continue
        labels = classify(seg, sr)
        speech = max((float(x["score"]) for x in labels if x["label"] in
                      {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation"}), default=0.)
        best = max((x for x in labels if x["label"] in EVENTS), key=lambda x: x["score"], default=None)
        if best and float(best["score"]) >= .4 and speech < .12:
            events.append({"start": round(a, 4), "end": round(b, 4), "label": best["label"],
                           "confidence": round(float(best["score"]), 3),
                           "before_words": sum(x["end"] <= a for x in words)})
    vocal_style = []
    if classify is not None and words:
        labels = classify(w, sr)
        for x in labels:
            if x["label"] in EVENTS and float(x["score"]) >= .4:
                vocal_style.append(x["label"])
        if any(x not in {e["label"] for e in events} for x in vocal_style):
            warnings.append("Неречевой звук не изолирован: передаётся подачей, точное воспроизведение требует проверки")
    if effects not in {"engine", "dry", "provided_stems", "auto", "unknown", "baked"}:
        warnings.append("Запечённые эффекты требуют отдельной дорожки или импульса помещения")
    feat = audio.speech_features(w, sr)
    return {"version": VERSION, "source": fingerprint(path), "duration": dur, "words": words,
            "events": events, "pauses": pauses, "features": feat, "effects": effects,
            "warnings": warnings, "vocal_style": vocal_style, "alignment": "word-gaps"}


def instruction(style: dict, delivery: dict | None = None, events: list | None = None) -> str:
    """Explicit English style instructions understood by instruction-capable engines."""
    emo = style.get("emotion", "neutral")
    parts = [{"angry": "restrained anger, firm and tense", "sad": "sad, subdued and weary",
              "happy": "warm and cheerful", "fearful": "anxious and tense", "surprised": "surprised",
              "disgust": "cold disapproval", "calm": "calm and soft"}.get(emo, "natural conversational delivery")]
    for tag in style.get("tags", []):
        if tag in {"fast", "slow", "pauses", "expressive", "flat"}:
            parts.append({"fast": "hurried pace", "slow": "slow deliberate pace", "pauses": "hesitant with pauses",
                          "expressive": "expressive intonation", "flat": "even restrained intonation"}[tag])
    if style.get("loud", 0) < -8:
        parts.append("very quiet, nearly whispering")
    elif style.get("loud", 0) > 6:
        parts.append("projected loud voice")
    if any("laugh" in str(x).lower() or "смех" in str(x).lower() for x in (events or [])):
        parts.append("smiling, lightly laughing delivery")
    note = (delivery or {}).get("note")
    if note:
        parts.append("Performance direction: " + re.sub(r"[\[\]()\r\n]", " ", str(note))[:160])
    return "; ".join(parts)


def compose(gen: np.ndarray, sr: int, orig: np.ndarray, osr: int, plan: dict,
            max_speedup: float = 1.1) -> tuple[np.ndarray, dict]:
    """Insert isolated original events at silence boundaries, never cut generated syllables.

    Internal placement is approximate across languages and is explicitly marked for review.
    No event is copied without recognised word boundaries and high-confidence classification.
    """
    dur = len(orig) / osr
    events = [e for e in plan.get("events", []) if 0 <= e["start"] < e["end"] <= dur
              and e.get("confidence", 0) >= .4]
    words = plan.get("words") or []
    events = [e for e in events if words and not any(e["start"] < w["end"] and e["end"] > w["start"] for w in words)]
    if not events:
        fitted, info = audio.fit_to_original(gen, sr, orig, osr, max_speedup=max_speedup)
        warnings = list(plan.get("warnings", []))
        return fitted, {**info, "performance_v": VERSION, "kept": [], "warnings": warnings,
                        "requires_review": bool(warnings)}
    # Fit against spoken content only: event duration must not speed up the Russian voice.
    mask = np.ones(len(orig), dtype=bool)
    for e in events:
        mask[int(e["start"] * osr):int(e["end"] * osr)] = False
    fitted, info = audio.fit_to_original(gen, sr, orig[mask], osr, max_speedup=max_speedup)
    islands = audio.sound_islands(fitted, osr)
    boundaries = [(islands[i][1] + islands[i + 1][0]) / 2 for i in range(len(islands) - 1)
                  if islands[i + 1][0] - islands[i][1] >= .06]
    inserts, skipped, internal = [], [], False
    for e in events:
        n = e.get("before_words", 0)
        if n == 0:
            pos = 0
        elif n >= len(words):
            pos = len(fitted)
        else:
            desired = n / len(words) * len(fitted) / osr
            near = min(boundaries, key=lambda x: abs(x - desired), default=None)
            if near is None or abs(near - desired) > .5:
                skipped.append(e["label"])
                continue
            pos = int(near * osr)
            internal = True
        inserts.append((pos, orig[int(e["start"] * osr):int(e["end"] * osr)].copy(), e["label"]))
    chunks, cursor, kept = [], 0, []
    for pos, seg, label in sorted(inserts, key=lambda x: x[0]):
        chunks.append(fitted[cursor:pos])
        # Fade only the inserted recording's edges; every original sample outside it stays out.
        fade = min(int(osr * .008), len(seg) // 2)
        if fade:
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        chunks.append(seg)
        cursor = pos
        kept.append(label)
    chunks.append(fitted[cursor:])
    result = np.concatenate(chunks).astype(np.float32)
    warnings = list(plan.get("warnings", []))
    if internal:
        warnings.append("Место неречевой вставки внутри перевода требует прослушивания")
    if skipped:
        warnings.append("Нет безопасной паузы для вставок: " + ", ".join(skipped))
    return result, {**info, "performance_v": VERSION, "kept": kept, "warnings": warnings,
                    "requires_review": bool(warnings), "ratio": round(len(result) / osr / max(dur, .3), 3)}


def mix_stems(speech: np.ndarray, sr: int, background: np.ndarray | None = None,
              impulse: np.ndarray | None = None, wet: float = .2) -> np.ndarray:
    """Apply an explicitly provided room response and/or a speech-free background stem."""
    out = np.asarray(speech, dtype=np.float32)
    if impulse is not None:
        from scipy.signal import fftconvolve
        h = np.asarray(impulse, dtype=np.float32)
        if not h.size or not np.all(np.isfinite(h)):
            raise ValueError("Некорректный импульс помещения")
        wet = float(np.clip(wet, 0, 1))
        convolved = fftconvolve(out, h)
        dry = np.pad(out, (0, len(convolved) - len(out)))
        out = dry * (1 - wet) + convolved * wet
    if background is not None:
        bg = np.asarray(background, dtype=np.float32)
        n = max(len(bg), len(out))
        out = np.pad(out, (0, n - len(out))) + np.pad(bg, (0, n - len(bg)))
    peak = np.max(np.abs(out)) if out.size else 0
    return out / max(1., float(peak) / .98)
