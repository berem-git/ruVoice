"""Донорский банк: русские голоса-образцы для ступени 1 конвейера v2.

Произношение рождается от русского образца, а не от английской записи персонажа, поэтому акцент не
переносится; тембр персонажа добавляет ступень 2 (RVC). Банк лежит в ruvoice/donors (manifest.json + FLAC):
у каждого донора пол, возраст, высота голоса (f0) и клипы в трёх подачах — calm / angry / sad — одним и тем же
голосом. Донор подбирается персонажу один раз (по высоте голоса, затем по отпечатку ECAPA) и закрепляется в
проекте; сменить его можно вручную на экране «Голоса»."""
import json
import math
import time
from pathlib import Path

import numpy as np

BANK_DIR = Path(__file__).parent / "donors"
STYLES = ("calm", "angry", "sad")
BAND_SEMITONES = 4.0     # кандидаты — доноры в пределах стольких полутонов от голоса персонажа
_STYLE_BY_EMOTION = {"angry": "angry", "disgust": "angry", "sad": "sad", "fearful": "sad"}
_emb_cache: dict[str, np.ndarray] = {}


def load(bank_dir=BANK_DIR) -> list[dict]:
    """Доноры из manifest.json (у каждого — путь к папке банка в поле dir); без банка — пустой список."""
    manifest = Path(bank_dir) / "manifest.json"
    if not manifest.is_file():
        return []
    try:
        data = json.loads(manifest.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return [{**d, "dir": str(bank_dir)} for d in data.get("donors", [])
            if d.get("id") and (d.get("clips") or {}).get("calm")]


def by_id(bank: list[dict], donor_id: str | None) -> dict | None:
    return next((d for d in bank if d["id"] == donor_id), None)


def style_for(emotion: str | None) -> str:
    """Подача донорского клипа по эмоции реплики: злость/отвращение → angry, грусть/страх → sad, иначе calm."""
    return _STYLE_BY_EMOTION.get((emotion or "").lower(), "calm")


def _clip(donor: dict, style: str) -> tuple[str, dict]:
    clips = donor.get("clips") or {}
    if style not in clips:
        style = "calm"
    return style, clips[style]


def clip_path(donor: dict, style: str, bank_dir=BANK_DIR) -> Path:
    style, clip = _clip(donor, style)
    return Path(donor.get("dir") or bank_dir) / clip["file"]


def reference(donor: dict, style: str, bank_dir=BANK_DIR) -> dict:
    """Образец для движка речи: русский клип с расшифровкой (ICL). Нет такой подачи — спокойная."""
    style, clip = _clip(donor, style)
    return {"audio": str(Path(donor.get("dir") or bank_dir) / clip["file"]), "text": clip.get("text") or "",
            "language": "ru", "speaker": None, "donor": donor["id"], "style": style}


def pitch_shift(char_f0: float, donor_f0: float) -> int:
    """Сдвиг высоты для переноса тембра (полутоны, ±12): от донора к персонажу; без высоты — 0."""
    if not char_f0 or not donor_f0:
        return 0
    return int(max(-12, min(12, round(12 * math.log2(char_f0 / donor_f0)))))


def _semitones(a: float, b: float) -> float:
    return abs(12 * math.log2(a / b))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def gender_for(f0: float | None) -> str | None:
    """Пол донора по высоте голоса персонажа: от 165 Гц — женские доноры (детские голоса тоже), ниже — мужские."""
    if not f0:
        return None
    return "female" if f0 >= 165 else "male"


def pick(bank: list[dict], f0: float, emb: np.ndarray | None, embed_fn) -> str | None:
    """Донор для голоса: сначала пол (по высоте), потом высота (±4 полутона, иначе три ближайших),
    среди них — похожесть ECAPA. Без отпечатка — ближайший по высоте; без высоты и отпечатка — первый мужской."""
    if not bank:
        return None
    g = gender_for(f0)
    bank = [d for d in bank if d.get("gender") == g] or bank
    with_f0 = [d for d in bank if d.get("f0")]
    cands = bank
    if f0 and with_f0:
        within = [d for d in with_f0 if _semitones(f0, d["f0"]) <= BAND_SEMITONES]
        cands = within or sorted(with_f0, key=lambda d: _semitones(f0, d["f0"]))[:3]
    if emb is None:
        if f0 and with_f0:
            return min(cands, key=lambda d: _semitones(f0, d["f0"]))["id"]
        return next((d["id"] for d in cands if d.get("gender") == "male"), cands[0]["id"])
    e = _unit(emb)

    def similarity(d: dict) -> float:
        key = str(clip_path(d, "calm"))
        if key not in _emb_cache:
            _emb_cache[key] = _unit(embed_fn(key))
        return float(np.dot(_emb_cache[key], e))
    return max(cands, key=similarity)["id"]


def _profile_embedding(jobs, p, char: str) -> np.ndarray | None:
    from . import judge
    prof = jobs.character_profile(p, char)
    if not prof or not judge.ecapa_ready():
        return None
    try:
        return np.asarray(judge.ecapa_embed([prof["audio"]])[0], dtype=np.float32)
    except Exception as e:
        print(f"донор для «{char}»: отпечаток профиля не посчитан ({e})")
        return None


def assigned(jobs, p, char: str) -> dict:
    """Закреплённый донор персонажа {"id", "pitch", "picked_at", "manual"}; подбирается один раз."""
    table = dict(p.get("donors") or {})
    bank = load()
    cur = table.get(char)
    if cur and by_id(bank, cur.get("id")):
        return cur
    f0 = float(jobs.profile_f0(p, char) or 0.0)
    emb = _profile_embedding(jobs, p, char)
    from . import judge
    donor_id = pick(bank, f0, emb, lambda path: judge.ecapa_embed([str(path)])[0])
    if donor_id is None:
        raise RuntimeError("Донорский банк пуст: нет ruvoice/donors/manifest.json")
    donor = by_id(bank, donor_id)
    entry = {"id": donor_id, "pitch": pitch_shift(f0, float(donor.get("f0") or 0)), "picked_at": time.time(),
             "manual": False}
    table[char] = entry
    p.set("donors", table)
    return entry


def set_manual(jobs, p, char: str, donor_id: str) -> dict:
    """Донор, выбранный пользователем: закрепляется и не пересчитывается."""
    donor = by_id(load(), donor_id)
    if not donor:
        raise ValueError("Нет такого донора")
    f0 = float(jobs.profile_f0(p, char) or 0.0)
    table = dict(p.get("donors") or {})
    entry = {"id": donor_id, "pitch": pitch_shift(f0, float(donor.get("f0") or 0)), "picked_at": time.time(),
             "manual": True}
    table[char] = entry
    p.set("donors", table)
    return entry
