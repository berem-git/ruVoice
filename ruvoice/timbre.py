"""Тембр персонажа (ступень 2 конвейера v2): кому обучать RVC-модель, на каком уровне, сколько это займёт,
обучение и его контроль.

Главный персонаж — тот, у кого речи не меньше `timbre_min_seconds` и реплик не меньше 12: ему обучается
своя модель тембра на его же записях (RVC/Applio в отдельном процессе). Остальные — эпизодические, идут
быстрым путём (клонирование или донорский голос). Пользователь может переопределить уровень вручную
(«обучать» / «эпизодический»). Обученная модель принимается только после контроля (calibrate) на
контрольных репликах, которые в обучение не попали: иначе остаётся «слабой» и не используется."""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .paths import DATA_DIR

TIMBRE_LEVELS = {
    "off": None,
    "fast": {"name": "Быстро", "epochs": 150, "minutes": 8},
    "normal": {"name": "Обычно", "epochs": 300, "minutes": 15},
    "deep": {"name": "Долго", "epochs": 500, "minutes": 25},
}
MIN_LINES = 12
HOLDOUT = 6                   # контрольных реплик вне обучения
MIN_DATASET_SEC = 45
SR = 40000                    # частота обучения RVC (HiFi-GAN f0G40k)
DISK_GB_PER_VOICE = 0.06
RVC_SEC_PER_MIN_EPOCH = 1.8   # секунд обучения на минуту записей за эпоху: RTX 5070 Ti, 8 мин записей — ~14 с/эпоха
SPEED_FILE = DATA_DIR / "train_speed.json"


def tier(char: dict, settings: dict, override: str | None) -> str:
    """«main» — обучать тембр, «episodic» — быстрый путь. char — строка Project.characters()."""
    if TIMBRE_LEVELS.get(settings.get("timbre_level", "normal")) is None:
        return "episodic"
    if override == "train":
        return "main"
    if override == "episodic":
        return "episodic"
    enough = (char.get("seconds") or 0) >= float(settings.get("timbre_min_seconds", 120))
    return "main" if enough and (char.get("lines") or 0) >= MIN_LINES else "episodic"


def level(settings: dict) -> dict | None:
    return TIMBRE_LEVELS.get(settings.get("timbre_level", "normal"))


# --- скорость обучения на этом компьютере ---

def speed_rate() -> float:
    """Секунд обучения на минуту записей за эпоху: медиана прошлых обучений RVC, до первого — константа."""
    try:
        hist = json.loads(SPEED_FILE.read_text("utf-8"))
        vals = sorted(float(x["rate"]) for x in hist[-10:] if x.get("method") == "rvc" and x.get("rate"))
        if vals:
            return vals[len(vals) // 2]
    except Exception:
        pass
    return RVC_SEC_PER_MIN_EPOCH


def record_speed(rate: float, batch: int):
    try:
        hist = json.loads(SPEED_FILE.read_text("utf-8")) if SPEED_FILE.exists() else []
    except Exception:
        hist = []
    hist = (hist + [{"rate": round(rate, 2), "method": "rvc", "batch": batch, "at": time.time()}])[-20:]
    SPEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEED_FILE.write_text(json.dumps(hist), "utf-8")


# --- обученные модели ---

def entry(p, char: str) -> dict | None:
    return (p.get("voices") or {}).get(char)


def voice(p, char: str) -> dict | None:
    """Принятая модель тембра с абсолютными путями или None (не обучена, не прошла контроль, файла нет)."""
    v = entry(p, char)
    if not v or v.get("method") != "rvc" or not v.get("val") or v.get("weak") or not v.get("pth"):
        return None
    pth = p.dir / v["pth"]
    if not pth.exists():
        return None
    idx = (p.dir / v["index"]) if v.get("index") else None
    return {"method": "rvc", "pth": str(pth), "index": str(idx) if idx and idx.exists() else None}


def needs_training(entry: dict | None, lvl: dict) -> bool:
    """Обучать ли (заново) на этом уровне: нет записи или контроль не завершён — да; принятая или слабая модель —
    только если уровень вырос (эпох больше, чем было). Слабую на том же уровне заново не мучаем."""
    if not entry or entry.get("val") is None:
        return True
    return int(entry.get("epochs") or 0) < int(lvl["epochs"])


def mains(p, settings: dict) -> list[dict]:
    """Главные персонажи (с учётом ручных переопределений), по убыванию числа реплик."""
    over = p.get("timbre_override") or {}
    chars = [c for c in p.characters() if tier(c, settings, over.get(c["id"])) == "main"]
    chars.sort(key=lambda c: -(c["lines"] or 0))
    return chars


def plan(jobs, gid: str) -> dict:
    """Кого обучать на текущем уровне: список с оценкой времени, кто уже обучен или слаб."""
    from . import engines
    p, s = jobs.project(gid), jobs.settings(gid)
    lvl = level(s)
    key = s.get("timbre_level", "normal")
    if not lvl:
        return {"level": key, "ready": engines.rvc_available(), "planned": [], "total_minutes": 0, "skipped": []}
    rate = speed_rate()
    planned = []
    for c in mains(p, s):
        v = entry(p, c["id"])
        minutes_used = min((c["seconds"] or 0) / 60, lvl["minutes"])
        planned.append({"id": c["id"], "name": c["name"], "minutes": round((c["seconds"] or 0) / 60, 1),
                        "trained": voice(p, c["id"]) is not None, "weak": bool(v and v.get("weak")),
                        "retrain": needs_training(v, lvl),
                        "reason": (v or {}).get("reason"), "epochs": lvl["epochs"], "max_seconds": lvl["minutes"] * 60,
                        "est_minutes": round(minutes_used * lvl["epochs"] * rate / 60)})
    ids = {c["id"] for c in planned}
    return {"level": key, "ready": engines.rvc_available(), "planned": planned,
            "total_minutes": sum(c["est_minutes"] for c in planned if c["retrain"]),
            "skipped": [c["name"] for c in p.characters() if c["id"] not in ids][:12]}


def estimate(jobs, gid: str) -> dict:
    """Для студии: сколько персонажей, времени и места займёт каждый уровень."""
    from . import engines
    p, s = jobs.project(gid), jobs.settings(gid)
    rate = speed_rate()
    out = {}
    for key, lvl in TIMBRE_LEVELS.items():
        if lvl is None:
            out[key] = {"name": "Без обучения", "chars": 0, "todo": 0, "minutes": 0, "disk_gb": 0, "names": []}
            continue
        chars = mains(p, {**s, "timbre_level": key})
        todo = [c for c in chars if needs_training(entry(p, c["id"]), lvl)]
        minutes = sum(min((c["seconds"] or 0) / 60, lvl["minutes"]) * lvl["epochs"] * rate / 60 for c in todo)
        out[key] = {"name": lvl["name"], "chars": len(chars), "todo": len(todo), "minutes": round(minutes),
                    "disk_gb": round(DISK_GB_PER_VOICE * len(chars), 2), "names": [c["name"] for c in chars][:6]}
    return {"levels": out, "current": s.get("timbre_level", "normal"), "ready": engines.rvc_available()}


# --- данные и обучение ---

def dataset(jobs, p, char: str, max_seconds: float) -> tuple[list[Path], list[dict], float]:
    """Записи персонажа для обучения (WAV 40 кГц в voices/<char>/rvc/dataset) и контрольные реплики вне обучения.
    Берутся реплики от 1 с после обрезки тишины, не пропущенные, где по звуку речь (speech ≥ 0.5), ближе к центру
    голоса (jobs._most_typical); контроль — 6 реплик с русским текстом длиной 2–7 с после первых восьми."""
    from . import audio
    from .dub import _safe
    cands = [l for l in p.lines(char=char)
             if l["status"] != "skipped" and l["orig"] and (l["duration"] or 0) >= 1.0
             and ((l["info"] or {}).get("speech") is None or (l["info"] or {}).get("speech") >= 0.5)]
    cands = jobs._most_typical(p, cands[:400])
    holdout = [l for l in cands[8:] if l["ru"] and 2.0 <= (l["duration"] or 0) <= 7.0][:HOLDOUT]
    hold_ids = {l["id"] for l in holdout}
    ddir = p.dir / "voices" / _safe(char) / "rvc" / "dataset"
    shutil.rmtree(ddir, ignore_errors=True)
    ddir.mkdir(parents=True)
    files, total = [], 0.0
    for l in cands:
        if l["id"] in hold_ids:
            continue
        w, sr = audio.read(p.dir / l["orig"])
        w = audio.trim_silence(w, sr)
        if len(w) < sr * 1.0:
            continue
        f = ddir / f"{_safe(l['id'])}.wav"
        audio.write(f, audio.resample(w, sr, SR), SR)
        files.append(f)
        total += len(w) / sr
        if total >= max_seconds:
            break
    return files, holdout, total


def train(jobs, gid: str, char: str) -> dict:
    """Обучить модель тембра персонажа на уровне из настроек; запись voices[char] ждёт контроля (calibrate)."""
    from . import accel, tts
    from .dub import _safe
    p, s, report = jobs.project(gid), jobs.settings(gid), jobs._progress(gid)
    lvl = level(s)
    if not lvl:
        raise RuntimeError("Обучение тембра выключено в настройках")
    name = next((c["name"] for c in p.characters() if c["id"] == char), char)
    report(f"Тембр «{name}»: подбор записей…")
    files, holdout, total = dataset(jobs, p, char, lvl["minutes"] * 60)
    if total < MIN_DATASET_SEC:
        raise RuntimeError(f"Слишком мало чистой речи персонажа для обучения тембра (нужно от {MIN_DATASET_SEC} с)")
    vdir = p.dir / "voices" / _safe(char) / "rvc"
    try:
        vram = accel.info().get("vram_gb") or 0
    except Exception:
        vram = 0
    batch = 8 if vram >= 12 else 4
    started, train_started = time.time(), [None]

    def on_progress(info: dict):
        step, tot = info.get("epoch", 0) or 0, info.get("total", lvl["epochs"])
        if info.get("message"):
            report(f"Тембр «{name}»: {info['message']}")
            return
        if info.get("phase") == "train" and train_started[0] is None:
            train_started[0] = time.time()
        msg = f"Обучение тембра «{name}»: эпоха {step}/{tot}"
        if step and train_started[0]:
            left = (time.time() - train_started[0]) / step * (tot - step)
            msg += f", осталось ~{int(left // 60)} мин {int(left % 60)} с"
        report(msg, step, tot)
    report(f"Обучение тембра «{name}»: {total / 60:.1f} мин записей, {lvl['epochs']} эпох…")
    res = tts.train_voice_rvc(str(files[0].parent), str(vdir), f"{p.id}__{_safe(char)}", lvl["epochs"], on_progress,
                              batch_size=batch)
    if jobs.cancel_flag.is_set():
        raise RuntimeError("остановлено")
    took = time.time() - started
    record_speed(took / max(total / 60 * lvl["epochs"], 1e-6), batch)
    rel = lambda path: Path(path).relative_to(p.dir).as_posix() if path else None
    row = {"method": "rvc", "pth": rel(res["pth"]), "index": rel(res.get("index")), "epochs": lvl["epochs"],
           "dataset_seconds": round(total), "train_seconds": round(took), "trained_at": time.time(),
           "holdout": [l["id"] for l in holdout], "val": None, "weak": True, "reason": "ждёт контроля"}
    (vdir / "meta.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), "utf-8")
    voices = p.get("voices") or {}
    voices[char] = row
    p.set("voices", voices)
    shutil.rmtree(files[0].parent, ignore_errors=True)
    calibrate(jobs, gid, char, holdout)
    return (p.get("voices") or {}).get(char) or row


# --- контроль обученной модели ---

PRON_TOLERANCE = 0.03      # произношение с тембром может быть хуже донорского не больше чем на столько (CER)
F0_TOLERANCE = 3.0         # полутонов от обычной высоты голоса персонажа


def calibrate(jobs, gid: str, char: str, holdout: list[dict]) -> dict:
    """Контрольные реплики озвучиваются донором без тембра и с тембром; модель принимается, если тембр сделал
    голос ближе к персонажу, не испортил произношение и не увёл высоту голоса. Принятая модель даёт
    фиксированный «русский голос» персонажа (центр ECAPA калибровочных записей) для контролёра."""
    from . import audio, judge, pron
    from .dub import _safe
    p, report = jobs.project(gid), jobs._progress(gid)
    name = next((c["name"] for c in p.characters() if c["id"] == char), char)
    voices = p.get("voices") or {}
    row = dict(voices.get(char) or {})
    res = {"accept": False, "reason": "", "sim_base": None, "sim_rvc": None, "pron_base": None, "pron_rvc": None,
           "f0_rvc": None, "f0_shift": None}
    if not holdout:
        res["reason"] = "нет контрольных реплик с русским текстом"
        row.update(val=res, weak=True, reason=res["reason"])
        voices[char] = row
        p.set("voices", voices)
        return res
    report(f"Тембр «{name}»: контроль на {len(holdout)} репликах…")
    calib = p.dir / "voices" / _safe(char) / "rvc" / "calib"
    shutil.rmtree(calib, ignore_errors=True)
    candidate = {"method": "rvc", "pth": str(p.dir / row["pth"]),
                 "index": str(p.dir / row["index"]) if row.get("index") and (p.dir / row["index"]).exists() else None}
    plan = jobs.voice_plan(p, char, rvc=candidate)  # план «как будто модель принята»: донор + тембр
    files = {}
    for tag, use in (("base", False), ("rvc", True)):
        files[tag], sim = jobs._render(gid, p, char, holdout, calib / tag, plan=plan, timbre=use)
        res[f"sim_{tag}"] = None if sim is None else round(float(sim), 3)
    if pron.ready():
        for tag in ("base", "rvc"):
            checks = pron.check([str(p.dir / f) for f in files[tag]], [l["ru"] for l in holdout])
            res[f"pron_{tag}"] = round(float(np.mean([c["cer"] for c in checks])), 3)
        pron.unload()
    f0s = [v for v in (audio.f0_median(*audio.read(p.dir / f)) for f in files["rvc"]) if v]
    res["f0_rvc"] = round(float(np.median(f0s))) if f0s else None
    profile_f0 = float(jobs.profile_f0(p, char) or 0)
    if res["f0_rvc"] and profile_f0:
        res["f0_shift"] = round(float(12 * math.log2(res["f0_rvc"] / profile_f0)), 1)
    problems = []
    if res["sim_base"] is None or res["sim_rvc"] is None:
        problems.append("похожесть не измерена (нет модели VoiceEncoder или профиля персонажа) — модель не проверена")
    elif not res["sim_rvc"] > res["sim_base"]:
        problems.append(f"похожесть не выросла ({res['sim_rvc']} против {res['sim_base']} без тембра)")
    if res["pron_base"] is not None and res["pron_rvc"] is not None and res["pron_rvc"] > res["pron_base"] + PRON_TOLERANCE:
        problems.append(f"произношение стало хуже (ошибок {res['pron_rvc']:.0%} против {res['pron_base']:.0%})")
    if res["f0_shift"] is not None and abs(res["f0_shift"]) > F0_TOLERANCE:
        problems.append(f"высота голоса ушла на {abs(res['f0_shift']):.0f} полутона от обычной для персонажа")
    res["accept"] = not problems
    res["reason"] = "; ".join(problems) if problems else "контроль пройден"
    row.update(val=res, weak=not res["accept"], reason=res["reason"])
    voices[char] = row
    p.set("voices", voices)
    if res["accept"] and judge.ecapa_ready():
        E = judge.ecapa_embed([str(p.dir / f) for f in files["rvc"]])
        centre = E.mean(axis=0)
        centre = centre / (np.linalg.norm(centre) + 1e-9)
        sims = sorted(float(e @ centre) for e in E)
        all_rv = p.get("ru_voice") or {}
        all_rv[char] = {"e": centre.tolist(), "fixed": True, "floor": round(sims[len(sims) // 2] - judge.DRIFT_MARGIN, 3),
                        "n": len(sims), "sims": []}
        p.set("ru_voice", all_rv)
    print(f"контроль тембра «{name}»: {res}", flush=True)
    return res
