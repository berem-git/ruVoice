"""Стенд движков: один и тот же набор реплик (несколько персонажей, разные эмоции и длины) озвучивается
несколькими вариантами конвейера и сравнивается цифрами и на слух.

Вариант — строка `бэкенд[@en][+rvc]`: `ru` — Qwen-RU с русским донором, `ru@en` — клон с английской записи
персонажа (быстрый путь), `silero+rvc` — Silero и перенос тембра персонажа, `voxcpm@en`, `omnivoice`…
Для каждой реплики и варианта — WAV, время синтеза и метрики на сыром звуке (до подгонки под оригинал):
pron_cer (wav2vec2 без словаря — прокси акцента), whisper_cer (разборчивость), ecapa_sim (похожесть на
оригинальный голос персонажа), f0_range (живость интонации), ratio (длина к оригиналу).
Результат — bench/<метка>/index.html (реплика × вариант с плеерами) и rows.json. Файлы игры не меняются.

CLI: python -m ruvoice.enginebench --game <gid> --variants ru,ru+rvc,silero+rvc,ru@en,voxcpm@en [--label]"""
import argparse
import html
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np

from . import audio

CHARS, PER_CHAR = 3, 10
METRICS = ("pron_cer", "whisper_cer", "ecapa_sim", "f0_range", "ratio", "seconds")
_VARIANT = re.compile(r"^([a-z_]+)(@en)?(\+rvc)?$")


def parse_variant(s: str) -> dict:
    from .engines import BACKENDS, CLONE_KEYS, SOURCE_KEYS
    m = _VARIANT.match((s or "").strip())
    if not m or m.group(1) not in BACKENDS:
        raise ValueError(f"Неизвестный вариант стенда: {s}")
    backend, english, timbre = m.group(1), bool(m.group(2)), bool(m.group(3))
    if english and backend not in CLONE_KEYS:
        raise ValueError(f"{backend} не клонирует с английской записи")
    if not english and backend not in SOURCE_KEYS:
        raise ValueError(f"{backend} работает только с английской записи персонажа (@en)")
    return {"backend": backend, "english": english, "timbre": timbre}


def variants_available(timbre: bool = False) -> list[str]:
    """Готовые варианты: источники русской речи (+rvc, если есть обученный тембр), клонировщики с @en."""
    from . import engines
    out = []
    for s in engines.sources():
        out.append(s)
        if timbre:
            out.append(s + "+rvc")
    out += [c + "@en" for c in engines.cloners()]
    return out


def pick_lines(jobs, p, chars: int = CHARS, per_char: int = PER_CHAR) -> list[dict]:
    """Закреплённый набор реплик (bench_lines в проекте): главные персонажи по числу реплик, у каждого —
    реплики разных эмоций (если эмоции уже посчитаны) и длин. Сохраняется, чтобы прогоны были сопоставимы."""
    saved = p.get("bench_lines")
    if saved:
        got = {l["id"]: l for l in p.lines(ids=saved)}
        lines = [got[i] for i in saved if i in got and got[i]["ru"]]
        if lines and len(lines) == len(saved):
            return lines
    ranked = sorted(p.characters(), key=lambda c: -(c["lines"] or 0))
    out = []
    for c in ranked[:chars]:
        cands = [l for l in p.lines(char=c["id"]) if l["ru"] and l["orig"] and 1.0 <= (l["duration"] or 0) <= 9
                 and l["status"] != "skipped"]
        if not cands:
            continue
        emos = jobs._emotions_for_char(p, c["id"]) if len(cands) > per_char else {}
        random.Random(13).shuffle(cands)
        cap = max(1, per_char // 3)
        chosen, seen = [], {}
        for l in sorted(cands, key=lambda l: -emos.get(l["id"], ("neutral", 0))[1]):  # сначала выраженные эмоции
            e = emos.get(l["id"], ("neutral", 0))[0]
            if seen.get(e, 0) >= cap:
                continue
            seen[e] = seen.get(e, 0) + 1
            chosen.append(l)
            if len(chosen) >= per_char:
                break
        for l in cands:  # добрать до нужного числа, если эмоций мало
            if len(chosen) >= per_char:
                break
            if l not in chosen:
                chosen.append(l)
        out += chosen
    p.set("bench_lines", [l["id"] for l in out])
    return out


def _reference(jobs, p, line: dict, spec: dict, profiles: dict) -> dict:
    """Образец для варианта: @en — профиль персонажа (английская запись), иначе донор / диктор."""
    backend = spec["backend"]
    if spec["english"]:
        prof = profiles.get(line["char"])
        if not prof:
            raise RuntimeError("нет профиля голоса персонажа")
        return {"audio": prof["audio"], "text": prof.get("text") if backend == "omnivoice" else None,
                "language": "en", "speaker": None}
    if backend in ("ru", "omnivoice"):
        from . import donors
        assigned = donors.assigned(jobs, p, line["char"])
        donor = donors.by_id(donors.load(), assigned["id"])
        if not donor:
            raise RuntimeError("донор не найден")
        emo = (jobs._emotions_for_char(p, line["char"]).get(line["id"]) or ("neutral", 0))[0]
        return donors.reference(donor, donors.style_for(emo))
    if backend == "custom":
        return {"speaker": jobs._custom_speaker_for(p, line["char"])}
    from . import basevoices, judge
    f0 = jobs.profile_f0(p, line["char"])
    gender = ("male" if f0 < 165 else "female") if f0 else None
    prof = profiles.get(line["char"])
    emb = judge.speaker_embed([prof["audio"]]) if prof and judge.VE_WEIGHTS.exists() else None
    return {"speaker": basevoices.pick_for(emb, gender)}


def _params(jobs, p, line: dict, backend: str) -> dict:
    params = dict(jobs._style_params(p, line, backend))
    if backend == "ru":
        params.setdefault("subtalker_temperature", 0.5)
    return params


def run(jobs, gid: str, variants: list[str], label: str | None = None, chars: int = CHARS,
        per_char: int = PER_CHAR) -> dict:
    from . import asr, dub, judge, pron, timbre, tts
    from .dub import _safe
    p = jobs.project(gid)
    specs = {v: parse_variant(v) for v in variants}
    label = label or time.strftime("%Y-%m-%d_%H-%M")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise ValueError("Недопустимое имя прогона стенда")
    out = p.dir / "bench" / label
    if out.exists():
        raise ValueError("Прогон стенда с таким именем уже существует")
    lines = pick_lines(jobs, p, chars, per_char)
    if not lines:
        raise RuntimeError("Нет реплик с русским текстом для стенда")
    out.mkdir(parents=True)
    report = jobs._progress(gid)
    names = {c["id"]: c["name"] for c in p.characters()}
    profiles = {char: jobs.character_profile(p, char) for char in {l["char"] for l in lines}}
    rows, total, done = [], len(lines) * len(variants), 0
    for v, spec in specs.items():
        backend = spec["backend"]
        for l in lines:
            done += 1
            if jobs.cancel_flag.is_set():
                raise RuntimeError("остановлено")
            report(f"Стенд: {v}, {names.get(l['char'], l['char'])} ({done}/{total})", done, total)
            row = {"id": l["id"], "variant": v, "char": l["char"], "name": names.get(l["char"], l["char"]),
                   "ru": l["ru"], "en": l["en"], "orig": l["orig"], "duration": l["duration"]}
            try:
                ref = _reference(jobs, p, l, spec, profiles)
                voice = timbre.voice(p, l["char"]) if spec["timbre"] else None
                if spec["timbre"] and not voice:
                    raise RuntimeError("нет модели тембра у персонажа")
                params = _params(jobs, p, l, backend)
                maxlen = max((l["duration"] or 2) * 2.0, 3.0) + len(l["ru"]) * 0.03
                t0 = time.time()
                (wav, sr), = tts.synthesize(backend, [l["ru"]], [ref], [maxlen], params, seed=dub.line_seed(l["id"], 0, 0))
                if voice:
                    pitch = 0 if spec["english"] else int(((p.get("donors") or {}).get(l["char"]) or {}).get("pitch") or 0)
                    wav, sr = tts.convert_timbre(wav, sr, voice, {"pitch": pitch, "auto_pitch": False, "index_rate": 0.0,
                                                                   "protect": 0.5})
                row["seconds"] = round(time.time() - t0, 2)
                f = out / _safe(v) / f"{_safe(l['id'])}.wav"
                audio.write(f, wav, sr)
                row["generated"] = f.relative_to(p.dir).as_posix()
                row["ratio"] = round(len(wav) / sr / max(l["duration"] or 0.3, 0.3), 3)
                row["f0_range"] = round(audio.f0_stats(wav, sr)[1], 2)
                row["_raw"] = (wav, sr)
            except Exception as e:
                row["error"] = str(e)[:200]
                print(f"стенд {v} / {l['id']}: {e}")
            rows.append(row)
    tts.unload()
    ok = [r for r in rows if not r.get("error")]
    if ok:
        report("Стенд: произношение, разборчивость, похожесть…")
        if pron.ready():
            for r, res in zip(ok, pron.check([r["_raw"] for r in ok], [r["ru"] for r in ok])):
                r["pron_cer"], r["pron_heard"] = res["cer"], res["heard"]
        heard = asr.transcribe([str(p.dir / r["generated"]) for r in ok], language="russian")
        for r, h in zip(ok, heard):
            r["whisper_cer"], r["heard"] = round(judge.cer(r["ru"], h), 3), h
        asr.unload()
        pron.unload()
        if judge.ecapa_ready():
            prof_emb = {c: judge.ecapa_embed([prof["audio"]])[0] for c, prof in profiles.items() if prof}
            for r, e in zip(ok, judge.ecapa_embed([r["_raw"] for r in ok])):
                if r["char"] in prof_emb:
                    r["ecapa_sim"] = round(float(e @ prof_emb[r["char"]]), 3)
        judge.unload()
    for r in rows:
        r.pop("_raw", None)
    summary = {}
    for v in variants:
        good = [r for r in rows if r["variant"] == v and not r.get("error")]
        s = {"rows": len(good), "errors": sum(1 for r in rows if r["variant"] == v and r.get("error"))}
        for k in METRICS:
            vals = [r[k] for r in good if r.get(k) is not None]
            s[k] = round(float(np.mean(vals)), 3) if vals else None
        summary[v] = s
    manifest = {"version": 1, "label": label, "game_id": gid, "variants": list(variants), "created_at": time.time(),
                "lines": len(lines), "rows": rows, "summary": summary}
    (out / "rows.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    render(out, manifest)
    hist = [b for b in (p.get("bench") or []) if b.get("label") != label]
    hist.append({"label": label, "variants": list(variants), "at": manifest["created_at"], "lines": len(lines), "summary": summary})
    p.set("bench", hist[-20:])
    return manifest


def render(out: Path, manifest: dict):
    e = html.escape
    gid = quote(manifest["game_id"], safe="")

    def player(path: str) -> str:
        url = f"/api/projects/{gid}/audio?" + urlencode({"kind": "file", "file": path})
        local = Path("..") / ".." / path
        return f'<audio controls preload="none" src="{e(local.as_posix())}" data-http="{e(url)}"></audio>'
    variants = manifest["variants"]
    head = ["<tr><th>Вариант</th><th>Реплик</th>"] + [f"<th>{e(k)}</th>" for k in METRICS] + ["<th>Ошибок</th></tr>"]
    summ = []
    for v in variants:
        s = manifest["summary"][v]
        summ.append(f"<tr><td><b>{e(v)}</b></td><td>{s['rows']}</td>"
                    + "".join(f"<td>{'' if s[k] is None else s[k]}</td>" for k in METRICS) + f"<td>{s['errors']}</td></tr>")
    by_line: dict[str, dict] = {}
    for r in manifest["rows"]:
        by_line.setdefault(r["id"], {})[r["variant"]] = r
    body = ["<table class=g><tr><th>Реплика</th><th>Оригинал</th>"] + [f"<th>{e(v)}</th>" for v in variants] + ["</tr>"]
    for lid, cells in by_line.items():
        first = next(iter(cells.values()))
        body.append(f"<tr><td><b>{e(first['name'])}</b><br>{e(first['ru'])}<br><small>{e(first.get('en') or '')}</small></td>"
                    f"<td>{player(first['orig'])}</td>")
        for v in variants:
            r = cells.get(v)
            if not r:
                body.append("<td>—</td>")
            elif r.get("error"):
                body.append(f"<td class=err>{e(r['error'])}</td>")
            else:
                meta = " · ".join(f"{k} {r[k]}" for k in ("pron_cer", "whisper_cer", "ecapa_sim", "f0_range", "ratio") if r.get(k) is not None)
                body.append(f"<td>{player(r['generated'])}<br><small>{e(meta)}</small>"
                            + (f"<br><small class=muted>слышно: {e(r.get('pron_heard') or r.get('heard') or '')}</small>" if r.get("pron_heard") or r.get("heard") else "")
                            + "</td>")
        body.append("</tr>")
    body.append("</table>")
    script = ('<script>if(location.protocol.startsWith("http"))document.querySelectorAll("audio")'
              '.forEach(a=>a.src=a.dataset.http)</script>')
    document = ('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                f'<title>Стенд движков · {e(manifest["label"])}</title>'
                '<style>body{background:#111827;color:#e5e7eb;font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 20px}'
                'table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #374151;padding:8px;vertical-align:top}'
                'th{background:#1f2937}audio{max-width:220px}.err{color:#f87171}.muted{color:#9ca3af}small{color:#aeb9ca}</style>'
                f'<h1>Стенд движков · {e(manifest["label"])}</h1>'
                '<p>Одни и те же реплики разными вариантами конвейера. pron_cer — произношение (ниже лучше, прокси акцента), '
                'whisper_cer — разборчивость (ниже лучше), ecapa_sim — похожесть на оригинальный голос (выше лучше), '
                'f0_range — живость интонации, ratio — длина к оригиналу. Файлы игры не изменены.</p>'
                '<table>' + "".join(head) + "".join(summ) + '</table>' + "".join(body) + script + '</html>')
    (out / "index.html").write_text(document, "utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True)
    parser.add_argument("--variants", required=True, help="через запятую: ru,ru+rvc,silero+rvc,ru@en,voxcpm@en")
    parser.add_argument("--label")
    parser.add_argument("--chars", type=int, default=CHARS)
    parser.add_argument("--per-char", type=int, default=PER_CHAR)
    args = parser.parse_args()
    from . import accel
    accel.activate()
    from .dub import DubJobs
    from .library import Library
    jobs = DubJobs(Library())
    jobs.state[args.game] = {"task": "enginebench", "message": "", "done": 0, "total": 0, "error": None, "queued": []}
    res = run(jobs, args.game, [v.strip() for v in args.variants.split(",") if v.strip()], args.label, args.chars, args.per_char)
    print(json.dumps({"label": res["label"], "summary": res["summary"]}, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
