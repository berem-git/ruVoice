"""Контрольная сцена: постоянный набор реплик (несколько персонажей, разные эмоции и длины), который
озвучивается текущим способом и складывается в projects/<id>/bench/<метка> вместе с оригиналами и оценками.
Любая смена модели или настроек сравнивается на нём — по цифрам и на слух (index.html рядом)."""
import json
import random
import time
from pathlib import Path

from . import audio

CHARS, PER_CHAR = 3, 10


def pick_lines(jobs, p) -> list[dict]:
    """Закреплённый набор: сохраняется в проекте, чтобы прогоны были сопоставимы."""
    saved = p.get("bench_lines")
    if saved:
        got = {l["id"]: l for l in p.lines(ids=saved)}
        lines = [got[i] for i in saved if i in got and got[i]["ru"]]
        if len(lines) >= PER_CHAR:
            return lines
    chars = [c for c in p.characters() if c["include"] and (c["lines"] or 0) >= 20]
    chars.sort(key=lambda c: -(c["lines"] or 0))
    out = []
    for c in chars[:CHARS]:
        cands = [l for l in p.lines(char=c["id"]) if l["ru"] and 1.5 <= (l["duration"] or 0) <= 9 and l["status"] != "skipped"]
        emos = jobs._emotions_for_char(p, c["id"]) if cands else {}
        random.Random(13).shuffle(cands)
        chosen, seen = [], {}
        for l in sorted(cands, key=lambda l: -emos.get(l["id"], ("neutral", 0))[1]):  # сначала выраженные эмоции
            e = emos.get(l["id"], ("neutral", 0))[0]
            if seen.get(e, 0) >= 3:
                continue
            seen[e] = seen.get(e, 0) + 1
            chosen.append(l)
            if len(chosen) >= PER_CHAR:
                break
        out += chosen
    p.set("bench_lines", [l["id"] for l in out])
    return out


def run(jobs, gid: str, label: str | None = None) -> dict:
    """Озвучить контрольную сцену текущим способом и оценить. Возвращает сводку (и пишет index.html)."""
    from . import asr, judge
    p = jobs.project(gid)
    report = jobs._progress(gid)
    lines = pick_lines(jobs, p)
    if not lines:
        raise RuntimeError("Нет реплик с русским текстом для контрольной сцены")
    label = label or time.strftime("%Y-%m-%d_%H-%M")
    out_dir = p.dir / "bench" / label
    engine = jobs._engine_for(gid)
    by_char: dict[str, list[dict]] = {}
    for l in lines:
        by_char.setdefault(l["char"], []).append(l)
    rows, done = [], 0
    for char, ls in by_char.items():
        if jobs.cancel_flag.is_set():
            return {}
        report(f"Контрольная сцена ({engine}): {char}", done, len(lines))
        files, sim = jobs._render(gid, p, char, ls, engine, None, out_dir / jobs._safe_name(char))
        heard = asr.transcribe([str(p.dir / f) for f in files], language="russian")
        for l, f, h in zip(ls, files, heard):
            orig, osr = audio.read(p.dir / l["orig"])
            gen, gsr = audio.read(p.dir / f)
            st = jobs._style_cache.get(l["id"], {})
            rows.append({"id": l["id"], "char": char, "ru": l["ru"], "en": l["en"], "orig": l["orig"], "gen": f,
                         "cer": round(judge.cer(l["ru"], h), 3), "heard": h, "emotion": st.get("emotion"),
                         "ratio": round((len(gen) / gsr) / max(len(orig) / osr, 0.3), 2), "sim": sim})
        done += len(ls)
    asr.unload()
    summary = {"label": label, "engine": engine, "at": time.time(), "lines": len(rows),
               "cer": round(sum(r["cer"] for r in rows) / len(rows), 3),
               "ratio": round(sum(r["ratio"] for r in rows) / len(rows), 2),
               "sim": round(sum(r["sim"] or 0 for r in rows) / len(rows), 3),
               "voices": {c: ("дообучен" if jobs._tts_voice(p, c) else "отпечаток") for c in by_char}}
    (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    _write_html(out_dir, p, summary, rows)
    hist = [b for b in (p.get("bench") or []) if b.get("label") != label] + [summary]
    p.set("bench", hist[-20:])
    return summary


def _write_html(out_dir: Path, p, summary: dict, rows: list[dict]):
    import html
    def rel(path):
        return html.escape(str(Path("..") / ".." / path).replace("\\", "/"))
    body = [f"<h1>Контрольная сцена {html.escape(summary['label'])}</h1>",
            f"<p>Способ: {summary['engine']} · ошибок текста {summary['cer']:.1%} · длина/оригинал {summary['ratio']} · "
            f"похожесть {summary['sim']} · голоса: {html.escape(json.dumps(summary['voices'], ensure_ascii=False))}</p>",
            "<table border=1 cellpadding=6><tr><th>Персонаж</th><th>Эмоция</th><th>Текст</th><th>Оригинал</th><th>Русский</th><th>CER</th><th>Длина</th></tr>"]
    for r in rows:
        body.append(f"<tr><td>{html.escape(r['char'])}</td><td>{r['emotion'] or ''}</td><td>{html.escape(r['ru'])}<br><small>{html.escape(r['en'] or '')}</small></td>"
                    f"<td><audio controls src='{rel(r['orig'])}'></audio></td><td><audio controls src='{rel(r['gen'])}'></audio></td>"
                    f"<td>{r['cer']:.2f}</td><td>{r['ratio']}</td></tr>")
    body.append("</table>")
    (out_dir / "index.html").write_text("<!doctype html><meta charset=utf-8><style>body{font-family:sans-serif;max-width:1200px;margin:20px auto}</style>"
                                        + "\n".join(body), "utf-8")
