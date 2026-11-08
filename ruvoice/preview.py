"""Small reproducible dubbing previews. This workflow never installs or approves game audio."""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote, urlencode

from . import audio


def run(jobs, gid: str, count: int = 2, engine: str | None = None,
        ids: list[str] | None = None, label: str | None = None) -> dict:
    from . import asr, judge, tts
    from .engines import BACKENDS
    p = jobs.project(gid)
    engine = engine or jobs.variant_info(gid)["source"]  # подпись прогона; сам синтез идёт по плану персонажа
    count = max(1, min(6, int(count)))
    label = label or time.strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise ValueError("Недопустимое имя предпросмотра")
    out = p.dir / "previews" / label
    if out.exists():
        raise ValueError("Предпросмотр с таким именем уже существует")
    candidates = [l for l in p.lines(ids=ids) if l["ru"] and l["en"] and l["status"] != "skipped"
                  and 2 <= (l["duration"] or 0) <= 9 and (p.dir / l["orig"]).is_file()]
    if ids:
        order = {value: i for i, value in enumerate(ids)}
        candidates.sort(key=lambda l: order[l["id"]])
    else:
        # Prefer one isolated-event case, then a different speaker. No game or character names.
        candidates.sort(key=lambda l: (not bool((l["info"] or {}).get("fit", {}).get("kept")), l["duration"]))
    chosen, seen = [], set()
    for l in candidates:
        if not ids and l["char"] in seen and len(seen) < min(count, len({x["char"] for x in candidates})):
            continue
        chosen.append(l)
        seen.add(l["char"])
        if len(chosen) == count:
            break
    if not chosen:
        raise ValueError("Нет реплик с исходным звуком и русскими субтитрами")
    out.mkdir(parents=True)
    rows = []
    report = jobs._progress(gid)
    jobs._release_translator()
    asr.unload()
    for i, line in enumerate(chosen):
        report(f"Предпросмотр {i + 1}/{len(chosen)}: {line['char']}", i, len(chosen))
        files, similarity = jobs._render(gid, p, line["char"], [line], out / str(i + 1), seed=23031 + i)
        orig = out / str(i + 1) / "original.wav"
        shutil.copy2(p.dir / line["orig"], orig)
        plan = (p.line(line["id"])["info"] or {}).get("performance") or {}
        w, sr = audio.read(p.dir / files[0])
        rows.append({"id": line["id"], "speaker": line["char"], "en": line["en"], "ru": line["ru"],
                     "original": str(orig.relative_to(p.dir)), "generated": files[0],
                     "voice_similarity": similarity, "seconds": round(len(w) / sr, 3),
                     "performance": plan, "seed": 23031 + i})
    tts.unload()
    heard = asr.transcribe([str(p.dir / r["generated"]) for r in rows], language="russian")
    for r, text in zip(rows, heard):
        r.update(heard=text, cer=round(judge.cer(r["ru"], text), 3))
    asr.unload()
    manifest = {"version": 1, "label": label, "game_id": gid, "engine": engine, "created_at": time.time(),
                "capabilities": BACKENDS.get(engine, {}).get("capabilities", {}), "installed": False, "rows": rows}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    _html(out, manifest)
    hist = p.get("previews") or []
    p.set("previews", (hist + [{"label": label, "engine": engine, "count": len(rows), "at": time.time()}])[-20:])
    return manifest


def _html(out: Path, manifest: dict):
    e = html.escape
    gid = quote(manifest["game_id"], safe="")
    body = []
    for r in manifest["rows"]:
        # Local file links work when opened directly; API links work in the running application.
        def player(path):
            rel = Path(path).relative_to(Path("previews") / manifest["label"]).as_posix()
            url = f"/api/projects/{gid}/audio?" + urlencode({"kind": "file", "file": path})
            return f'<audio controls preload="none" src="{e(rel)}" data-http="{e(url)}"></audio>'
        warnings = r["performance"].get("warnings", [])
        body.append(f'<article><h2>{e(r["speaker"])}</h2><p>{e(r["ru"])}</p><p class="muted">{e(r["en"])}</p>'
                    f'<div class="pair"><div>Оригинал<br>{player(r["original"])}</div>'
                    f'<div>Русский<br>{player(r["generated"])}</div></div>'
                    f'<p>CER: {r["cer"]:.1%} · Похожесть: {r["voice_similarity"]}</p>'
                    f'<p class="muted">{e("; ".join(warnings))}</p></article>')
    document = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">' \
        '<title>RuVoice — пробы озвучки</title><style>body{background:#111827;color:#e5e7eb;font:16px system-ui;' \
        'max-width:960px;margin:40px auto;padding:0 20px}article{background:#1f2937;padding:24px;border-radius:14px;' \
        'margin:24px 0}.pair{display:flex;flex-wrap:wrap;gap:24px}.muted{color:#aeb9ca}audio{max-width:100%}</style>' \
        f'<h1>Пробы озвучки · {e(manifest["engine"])}</h1><p>Отдельный предпросмотр. Файлы игры не изменены.</p>' \
        + "".join(body) + '<script>if(location.protocol.startsWith("http"))document.querySelectorAll("audio")' \
        '.forEach(a=>a.src=a.dataset.http)</script></html>'
    (out / "index.html").write_text(document, "utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, help="Game id from RuVoice library")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--engine", choices=["voxcpm", "ru", "custom"])
    parser.add_argument("--label")
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    from . import accel
    accel.activate()
    from .library import Library
    from .dub import DubJobs
    jobs = DubJobs(Library())
    jobs.state[args.game] = {"task": "preview", "message": "", "done": 0, "total": 0, "error": None, "queued": []}
    result = run(jobs, args.game, args.count, args.engine, args.ids, args.label)
    print(json.dumps({"label": result["label"], "engine": result["engine"],
                      "rows": [{k: r[k] for k in ("id", "speaker", "generated", "cer")} for r in result["rows"]]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
