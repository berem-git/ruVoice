"""Russian anchor auditions across several scenes; never change generated/game audio or approve a voice."""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np

from . import audio, identity


def select_lines(p, seed: dict, count: int = 3) -> tuple[list[dict], dict]:
    """Use original acoustic evidence, not a character name guessed from subtitles."""
    from . import judge
    candidates = [l for l in p.lines(char=seed["char"]) if l["ru"] and l["orig"]
                  and l["status"] != "skipped" and 2.5 <= (l["duration"] or 0) <= 8
                  and (p.dir / l["orig"]).is_file()]
    cache = p.dir / "speakers.npz"
    have = {}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            have = dict(zip(z["ids"].tolist(), z["E"]))
    if seed["id"] not in have:
        have[seed["id"]] = judge.ecapa_embed([str(p.dir / seed["orig"])])[0]
    e = have[seed["id"]]
    missing = [l for l in candidates if l["id"] not in have][:24]
    if missing:
        have.update(zip([l["id"] for l in missing], judge.ecapa_embed([str(p.dir / l["orig"]) for l in missing])))
    scored = [(l, float(have[l["id"]] @ e)) for l in candidates if l["id"] in have]
    scored.sort(key=lambda x: -x[1])
    # Conservative proposal only. Low similarity is not proof of another speaker.
    accepted = [(l, s) for l, s in scored if s >= 0.65 and l["id"] != seed["id"]]
    chosen, texts = [], set()
    for l, score in accepted:
        if l["ru"] in texts:
            continue
        chosen.append(l)
        texts.add(l["ru"])
        if len(chosen) >= count:
            break
    if len(chosen) < 2:
        raise ValueError("Недостаточно похожих исходных реплик: сначала проверьте разметку говорящих")
    return chosen, {"seed": seed["id"], "compared": len(scored), "below_threshold": sum(s < 0.65 for _, s in scored),
                    "selected": [{"id": l["id"], "similarity": s} for l, s in accepted if l in chosen],
                    "note": "Акустическая похожесть — подсказка, принадлежность голоса нужно проверить на слух"}


def run(p, seed_ids: list[str], label: str | None = None, report=print, cancel=lambda: False) -> dict:
    from . import asr, judge, tts
    label = label or time.strftime("fixed_voice_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise ValueError("Недопустимое имя предпросмотра")
    out = p.dir / "previews" / label
    if out.exists():
        raise ValueError("Предпросмотр уже существует")
    if not 1 <= len(seed_ids) <= 4:
        raise ValueError("Выберите от одного до четырёх исходных голосов")
    out.mkdir(parents=True)
    groups = []
    try:
        tts.unload()
        asr.unload()
        for i, lid in enumerate(seed_ids):
            seed = p.line(lid)
            if not seed or not seed["ru"] or not 2 <= (seed["duration"] or 0) <= 15:
                raise ValueError("Для исходного голоса нужна реплика с текстом длиной от 2 до 15 секунд")
            report(f"Проверка исходного голоса {i + 1}/{len(seed_ids)}")
            lines, audit = select_lines(p, seed)
            folder = out / str(i + 1)
            folder.mkdir()
            shutil.copy2(p.dir / seed["orig"], folder / "source.wav")
            # One Russian utterance long enough to cover more than one short exclamation.
            text = " ".join(l["ru"] for l in [seed, lines[0]])
            groups.append({"char": seed["char"], "seed": seed["id"], "text": text, "audit": audit,
                           "source": (folder / "source.wav").relative_to(p.dir).as_posix(), "anchors": [], "rows": [],
                           "_folder": folder, "_lines": lines})
        judge.unload()
        for i, group in enumerate(groups):
            seed = p.line(group["seed"])
            for take in range(2):
                if cancel():
                    raise InterruptedError("Пробы остановлены")
                report(f"Русский эталон: голос {i + 1}, вариант {take + 1}/2")
                wav, sr = tts.synthesize("ru", [group["text"]],
                    [{"audio": str(p.dir / seed["orig"]), "text": None}], [25],
                    {"temperature": 0.7, "subtalker_temperature": 0.45}, seed=42101 + take)[0]
                path = group["_folder"] / f"anchor_{take + 1}.wav"
                audio.write(path, wav, sr)
                row = identity.propose(p, seed["char"], path.relative_to(p.dir).as_posix(), group["text"], "voxcpm", seed["id"])
                group["anchors"].append(row)
        tts.unload()  # Qwen and VoxCPM do not occupy VRAM together.
        for i, group in enumerate(groups):
            row = identity.get(p, group["anchors"][0]["id"])
            for j, line in enumerate(group["_lines"]):
                if cancel():
                    raise InterruptedError("Пробы остановлены")
                report(f"Фиксированный русский голос: {i + 1}/{len(groups)}, реплика {j + 1}/3")
                wav, sr = tts.synthesize("voxcpm", [line["ru"]], [identity.reference(row)],
                    [max(8, len(line["ru"]) * 0.14)],
                    {"instruct": "Естественная русская речь, чёткое произношение, спокойный темп.", "steps": 15}, seed=58217)[0]
                path = group["_folder"] / f"trial_{j + 1}.wav"
                audio.write(path, wav, sr)
                orig = group["_folder"] / f"original_{j + 1}.wav"
                shutil.copy2(p.dir / line["orig"], orig)
                old = None
                if line["gen"] and (p.dir / line["gen"]).is_file():
                    old = group["_folder"] / f"previous_{j + 1}.wav"
                    shutil.copy2(p.dir / line["gen"], old)
                group["rows"].append({"id": line["id"], "ru": line["ru"], "anchor": row["id"],
                    "original": orig.relative_to(p.dir).as_posix(), "generated": path.relative_to(p.dir).as_posix(),
                    "previous": old.relative_to(p.dir).as_posix() if old else None})
        tts.unload()
        for group in groups:
            paths = [a["audio"] for a in group["anchors"]] + [r["generated"] for r in group["rows"]]
            heard = asr.transcribe([str(p.dir / x) for x in paths], language="russian")
            texts = [a["text"] for a in group["anchors"]] + [r["ru"] for r in group["rows"]]
            checks = [{"heard": h, "cer": round(judge.cer(t, h), 3)} for t, h in zip(texts, heard)]
            group["checks"] = checks
        asr.unload()
        for group in groups:
            paths = [group["anchors"][0]["audio"]] + [r["generated"] for r in group["rows"]]
            E = judge.ecapa_embed([str(p.dir / x) for x in paths])
            for row, e in zip(group["rows"], E[1:]):
                row["identity_similarity"] = round(float(e @ E[0]), 3)
            del group["_folder"], group["_lines"]
        manifest = {"version": 1, "kind": "voice_identity_trials", "label": label, "game_id": p.game_id,
                    "engine": "voxcpm", "created_at": time.time(), "installed": False, "approved": False, "groups": groups}
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        render(out, manifest)
        hist = p.get("previews") or []
        p.set("previews", (hist + [{"label": label, "engine": "voxcpm", "count": sum(len(g["rows"]) for g in groups),
                                    "at": time.time()}])[-20:])
        return manifest
    finally:
        tts.unload()
        asr.unload()
        judge.unload()


def render(out: Path, manifest: dict):
    e = html.escape
    gid = quote(manifest["game_id"], safe="")
    def player(path):
        if not path:
            return ""
        url = f"/api/projects/{gid}/audio?" + urlencode({"kind": "file", "file": path})
        # Relative to the project previews directory; anchors live alongside it.
        local = Path("../..") / path
        return f'<audio controls preload="none" src="{e(local.as_posix())}" data-http="{e(url)}"></audio>'
    body = []
    for group in manifest["groups"]:
        label = group.get("display_name") or group["char"]
        body.append(f'<section><h2>{e(label)}</h2><p>Исходный голос (проверьте говорящего)</p>{player(group["source"])}')
        for i, anchor in enumerate(group["anchors"]):
            body.append(f'<p>Русский эталон {i + 1} · {"использован в пробах ниже" if i == 0 else "альтернатива без контрольных проб"}</p>'
                        f'{player(anchor["audio"])}<p>{e(anchor["text"])}</p>'
                        f'<button data-anchor="{e(anchor["id"])}">Закрепить этот голос после прослушивания</button>')
        for row in group["rows"]:
            body.append(f'<article><h3>{e(row["ru"])}</h3><p>Оригинал</p>{player(row["original"])}'
                        f'<p>Ранее установленная озвучка</p>{player(row["previous"])}'
                        + (f'<p>Русский образец без продолжения по тексту</p>{player(row["reference_only"])}' if row.get("reference_only") else '') +
                        f'<p>Проба с фиксированным русским эталоном</p>{player(row["generated"])}'
                        f'<small>Сходство с эталоном ECAPA: {row["identity_similarity"]}. Это не оценка акцента.</small></article>')
        body.append('</section>')
    script = '''<script>if(location.protocol.startsWith('http')) {
      document.querySelectorAll('audio').forEach(a=>a.src=a.dataset.http);
      document.querySelectorAll('[data-anchor]').forEach(b=>b.onclick=async()=>{
        b.disabled=true;try { const r=await fetch(ENDPOINT,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate:b.dataset.anchor})});
          if(!r.ok)throw Error((await r.json()).detail||'Ошибка');b.textContent='Голос закреплён';
        }catch(e){alert(e.message);b.disabled=false;}});
    } else document.querySelectorAll('button').forEach(b=>{b.disabled=true;b.title='Откройте пробы в RuVoice для выбора эталона';});</script>'''
    script = script.replace('ENDPOINT', json.dumps(f'/api/projects/{gid}/voice-identities'))
    document = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пробы постоянного голоса</title>' \
        '<style>body{background:#111827;color:#eee;font:16px system-ui;max-width:1000px;margin:30px auto;padding:20px}section,article{background:#1f2937;padding:20px;margin:20px 0;border-radius:12px}article{background:#111827}button{padding:12px}audio{max-width:100%}</style>' \
        '<h1>Пробы постоянного русского голоса</h1><p>Эталоны ещё не одобрены. ' \
        'Пробы созданы в спокойной подаче, без подгонки времени и вклеивания оригинальной речи. ' \
        'Проверьте акцент, тембр и постоянство на разных фразах. Файлы игры не изменены.</p>' + ''.join(body) + script + '</html>'
    (out / "index.html").write_text(document, "utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', required=True)
    parser.add_argument('--seed-lines', nargs='+', required=True)
    parser.add_argument('--label')
    args = parser.parse_args()
    from . import accel
    from .project import Project
    accel.activate()
    result = run(Project(args.game), args.seed_lines, args.label, report=lambda s: print(s, flush=True))
    print(json.dumps({"label": result["label"], "groups": len(result["groups"])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
