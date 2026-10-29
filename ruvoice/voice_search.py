"""Compare cloning recipes against unchanged originals before changing production defaults.

Outputs are project previews only: no line is approved, replaced, or installed.
Speaker similarity is an imperfect measurement, not a percentage of audible identity.
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np

RECIPES = [
    {"name": "source_reference", "reference": "source", "cfg_value": 2., "steps": 20},
    {"name": "profile_reference", "reference": "profile", "style": True, "cfg_value": 2., "steps": 10},
    {"name": "profile_reference_seed2", "reference": "profile", "style": True, "cfg_value": 2., "steps": 10, "seed_offset": 17},
    {"name": "profile_reference_plain", "reference": "profile", "cfg_value": 2., "steps": 20},
    {"name": "profile_reference_soft", "reference": "profile", "cfg_value": 1.5, "steps": 20},
]


def select(candidates: list[dict]) -> dict | None:
    """Reject wrong words, missing measurements, clipping and runaway generation before ranking."""
    valid = [c for c in candidates if c.get("cer", 1.) <= .05
             and c.get('pron_cer', 1.) <= .08 and not c.get('rejected')
             and .45 <= c.get("duration_ratio", 0.) <= 1.8
             and c.get("clipped_fraction", 1.) < .01
             and all(np.isfinite(c.get(k, float("nan"))) for k in ("ve_source", "ve_holdout", "ecapa_source"))]
    return max(valid, key=lambda c: .55 * c["ve_source"] + .25 * c["ve_holdout"] + .20 * c["ecapa_source"], default=None)


def write_report(out: Path, manifest: dict):
    e = html.escape
    gid = quote(manifest["game_id"], safe="")
    def player(path):
        query = urlencode({"kind": "file", "file": path})
        return f'<audio controls preload="metadata" src="/api/projects/{gid}/audio?{e(query)}"></audio>'
    rows = []
    for item in manifest["lines"]:
        rows.append(f'<section><h2>{e(item["speaker"])} · {e(item["ru"])}</h2><p>Оригинал</p>{player(item["original"])}')
        for candidate in item["candidates"]:
            best = " · выбран по проверкам" if candidate["name"] == item.get("selected") else ""
            pronunciation = (f' · CTC {candidate["pron_cer"]:.1%}' if 'pron_cer' in candidate else '')
            rows.append(f'<article><h3>{e(candidate["name"])}{best}</h3>{player(candidate["file"])}'
                        f'<p>CER {candidate["cer"]:.1%} · сходство с репликой {candidate["ve_source"]:.3f}'
                        f' · с другими записями {candidate["ve_holdout"]:.3f}'
                        f' · ECAPA {candidate["ecapa_source"]:.3f}{pronunciation}</p>'
                        f'<p>{e(candidate["heard"])}</p></article>')
        rows.append('</section>')
    (out / "index.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>RuVoice — сравнение голосов</title>'
        '<style>body{background:#111827;color:#e5e7eb;font:16px system-ui;max-width:960px;margin:32px auto;padding:16px}'
        'section{margin:32px 0}article{background:#1f2937;padding:20px;margin:16px 0;border-radius:12px}'
        'audio{max-width:100%}p{color:#cbd5e1}</style><h1>Сравнение способов клонирования</h1>'
        '<p>ИИ-озвучка. Метрики помогают выбирать варианты, но естественность и сходство проверяются на слух.</p>'
        + ''.join(rows) + '</html>', 'utf-8')


def run(gid: str, ids: list[str], label: str, baseline: str | None = None, recipes=None, jobs=None):
    from . import accel
    accel.activate()
    from . import asr, audio, judge, tts, pron
    from .dub import DubJobs
    from .engines import client
    from .library import Library
    from .performance import compose, fingerprint
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", label):
        raise ValueError("Invalid preview label")
    owns_jobs = jobs is None
    jobs = jobs or DubJobs(Library())
    if owns_jobs:
        jobs.state[gid] = {"task": "voice_search", "message": "", "done": 0, "total": 0, "error": None, "queued": []}
    p = jobs.project(gid)
    if not pron.ready():
        raise RuntimeError('Для подбора голоса нужна модель проверки русского произношения')
    recipes = recipes or RECIPES
    if not jobs.settings(gid).get('allow_cross_language_prompt', False):
        recipes = [r for r in recipes if not r.get('continuation')]
    if not recipes:
        raise ValueError('Английский речевой контекст отключён: он может переносить акцент')
    out = p.dir / "previews" / label
    out.mkdir(parents=True, exist_ok=False)
    previous = {}
    if baseline:
        old = json.loads((p.dir / "previews" / baseline / "manifest.json").read_text('utf-8'))
        if 'rows' in old:
            previous = {r['id']: r for r in old['rows']}
        else:
            for row in old['lines']:
                candidate = next((c for c in row['candidates'] if c['name'] == row.get('selected')), None)
                if candidate:
                    previous[row['id']] = {"generated": candidate['file']}
    manifest = {"version": 1, "game_id": gid, "label": label, "created_at": time.time(), "installed": False, "lines": []}
    def checkpoint():
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    try:
        for index, ident in enumerate(ids):
            line = p.line(ident)
            if not line or not line['en'] or not line['ru']:
                raise ValueError("A source recording and both texts are required")
            folder = out / str(index + 1)
            folder.mkdir()
            source = p.dir / line['orig']
            prof = jobs.character_profile(p, line['char'])
            others = [x for x in p.lines(char=line['char']) if x['id'] != ident and x['en']
                      and 3.0 <= (x['duration'] or 0) <= 7.0 and (p.dir / x['orig']).is_file()]
            # Different source assets, not repeated copies from overlapping banks.
            seen, unique = {line['en'].strip().lower()}, []
            for x in others:
                if x['en'].strip().lower() not in seen:
                    seen.add(x['en'].strip().lower())
                    unique.append(x)
            if len(unique) < 3:
                raise ValueError("Need at least three independent reference lines to evaluate identity")
            other = unique[0]
            used_prompts = {unique[min(r.get('other_index', 0), len(unique) - 1)]['id']
                            for r in recipes or RECIPES if r.get('prompt_other')}
            heldout = [x for x in unique if x['id'] not in used_prompts
                       and x['en'].strip() not in prof['text']][:5]
            if len(heldout) < 3:
                raise ValueError("Need three evaluation recordings absent from the cloning references")
            plan = jobs._source_performance(p, line)
            style = jobs._style_params(p, line, 'voxcpm')
            asr.unload()
            item = {"id": ident, "speaker": line['char'], "ru": line['ru'], "en": line['en'],
                    "original": line['orig'], "heldout_ids": [x['id'] for x in heldout], "candidates": []}
            manifest['lines'].append(item)
            if ident in previous:
                item['candidates'].append({"name": "previous", "file": previous[ident]['generated'],
                                           "raw_file": previous[ident]['generated'], "recipe": {"baseline": baseline}})
            elif not baseline:
                saved = (line.get('info') or {}).get('voice_recipe') or {}
                existing = saved.get('output') or line.get('gen')
                raw_existing = saved.get('raw_output') or existing
                if existing and raw_existing:
                    paths = [(p.dir / name).resolve() for name in (existing, raw_existing)]
                    if all(path.is_relative_to(p.dir.resolve()) and path.is_file() for path in paths):
                        item['candidates'].append({"name": "previous", "file": existing,
                            "raw_file": raw_existing, "recipe": {"baseline": "current"}})
            original, osr = audio.read(source)
            for recipe in recipes or RECIPES:
                if jobs.cancel_flag.is_set():
                    raise RuntimeError("Подбор голоса остановлен")
                name = recipe['name']
                print(f"GENERATE {index + 1}/{len(ids)} {line['char']} {name}", flush=True)
                jobs._progress(gid)(f"Подбор голоса: {index + 1}/{len(ids)} · {name}")
                ref_audio = {'source': str(source), 'profile': prof['audio'], 'other': str(p.dir / other['orig'])}[recipe['reference']]
                prompt_line = unique[min(recipe.get('other_index', 0), len(unique) - 1)] if recipe.get('prompt_other') else line
                req = {"op": "generate", "text": line['ru'], "ref_audio": ref_audio,
                       "max_seconds": max(line['duration'] * 2.5, 5), "seed": 23031 + index + recipe.get('seed_offset', 0),
                       "params": {"cfg_value": recipe['cfg_value'], "steps": recipe['steps'],
                                  "instruct": style.get('instruct') if recipe.get('style') else ''},
                       "out": str(folder / (name + '_raw.wav'))}
                if recipe.get('continuation'):
                    req.update(prompt_audio=prof['audio'] if recipe.get('prompt_profile') else str(p.dir / prompt_line['orig']),
                               prompt_text=prof['text'] if recipe.get('prompt_profile') else prompt_line['en'])
                client.get('voxcpm').request(req)
                w, sr = audio.read(req['out'])
                fitted, fit = compose(w, sr, original, osr, plan)
                path = folder / (name + '.wav')
                audio.write(path, fitted, osr)
                item['candidates'].append({"name": name, "file": str(path.relative_to(p.dir)),
                    "raw_file": str(Path(req['out']).relative_to(p.dir)), "recipe": recipe, "fit": fit,
                    "reproduce": {"audio": str(Path(ref_audio).relative_to(p.dir)),
                        "prompt_audio": str(Path(req['prompt_audio']).relative_to(p.dir)) if req.get('prompt_audio') else None,
                        "prompt_text": req.get('prompt_text'), "params": req['params'], "seed": req['seed']}})
                checkpoint()
        tts.unload()
        client.stop('voxcpm')
        print('CHECK words and speaker identity', flush=True)
        for item in manifest['lines']:
            source_embed = judge.speaker_embed([str(p.dir / item['original'])])
            heldout_paths = [str(p.dir / p.line(i)['orig']) for i in item['heldout_ids']]
            heldout_embed = judge.speaker_embed(heldout_paths)
            paths = [str(p.dir / c['raw_file']) for c in item['candidates']]
            ecapa = judge.ecapa_embed([str(p.dir / item['original'])] + paths)
            source_w, source_sr = audio.read(p.dir / item['original'])
            source_duration = max(.3, len(audio.trim_silence(source_w, source_sr)) / source_sr)
            for i, (c, path) in enumerate(zip(item['candidates'], paths)):
                w, sr = audio.read(path)
                c.update(ve_source=round(judge.voice_similarity(path, source_embed), 4),
                         ve_holdout=round(judge.voice_similarity(path, heldout_embed), 4),
                         ecapa_source=round(float(np.dot(ecapa[0], ecapa[i + 1])), 4),
                         duration_ratio=round(len(audio.trim_silence(w, sr)) / sr / source_duration, 3),
                         clipped_fraction=float(np.mean(np.abs(w) >= .999)))
            judge.unload()
            heard = asr.transcribe(paths, language='russian')
            for c, text in zip(item['candidates'], heard):
                c.update(heard=text, cer=round(judge.cer(item['ru'], text), 4))
                print(item['speaker'], c['name'], c['cer'], c['ve_source'], c['ve_holdout'], c['ecapa_source'], flush=True)
            asr.unload()
            pronunciation = pron.check(paths, [item['ru']] * len(paths))
            pron.unload()
            rejected = ((p.line(item['id']).get('info') or {}).get('voice_feedback') or {}).get('rejected_output')
            for c, result in zip(item['candidates'], pronunciation):
                c.update(pron_cer=result['cer'], pron_heard=result['heard'], pron_words=result['words'],
                         rejected=bool(rejected and c['file'] == rejected))
            selected = select(item['candidates'])
            item['selected'] = selected['name'] if selected else None
            if selected and selected.get('reproduce'):
                line = p.line(item['id'])
                info = dict(line.get('info') or {})
                info['voice_recipe'] = {**selected['reproduce'], 'source': fingerprint(p.dir / line['orig']),
                                        'ru': line['ru'], 'search': label, 'candidate': selected['name'],
                                        'output': selected['file'], 'raw_output': selected['raw_file']}
                p.update_line(line['id'], info=info)
            checkpoint()
        write_report(out, manifest)
        hist = p.get('previews') or []
        p.set('previews', (hist + [{"label": label, "engine": "voxcpm search", "count": len(ids), "at": time.time()}])[-20:])
        print(json.dumps({"label": label, "selected": [{"speaker": x['speaker'], "name": x['selected']} for x in manifest['lines']]}, ensure_ascii=False), flush=True)
        return manifest
    finally:
        client.stop('voxcpm')
        asr.unload()
        judge.unload()
        pron.unload()
        if owns_jobs:
            p.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--baseline')
    args = parser.parse_args()
    run(args.game, args.ids, args.label, args.baseline)


if __name__ == '__main__':
    main()
