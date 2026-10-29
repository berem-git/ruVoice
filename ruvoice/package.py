"""Экспорт и импорт озвучки: файл .ruvoice (zip) — готовая русская озвучка игры, которую можно перенести на другой
компьютер или отдать другу. Внутри: manifest.json (какая игра и чем сделано), lines.jsonl (русский текст,
статусы, оценки контролёра), audio\\*.opus (озвученные реплики, Opus 96 кбит/с — для речи без потерь на слух).
Оригинальных звуков игры в пакете нет: при импорте реплики извлекаются из своей копии игры, а русские
файлы сопоставляются по идентификаторам реплик.

Установке пакета нейросети не нужны — только сама игра."""
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import audio, tools

FORMAT = 1
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", "", (s or "").lower())


def _encode(src: Path, dst: Path) -> bool:
    ff = tools.ffmpeg()
    r = subprocess.run([str(ff), "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-c:a", "libopus",
                        "-b:a", "96k", "-application", "voip", str(dst)], capture_output=True, creationflags=NO_WINDOW)
    return r.returncode == 0 and dst.exists()


def _decode(src: Path, dst: Path, sr: int) -> bool:
    ff = tools.ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(ff), "-y", "-loglevel", "error", "-i", str(src), "-ar", str(sr), "-ac", "1",
                        "-c:a", "pcm_s16le", str(dst)], capture_output=True, creationflags=NO_WINDOW)
    return r.returncode == 0 and dst.exists()


def export_project(p, game: dict, adapter_id: str, out_dir: Path, progress=lambda *a: None, only_approved: bool = False) -> Path:
    """Собрать пакет .ruvoice из проекта. Возвращает путь к файлу."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {c["id"]: c["name"] for c in p.characters()}
    statuses = ("approved",) if only_approved else ("generated", "approved")
    lines = p.lines()
    dubbed = [l for l in lines if l["status"] in statuses and l["gen"] and (p.dir / l["gen"]).exists()]
    if not dubbed:
        raise RuntimeError("В проекте нет озвученных реплик — экспортировать нечего")
    stamp = time.strftime("%Y-%m-%d_%H-%M")
    target = out_dir / f"{_safe(game.get('name') or p.id)}_{stamp}.ruvoice"
    with tempfile.TemporaryDirectory(prefix="ruvoice_exp_") as td:
        td = Path(td)
        (td / "audio").mkdir()
        jobs = {l["id"]: (p.dir / l["gen"], td / "audio" / (_safe(l["id"]) + ".opus")) for l in dubbed}
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            for lid, ok in zip(jobs, ex.map(lambda a: _encode(*a), jobs.values())):
                done += 1
                if done % 50 == 0:
                    progress(f"Экспорт: сжатие реплик {done}/{len(jobs)}", done, len(jobs))
                if not ok:
                    jobs[lid] = None
        rows = []
        for l in lines:
            if not l["ru"]:
                continue
            j = jobs.get(l["id"])
            info = l["info"] or {}
            rows.append({"id": l["id"], "char": l["char"], "char_name": names.get(l["char"], l["char"]),
                         "ru": l["ru"], "ru_source": l["ru_source"], "status": l["status"] if j else
                         ("translated" if l["status"] in ("generated", "approved", "rejected") else l["status"]),
                         "audio": f"audio/{j[1].name}" if j else None, "duration": l["duration"],
                         "gen_duration": l["gen_duration"] if j else None,
                         "judge": {k: (info.get("judge") or {}).get(k) for k in ("score", "verdict", "reasons")} if j else None})
        manifest = {"format": FORMAT, "app": "RuVoice", "created": time.time(),
                    "game": {"name": game.get("name"), "appid": game.get("appid"), "adapter": adapter_id,
                             "engine": (game.get("analysis") or {}).get("engine")},
                    "characters": [{"id": c["id"], "name": c["name"]} for c in p.characters()],
                    "char_notes": p.get("char_notes") or {}, "selected_chars": p.get("selected_chars") or [],
                    "counts": {"lines_with_text": len(rows), "dubbed": sum(1 for r in rows if r["audio"])},
                    "voice": {"engine": "qwen3-tts-ru", "settings": {k: (p.get("settings") or {}).get(k)
                                                                     for k in ("quality", "event_gap", "tts_finetune")}}}
        (td / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
        (td / "lines.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
        progress("Экспорт: упаковка…")
        tmp = target.with_suffix(".part")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # opus уже сжат
            z.write(td / "manifest.json", "manifest.json")
            z.write(td / "lines.jsonl", "lines.jsonl")
            for f in sorted((td / "audio").iterdir()):
                z.write(f, f"audio/{f.name}")
        tmp.replace(target)
    return target


def read_manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        m = json.loads(z.read("manifest.json").decode("utf-8"))
    if m.get("app") != "RuVoice":
        raise RuntimeError("Это не пакет озвучки RuVoice")
    if int(m.get("format", 0)) > FORMAT:
        raise RuntimeError("Пакет сделан более новой версией RuVoice — обновите программу")
    return m


def match_game(lib, manifest: dict) -> dict | None:
    """Найти в библиотеке игру из пакета: по Steam appid, иначе по названию и способу озвучки."""
    from . import adapters
    g = manifest.get("game") or {}
    games = list(lib.games.values())
    if g.get("appid"):
        hit = next((x for x in games if str(x.get("appid") or "") == str(g["appid"])), None)
        if hit:
            return hit
    want = _norm_name(g.get("name"))
    cands = [x for x in games if want and (want in _norm_name(x["name"]) or _norm_name(x["name"]) in want)]
    for x in cands:
        a = adapters.for_game(x)
        if a and a.id == g.get("adapter"):
            return x
    return cands[0] if cands else None


def import_package(p, path: Path, progress=lambda *a: None) -> dict:
    """Положить озвучку из пакета в проект (реплики уже должны быть извлечены из игры)."""
    manifest = read_manifest(path)
    by_id = {l["id"]: l for l in p.lines()}
    if not by_id:
        raise RuntimeError("В проекте нет реплик — сначала нужно извлечь их из игры")
    p.gen_dir.mkdir(parents=True, exist_ok=True)
    matched, missing, audio_ok = 0, 0, 0
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory(prefix="ruvoice_imp_") as td:
        td = Path(td)
        rows = [json.loads(x) for x in z.read("lines.jsonl").decode("utf-8").splitlines() if x.strip()]
        tasks = []
        for r in rows:
            l = by_id.get(r["id"])
            if not l:
                missing += 1
                continue
            matched += 1
            fields = {"ru": r["ru"], "ru_source": r.get("ru_source") or "import"}
            if r.get("audio"):
                src = td / r["audio"]
                z.extract(r["audio"], td)
                sr = audio.read(p.dir / l["orig"])[1] if (p.dir / l["orig"]).exists() else 48000
                dst = p.gen_dir / Path(l["orig"]).name
                tasks.append((l, r, src, dst, sr, fields))
            else:
                fields["status"] = "translated" if l["status"] not in ("skipped",) else l["status"]
                p.update_line(l["id"], **fields)
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            for (l, r, src, dst, sr, fields), ok in zip(tasks, ex.map(lambda t: _decode(t[2], t[3], t[4]), tasks)):
                done += 1
                if done % 50 == 0:
                    progress(f"Импорт: реплики {done}/{len(tasks)}", done, len(tasks))
                if not ok:
                    continue
                audio_ok += 1
                info = dict(l["info"] or {})
                if r.get("judge"):
                    info["judge"] = r["judge"]
                info["imported"] = True
                p.update_line(l["id"], **fields, gen=str(dst.relative_to(p.dir)), gen_duration=r.get("gen_duration"),
                              status="approved" if r.get("status") == "approved" else "generated", info=info, error=None)
    names = {c["id"]: c["name"] for c in p.characters()}
    for c in manifest.get("characters") or []:
        if c["id"] in names and c["name"] != names[c["id"]]:
            p.update_character(c["id"], name=c["name"])
    if manifest.get("char_notes"):
        p.set("char_notes", {**(p.get("char_notes") or {}), **manifest["char_notes"]})
    return {"matched": matched, "missing": missing, "audio": audio_ok, "game": (manifest.get("game") or {}).get("name")}
