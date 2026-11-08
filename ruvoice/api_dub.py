"""API озвучки: проекты, реплики, прослушивание, одобрение, установка; языки; ускоритель и модели."""
import os
import re
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import accel, adapters, engines, langmgr, models, timbre, tools
from .dub import DubJobs
from .library import Library
from .project import Project


class Steps(BaseModel):
    steps: list[str]
    ids: list[str] | None = None


class Sample(BaseModel):
    char: str
    count: int = 5


class LineEdit(BaseModel):
    ru: str | None = None
    status: str | None = None


class Mark(BaseModel):
    status: str
    ids: list[str] | None = None
    char: str | None = None
    from_status: str | None = "generated"
    min_score: int | None = None


class Ids(BaseModel):
    ids: list[str]


class ReviewReq(BaseModel):
    char: str
    note: str | None = None


class CharEdit(BaseModel):
    name: str | None = None
    include: bool | None = None
    ref_line: str | None = None


class Merge(BaseModel):
    into: str
    others: list[str]


class Kinds(BaseModel):
    kinds: list[str]


class Variant(BaseModel):
    variant: str


class TrainReq(BaseModel):
    char: str
    steps: int = 300       # эпохи обучения RVC (method=rvc)
    method: str = "tts"    # tts — дообучить модель речи под голос (LoRA); rvc — модель переноса тембра


class DownloadReq(BaseModel):
    keys: list[str] | None = None
    tier: str | None = None


ALLOWED_STEPS = {"extract", "transcribe", "translate", "generate", "judge", "install", "uninstall", "prepare", "events_pass", "export",
                 "samples", "preview"}


def _proposal_view(prop: dict | None) -> dict | None:
    """Предложение «кто говорит» для интерфейса — без длинных списков реплик."""
    if not prop:
        return None
    return {"at": prop.get("at"), "covered": prop.get("covered"), "total": prop.get("total"),
            "voices": [{k: v[k] for k in ("key", "name", "count", "f0", "voice_type", "from", "samples", "auto_name",
                                          "kind", "groups", "moves", "hero")
                        if k in v} | {"texts": v.get("texts", [])[:3], "llm": v.get("llm")} for v in prop.get("voices", [])]}


def create_router(lib: Library, dub: DubJobs) -> APIRouter:
    r = APIRouter(prefix="/api")
    system = {"accel": None, "amd_install": {"running": False, "message": ""}}

    def warm():
        try:
            system["accel"] = accel.info()
        except Exception as e:
            system["accel"] = {"error": str(e)}
    threading.Thread(target=warm, daemon=True).start()

    def _game(gid: str) -> dict:
        g = lib.get(gid)
        if not g:
            raise HTTPException(404, "Игра не найдена")
        return g

    def _project(gid: str) -> Project:
        _game(gid)
        return dub.project(gid)

    # --- система ---

    @r.get("/system")
    def system_info():
        from . import power
        from .engines import install
        return {"accel": system["accel"], "tools": tools.status(), "models": models.status(system["accel"]),
                "engines": engines.describe(), "engine_envs": install.status(), "amd_install": system["amd_install"],
                "variants": accel.VARIANTS, "power": power.describe()}

    @r.post("/system/engines/install")
    def install_engine(body: dict):
        """Поставить среду движка (rvc / omnivoice / voxcpm) в фоне; ход — в GET /system → engine_envs."""
        from .engines import install
        engine = str((body or {}).get("engine") or "")
        if engine not in install.ENVS:
            raise HTTPException(400, "Неизвестная среда движка")
        return install.install_async(engine)

    def _dir_gb(path) -> float:
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return 0.0
        return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30, 2)

    @r.get("/system/storage")
    def storage():
        """Что занимает место: модели, кэш, проекты озвучки — и у каких проектов игры больше нет на диске."""
        from pathlib import Path
        from .paths import PROJECT_DIR
        from .project import PROJECTS_DIR
        import json as _json
        projects = []
        if PROJECTS_DIR.exists():
            for d in sorted(PROJECTS_DIR.iterdir()):
                if not (d / "project.db").exists():
                    continue
                try:
                    import sqlite3
                    c = sqlite3.connect(d / "project.db", timeout=5)
                    meta = {k: _json.loads(v) for k, v in c.execute("SELECT key, value FROM meta")}
                    c.close()
                except Exception:
                    meta = {}
                gid = meta.get("game_id") or next((g["id"] for g in lib.games.values()
                                                    if __import__("re").sub(r"[^A-Za-z0-9_-]+", "_", g["id"]) == d.name), None)
                game = lib.get(gid) if gid else None
                from . import cleanup
                g = cleanup.project_garbage(d, dry=True)
                parts = {k: _dir_gb(d / k) for k in ("orig", "gen", "voices", "profiles", "backup", "samples", "bench")}
                projects.append({"dir": d.name, "game_id": gid, "name": meta.get("game_name") or d.name,
                                 "size_gb": _dir_gb(d), "parts_gb": parts, "garbage_gb": round(g["bytes"] / 2**30, 2),
                                 "garbage": g["items"][:12], "broken_lines": g["fixed_lines"],
                                 "game_missing": not (game and Path(game["path"]).is_dir()),
                                 "installed_count": meta.get("installed_count") or 0,
                                 "lines_done": bool(meta.get("prepared_at"))})
        from . import cleanup
        cache_garbage = round(cleanup.cache_garbage(dry=True) / 2**30, 2)
        return {"models_gb": _dir_gb(PROJECT_DIR / "models"), "cache_gb": _dir_gb(PROJECT_DIR / "cache"),
                "garbage_gb": round(cache_garbage + sum(x["garbage_gb"] for x in projects), 2),
                "runtime_gb": _dir_gb(PROJECT_DIR / "runtime"), "tools_gb": _dir_gb(PROJECT_DIR / "tools"),
                "projects": projects, "projects_gb": round(sum(x["size_gb"] for x in projects), 2),
                "disk_free_gb": round(shutil.disk_usage(PROJECT_DIR).free / 2**30, 1)}

    @r.delete("/system/storage/project/{pid}")
    def delete_project_dir(pid: str):
        """Удалить проект озвучки по имени папки (для игр, которых уже нет на диске)."""
        from .project import PROJECTS_DIR
        d = PROJECTS_DIR / pid
        if not d.is_dir() or ".." in pid or "/" in pid or "\\" in pid:
            raise HTTPException(404, "Проект не найден")
        for gid, pr in list(dub._projects.items()):
            if pr.dir == d:
                if dub.current == gid:
                    raise HTTPException(400, "Сначала остановите текущую задачу")
                pr.close()
                dub._projects.pop(gid, None)
        shutil.rmtree(d, ignore_errors=True)
        return {"ok": True, "left": d.exists()}

    @r.post("/system/storage/cleanup")
    def storage_cleanup():
        """Убрать весь мусор (проекты + кэш); проект, который сейчас озвучивается, не трогается."""
        res = dub.collect_garbage(dry=False)
        return {"freed_gb": round(res["bytes"] / 2**30, 2), "fixed_lines": res["fixed_lines"]}

    @r.delete("/system/storage/cache")
    def clear_cache():
        from .paths import PROJECT_DIR
        c = PROJECT_DIR / "cache"
        for x in c.iterdir() if c.exists() else []:
            if x.name == "uv":
                continue
            shutil.rmtree(x, ignore_errors=True) if x.is_dir() else x.unlink(missing_ok=True)
        return {"ok": True}

    @r.post("/system/power")
    def set_power(body: Variant):
        """Нагрузка на систему: low | medium | full — действует сразу, со следующей пачки работы."""
        from . import power
        try:
            power.set_level(body.variant)
        except ValueError:
            raise HTTPException(400, "Уровень: low, medium или full")
        return power.describe()

    @r.post("/system/accel")
    def set_accel(body: Variant):
        try:
            accel.set_preferred(body.variant)
        except ValueError:
            raise HTTPException(400, "Неизвестный вариант")
        return {"ok": True, "restart": True}

    @r.post("/system/restart")
    def restart_app():
        """Перезапуск программы (нужен после смены ускорителя)."""
        if dub.current:
            raise HTTPException(400, "Сначала остановите текущую задачу")
        accel.restart_app()
        return {"ok": True}

    @r.post("/system/install-amd")
    def install_amd():
        st = system["amd_install"]
        if st["running"]:
            return st

        def run():
            st.update(running=True, message="Подготовка…", error=None)
            try:
                accel.install_rocm(lambda m: st.update(message=m))
            except Exception as e:
                st.update(error=str(e), message="Ошибка установки")
            finally:
                st["running"] = False
        threading.Thread(target=run, daemon=True).start()
        return st

    @r.post("/system/models/download")
    def download_models(body: DownloadReq):
        """Скачать набор (tier) или отдельные модели; уже скачанные пропускаются."""
        keys = list(body.keys or [])
        if body.tier:
            group = models.TIERS.get(body.tier) or models.ADDONS.get(body.tier)
            if not group:
                raise HTTPException(400, "Неизвестный набор")
            keys += group["models"]
        if not keys:
            raise HTTPException(400, "Не указано, что скачивать")
        need = sum(models.ALL[k]["gb"] for k in keys if k in models.ALL and not models.path(k))
        free = models.status()["disk_free_gb"]
        if need > free - 1:
            raise HTTPException(400, f"Не хватает места на диске: нужно ~{need:.1f} ГБ, свободно {free} ГБ")
        models.download_async(keys)
        return models.status(system["accel"])

    @r.delete("/system/models/{key}")
    def delete_model(key: str):
        if key not in models.ALL:
            raise HTTPException(404, "Нет такой модели")
        if dub.current or models.status()["download"]["running"]:
            raise HTTPException(400, "Дождитесь окончания текущих задач")
        from . import asr, tts
        dub._release_translator()
        tts.unload()
        asr.unload()
        models.delete(key)
        return models.status(system["accel"])

    # --- проект озвучки ---

    @r.get("/games/{gid:path}/resources")
    def resources(gid: str):
        from .discovery import scan
        game = _game(gid)
        result = scan(Path(game["path"]))
        adapter = adapters.for_game(game)
        result["adapter"] = adapter.id if adapter else None
        return result

    @r.get('/support')
    def support_catalog():
        from .engine_catalog import catalog
        result = catalog()
        result['adapters'] = [{'id': cls.id, 'name': getattr(cls, 'name', cls.id)} for cls in adapters._all()]
        return result

    @r.get("/projects/{gid:path}/preview")
    def preview_report(gid: str, label: str):
        p = _project(gid)
        target = (p.dir / "previews" / label / "index.html").resolve()
        if not target.is_relative_to((p.dir / "previews").resolve()) or not target.is_file():
            raise HTTPException(404, "Нет такого предпросмотра")
        return FileResponse(target, media_type="text/html")

    @r.post("/projects/{gid:path}/preview")
    def create_preview(gid: str, body: dict | None = None):
        _project(gid)
        body = body or {}
        engine = body.get("engine")
        if engine and engine not in engines.available_tts():
            raise HTTPException(400, "Движок не готов")
        count = max(1, min(6, int(body.get("count", 2))))
        dub.enqueue(gid, "preview", count=count, engine=engine)
        return dub.status(gid)

    @r.get("/projects/{gid:path}/lines")
    def lines(gid: str, status: str | None = None, char: str | None = None, q: str | None = None,
              limit: int = 100, offset: int = 0, rework: bool = False):
        p = _project(gid)
        items = p.lines(status=status, char=char, query=q, limit=limit, offset=offset, rework=rework)
        return {"items": items, "total": p.count(status=status, char=char, query=q, rework=rework)}

    @r.get("/projects/{gid:path}/voice-identities")
    def voice_identities(gid: str):
        from . import identity
        return identity.view(_project(gid))

    @r.post("/projects/{gid:path}/voice-trials")
    def voice_trials(gid: str, body: Ids):
        p = _project(gid)
        if not 1 <= len(body.ids) <= 4 or any(not p.line(i) for i in body.ids):
            raise HTTPException(400, "Выберите от одной до четырёх исходных реплик")
        if not all(e in engines.available_tts() for e in ("ru", "voxcpm")):
            raise HTTPException(400, "Для этих проб нужны Qwen-RU и VoxCPM2")
        dub.enqueue(gid, "voice_trials", seed_ids=body.ids)
        return dub.status(gid)

    @r.post("/projects/{gid:path}/voice-identities")
    def choose_voice_identity(gid: str, body: dict):
        from . import identity
        p = _project(gid)
        if dub.current:
            raise HTTPException(409, "Дождитесь окончания текущей задачи")
        try:
            if body.get("clear"):
                identity.clear(p, str(body.get("char") or ""))
            else:
                token = body.get("candidate")
                if not token:
                    line = p.line(str(body.get("line") or ""))
                    if not line or not line["gen"]:
                        raise ValueError("Выберите озвученную русскую реплику")
                    row = identity.propose(p, line["char"], line["gen"],
                                           (line["info"] or {}).get("spoken") or line["ru"],
                                           (line["info"] or {}).get("engine") or dub._engine_for(gid), line["id"])
                    token = row["id"]
                identity.activate(p, token)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error))
        return identity.view(p)

    @r.get("/projects/{gid:path}/audio")
    def line_audio(gid: str, id: str = "", kind: str = "orig", file: str = ""):
        p = _project(gid)
        if kind == "file":  # файлы сравнения голосов и профили: относительный путь внутри проекта
            target = (p.dir / file).resolve()
            if not target.is_relative_to(p.dir.resolve()) or not target.is_file():
                raise HTTPException(404, "Нет звука")
            return FileResponse(target, media_type="audio/wav", headers={"Cache-Control": "no-store"})
        line = p.line(id)
        rel = line and line.get("gen" if kind == "gen" else "orig")
        if not rel or not (p.dir / rel).exists():
            raise HTTPException(404, "Нет звука")
        return FileResponse(p.dir / rel, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    # --- голоса персонажей: уровень, донор, тембр ---

    @r.get("/projects/{gid:path}/voices")
    def voices_view(gid: str):
        """Экран «Голоса»: кто главный, какой донор и тембр у персонажа, 3 самые длинные реплики на прослушивание."""
        from . import donors, judge
        p = _project(gid)
        s = dub.settings(gid)
        over = p.get("timbre_override") or {}
        bank = donors.load()
        assigned = p.get("donors") or {}
        f0s = p.get("voice_f0") or {}
        out = []
        for c in p.characters():
            entry = timbre.entry(p, c["id"])
            if timbre.voice(p, c["id"]):
                state = "trained"
            elif entry and entry.get("val") is not None:
                state = "weak"
            elif entry:
                state = "training"
            else:
                state = "none"
            d = assigned.get(c["id"])
            donor = donors.by_id(bank, d["id"]) if d else None
            f0 = f0s.get(c["id"])
            samples = sorted((l for l in p.lines(char=c["id"]) if l["orig"]), key=lambda l: -(l["duration"] or 0))[:3]
            out.append({"id": c["id"], "name": c["name"], "lines": c["lines"] or 0, "seconds": round(c["seconds"] or 0),
                        "f0": f0, "voice_type": judge.voice_type(f0) if f0 else None,
                        "tier": timbre.tier(c, s, over.get(c["id"])), "override": over.get(c["id"]),
                        "donor": {"id": donor["id"], "gender": donor["gender"], "age": donor["age"], "pitch": d.get("pitch", 0),
                                  "manual": bool(d.get("manual"))} if donor else None,
                        "timbre": {"state": state, "reason": (entry or {}).get("reason")},
                        "samples": [l["id"] for l in samples]})
        return out

    @r.post("/projects/{gid:path}/voices/donor")
    def voices_donor(gid: str, body: dict):
        from . import donors
        p = _project(gid)
        if dub.current == gid:
            raise HTTPException(409, "Дождитесь окончания текущей задачи")
        try:
            return donors.set_manual(dub, p, str(body.get("char") or ""), str(body.get("donor") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @r.post("/projects/{gid:path}/voices/tier")
    def voices_tier(gid: str, body: dict):
        """Уровень персонажа вручную: train — обучать тембр, episodic — быстрый путь, auto — по правилу."""
        p = _project(gid)
        if dub.current == gid:
            raise HTTPException(409, "Дождитесь окончания текущей задачи")
        char, tier = str(body.get("char") or ""), str(body.get("tier") or "")
        if tier not in ("train", "episodic", "auto") or not char:
            raise HTTPException(400, "Уровень: train, episodic или auto")
        over = p.get("timbre_override") or {}
        if tier == "auto":
            over.pop(char, None)
        else:
            over[char] = tier
        p.set("timbre_override", over)
        return {"char": char, "override": over.get(char)}

    @r.get("/donors")
    def donors_list():
        from . import donors
        return [{"id": d["id"], "gender": d["gender"], "age": d.get("age"), "f0": d.get("f0"), "styles": list(d["clips"]),
                 "url": {style: f"/api/donors/audio?id={d['id']}&style={style}" for style in d["clips"]}}
                for d in donors.load()]

    @r.get("/donors/audio")
    def donor_audio(id: str, style: str = "calm"):
        from . import donors
        donor = donors.by_id(donors.load(), id)
        if not donor:
            raise HTTPException(404, "Нет такого донора")
        path = donors.clip_path(donor, style)
        if not path.is_file():
            raise HTTPException(404, "Нет такого клипа")
        return FileResponse(path, media_type="audio/flac", headers={"Cache-Control": "no-store"})

    @r.get("/projects/{gid:path}/enginebench/variants")
    def enginebench_variants(gid: str):
        """Готовые варианты стенда: источники речи (+rvc, если у кого-то обучен тембр) и клонировщики @en."""
        from . import enginebench, timbre
        p = _project(gid)
        has_timbre = any(timbre.voice(p, c["id"]) for c in p.characters())
        return {"variants": enginebench.variants_available(timbre=has_timbre)}

    @r.post("/projects/{gid:path}/enginebench")
    def enginebench_run(gid: str, body: dict | None = None):
        """Прогон стенда движков по выбранным вариантам (в очередь задач)."""
        from . import enginebench
        _project(gid)
        variants = [str(v) for v in ((body or {}).get("variants") or []) if v]
        if not variants:
            raise HTTPException(400, "Выберите хотя бы один вариант стенда")
        try:
            for v in variants:
                enginebench.parse_variant(v)
        except ValueError as e:
            raise HTTPException(400, str(e))
        dub.enqueue(gid, "enginebench", variants=variants, label=(body or {}).get("label"))
        return dub.status(gid)

    @r.get("/projects/{gid:path}/bench")
    def bench_page(gid: str, label: str):
        p = _project(gid)
        target = (p.dir / "bench" / label / "index.html").resolve()
        if not target.is_relative_to((p.dir / "bench").resolve()) or not target.is_file():
            raise HTTPException(404, "Нет такого прогона стенда")
        return FileResponse(target, media_type="text/html")

    @r.get("/projects/{gid:path}/review")
    def review(gid: str):
        """Персонажи, озвученные целиком и ждущие прослушивания: по 4 реплики на пробу, комментарий."""
        p = _project(gid)
        review = p.get("review") or {}
        notes = p.get("char_notes") or {}
        names = {c["id"]: c["name"] for c in p.characters()}
        out = []
        for char, st in sorted(review.items(), key=lambda kv: kv[1].get("at", 0)):
            if st.get("state") != "ready":
                continue
            lines = [l for l in p.lines(char=char, status="generated") if l["gen"]]
            if not lines:
                continue
            step = max(1, len(lines) // 4)
            picks = lines[::step][:4]
            out.append({"char": char, "name": names.get(char, char), "lines": len(lines), "at": st.get("at"),
                        "note": notes.get(char, ""),
                        "rework": sum(1 for l in lines if (l["info"] or {}).get("judge", {}).get("verdict") == "rework"),
                        "samples": [{"id": l["id"], "ru": l["ru"], "en": l["en"],
                                     "score": (l["info"] or {}).get("judge", {}).get("score")} for l in picks]})
        return {"ready": out, "approved": [names.get(c, c) for c, st in review.items() if st.get("state") == "approved"]}

    @r.post("/projects/{gid:path}/review/approve")
    def review_approve(gid: str, body: ReviewReq):
        p = _project(gid)
        if body.note is not None:
            notes = p.get("char_notes") or {}
            notes[body.char] = body.note.strip()
            p.set("char_notes", notes)
        return {"approved": dub.review_approve(p, body.char)}

    @r.post("/projects/{gid:path}/review/redo")
    def review_redo(gid: str, body: ReviewReq):
        """Переозвучить персонажа заново с учётом комментария (ставится в очередь после текущей задачи)."""
        p = _project(gid)
        notes = p.get("char_notes") or {}
        notes[body.char] = (body.note or "").strip()
        p.set("char_notes", notes)
        ids = dub.review_reset(p, body.char)
        if ids:
            dub.enqueue(gid, "generate", ids=ids)
        return {"reset": len(ids), **dub.status(gid)}

    @r.post("/projects/{gid:path}/character-note")
    def character_note(gid: str, body: ReviewReq):
        p = _project(gid)
        notes = p.get("char_notes") or {}
        notes[body.char] = (body.note or "").strip()
        p.set("char_notes", notes)
        return notes

    @r.post("/projects/{gid:path}/prepare")
    def prepare(gid: str):
        """Всё автоматически: реплики → русский текст → обучение голосов главных персонажей → пробы."""
        _project(gid)
        dub.enqueue(gid, "prepare")
        return dub.status(gid)

    @r.post("/projects/{gid:path}/samples")
    def samples(gid: str, body: dict | None = None):
        """Пробы голосов главных персонажей; next=true — следующий вариант звучания."""
        _project(gid)
        dub.enqueue(gid, "samples", next_variant=bool((body or {}).get("next")))
        return dub.status(gid)

    @r.post("/projects/{gid:path}/restart")
    def restart(gid: str):
        """Начать заново без обучения: сбросить текст и озвучку (кроме правок пользователя), снова подготовить."""
        p = _project(gid)
        if dub.current == gid:
            raise HTTPException(400, "Сначала остановите текущую задачу")
        with p.db() as c:
            c.execute("UPDATE lines SET ru=NULL, ru_source=NULL, gen=NULL, gen_duration=NULL, info=NULL, "
                      "status='new' WHERE ru_source IS NULL OR ru_source != 'user'")
            c.execute("UPDATE lines SET gen=NULL, gen_duration=NULL, info=NULL, status='translated' "
                      "WHERE ru_source = 'user'")
        p.set("samples", None)
        p.set("prepared_at", None)
        dub.enqueue(gid, "prepare")
        return dub.status(gid)

    @r.post("/projects/{gid:path}/train-voice")
    def train_voice(gid: str, body: TrainReq):
        """Обучить RVC-голос персонажа на его записях."""
        _project(gid)
        if body.method == "rvc":
            dub.enqueue(gid, "train_voice", char=body.char, steps=max(20, min(body.steps, 3000)))
        else:
            s = dub.settings(gid)
            dub.enqueue(gid, "train_tts", char=body.char, max_seconds=float(s.get("ft_minutes", 15)) * 60,
                        epochs=int(s.get("ft_epochs", 4)))
        return dub.status(gid)

    @r.post("/projects/{gid:path}/judge")
    def run_judge(gid: str, body: Ids | None = None):
        _project(gid)
        dub.enqueue(gid, "judge", ids=body.ids if body and body.ids else None)
        return dub.status(gid)

    @r.patch("/projects/{gid:path}/line")
    def edit_line(gid: str, id: str, body: LineEdit):
        p = _project(gid)
        line = p.line(id)
        if not line:
            raise HTTPException(404, "Реплика не найдена")
        fields = {}
        if body.ru is not None and body.ru.strip() != (line["ru"] or ""):
            fields.update(ru=body.ru.strip(), ru_source="user")
            if line["status"] in ("generated", "approved", "rejected", "new", "error"):
                fields["status"] = "translated"  # текст изменён — нужно озвучить заново
        if body.status:
            fields["status"] = body.status
        if fields:
            p.update_line(id, **fields)
        return p.line(id)

    @r.post("/projects/{gid:path}/mark")
    def mark(gid: str, body: Mark):
        if body.status not in ("approved", "rejected", "generated", "skipped", "translated"):
            raise HTTPException(400, "Неверный статус")
        p = _project(gid)
        targets = p.lines(ids=body.ids) if body.ids else p.lines(status=body.from_status, char=body.char)
        if body.status == "approved":
            targets = [l for l in targets if l["gen"]]
            if body.min_score is not None:  # только те, что контролёр счёл хорошими
                targets = [l for l in targets if ((l["info"] or {}).get("judge") or {}).get("score", 0) >= body.min_score
                           and ((l["info"] or {}).get("judge") or {}).get("verdict") != "rework"]
        with p.db() as c:
            c.executemany("UPDATE lines SET status=? WHERE id=?", [(body.status, l["id"]) for l in targets])
        return {"changed": len(targets)}

    @r.post("/projects/{gid:path}/regenerate")
    def regenerate(gid: str, body: Ids):
        _project(gid)
        dub.enqueue(gid, "generate", ids=body.ids)
        return dub.status(gid)

    @r.post("/projects/{gid:path}/run")
    def run(gid: str, body: Steps):
        _project(gid)
        for step in body.steps:
            if step not in ALLOWED_STEPS:
                raise HTTPException(400, f"Неизвестный шаг {step}")
            kw = {"ids": body.ids} if body.ids and step in ("translate", "generate") else {}
            dub.enqueue(gid, step, **kw)
        return dub.status(gid)

    @r.post("/projects/{gid:path}/sample")
    def sample(gid: str, body: Sample):
        """Проба голоса: несколько характерных реплик персонажа — перевести и озвучить."""
        p = _project(gid)
        cands = [l for l in p.lines(char=body.char) if l["status"] != "skipped" and (l["en"] or l["ru_official"])
                 and 1.5 <= (l["duration"] or 0) <= 9]
        cands.sort(key=lambda l: -(l["duration"] or 0))
        step = max(1, len(cands) // max(body.count, 1))
        ids = [l["id"] for l in cands[::step][:body.count]]
        if not ids:
            raise HTTPException(400, "У персонажа нет подходящих реплик с текстом")
        dub.enqueue(gid, "translate", ids=ids)
        dub.enqueue(gid, "generate", ids=ids)
        p.set("last_sample", ids)
        return {"ids": ids, "status": dub.status(gid)}

    @r.post("/projects/{gid:path}/cancel")
    def cancel(gid: str):
        dub.cancel(gid)
        return dub.status(gid)

    @r.post("/projects/{gid:path}/settings")
    def settings(gid: str, body: dict):
        _project(gid)
        return dub.save_settings(gid, body)

    @r.patch("/projects/{gid:path}/character")
    def edit_char(gid: str, id: str, body: CharEdit):
        p = _project(gid)
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        if "include" in fields:
            fields["include"] = int(fields["include"])
        if fields:
            p.update_character(id, **fields)
        return p.characters()

    @r.post("/import")
    async def import_upload(request: Request, name: str = "package.ruvoice", gid: str | None = None):
        """Загрузить пакет .ruvoice (тело запроса — сам файл), найти игру в библиотеке и поставить импорт в очередь."""
        from . import package
        from .paths import DATA_DIR
        inbox = DATA_DIR / "imports"
        inbox.mkdir(parents=True, exist_ok=True)
        dst = inbox / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "package.ruvoice")
        with open(dst, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        try:
            m = package.read_manifest(dst)
        except Exception as e:
            dst.unlink(missing_ok=True)
            raise HTTPException(400, f"Не удалось прочитать пакет: {e}")
        game = lib.get(gid) if gid else package.match_game(lib, m)
        if not game:
            raise HTTPException(404, f"Игра «{(m.get('game') or {}).get('name')}» не найдена в библиотеке — добавьте её и повторите")
        dub.enqueue(game["id"], "import", path=str(dst))
        return {"game_id": game["id"], "game": game["name"], "counts": m.get("counts"), **dub.status(game["id"])}

    @r.post("/projects/{gid:path}/open-exports")
    def open_exports(gid: str):
        from .paths import DATA_DIR
        d = DATA_DIR / "exports"
        d.mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))
        return {"ok": True}

    @r.post("/projects/{gid:path}/speakers")
    def find_speakers(gid: str):
        """Разобрать реплики по голосам: кто на самом деле говорит (результат — предложение для проверки)."""
        _project(gid)
        dub.enqueue(gid, "speakers")
        return dub.status(gid)

    @r.post("/projects/{gid:path}/speakers/apply")
    def apply_speakers(gid: str, body: dict):
        from . import speakers
        p = _project(gid)
        if dub.current == gid:
            raise HTTPException(400, "Сначала остановите текущую задачу")
        prop = p.get("speaker_proposal")
        if not prop:
            raise HTTPException(400, "Сначала разберите реплики по голосам")
        moved = speakers.apply(p, prop, body.get("names") or {}, set(body.get("skip") or []))
        p.set("speaker_proposal", None)
        return {"moved": moved, "characters": len(p.characters())}

    @r.post("/projects/{gid:path}/speakers/discard")
    def discard_speakers(gid: str):
        _project(gid).set("speaker_proposal", None)
        return {"ok": True}

    @r.post("/projects/{gid:path}/line-note")
    def line_note(gid: str, body: dict):
        """Указание к реплике («злее, на крике», «тише, с паузой после обращения») → режиссёр → переозвучка."""
        p = _project(gid)
        l = p.line(body.get("id") or "")
        if not l:
            raise HTTPException(404, "Реплика не найдена")
        info = dict(l["info"] or {})
        info["user_note"] = (body.get("note") or "").strip()[:300]
        p.update_line(l["id"], info=info)
        if info["user_note"] and body.get("redo", True):
            dub.enqueue(gid, "redirect", ids=[l["id"]])
        return dub.status(gid)

    @r.post("/projects/{gid:path}/selection")
    def selection(gid: str, body: dict):
        """Каких персонажей озвучивать: {"chars": [id…]}; пустой список — всех."""
        p = _project(gid)
        sel = dub.set_selected(p, list(body.get("chars") or []))
        return {"selected": sel}

    @r.post("/projects/{gid:path}/merge")
    def merge(gid: str, body: Merge):
        p = _project(gid)
        p.merge_characters(body.into, body.others)
        return p.characters()

    @r.delete("/projects/{gid:path}/project")
    def delete_project(gid: str):
        game = _game(gid)
        if not Project.exists(gid):
            return {"ok": True}
        p = dub.project(gid)
        a = adapters.by_id(p.get("adapter")) if p.get("adapter") else None
        if a and a.is_installed(p, game):
            raise HTTPException(400, "Сначала удалите установленную озвучку из игры")
        if dub.current == gid:
            raise HTTPException(400, "Сначала остановите текущую задачу")
        dub.cancel(gid)
        dub._projects.pop(gid, None)
        p.close()
        shutil.rmtree(p.dir, ignore_errors=True)
        if p.dir.exists():
            raise HTTPException(500, "Не удалось удалить папку проекта — возможно, файлы заняты")
        return {"ok": True}

    @r.get("/dub-overview")
    def dub_overview():
        """Для главного экрана: где сейчас идёт озвучка (задача, прогресс) и где уже есть озвучка RuVoice —
        сколько реплик озвучено, одобрено, установлено в игру. Только игры, для которых есть проект."""
        out = {}
        for gid, game in list(lib.games.items()):
            if not Project.exists(gid):
                continue
            try:
                p = dub.project(gid)
                st = p.stats()
                by = st.get("by_status") or {}
                a = adapters.by_id(p.get("adapter")) if p.get("adapter") else adapters.for_game(game)
                try:
                    installed = bool(a and a.is_installed(p, game)) or bool(p.get("installed_at"))
                except Exception:
                    installed = bool(p.get("installed_at"))
                s = dub.status(gid)
                out[gid] = {"running": bool(s.get("running")), "queued": len(s.get("queued") or []),
                            "task": s.get("task"), "message": s.get("message") or "", "done": s.get("done") or 0,
                            "total": s.get("total") or 0, "error": s.get("error"),
                            "dubbed": (by.get("generated") or 0) + (by.get("approved") or 0),
                            "approved": by.get("approved") or 0, "to_dub": by.get("translated") or 0,
                            "installed": installed, "installed_count": p.get("installed_count") or 0}
            except Exception as e:
                out[gid] = {"error": str(e)[:200]}
        return out

    @r.get("/projects/{gid:path}")
    def project_info(gid: str):
        game = _game(gid)
        exists = Project.exists(gid)
        adapter = adapters.for_game(game)
        info = {"exists": exists, "adapter": adapter.describe(game) if adapter else None,
                "status": dub.status(gid)}
        if exists:
            p = dub.project(gid)
            a = adapters.by_id(p.get("adapter")) if p.get("adapter") else adapter
            info.update(settings=dub.settings(gid), stats=p.stats(), characters=p.characters(),
                        installed=bool(a and a.is_installed(p, game)) or bool(p.get("installed_at")),
                        installed_count=p.get("installed_count", 0),
                        extracted=bool(p.get("extracted_at")), last_sample=p.get("last_sample", []),
                        engines=engines.describe(),
                        voices={k: {**v, "trained": timbre.voice(p, k) is not None} for k, v in (p.get("voices") or {}).items()},
                        samples=p.get("samples"), prepared=bool(p.get("prepared_at")),
                        voice_variant=dub.variant_info(gid),
                        train_plan=timbre.plan(dub, gid), timbre=timbre.estimate(dub, gid),
                        bench=p.get("bench") or [], previews=p.get("previews") or [], voice_f0=p.get("voice_f0") or {},
                        review=p.get("review") or {}, char_notes=p.get("char_notes") or {},
                        selected_chars=(dub.selected(p) and p.get("selected_chars")) or [],
                        last_export=p.get("last_export"), last_import=p.get("last_import"),
                        speaker_proposal=_proposal_view(p.get("speaker_proposal")))
        return info

    # --- языки ---

    @r.get("/games/{gid:path}/language-files")
    def language_files(gid: str, code: str):
        files = langmgr.language_files(_game(gid)["path"], code)
        by_kind = {k: {"count": 0, "size": 0} for k in langmgr.KINDS}
        for f in files:
            by_kind[f["kind"]]["count"] += 1
            by_kind[f["kind"]]["size"] += f["size"]
        return {"by_kind": by_kind, "files": [{k: f[k] for k in ("rel", "size", "kind")} for f in files[:200]],
                "total": len(files)}

    @r.post("/games/{gid:path}/language-delete")
    def language_delete(gid: str, code: str, body: Kinds):
        if code == "ru":
            raise HTTPException(400, "Русский язык удалять не будем")
        return langmgr.delete(_game(gid)["path"], code, [k for k in body.kinds if k in langmgr.KINDS])

    return r
