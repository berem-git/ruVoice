"""Уборка мусора: всё, что программа создаёт по ходу работы и что потом не нужно.

В проекте озвучки (data\\projects\\<id>) мусор — это:
  • варианты реплик, оставшиеся после выбора лучшего (gen\\_cand_*.wav);
  • русские файлы, на которые не ссылается ни одна реплика (gen\\*.wav-сироты);
  • рабочие папки обучения голоса (voices\\*\\tts_data, val, dataset, before, after, rvc\\dataset);
  • старые дообученные модели речи 1.0.0 (по ~4 ГБ на персонажа), модели RVC без контроля, незавершённое
    обучение тембра (voices\\*\\rvc без meta.json), старые эталоны голоса (voice_identities);
  • старые прогоны контрольной сцены (кроме трёх последних), пробные папки samples\\test_*, compare\\;
  • профили голоса, на которые больше нет ссылок.
Плюс общий кэш программы (cache\\, кроме менеджера пакетов uv).

Заодно чинится целостность: реплики, у которых пропал файл озвучки (например, папку удалили руками),
возвращаются в «ждут озвучки», чтобы программа не считала их готовыми."""
import json
import shutil
import sqlite3
import time
from pathlib import Path

from .paths import PROJECT_DIR
from .project import PROJECTS_DIR

WORK_DIRS = ("tts_data", "val", "dataset", "before", "after")


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def _remove(path: Path, dry: bool) -> int:
    """Удалить файл/папку; занятые другим процессом файлы пропускаются (уборка не должна падать)."""
    try:
        n = _size(path)
    except OSError:
        return 0
    if not dry and path.exists():
        try:
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
        except OSError:
            return 0
    return n


def project_garbage(pdir: Path, dry: bool = True) -> dict:
    """Найти (dry) или убрать мусор в папке проекта. Работает напрямую с базой — проект не обязан быть открыт."""
    db = pdir / "project.db"
    if not db.exists():
        return {"bytes": 0, "items": [], "fixed_lines": 0}
    c = sqlite3.connect(db, timeout=30)
    c.row_factory = sqlite3.Row
    meta = {r["key"]: json.loads(r["value"]) for r in c.execute("SELECT key, value FROM meta")}
    settings = meta.get("settings") or {}
    items: list[tuple[str, int]] = []

    def drop(path: Path, why: str):
        n = _remove(path, dry)
        if n:
            items.append((f"{path.relative_to(pdir)} — {why}", n))

    gen = pdir / "gen"
    used = {r["gen"].replace("\\", "/") for r in c.execute("SELECT gen FROM lines WHERE gen IS NOT NULL")}
    if gen.is_dir():
        for f in gen.iterdir():
            rel = f"gen/{f.name}"
            if f.name.startswith("_cand_"):
                drop(f, "вариант после выбора")
            elif f.suffix.lower() == ".wav" and rel not in used:
                drop(f, "файл без реплики")
    voices = pdir / "voices"
    rvc_voices = meta.get("voices") or {}
    if voices.is_dir():
        for v in voices.iterdir():
            if not v.is_dir():
                continue
            for w in WORK_DIRS:
                drop(v / w, "рабочая папка обучения")
            for old in ("tts", "tts_full"):  # дообучение Qwen из 1.0.0 (по ~4 ГБ) конвейером v2 не используется
                drop(v / old, "старое дообучение модели речи")
            for old in ("model.pth", "model.index"):  # модели RVC 1.0.0 без контроля — не используются
                drop(v / old, "старая модель тембра без контроля")
            rvc = v / "rvc"
            if rvc.is_dir():
                drop(rvc / "dataset", "рабочая папка обучения тембра")
                if not (rvc / "meta.json").exists():
                    drop(rvc, "незавершённое обучение тембра")
    drop(pdir / "voice_identities", "старые эталоны голоса (1.0.0)")
    samples = pdir / "samples"
    if samples.is_dir():
        for d in samples.iterdir():
            if d.name.startswith("test_"):
                drop(d, "пробная папка")
    drop(pdir / "compare", "старое сравнение голосов")
    bench = pdir / "bench"
    if bench.is_dir():
        runs = sorted((d for d in bench.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
        for d in runs[:-3]:
            drop(d, "старый прогон контрольной сцены")
    profiles = pdir / "profiles"
    refs = {str(Path(x.get("audio") or "")).replace("\\", "/") for x in (meta.get("profiles") or {}).values() if x.get("audio")}
    if profiles.is_dir():
        for f in profiles.iterdir():
            if f"profiles/{f.name}" not in refs:
                drop(f, "профиль голоса без ссылки")
    # целостность: реплики с пропавшей озвучкой
    fixed = 0
    rows = c.execute("SELECT id, gen, status FROM lines WHERE gen IS NOT NULL").fetchall()
    missing = [r["id"] for r in rows if not (pdir / r["gen"]).exists()]
    if missing:
        fixed = len(missing)
        if not dry:
            with c:
                c.executemany("UPDATE lines SET gen=NULL, gen_duration=NULL, "
                              "status=CASE WHEN status IN ('generated','approved','rejected') THEN 'translated' ELSE status END "
                              "WHERE id=?", [(i,) for i in missing])
    if not dry:
        # записи о тембре: остаются только модели конвейера v2 (voices/<char>/rvc с meta.json), чьи файлы на месте
        alive = {k: e for k, e in rvc_voices.items()
                 if e.get("pth") and (pdir / e["pth"]).exists() and (pdir / e["pth"]).parent.name == "rvc"
                 and ((pdir / e["pth"]).parent / "meta.json").exists()}
        with c:
            if len(alive) != len(rvc_voices):
                c.execute("INSERT OR REPLACE INTO meta VALUES ('voices', ?)", (json.dumps(alive, ensure_ascii=False),))
            if "tts_voices" in meta:
                c.execute("DELETE FROM meta WHERE key = 'tts_voices'")
    c.close()
    return {"bytes": sum(n for _, n in items), "items": [f"{t} ({n / 2**20:.0f} МБ)" for t, n in items[:50]],
            "fixed_lines": fixed}


# --- после установки в игру: «минимум» ---
# Что можно удалить после установки и потом вернуть (извлечь из игры заново): оригиналы, пробы, стенд.
# Остаются: project.db (переводы, оценки), gen/ (готовая озвучка — для переустановки и экспорта),
# backup/ (для «Удалить озвучку»), voices/ (модели тембра), profiles/, orig_chunks/ (липсинк Source при установке).
PURGE_DIRS = ("orig", "samples", "bench")


def purge_size(pdir: Path) -> int:
    """Сколько байт освободит purge_originals."""
    return sum(_size(Path(pdir) / d) for d in PURGE_DIRS)


_size_cache: dict[str, tuple[float, int]] = {}


def purge_size_cached(pdir: Path, ttl: float = 60.0) -> int:
    """То же с кэшем на ttl секунд: папка оригиналов большая, опрос студии частый."""
    key = str(pdir)
    now = time.time()
    hit = _size_cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    n = purge_size(pdir)
    _size_cache[key] = (now, n)
    return n


def purge_originals(pdir: Path) -> int:
    """Удалить оригиналы, пробы и стенд; вернуть освобождённые байты."""
    freed = 0
    for d in PURGE_DIRS:
        freed += _remove(Path(pdir) / d, dry=False)
    _size_cache.pop(str(pdir), None)
    return freed


CACHE_MIN_AGE = 24 * 3600  # свежее суток в кэше не трогаем: им может пользоваться идущая задача


def _mtime(path: Path) -> float:
    try:
        if path.is_file():
            return path.stat().st_mtime
        return max([path.stat().st_mtime] + [f.stat().st_mtime for f in path.rglob("*")])
    except OSError:
        return time.time()


def cache_garbage(dry: bool = True) -> int:
    """Временные файлы в cache\\ (кроме пакетов uv), не менявшиеся больше суток."""
    cache = PROJECT_DIR / "cache"
    total = 0
    now = time.time()
    if cache.is_dir():
        for x in cache.iterdir():
            if x.name != "uv" and now - _mtime(x) > CACHE_MIN_AGE:
                total += _remove(x, dry)
    return total


def all_garbage(dry: bool = True, skip: set[str] | None = None) -> dict:
    """Мусор по всем проектам и кэшу. skip — папки проектов, которые сейчас в работе."""
    res = {"projects": {}, "cache_bytes": cache_garbage(dry), "fixed_lines": 0}
    if PROJECTS_DIR.exists():
        for d in sorted(PROJECTS_DIR.iterdir()):
            if d.is_dir() and d.name not in (skip or set()):
                try:
                    g = project_garbage(d, dry)
                except Exception as e:
                    g = {"bytes": 0, "items": [f"ошибка: {e}"], "fixed_lines": 0}
                res["projects"][d.name] = g
                res["fixed_lines"] += g["fixed_lines"]
    res["bytes"] = res["cache_bytes"] + sum(g["bytes"] for g in res["projects"].values())
    return res
