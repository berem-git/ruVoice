"""Библиотека игр: хранение в data/library.json и сводка по языкам для интерфейса."""
import hashlib
import json
import os
import threading
from pathlib import Path

from .languages import STEAM_LANGS, display_name
from .paths import DATA_DIR

VOICE_MIN_BYTES = 50 * 2**20


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)).rstrip("\\/")


def manual_id(path: str) -> str:
    return "manual:" + hashlib.sha1(_norm(path).encode("utf-8")).hexdigest()[:12]


class Library:
    def __init__(self, file: Path = DATA_DIR / "library.json"):
        self.file = file
        self.lock = threading.RLock()
        self.data = {"games": {}, "settings": {"library_dirs": []}, "steam_store": {}}
        if file.exists():
            self.data.update(json.loads(file.read_text("utf-8")))

    def save(self):
        with self.lock:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), "utf-8")
            os.replace(tmp, self.file)

    @property
    def games(self) -> dict[str, dict]:
        return self.data["games"]

    def get(self, gid: str) -> dict | None:
        return self.games.get(gid)

    def find_by_path(self, path: str) -> dict | None:
        norm = _norm(path)
        return next((g for g in self.games.values() if _norm(g["path"]) == norm), None)

    def merge_found(self, found: list[dict]):
        """Обновить игры из лаунчеров, сохранив анализ, ручное имя и скрытие."""
        with self.lock:
            for g in found:
                old = self.games.get(g["id"])
                if old is None:
                    # Игру уже добавили вручную — переводим её на лаунчер, не плодим дубль
                    dup = self.find_by_path(g["path"])
                    if dup and dup["source"] in ("manual", "folder"):
                        del self.games[dup["id"]]
                        g = {**dup, **g, "name": dup["name"] if dup.get("custom_name") else g["name"]}
                    self.games[g["id"]] = {"hidden": False, "analysis": None, **g}
                    continue
                if _norm(old["path"]) != _norm(g["path"]):
                    old["analysis"] = None
                for k, v in g.items():
                    if k == "name" and old.get("custom_name"):
                        continue
                    old[k] = v
            # Удалённые с диска игры лаунчеров убираем; ручные оставляем с пометкой
            for gid in [gid for gid, g in self.games.items()
                        if g["source"] not in ("manual", "folder") and not Path(g["path"]).is_dir()]:
                del self.games[gid]
            self.save()

    def add_manual(self, path: str, name: str | None = None, source: str = "manual") -> dict:
        p = Path(path.strip().strip('"'))
        if not p.is_dir():
            raise ValueError(f"Папка не найдена: {p}")
        if p.parent == p:
            raise ValueError("Нельзя добавить корень диска как игру — добавьте его как папку с играми")
        with self.lock:
            existing = self.find_by_path(str(p))
            if existing:
                existing["hidden"] = False
                if name:
                    existing["name"], existing["custom_name"] = name, True
                self.save()
                return existing
            g = {"id": manual_id(str(p)), "name": name or p.name, "path": str(p), "source": source,
                 "hidden": False, "analysis": None, "custom_name": bool(name)}
            self.games[g["id"]] = g
            self.save()
            return g

    def update(self, gid: str, **fields):
        with self.lock:
            g = self.games[gid]
            g.update(fields)
            self.save()
            return g

    def remove(self, gid: str):
        """Ручные игры удаляются, игры лаунчеров скрываются (иначе вернутся при сканировании)."""
        with self.lock:
            g = self.games.get(gid)
            if not g:
                return
            if g["source"] in ("manual", "folder"):
                del self.games[gid]
            else:
                g["hidden"] = True
            self.save()

    def unhide_all(self):
        with self.lock:
            for g in self.games.values():
                g["hidden"] = False
            self.save()

    # --- Папки-библиотеки ---

    @property
    def library_dirs(self) -> list[str]:
        return self.data["settings"]["library_dirs"]

    def add_library_dir(self, path: str):
        p = Path(path.strip().strip('"'))
        if not p.is_dir():
            raise ValueError(f"Папка не найдена: {p}")
        with self.lock:
            if all(_norm(d) != _norm(str(p)) for d in self.library_dirs):
                self.library_dirs.append(str(p))
                self.save()
        return str(p)

    def remove_library_dir(self, path: str):
        with self.lock:
            norm = _norm(path)
            self.data["settings"]["library_dirs"] = [d for d in self.library_dirs if _norm(d) != norm]
            for gid in [gid for gid, g in self.games.items()
                        if g["source"] == "folder" and _norm(str(Path(g["path"]).parent)) == norm]:
                del self.games[gid]
            self.save()

    # --- Данные Steam ---

    def steam_store(self, appid: str) -> dict | None:
        return self.data["steam_store"].get(str(appid))

    def set_steam_store(self, appid: str, info: dict):
        with self.lock:
            self.data["steam_store"][str(appid)] = info
            self.save()


def game_view(lib: Library, g: dict) -> dict:
    """Сводка для интерфейса: статус русского языка и таблица языков."""
    a = g.get("analysis") or {}
    langs: dict[str, dict] = {}

    def row(code):
        return langs.setdefault(code, {"code": code, "name": display_name(code), "audio": 0, "text": 0,
                                       "video": 0, "other": 0, "store_audio": False, "store_text": False})

    for code, b in (a.get("languages") or {}).items():
        r = row(code)
        for k in ("audio", "text", "video", "other"):
            r[k] += b.get(k, 0)

    store = lib.steam_store(g["appid"]) if g.get("appid") else None
    for steam_name, flags in ((store or {}).get("languages") or {}).items():
        r = row(STEAM_LANGS.get(steam_name, steam_name))
        r["store_audio"] |= flags["audio"]
        r["store_text"] |= flags["text"]

    for r in langs.values():
        r["size"] = r["audio"] + r["text"] + r["video"] + r["other"]

    ru = langs.get("ru")
    # Мелкие звуки (интерфейс) и локализованные ролики — ещё не озвучка
    has_ru_audio = bool(ru) and ru["audio"] >= VOICE_MIN_BYTES
    voices_split_by_language = any(r["audio"] >= VOICE_MIN_BYTES for r in langs.values())
    if has_ru_audio:
        status = "ru_audio"
    elif ru and ru["store_audio"] and voices_split_by_language:
        status = "ru_audio_store"  # озвучка официально есть, но этот язык не скачан
    elif ru and ru["store_audio"]:
        status = "ru_audio"
    elif ru and (ru["text"] or ru["store_text"]):
        status = "ru_text"
    elif ru and ru["other"]:
        status = "ru_unknown"
    elif langs:
        status = "no_ru"
    elif a:
        status = "unknown"
    else:
        status = "pending"

    order = {"ru": 0, "en": 1}
    lang_list = sorted(langs.values(), key=lambda r: (order.get(r["code"], 2), -r["size"], r["name"]))
    cover = (f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{g['appid']}/header.jpg"
             if g.get("appid") else None)
    return {
        "id": g["id"], "name": g["name"], "path": g["path"], "source": g["source"],
        "hidden": g.get("hidden", False), "missing": not Path(g["path"]).is_dir(),
        "cover": cover, "status": status, "languages": lang_list,
        "engine": a.get("engine"), "middleware": a.get("middleware", []),
        "size": a.get("size"), "files": a.get("files"), "exes": a.get("exes", []),
        "truncated": a.get("truncated", False), "analyzed": bool(a),
        "steam_language": g.get("steam_language"),
        "dub_support": _dub_support(g) if a else None,
        "dub_note": _dub_note(g) if a else None,
        "aes_key": g.get("aes_key") or "",
        "ru_installed": _ru_installed(g["id"]),
    }


def _ru_installed(game_id: str) -> int:
    """Сколько реплик русской озвучки установлено этой программой (0 — не установлена / проекта нет)."""
    try:
        from .project import Project
        if not Project.exists(game_id):
            return 0
        p = Project(game_id)
        return int(p.get("installed_count") or 0) if p.get("installed_at") else 0
    except Exception:
        return 0


def _dub_support(g: dict) -> str | None:
    """Каким способом можно озвучить игру (id адаптера) или None, если пока никаким."""
    from . import adapters
    try:
        adapter = adapters.for_game(g)
    except Exception:
        return None
    return adapter.id if adapter else None


def _dub_note(g: dict) -> str | None:
    """Read cached metadata; detailed archive diagnostics belong to the project endpoint."""
    return (g.get("analysis") or {}).get("dub_note")
