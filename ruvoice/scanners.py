"""Поиск установленных игр: Steam, Epic, GOG, Ubisoft и папки-библиотеки пользователя."""
import json
import os
import re
import time
import urllib.request
import winreg
from pathlib import Path

import vdf

from .detect import JUNK_EXE

STEAM_SKIP_APPIDS = {"228980"}  # Steamworks Common Redistributables
STEAM_SKIP_PREFIXES = ("proton", "steam linux runtime", "steamworks", "steamvr")


def _reg(hive, key: str, value: str) -> str | None:
    try:
        with winreg.OpenKey(hive, key) as k:
            return str(winreg.QueryValueEx(k, value)[0])
    except OSError:
        return None


def _reg_subkeys(hive, key: str):
    try:
        with winreg.OpenKey(hive, key) as k:
            i = 0
            while True:
                try:
                    name = winreg.EnumKey(k, i)
                except OSError:
                    return
                yield name
                i += 1
    except OSError:
        return


def _load_vdf(path: Path) -> dict:
    with open(path, encoding="utf-8", errors="replace") as f:
        return vdf.load(f)


def scan_steam() -> list[dict]:
    root = _reg(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath") \
        or _reg(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath")
    if not root:
        return []
    libraries = {Path(root)}
    lf = Path(root) / "steamapps" / "libraryfolders.vdf"
    if lf.exists():
        for entry in _load_vdf(lf).get("libraryfolders", {}).values():
            path = entry.get("path") if isinstance(entry, dict) else entry
            if path:
                libraries.add(Path(path))

    games = []
    for lib in libraries:
        for acf in (lib / "steamapps").glob("appmanifest_*.acf"):
            try:
                state = _load_vdf(acf).get("AppState", {})
            except Exception:
                continue
            appid, name = state.get("appid"), state.get("name", "")
            path = lib / "steamapps" / "common" / state.get("installdir", "")
            if not appid or appid in STEAM_SKIP_APPIDS or name.lower().startswith(STEAM_SKIP_PREFIXES):
                continue
            if not path.is_dir():
                continue
            games.append({"id": f"steam:{appid}", "name": name, "path": str(path),
                          "source": "steam", "appid": appid,
                          "steam_language": state.get("UserConfig", {}).get("language")})
    return games


def scan_epic() -> list[dict]:
    manifests = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Epic/EpicGamesLauncher/Data/Manifests"
    games = []
    for item in manifests.glob("*.item"):
        try:
            m = json.loads(item.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if m.get("bIsIncompleteInstall"):
            continue
        if m.get("MainGameAppName") and m.get("MainGameAppName") != m.get("AppName"):
            continue  # DLC
        cats = m.get("AppCategories") or []
        if cats and "games" not in cats:
            continue
        path = m.get("InstallLocation")
        if path and Path(path).is_dir():
            games.append({"id": f"epic:{m.get('AppName')}", "name": m.get("DisplayName") or Path(path).name,
                          "path": path, "source": "epic", "exe": m.get("LaunchExecutable")})
    return games


def scan_gog() -> list[dict]:
    base = r"SOFTWARE\WOW6432Node\GOG.com\Games"
    games = []
    for gid in _reg_subkeys(winreg.HKEY_LOCAL_MACHINE, base):
        key = f"{base}\\{gid}"
        if _reg(winreg.HKEY_LOCAL_MACHINE, key, "dependsOn"):
            continue  # DLC
        path = _reg(winreg.HKEY_LOCAL_MACHINE, key, "path")
        if path and Path(path).is_dir():
            games.append({"id": f"gog:{gid}",
                          "name": _reg(winreg.HKEY_LOCAL_MACHINE, key, "gameName") or Path(path).name,
                          "path": path, "source": "gog", "exe": _reg(winreg.HKEY_LOCAL_MACHINE, key, "exe")})
    return games


def scan_ubisoft() -> list[dict]:
    base = r"SOFTWARE\WOW6432Node\Ubisoft\Launcher\Installs"
    games = []
    for gid in _reg_subkeys(winreg.HKEY_LOCAL_MACHINE, base):
        path = _reg(winreg.HKEY_LOCAL_MACHINE, f"{base}\\{gid}", "InstallDir")
        if path and Path(path).is_dir():
            p = Path(path)
            games.append({"id": f"ubisoft:{gid}", "name": p.name, "path": str(p), "source": "ubisoft"})
    return games


LAUNCHERS = {"steam": scan_steam, "epic": scan_epic, "gog": scan_gog, "ubisoft": scan_ubisoft}


def scan_launchers() -> tuple[list[dict], dict[str, str]]:
    """Возвращает найденные игры и ошибки по лаунчерам (если были)."""
    found, errors = [], {}
    for name, fn in LAUNCHERS.items():
        try:
            found.extend(fn())
        except Exception as e:  # один сломанный лаунчер не должен ломать весь поиск
            errors[name] = str(e)
    return found, errors


def _has_exe(folder: Path, depth: int = 2) -> bool:
    try:
        for e in os.scandir(folder):
            low = e.name.lower()
            if e.is_file() and low.endswith(".exe") and not low.startswith(JUNK_EXE):
                return True
            if depth > 0 and e.is_dir() and _has_exe(Path(e.path), depth - 1):
                return True
    except OSError:
        pass
    return False


def scan_library_dir(folder: str) -> list[str]:
    """Папка с играми (например D:\\Games): каждая подпапка с .exe — отдельная игра."""
    result = []
    try:
        for e in os.scandir(folder):
            if e.is_dir() and e.name.lower() not in {"$recycle.bin", "system volume information"} \
                    and _has_exe(Path(e.path)):
                result.append(e.path)
    except OSError:
        pass
    return sorted(result)


def parse_steam_languages(html: str) -> dict[str, dict[str, bool]]:
    """'English<strong>*</strong>, Russian<br><strong>*</strong>languages with full audio support'."""
    html = re.split(r"<br\s*/?>", html)[0]
    result = {}
    for part in html.split(","):
        name = re.sub(r"<[^>]+>|\*", "", part).strip()
        if name:
            result[name] = {"text": True, "audio": "*" in part}
    return result


def fetch_steam_store(appid: str) -> dict | None:
    """Языки из магазина Steam. None — сеть недоступна, попробуем в следующий раз."""
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
    req = urllib.request.Request(url, headers={"User-Agent": "RuVoice/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except Exception:
        return None
    entry = (data or {}).get(str(appid)) or {}
    if not entry.get("success"):
        return {"ok": False, "languages": {}, "fetched_at": time.time()}
    return {"ok": True, "fetched_at": time.time(),
            "languages": parse_steam_languages(entry["data"].get("supported_languages", ""))}
