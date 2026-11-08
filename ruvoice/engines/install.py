"""Установка сред движков в runtime\\engines\\<env> из программы (кнопка в «Система и нейросети»).

Среда — папка с закреплёнными версиями библиотек (requirements-<env>.txt, ставятся uv без зависимостей:
PyTorch, numpy и прочее берутся из основной среды runtime\\python; у движка могут быть свои transformers).
RVC (Applio) дополнительно получает исходники закреплённого релиза и претрейны (HiFi-GAN, RMVPE, ContentVec);
VoxCPM2 и OmniVoice — модели через обычную очередь скачивания (models.download_async)."""
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from ..paths import PROJECT_DIR
from . import ENGINES_DIR

APPLIO_TAG = "3.7.1"   # релиз Applio от 2026-10-26
APPLIO_URL = f"https://github.com/IAHispano/Applio/archive/refs/tags/{APPLIO_TAG}.zip"
APPLIO_SHA256 = "2a341a81283826957a08f6eb2a2739e4d92dd505d38e8baa792df10c3626f58e"  # архив 3.7.1, 18 652 781 байт
_HF = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
ENVS = {
    "voxcpm": {"name": "VoxCPM2", "requirements": "requirements-voxcpm.txt", "models": ["tts-voxcpm"]},
    "omnivoice": {"name": "OmniVoice", "requirements": "requirements-omnivoice.txt", "models": ["omnivoice"]},
    "rvc": {"name": "Перенос тембра (RVC / Applio)", "requirements": "requirements-rvc.txt", "models": [],
            "applio": {"url": APPLIO_URL, "sha256": APPLIO_SHA256},
            "pretrained": [
                (f"{_HF}/pretrained_v2/f0G40k.pth", "applio/rvc/models/pretraineds/hifi-gan/f0G40k.pth"),
                (f"{_HF}/pretrained_v2/f0D40k.pth", "applio/rvc/models/pretraineds/hifi-gan/f0D40k.pth"),
                (f"{_HF}/predictors/rmvpe.pt", "applio/rvc/models/predictors/rmvpe.pt"),
                (f"{_HF}/embedders/contentvec/pytorch_model.bin", "applio/rvc/models/embedders/contentvec/pytorch_model.bin"),
                (f"{_HF}/embedders/contentvec/config.json", "applio/rvc/models/embedders/contentvec/config.json"),
            ]},
}
_state = {"running": False, "engine": None, "message": "", "error": None, "thread": None}
_lock = threading.Lock()


def _installed(engine: str) -> bool:
    from . import ready, rvc_available
    return rvc_available() if engine == "rvc" else ready(engine)


def status() -> dict:
    out = {}
    for engine, spec in ENVS.items():
        mine = _state["engine"] == engine
        out[engine] = {"name": spec["name"], "installed": _installed(engine), "running": bool(_state["running"] and mine),
                       "message": _state["message"] if mine else "", "error": _state["error"] if mine else None}
    return out


def _python() -> str:
    return str(Path(sys.executable).with_name("python.exe"))


def _pip_target(engine: str, requirements: Path, progress):
    """Закреплённые пакеты среды — в runtime\\engines\\<env>, без зависимостей (они в основной среде)."""
    uv = PROJECT_DIR / "tools" / "uv" / "uv.exe"
    target = ENGINES_DIR / engine
    target.mkdir(parents=True, exist_ok=True)
    progress(f"{ENVS[engine]['name']}: установка библиотек…")
    env = {**os.environ, "UV_CACHE_DIR": str(PROJECT_DIR / "cache" / "uv")}
    r = subprocess.run([str(uv), "pip", "install", "--python", _python(), "--target", str(target), "--no-deps",
                        "--link-mode", "copy", "-r", str(requirements)], capture_output=True, text=True, env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode:
        raise RuntimeError(f"установка библиотек {engine} не удалась: " + ((r.stderr or r.stdout) or "")[-600:])


def _download(url: str, dest: Path, progress, sha256: str | None = None):
    """Скачать в .part с ходом загрузки, проверить хэш (если задан), переименовать."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last = [0.0]

    def hook(blocks, block_size, total):
        now = time.time()
        if total > 0 and now - last[0] > 1.0:
            last[0] = now
            progress(f"Скачивание {dest.name}: {min(100, int(blocks * block_size * 100 / total))}%")
    progress(f"Скачивание {dest.name}…")
    urllib.request.urlretrieve(url, str(part), hook)
    if sha256:
        digest = hashlib.sha256(part.read_bytes()).hexdigest()
        if digest != sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{dest.name}: контрольная сумма не совпала ({digest[:12]}…)")
    os.replace(part, dest)


def _extract_zip_strip(zip_path: Path, dest: Path):
    """Распаковать архив релиза, срезав верхнюю папку (Applio-<tag>/…) → dest; старая папка заменяется."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        tops = {n.split("/", 1)[0] for n in names if "/" in n}
        strip = tops.pop() + "/" if len(tops) == 1 and all(n.startswith(next(iter(tops)) + "/") for n in names) else ""
        for n in names:
            rel = n[len(strip):] if strip and n.startswith(strip) else n
            if not rel or rel.startswith("/") or ".." in Path(rel).parts:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def install(engine: str, progress=lambda m: None):
    """Поставить среду движка целиком; вызывать из фонового потока (install_async)."""
    from .. import models
    if engine not in ENVS:
        raise ValueError(f"Неизвестная среда {engine}")
    spec = ENVS[engine]
    env_dir = ENGINES_DIR / engine
    _pip_target(engine, PROJECT_DIR / spec["requirements"], progress)
    if spec.get("applio"):
        zip_path = PROJECT_DIR / "cache" / f"applio-{APPLIO_TAG}.zip"
        if not zip_path.exists():
            _download(spec["applio"]["url"], zip_path, progress, spec["applio"].get("sha256"))
        progress("Распаковка Applio…")
        _extract_zip_strip(zip_path, env_dir / "applio")
    for url, rel in spec.get("pretrained", []):
        dest = env_dir / rel
        if not dest.exists():
            _download(url, dest, progress)
    if spec["models"]:
        models.download_async(spec["models"])
        progress(f"{spec['name']}: среда готова, модели ставятся в очередь скачивания")
    else:
        progress(f"{spec['name']}: готово")


def install_async(engine: str) -> dict:
    """Запустить установку в фоне; повторный вызов во время работы ничего не меняет."""
    if engine not in ENVS:
        raise ValueError(f"Неизвестная среда {engine}")
    with _lock:
        if _state["running"]:
            return status()
        _state.update(running=True, engine=engine, message="Подготовка…", error=None)

        def run():
            try:
                install(engine, lambda m: _state.update(message=m))
            except Exception as e:
                _state.update(error=str(e), message="Ошибка установки")
            finally:
                _state["running"] = False
        t = threading.Thread(target=run, daemon=True)
        _state["thread"] = t
        t.start()
    return status()
