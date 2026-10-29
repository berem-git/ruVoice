"""Пути к внешним программам из G:\\RuVoice\\tools. Если программы нет — None."""
import os
import shutil
from pathlib import Path

from .paths import PROJECT_DIR

TOOLS_DIR = PROJECT_DIR / "tools"
WWISE_PROJECT = TOOLS_DIR / "wwise_project" / "RuVoiceConvert" / "RuVoiceConvert.wproj"


def _first(*candidates) -> Path | None:
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def ffmpeg() -> Path | None:
    return _first(TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg.exe", shutil.which("ffmpeg"))


def vgmstream() -> Path | None:
    return _first(TOOLS_DIR / "vgmstream" / "vgmstream-cli.exe", shutil.which("vgmstream-cli"))


def quickbms() -> Path | None:
    return _first(TOOLS_DIR / "quickbms" / "quickbms.exe", shutil.which("quickbms"))


def wwise_versions() -> list[Path]:
    """Все установленные версии Wwise (новые первыми) — старым играм может понадобиться старая."""
    return sorted((TOOLS_DIR / "Wwise").glob("Wwise_*/Authoring/x64/Release/bin/WwiseConsole.exe"), reverse=True)


def wwise_console() -> Path | None:
    versions = wwise_versions()
    return _first(os.environ.get("WWISE_CONSOLE"), versions[0] if versions else None)


def repak() -> Path | None:
    """Сборка .pak для Unreal Engine (tools\repak, MIT/Apache-2.0)."""
    return _first(TOOLS_DIR / "repak" / "repak.exe")


def retoc() -> Path | None:
    """Контейнеры IoStore (.utoc/.ucas) Unreal Engine (tools\retoc, MIT)."""
    return _first(TOOLS_DIR / "retoc" / "retoc.exe")


def oodle() -> Path | None:
    cands = sorted((TOOLS_DIR / "oodle").glob("oo2core_*_win64.dll"), reverse=True)
    return cands[0] if cands else None


def llama_server() -> Path | None:
    return _first(TOOLS_DIR / "llama.cpp" / "llama-server.exe")


def status() -> dict[str, str | None]:
    """Что входит в комплект программы (QuickBMS необязателен и здесь не показывается)."""
    return {name: str(p) if (p := fn()) else None
            for name, fn in [("ffmpeg", ffmpeg), ("vgmstream", vgmstream),
                             ("wwise", wwise_console), ("llama.cpp", llama_server),
                             ("repak", repak), ("oodle", oodle)]}
