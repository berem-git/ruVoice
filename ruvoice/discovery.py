"""Bounded, read-only asset discovery. No game names or installation rules live here."""
from __future__ import annotations

import os
import zipfile
from collections import Counter
from pathlib import Path

AUDIO = {".wav", ".ogg", ".opus", ".flac", ".mp3", ".wem", ".m4a", ".hca", ".adx"}
TEXT = {".srt", ".vtt", ".ass", ".ssa", ".json", ".csv", ".tsv", ".po", ".xml",
        ".locres", ".fmg", ".ini", ".strings", ".lang", ".resx"}
CONTAINERS = {".pak", ".utoc", ".ucas", ".vpk", ".fsb", ".bank", ".bnk", ".pck",
              ".bhd", ".bdt", ".bnd", ".bundle", ".assets", ".rpa", ".ba2", ".bsa",
              ".cpk", ".awb", ".acb", ".zip", ".archive"}
SKIP = {".git", "__pycache__", "node_modules", "_commonredist", "$recycle.bin"}
MAGIC = [(b"RIFF", "riff"), (b"OggS", "ogg"), (b"fLaC", "flac"), (b"FSB5", "fmod-fsb5"),
         (b"BKHD", "wwise-bank"), (b"AKPK", "wwise-package"), (b"BHD5", "bhd5"),
         (b"UnityFS", "unity-bundle"), (b"BND4", "bnd4"), (b"PK\x03\x04", "zip")]


def scan(root: Path, max_files: int = 100_000, max_examples: int = 150,
         cancelled=lambda: False) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Папка игры не найдена")
    counts: Counter = Counter()
    assets, warnings, seen, truncated = [], [], 0, False
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            if seen >= max_files or cancelled():
                truncated = True
                break
            seen += 1
            p = Path(directory) / name
            if p.is_symlink():
                continue
            ext = p.suffix.lower()
            kind = "audio" if ext in AUDIO else "subtitles" if ext in TEXT else "container" if ext in CONTAINERS else None
            # Also recognise renamed containers (.dat/.bin), without loading whole files.
            magic = None
            if kind or ext in {".dat", ".bin", ".resource"}:
                try:
                    with p.open("rb") as f:
                        head = f.read(16)
                    magic = next((fmt for signature, fmt in MAGIC if head.startswith(signature)), None)
                except OSError:
                    warnings.append(f"Не прочитан: {p.relative_to(root).as_posix()}")
                    continue
            if magic and not kind:
                kind = "audio" if magic in {"riff", "ogg", "flac"} else "container"
            if not kind:
                continue
            counts[kind] += 1
            if len(assets) < max_examples:
                assets.append({"path": p.relative_to(root).as_posix(), "kind": kind, "format": magic or ext[1:]})
            if magic == "zip":
                try:
                    with zipfile.ZipFile(p) as archive:
                        for entry in archive.infolist()[:max_files - seen]:
                            if entry.is_dir():
                                continue
                            seen += 1
                            suffix = Path(entry.filename).suffix.lower()
                            nested_kind = "audio" if suffix in AUDIO else "subtitles" if suffix in TEXT else None
                            if nested_kind:
                                counts["archived_" + nested_kind] += 1
                                if len(assets) < max_examples:
                                    assets.append({"path": p.relative_to(root).as_posix() + "!/" + entry.filename,
                                                   "kind": nested_kind, "format": suffix[1:],
                                                   "encrypted": bool(entry.flag_bits & 1)})
                except (OSError, zipfile.BadZipFile):
                    warnings.append(f"Не удалось прочитать ZIP: {p.name}")
        if truncated:
            break
    return {"version": 1, "root": str(root), "scanned": seen, "truncated": truncated,
            "counts": dict(counts), "assets": assets, "warnings": warnings[:30],
            "note": "Обнаружение ресурсов не означает возможность распаковки, сопоставления текста и установки."
                    " Неизвестному или зашифрованному формату нужен отдельный адаптер."}
