"""Bounded, read-only asset discovery. No game names or installation rules live here."""
from __future__ import annotations

import os
import zipfile
from collections import Counter
from pathlib import Path
from .engine_catalog import signature

AUDIO = {".wav", ".ogg", ".opus", ".flac", ".mp3", ".wem", ".m4a", ".hca", ".adx"}
TEXT = {".srt", ".vtt", ".ass", ".ssa", ".json", ".csv", ".tsv", ".po", ".xml",
        ".locres", ".fmg", ".ini", ".strings", ".lang", ".resx", '.rpy'}
CONTAINERS = {".pak", ".utoc", ".ucas", ".vpk", ".fsb", ".bank", ".bnk", ".pck",
              ".bhd", ".bdt", ".bnd", ".bundle", ".assets", ".rpa", ".ba2", ".bsa",
              ".cpk", ".awb", ".acb", ".zip", ".archive", ".pk3", ".pk4", ".iwd", ".love", ".nw",
              ".rpf", ".mpq", ".forge", ".rpkg", ".rgssad", ".rgss2a", ".rgss3a", ".cas", ".cat", ".fat"}
SKIP = {".git", "__pycache__", "node_modules", "_commonredist", "$recycle.bin"}


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
                    magic = signature(head)
                except OSError:
                    warnings.append(f"Не прочитан: {p.relative_to(root).as_posix()}")
                    continue
            if magic and not kind:
                kind = "audio" if magic in {"wav", "ogg", "flac"} else "container"
            if not kind:
                continue
            counts[kind] += 1
            if len(assets) < max_examples:
                assets.append({"path": p.relative_to(root).as_posix(), "kind": kind, "format": magic or ext[1:]})
            if magic == "zip":
                try:
                    with zipfile.ZipFile(p) as archive:
                        for entry in archive.infolist():
                            if entry.is_dir():
                                continue
                            if seen >= max_files or cancelled():
                                truncated = True
                                break
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
            elif magic in {'godot-pck', 'renpy-rpa'}:
                from .adapters.open_packages import PCK, RPA
                import pickle, zlib
                try:
                    archive = (PCK if magic == 'godot-pck' else RPA)(p)
                    for entry in archive.entries:
                        if seen >= max_files or cancelled():
                            truncated = True
                            break
                        seen += 1
                        suffix = Path(entry.name).suffix.lower()
                        nested = 'audio' if suffix in AUDIO else 'subtitles' if suffix in TEXT else None
                        if nested:
                            counts['archived_' + nested] += 1
                            if len(assets) < max_examples:
                                assets.append({'path': p.relative_to(root).as_posix() + '!/' + entry.name,
                                               'kind': nested, 'format': suffix[1:], 'encrypted': bool(entry.flags & 1)})
                except (OSError, ValueError, EOFError, TypeError, pickle.UnpicklingError, zlib.error) as error:
                    warnings.append(f'{p.name}: {error}')
            if truncated:
                break
        if truncated:
            break
    return {"version": 1, "root": str(root), "scanned": seen, "truncated": truncated,
            "counts": dict(counts), "assets": assets, "warnings": warnings[:30],
            "note": "Обнаружение ресурсов не означает возможность распаковки, сопоставления текста и установки."
                    " Неизвестному или зашифрованному формату нужен отдельный адаптер."}
