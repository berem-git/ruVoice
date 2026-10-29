"""Удаление лишних языков озвучки и субтитров. Файлы уходят в Корзину Windows — их можно вернуть."""
import os
from pathlib import Path

from .detect import SKIP_DIRS, _kind, _lang_for

KINDS = ("audio", "video", "text", "other")


def language_files(root: str, code: str) -> list[dict]:
    result = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x.lower() not in SKIP_DIRS]
        rel = tuple(Path(d).relative_to(root).parts)
        for f in files:
            stem, ext = os.path.splitext(f.lower())
            if _lang_for(rel, stem, ext) != code:
                continue
            p = Path(d) / f
            try:
                size = p.stat().st_size
            except OSError:
                continue
            result.append({"path": str(p), "rel": str(Path(*rel, f)), "size": size,
                           "kind": _kind(tuple(x.lower() for x in rel), stem, ext)})
    return sorted(result, key=lambda x: -x["size"])


def delete(root: str, code: str, kinds: list[str]) -> dict:
    from send2trash import send2trash
    files = [f for f in language_files(root, code) if f["kind"] in kinds]
    deleted, freed, errors = 0, 0, []
    for f in files:
        try:
            send2trash(f["path"])
            deleted += 1
            freed += f["size"]
        except Exception as e:
            errors.append(f"{f['rel']}: {e}")
    return {"deleted": deleted, "freed": freed, "errors": errors[:20]}
