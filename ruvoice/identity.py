"""Immutable Russian voice anchors on disk. Approval is explicit, never inferred from ASR."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import audio


def _path(p, relative: str) -> Path:
    path = (p.dir / relative).resolve()
    if not path.is_relative_to(p.dir.resolve()) or not path.is_file():
        raise ValueError("Файл эталона отсутствует или находится вне проекта")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def propose(p, char: str, relative: str, text: str, engine: str, line_id: str) -> dict:
    """Copy a candidate so later regeneration cannot silently change the reference."""
    line = p.line(line_id)
    if not line or line["char"] != char:
        raise ValueError("Реплика не принадлежит выбранному голосу")
    if engine not in ("voxcpm", "ru"):
        raise ValueError("Эталон поддерживается только движками клонирования голоса")
    if not re.search(r"[А-Яа-яЁё]", text or ""):
        raise ValueError("Для эталона нужен русский текст")
    src = _path(p, relative)
    # No arbitrary source files or original English recordings can become Russian anchors.
    if not any(src.is_relative_to((p.dir / d).resolve()) for d in ("gen", "previews")):
        raise ValueError("Выберите русскую запись из озвучки или предпросмотра")
    if src.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Слишком большой файл эталона")
    wav, sr = audio.read(src)
    seconds = len(wav) / sr
    if not 2 <= seconds <= 30 or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 0.01:
        raise ValueError("Нужна непустая чистая русская запись длиной от 2 до 30 секунд")
    sha = _digest(src)
    token = hashlib.sha256(json.dumps([char, engine, text, line_id, sha], ensure_ascii=False).encode()).hexdigest()[:24]
    folder = p.dir / "voice_identities"
    folder.mkdir(exist_ok=True)
    dst = folder / f"{token}.wav"
    if not dst.exists():
        shutil.copyfile(src, dst)
    elif _digest(dst) != sha:
        raise ValueError("Сохранённый эталон повреждён")
    row = {"id": token, "char": char, "engine": engine, "text": text, "line": line_id,
           "audio": dst.relative_to(p.dir).as_posix(), "sha256": sha,
           "seconds": round(seconds, 3), "created_at": time.time()}
    candidates = p.get("voice_candidates") or {}
    candidates[token] = row
    p.set("voice_candidates", candidates)
    return row


def get(p, token: str) -> dict:
    row = (p.get("voice_candidates") or {}).get(token)
    if not row:
        raise ValueError("Нет такого эталона")
    line = p.line(row["line"])
    if not line or line["char"] != row["char"]:
        raise ValueError("Разметка говорящих изменилась: выберите эталон заново")
    path = _path(p, row["audio"])
    if _digest(path) != row["sha256"]:
        raise ValueError("Файл эталона изменился: требуется новое прослушивание")
    return {**row, "path": str(path)}


def activate(p, token: str) -> dict:
    row = get(p, token)
    if row.get("rejected_at"):
        raise ValueError("Этот образец отклонён после прослушивания: " + row.get("rejection_reason", "качество речи"))
    anchors = p.get("voice_anchors") or {}
    anchors[row["char"]] = {"id": token, "approved_at": time.time()}
    p.set("voice_anchors", anchors)
    return row


def reject(p, token: str, reason: str):
    candidates = p.get("voice_candidates") or {}
    if token not in candidates:
        raise ValueError("Нет такого эталона")
    row = candidates[token]
    row.update(rejected_at=time.time(), rejection_reason=reason)
    p.set("voice_candidates", candidates)
    if ((p.get("voice_anchors") or {}).get(row["char"]) or {}).get("id") == token:
        clear(p, row["char"])


def clear(p, char: str):
    anchors = p.get("voice_anchors") or {}
    anchors.pop(char, None)
    p.set("voice_anchors", anchors)


def active(p, char: str, engine: str | None = None) -> dict | None:
    entry = (p.get("voice_anchors") or {}).get(char)
    if not entry:
        return None
    row = get(p, entry["id"])  # Fail closed if changed; never silently switch to English.
    if engine and row["engine"] != engine:
        raise ValueError("Закреплённый голос создан для другого движка; выберите его или снимите эталон")
    return row


def reference(row: dict) -> dict:
    return {"audio": row["path"], "text": row["text"], "speaker": None,
            "identity": row["id"], "language": "ru"}


def centroid(p, row: dict, embed) -> tuple[np.ndarray, float]:
    """A fixed CPU/disk vector, not an average of potentially bad generated takes."""
    cache = p.get("voice_anchor_embeddings") or {}
    value = cache.get(row["id"])
    if not value or value.get("sha256") != row["sha256"]:
        e = np.asarray(embed([row["path"]])[0], dtype=np.float32)
        e /= np.linalg.norm(e) + 1e-9
        value = {"sha256": row["sha256"], "e": e.tolist()}
        cache[row["id"]] = value
        p.set("voice_anchor_embeddings", cache)
    e = np.asarray(value["e"], dtype=np.float32)
    # Conservative initial gate; listening is still required, this is not an accent detector.
    return e / (np.linalg.norm(e) + 1e-9), 0.55


def view(p) -> dict:
    anchors = p.get("voice_anchors") or {}
    rows = []
    for row in (p.get("voice_candidates") or {}).values():
        rows.append({**row, "active": (anchors.get(row["char"]) or {}).get("id") == row["id"]})
    return {"anchors": anchors, "candidates": rows}
