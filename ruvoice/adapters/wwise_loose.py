"""Любая игра со звуком Wwise в папке игры (не в архивах движка): Unity, собственные движки и т. п.

Ищет файлы Wwise по содержимому: пакеты .pck (AKPK), банки .bnk (BKHD), отдельные .wem в папке английского
языка («English(US)», «en-us»…; имя файла — id звука). Английская озвучка и её замена — wwisevoice.py (язык
определяется внутри файлов, имена не нужны). Текст — общий шаг подготовки: субтитры из файлов игры
(subtitles.harvest) по распознанной речи, без них — перевод нейросетью.
Установка заменяет изменённые файлы, оригиналы — в резервной копии проекта (backup\\), удаление их возвращает."""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from . import wwisevoice
from .wwisepck import is_english

SKIP_DIRS = {"_commonredist", "redist", "directx", "__pycache__"}
EXT = {".pck", ".bnk", ".wem"}
MAX_FILE = 4 << 30


def _files(root: Path) -> list[Path]:
    out = []
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x.lower() not in SKIP_DIRS]
        for n in names:
            if os.path.splitext(n)[1].lower() in EXT:
                out.append(Path(d) / n)
    return out


def _lang_dir(p: Path) -> str | None:
    for part in reversed(p.parent.parts):
        if is_english(part):
            return "en"
    return None


class WwiseLooseAdapter:
    id = "wwise"
    name = "Wwise в папке игры (любой движок)"

    @staticmethod
    def detect(game: dict) -> bool:
        root = Path(game["path"])
        if not root.is_dir():
            return False
        for p in _files(root)[:4000]:
            try:
                with open(p, "rb") as f:
                    head = f.read(4)
            except OSError:
                continue
            if head in (b"AKPK", b"BKHD") or (p.suffix.lower() == ".wem" and _lang_dir(p)):
                return True
        return False

    def describe(self, game: dict) -> dict:
        return {"adapter": self.id, "name": self.name, "supports_official_ru": False,
                "notes": ["Озвучка — английские файлы Wwise из папки игры (пакеты .pck, банки .bnk, отдельные .wem); "
                          "язык определяется внутри файлов. Русский текст — из субтитров игры (по распознанной речи), "
                          "если их нет — перевод нейросетью.",
                          "Установка заменяет файлы, оригиналы хранятся в проекте и возвращаются кнопкой «Удалить озвучку»."]}

    def _items(self, root: Path, project=None):
        """(путь относительно игры, данные) файлов Wwise; при установке — оригиналы из резервной копии."""
        for p in _files(root):
            try:
                if p.stat().st_size > MAX_FILE:
                    continue
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            src = (project.backup_dir / rel) if project is not None and (project.backup_dir / rel).exists() else p
            try:
                with open(src, "rb") as f:
                    head = f.read(4)
                    if head not in (b"AKPK", b"BKHD") and src.suffix.lower() != ".wem":
                        continue
                    f.seek(0)
                    data = f.read()
            except OSError:
                continue
            yield rel, data

    def extract(self, project, game: dict, progress):
        root = Path(game["path"])
        t0 = time.time()
        progress("Поиск файлов Wwise…")
        loose: dict[int, tuple[bytes, str]] = {}

        def sound():
            for rel, data in self._items(root):
                if data[:4] in (b"AKPK", b"BKHD"):
                    yield Path(rel).stem, data
                elif rel.lower().endswith(".wem") and _lang_dir(Path(rel)) and re.fullmatch(r"\d+", Path(rel).stem):
                    loose[int(Path(rel).stem)] = (data, f"Папка {Path(rel).parent.name}")

        streams, embedded, has_ru = wwisevoice.scan(sound(), progress)
        for sid, v in loose.items():
            streams.setdefault(sid, v)
        if not streams and not embedded:
            raise RuntimeError("Английской озвучки Wwise в папке игры не найдено")
        project.reset()
        lines, chars = wwisevoice.to_lines(project, streams, embedded, progress)
        for l in lines:  # отдельный .wem — запомнить, какой файл заменять
            sid = l["src"]["sid"]
            if sid in loose and streams.get(sid) is loose[sid]:
                l["src"]["loose"] = True
        project.add_lines(lines)
        project.set_characters([{"id": c["id"], "name": c["name"], "ref_line": c["best"]} for c in chars.values()])
        project.set("wwise_loose", {"voices": len(lines), "has_ru_voice": has_ru, "seconds": round(time.time() - t0)})
        return len(lines)

    def install(self, project, game: dict, progress) -> int:
        root = Path(game["path"])
        approved = [l for l in project.lines(status="approved") if l["src"].get("sid") is not None]
        if not approved:
            return 0
        work = project.dir / "stage_ww"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        new = wwisevoice.encode(project, approved, work, progress)
        kinds = {int(l["src"]["sid"]): l["src"]["kind"] for l in approved}
        progress("Сборка файлов Wwise с русскими репликами…")
        changed: dict[str, bytes] = {}

        def files():  # по одному файлу; отдельные .wem с id реплики заменяются целиком
            for rel, data in self._items(root, project):
                stem = Path(rel).stem
                if rel.lower().endswith(".wem") and stem.isdigit() and int(stem) in new and _lang_dir(Path(rel)):
                    changed[rel] = new[int(stem)]
                    continue
                yield rel, data

        for rel, data in wwisevoice.patch(files(), new, kinds):
            changed[rel] = data
        for i, (rel, data) in enumerate(changed.items(), 1):
            progress(f"Установка: {i}/{len(changed)}", i, len(changed))
            target = root / rel
            backup = project.backup_dir / rel
            if target.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            tmp = target.with_name(target.name + ".ruvoice.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        shutil.rmtree(work, ignore_errors=True)
        project.set("installed_count", len(approved))
        return len(approved)

    def uninstall(self, project, game: dict, progress):
        root = Path(game["path"])
        if project.backup_dir.is_dir():
            backups = [p for p in project.backup_dir.rglob("*") if p.is_file()]
            for i, b in enumerate(backups, 1):
                progress(f"Восстановление оригиналов: {i}/{len(backups)}", i, len(backups))
                target = root / b.relative_to(project.backup_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(b, target)
            shutil.rmtree(project.backup_dir, ignore_errors=True)
        project.set("installed_count", 0)

    def is_installed(self, project, game: dict) -> bool:
        return project.backup_dir.is_dir() and any(project.backup_dir.rglob("*.*"))
