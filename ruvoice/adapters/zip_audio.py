"""Audio in standard ZIP containers, independent of game/engine; atomic archive replacement."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .loose import LooseAdapter, AUDIO_EXT, MUSIC_TOKENS, _tokens, MIN_SEC, MAX_SEC
from .. import audio, subtitles
from ..languages import code_from_component, code_from_dirname, code_from_filename


def archives(root: Path):
    seen = 0
    for d, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [n for n in dirs if n.lower() not in {".git", "node_modules", "_commonredist"}
                   and not (Path(d) / n).is_symlink()]
        for name in names:
            seen += 1
            if seen > 100_000:
                return
            p = Path(d) / name
            if p.suffix.lower() in {".zip", ".dat", ".bin", ".pak", ".pk3", ".pk4", ".iwd", ".love", ".nw"} and not p.is_symlink():
                try:
                    with p.open("rb") as f:
                        if f.read(4) == b"PK\x03\x04":
                            yield p
                except OSError:
                    continue


class ZipAudioAdapter:
    id = "zip-audio"
    name = "Звук в стандартных ZIP-контейнерах"

    @staticmethod
    def detect(game):
        for p in archives(Path(game["path"])):
            try:
                with zipfile.ZipFile(p) as z:
                    if any(Path(i.filename).suffix.lower() in AUDIO_EXT and not i.flag_bits & 1 for i in z.infolist()):
                        return True
            except (OSError, zipfile.BadZipFile):
                continue
        return False

    def describe(self, game):
        return {"adapter": self.id, "name": self.name, "supports_official_ru": True}

    def extract(self, project, game, progress):
        root = Path(game["path"])
        pool = subtitles.harvest(root, progress)
        lines, chars = [], {}
        project.orig_dir.mkdir(parents=True, exist_ok=True)
        for p in archives(root):
            with zipfile.ZipFile(p) as z, tempfile.TemporaryDirectory() as td:
                for entry in z.infolist():
                    rel = Path(entry.filename)
                    if (entry.flag_bits & 1 or entry.file_size > 256 * 2**20 or
                            rel.suffix.lower() not in AUDIO_EXT or _tokens(str(rel)) & MUSIC_TOKENS):
                        continue
                    lang = next((code_from_component(x) or code_from_dirname(x) for x in reversed(rel.parts[:-1])
                                 if code_from_component(x) or code_from_dirname(x)),
                                code_from_filename(rel.stem))
                    if lang not in (None, "en"):
                        continue
                    identity = p.relative_to(root).as_posix() + "!/" + entry.filename
                    dest = project.orig_dir / (hashlib.sha256(identity.encode()).hexdigest()[:24] + ".wav")
                    # Never extract member paths: traversal and absolute paths stay inert ZIP metadata.
                    src = Path(td) / ("source" + rel.suffix)
                    try:
                        with z.open(entry) as r, src.open("wb") as w:
                            shutil.copyfileobj(r, w)
                        audio.decode_to_wav(src, dest)
                        inf = audio.sf.info(str(dest))
                        if not MIN_SEC <= inf.duration <= MAX_SEC:
                            dest.unlink(missing_ok=True)
                            continue
                    except Exception:
                        dest.unlink(missing_ok=True)
                        continue
                    char = LooseAdapter.speaker_of(rel)
                    text = subtitles.exact_asset_match(pool, entry.filename)
                    chars.setdefault(char, {"id": char, "name": char, "ref_line": identity})
                    lines.append({"id": identity, "char": char, "orig": str(dest.relative_to(project.dir)),
                                  "duration": inf.duration, "en": text["en"] if text else None,
                                  "ru_official": text["ru"] if text else None,
                                  "src": {"archive": p.relative_to(root).as_posix(), "member": entry.filename,
                                          "ext": rel.suffix.lower(), "sr": inf.samplerate, "channels": inf.channels}})
                    progress(f"ZIP: извлечено {len(lines)} реплик")
        project.reset()
        project.add_lines(lines)
        project.set_characters(list(chars.values()))
        subtitles.save_pool(project, pool)
        return len(lines)

    def install(self, project, game, progress):
        root = Path(game["path"]).resolve()
        grouped = {}
        for line in project.lines(status="approved"):
            grouped.setdefault(line["src"]["archive"], []).append(line)
        for rel, lines in grouped.items():
            target = (root / rel).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Архив вне папки игры")
            backup = project.backup_dir / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(target, backup)
            with tempfile.TemporaryDirectory(dir=target.parent, prefix="ruvoice_zip_") as td:
                replacements = {}
                for i, line in enumerate(lines):
                    src = line["src"]
                    dst = Path(td) / (str(i) + src["ext"])
                    if src["ext"] == ".wem":
                        audio.encode_wem_batch([(project.dir / line["gen"], dst)])
                    else:
                        audio.encode_like(project.dir / line["gen"], src["ext"], dst, src["sr"], src["channels"])
                    replacements[src["member"]] = dst
                rebuilt = Path(td) / "rebuilt.zip"
                with zipfile.ZipFile(backup) as zin, zipfile.ZipFile(rebuilt, "w") as zout:
                    zout.comment = zin.comment
                    for entry in zin.infolist():
                        source = replacements.get(entry.filename)
                        with (source.open("rb") if source else zin.open(entry)) as r, zout.open(entry, "w") as w:
                            shutil.copyfileobj(r, w)
                os.replace(rebuilt, target)
            progress(f"Установлен контейнер {rel}")
        project.set("zip_audio_installed", bool(grouped))
        return sum(map(len, grouped.values()))

    def uninstall(self, project, game, progress):
        root = Path(game["path"]).resolve()
        for f in project.backup_dir.rglob("*"):
            if f.is_file():
                target = (root / f.relative_to(project.backup_dir)).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Файл вне папки игры")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
        # Retain backups until project cleanup explicitly removes them.
        project.set("zip_audio_installed", False)

    def is_installed(self, project, game):
        return bool(project.get("zip_audio_installed", False))
