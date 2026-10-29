"""Игры со звуком в отдельных файлах (.wav/.ogg/.mp3/.flac/.opus/.wem прямо в папках игры).

Подходит для большинства самодельных и инди-игр (Ren'Py без архивов, RPG Maker, GameMaker с внешним
звуком, Unity StreamingAssets, Wwise с отдельными .wem). Текста реплик нет — его распознаёт Whisper.
Установка заменяет файлы, оригиналы сохраняются в проекте (backup/) и возвращаются при удалении озвучки.
"""
import os
import re
import shutil
from pathlib import Path

from .. import audio
from .. import subtitles
from ..languages import code_from_component, code_from_dirname, code_from_filename

AUDIO_EXT = {".wav", ".ogg", ".mp3", ".flac", ".opus", ".wem"}
VOICE_TOKENS = {"voice", "voices", "vo", "dialog", "dialogue", "dialogues", "speech", "dub", "dubbing",
                "lines", "talk", "npc", "barks", "voiceover", "voiceovers", "vox"}
GENERIC_DIRS = {"voice", "voices", "vo", "audio", "sound", "sounds", "sfx", "dialog", "dialogue", "speech",
                "en", "eng", "english", "en-us", "streamingassets", "data", "assets", "media", "resources"}
MUSIC_TOKENS = {"music", "bgm", "ost", "soundtrack", "ambient", "ambience", "amb"}
SKIP_DIRS = {"_commonredist", "redist", "directx", "$recycle.bin", "__pycache__"}
MIN_SEC, MAX_SEC, MAX_FILES = 0.6, 40.0, 20000


def _tokens(s: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", s.lower()))


def _quick_has_audio(root: Path, max_files: int = 60_000) -> bool:
    """Быстрая проверка без полного анализа: останавливается на первом звуковом файле."""
    seen = 0
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x.lower() not in SKIP_DIRS]
        for n in names:
            seen += 1
            if os.path.splitext(n)[1].lower() in AUDIO_EXT:
                try:
                    if os.path.getsize(os.path.join(d, n)) > 8_000:
                        return True
                except OSError:
                    pass
            if seen >= max_files:
                return False
    return False


class LooseAdapter:
    id = "loose"
    name = "Отдельные звуковые файлы (замена с резервной копией)"

    @staticmethod
    def detect(game: dict) -> bool:
        """Подходит, только если в папке игры есть отдельные звуковые файлы (не внутри архивов)."""
        root = Path(game["path"])
        if not root.is_dir():
            return False
        analysis = game.get("analysis") or {}
        if "loose_audio" in analysis:
            return analysis["loose_audio"] > 0
        return _quick_has_audio(root)

    def describe(self, game: dict) -> dict:
        return {"adapter": self.id, "name": self.name, "supports_official_ru": True,
                "text_matching": "localisation key, then recognised source speech"}

    def _candidates(self, root: Path) -> list[Path]:
        files = []
        for d, dirs, names in os.walk(root):
            dirs[:] = [x for x in dirs if x.lower() not in SKIP_DIRS]
            for n in names:
                p = Path(d) / n
                if p.suffix.lower() in AUDIO_EXT:
                    try:
                        if p.stat().st_size > 8_000:
                            files.append(p)
                    except OSError:
                        pass
            if len(files) > MAX_FILES * 3:
                break
        files = [f for f in files if not (_tokens(str(f.relative_to(root))) & MUSIC_TOKENS)]
        def source_language(f):
            rel = f.relative_to(root)
            for part in reversed(rel.parts[:-1]):
                lang = code_from_component(part) or code_from_dirname(part)
                if lang:
                    return lang
            return code_from_filename(f.stem)
        files = [f for f in files if source_language(f) in (None, "en")]
        voiced = [f for f in files if _tokens(str(f.relative_to(root).parent)) & VOICE_TOKENS]
        return (voiced or files)[:MAX_FILES]

    @staticmethod
    def speaker_of(rel: Path) -> str:
        for part in reversed(rel.parts[:-1]):
            if part.lower() not in GENERIC_DIRS and not re.fullmatch(r"\d+", part):
                return part
        m = re.match(r"([A-Za-z]+)[_\-\d]", rel.stem)
        return m.group(1) if m else "Голос"

    def extract(self, project, game: dict, progress):
        root = Path(game["path"])
        import hashlib
        import json
        from ..discovery import scan
        inventory = scan(root)
        progress("Поиск звуковых файлов…")
        files = self._candidates(root)
        pool = subtitles.harvest(root, progress)
        lines, chars = [], {}
        for i, f in enumerate(files, 1):
            if i % 10 == 0 or i == len(files):
                progress(f"Извлечение звука: {i}/{len(files)}", i, len(files))
            rel = f.relative_to(root)
            line_id = rel.as_posix()
            orig = project.orig_dir / (hashlib.sha256(line_id.encode("utf-8")).hexdigest()[:24] + ".wav")
            try:
                audio.decode_to_wav(f, orig)
                info = audio.sf.info(str(orig))
            except Exception:
                continue
            if not MIN_SEC <= info.duration <= MAX_SEC:
                orig.unlink(missing_ok=True)
                continue
            spk = self.speaker_of(rel)
            ch = chars.setdefault(spk, {"id": spk, "name": spk, "best": None, "best_d": 0})
            if 3.5 <= info.duration <= 12 and info.duration > ch["best_d"]:
                ch["best"], ch["best_d"] = line_id, info.duration
            text = subtitles.exact_asset_match(pool, line_id)
            lines.append({"id": line_id, "char": spk, "orig": str(orig.relative_to(project.dir)),
                          "duration": round(info.duration, 3), "en": text["en"] if text else None,
                          "ru_official": text["ru"] if text else None,
                          "src": {"rel": line_id, "ext": f.suffix.lower(), "sr": info.samplerate,
                                  "channels": info.channels, "subtitle_key": text["key"] if text else None}})
        project.reset()
        subtitles.save_pool(project, pool)
        (project.dir / "assets.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), "utf-8")
        project.add_lines(lines)
        project.set_characters([{"id": c["id"], "name": c["name"], "ref_line": c["best"]} for c in chars.values()])
        return len(lines)

    def install(self, project, game: dict, progress) -> int:
        root = Path(game["path"])
        approved = project.lines(status="approved")
        wem_pairs = []
        for i, l in enumerate(approved, 1):
            if i % 10 == 0 or i == len(approved):
                progress(f"Установка: {i}/{len(approved)}", i, len(approved))
            src = l["src"]
            target = root / src["rel"]
            backup = project.backup_dir / src["rel"]
            if target.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            gen = project.dir / l["gen"]
            if src["ext"] == ".wem":
                wem_pairs.append((gen, target))
            else:
                audio.encode_like(gen, src["ext"], target, sr=src["sr"], channels=src["channels"])
        if wem_pairs:
            progress("Конвертация в формат Wwise…")
            audio.encode_wem_batch(wem_pairs)
        return len(approved)

    def uninstall(self, project, game: dict, progress):
        root = Path(game["path"])
        if not project.backup_dir.is_dir():
            return
        backups = [p for p in project.backup_dir.rglob("*") if p.is_file()]
        for i, b in enumerate(backups, 1):
            progress(f"Восстановление оригиналов: {i}/{len(backups)}", i, len(backups))
            target = root / b.relative_to(project.backup_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b, target)
        shutil.rmtree(project.backup_dir, ignore_errors=True)

    def is_installed(self, project, game: dict) -> bool:
        return project.backup_dir.is_dir() and any(project.backup_dir.rglob("*.*"))
