"""Format-based game audio adapters. PCK/RPA originals are backed up before atomic replacement."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import pickle
import zlib
from functools import lru_cache
from pathlib import Path

from .open_packages import PCK, RPA
from .loose import AUDIO_EXT, MIN_SEC, MAX_SEC, MUSIC_TOKENS, _tokens, LooseAdapter
from .. import audio, subtitles
from ..languages import code_from_component, code_from_dirname, code_from_filename

READERS = {'.pck': PCK, '.rpa': RPA}


def packages(root, suffix):
    seen = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in {'.git', '__pycache__', 'node_modules', '_commonredist'}
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            seen += 1
            if seen > 100_000:
                return
            path = Path(directory) / name
            if path.suffix.lower() == suffix and not path.is_symlink():
                yield path


def usable(entry):
    rel = Path(entry.name)
    if entry.flags or rel.suffix.lower() not in AUDIO_EXT or _tokens(entry.name) & MUSIC_TOKENS or entry.size > 256 << 20:
        return False
    for part in reversed(rel.parts[:-1]):
        lang = code_from_component(part) or code_from_dirname(part)
        if lang:
            return lang == 'en'
    return code_from_filename(rel.stem) in (None, 'en')


@lru_cache(maxsize=64)
def _supported(path, size, modified):
    try:
        archive = READERS[Path(path).suffix.lower()](path)
        return archive.writable and any(usable(e) for e in archive.entries)
    except (OSError, ValueError, EOFError, TypeError, pickle.UnpicklingError, zlib.error):
        return False


class PackedAudioAdapter:
    suffix = ''

    @classmethod
    def detect(cls, game):
        for path in packages(Path(game['path']), cls.suffix):
            try:
                stat = path.stat()
            except OSError:
                continue
            if _supported(str(path), stat.st_size, stat.st_mtime_ns):
                return True
        return False

    def describe(self, game):
        return {'adapter': self.id, 'name': self.name, 'supports_official_ru': True,
                'notes': ['Поддерживаются открытые архивы с обычными аудиофайлами. Зашифрованные, встроенные в exe '
                          'и внутренние сериализованные аудиоресурсы требуют отдельного адаптера.',
                          'Архив пересобирается с заменой только одобренных реплик. Оригинал сохраняется для восстановления.']}

    def extract(self, project, game, progress):
        root = Path(game['path']).resolve()
        pool = subtitles.harvest(root, progress)
        lines, chars = [], {}
        project.orig_dir.mkdir(parents=True, exist_ok=True)
        for path in packages(root, self.suffix):
            try:
                archive = READERS[self.suffix](path)
            except (ValueError, OSError, EOFError, TypeError, pickle.UnpicklingError, zlib.error):
                continue
            if not archive.writable:
                continue
            for entry in archive.entries:
                if not usable(entry):
                    continue
                identity = path.relative_to(root).as_posix() + '!/' + entry.name
                original = project.orig_dir / (hashlib.sha256(identity.encode()).hexdigest()[:24] + '.wav')
                ext = Path(entry.name).suffix.lower()
                with tempfile.TemporaryDirectory() as td:
                    source = Path(td) / ('source' + ext)
                    try:
                        source.write_bytes(archive.read(entry))
                        audio.decode_to_wav(source, original)
                        data = audio.sf.info(str(original))
                    except (OSError, ValueError, RuntimeError):
                        original.unlink(missing_ok=True)
                        continue
                if not MIN_SEC <= data.duration <= MAX_SEC:
                    original.unlink(missing_ok=True)
                    continue
                char = LooseAdapter.speaker_of(Path(entry.name))
                match = subtitles.exact_asset_match(pool, entry.name)
                chars.setdefault(char, {'id': char, 'name': char, 'ref_line': identity})
                lines.append(dict(id=identity, char=char, orig=str(original.relative_to(project.dir)), duration=data.duration,
                                  en=match['en'] if match else None, ru_official=match['ru'] if match else None,
                                  src=dict(archive=path.relative_to(root).as_posix(), member=entry.name, ext=ext,
                                           sr=data.samplerate, channels=data.channels)))
                progress(f'{self.name}: извлечено {len(lines)} реплик')
        if not lines:
            raise ValueError('В архиве нет поддерживаемых английских звуковых файлов')
        project.reset()
        project.add_lines(lines)
        project.set_characters(list(chars.values()))
        subtitles.save_pool(project, pool)
        return len(lines)

    def install(self, project, game, progress):
        root = Path(game['path']).resolve()
        grouped = {}
        for line in project.lines(status='approved'):
            if line['gen']:
                grouped.setdefault(line['src']['archive'], []).append(line)
        installed = project.get(self.id + '_installed') or []
        for relative, lines in grouped.items():
            target = (root / relative).resolve()
            backup = (project.backup_dir / relative).resolve()
            if not target.is_relative_to(root) or not backup.is_relative_to(project.backup_dir.resolve()):
                raise ValueError('Package path outside project/game')
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(target, backup)
            original = READERS[self.suffix](backup)
            with tempfile.TemporaryDirectory(dir=target.parent, prefix='ruvoice_pack_') as td:
                replacements = {}
                for index, line in enumerate(lines):
                    src = line['src']
                    generated = (project.dir / line['gen']).resolve()
                    if not generated.is_relative_to(project.dir.resolve()):
                        raise ValueError('Generated audio outside project')
                    dest = Path(td) / (str(index) + src['ext'])
                    if src['ext'] == '.wem':
                        audio.encode_wem_batch([(generated, dest)])
                    else:
                        audio.encode_like(generated, src['ext'], dest, src['sr'], src['channels'])
                    replacements[src['member']] = dest
                rebuilt = Path(td) / ('rebuilt' + self.suffix)
                original.rebuild(rebuilt, replacements)
                # Validate the directory before committing the new archive.
                READERS[self.suffix](rebuilt)
                os.replace(rebuilt, target)
            if relative not in installed:
                installed.append(relative)
            project.set(self.id + '_installed', installed)
            progress(f'Установлен {relative}')
        return sum(map(len, grouped.values()))

    def uninstall(self, project, game, progress):
        root = Path(game['path']).resolve()
        remaining = list(project.get(self.id + '_installed') or [])
        for relative in list(remaining):
            target = (root / relative).resolve()
            backup = (project.backup_dir / relative).resolve()
            if not target.is_relative_to(root) or not backup.is_relative_to(project.backup_dir.resolve()):
                raise ValueError('Package path outside project/game')
            if not backup.is_file():
                raise ValueError('Original package backup is missing')
            with tempfile.TemporaryDirectory(dir=target.parent) as td:
                restored = Path(td) / 'original'
                shutil.copy2(backup, restored)
                os.replace(restored, target)
            remaining.remove(relative)
            project.set(self.id + '_installed', remaining)
            progress(f'Восстановлен {relative}')

    def is_installed(self, project, game):
        return bool(project.get(self.id + '_installed'))


class GodotPCKAdapter(PackedAudioAdapter):
    id, suffix, name = 'godot-pck', '.pck', 'Godot PCK (открытые архивы v1/v2)'


class RenpyRPAAdapter(PackedAudioAdapter):
    id, suffix, name = 'renpy-rpa', '.rpa', "Ren'Py RPA (открытые архивы v2/v3)"
