"""Анализ папки игры: движок, звуковые библиотеки, языки озвучки и текста, размеры."""
import os
import re
import time
from collections import Counter

from .languages import code_from_component, code_from_dirname, code_from_filename
from .engine_catalog import PROBE_EXT, signature, candidates

AUDIO_EXT = {".wem", ".bnk", ".bank", ".fsb", ".ogg", ".wav", ".mp3", ".opus", ".flac", ".acb", ".awb",
             ".xwb", ".xsb", ".wma", ".m4a", ".aac", ".snd", ".hca", ".adx"}
VIDEO_EXT = {".bik", ".bk2", ".usm", ".mp4", ".webm", ".wmv", ".avi", ".mkv", ".ogv"}
TEXT_EXT = {".locres", ".json", ".csv", ".txt", ".xml", ".po", ".mo", ".loc", ".strings", ".tsv",
            ".yaml", ".yml", ".lang", ".tab", ".srt", ".vtt", ".ass", ".ssa", ".rpy", ".rpyc"}
# Пакеты, у которых язык может быть только в имени файла
PACKAGE_EXT = {".pak", ".utoc", ".ucas", ".pck", ".cpk", ".assets", ".bundle", ".vpk", ".ba2", ".bsa",
               ".bin", ".dat", ".arc", ".archive", ".rpa", ".pk3", ".pk4", ".iwd", ".love", ".rpf", ".mpq", ".forge", ".rpkg"}
LANG_FILE_EXT = AUDIO_EXT | VIDEO_EXT | TEXT_EXT | PACKAGE_EXT
LOOSE_AUDIO_EXT = {".wav", ".ogg", ".mp3", ".flac", ".opus", ".wem"}

AUDIO_DIRS = {"audio", "sound", "sounds", "voice", "voices", "vo", "speech", "dialog", "dialogue",
              "wwise", "wwiseaudio", "fmod", "soundbanks", "generatedsoundbanks", "sfx", "music"}
TEXT_DIRS = {"localization", "localisation", "loc", "text", "texts", "strings", "subtitles", "l10n",
             "i18n", "lang", "languages", "translations"}
SKIP_DIRS = {"_commonredist", "__redist", "redist", "redistributables", "directx", "vcredist",
             "__installer", "$recycle.bin", "system volume information"}
JUNK_EXE = ("unins", "crash", "setup", "redist", "vc_redist", "dxsetup", "dotnet", "uninstall",
            "unitycrashhandler", "ue4prereq", "ueprereq", "easyanticheat", "battleye", "launcherpatcher")


# Слова в именах архивов: hl2_sound_vo_russian.vpk, lang_ru_text.archive, pakchunk11-...-Vo-Rest-RU.pak
VOICE_TOKENS = {"vo", "voice", "voices", "voiceover", "speech", "dialog", "dialogue", "dub", "dubbing"}
TEXT_TOKENS = {"text", "texts", "txt", "loc", "locres", "subtitles", "subtitle", "subs", "strings",
               "localization", "localisation", "l10n", "ui"}
SOUND_TOKENS = {"snd", "sound", "sounds", "audio", "sfx", "banks", "soundbanks", "wwise", "fmod"}
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _kind(parts_lower: tuple[str, ...], stem: str, ext: str) -> str:
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in TEXT_EXT:
        return "text"
    tokens = set(_TOKEN_SPLIT.split(stem))
    for p in parts_lower:
        tokens.update(_TOKEN_SPLIT.split(p))
    if tokens & VOICE_TOKENS:
        return "audio"
    if tokens & TEXT_TOKENS:
        return "text"
    if tokens & SOUND_TOKENS or any(p in AUDIO_DIRS for p in parts_lower):
        return "audio"
    if any(p in TEXT_DIRS for p in parts_lower):
        return "text"
    return "other"


def _lang_for(rel: tuple[str, ...], stem: str, ext: str) -> str | None:
    # Сначала папки (ближайшая к файлу важнее), потом имя файла
    for part in reversed(rel):
        code = code_from_component(part) or code_from_dirname(part)
        if code:
            return code
    if ext in LANG_FILE_EXT:
        return code_from_filename(stem)
    return None


def _detect_engine(names: set[str], exts: Counter, root_exe_stems: list[str]) -> str | None:
    has = names.__contains__
    if has("unityplayer.dll") or has("globalgamemanagers") or has("data.unity3d"):
        return "Unity"
    if exts[".utoc"] or exts[".ucas"]:
        return "Unreal Engine 4.26+ / 5"
    if exts[".pak"] and has("paks"):
        return "Unreal Engine"
    if has("renpy") or exts[".rpa"] or exts[".rpyc"]:
        return "Ren'Py"
    if has("rmmz_core.js"):
        return "RPG Maker MZ"
    if has("rpg_core.js"):
        return "RPG Maker MV"
    if exts[".rgss3a"]:
        return "RPG Maker VX Ace"
    if exts[".rgss2a"] or exts[".rgssad"]:
        return "RPG Maker XP/VX"
    if has("data.win"):
        return "GameMaker"
    if has("gameinfo.gi"):
        return "Source 2"
    if has("gameinfo.txt") and exts[".vpk"]:
        return "Source"
    if has('project.godot'):
        return "Godot"
    if has("re_chunk_000.pak"):
        return "RE Engine"
    if exts[".ba2"] or exts[".bsa"]:
        return "Creation Engine (Bethesda)"
    if exts[".archive"] and has("archive"):
        return "REDengine"
    if exts[".cas"] and exts[".cat"]:
        return "Frostbite"
    if exts[".xnb"]:
        return "XNA / FNA / MonoGame"
    if has("love.dll"):
        return "LÖVE"
    if has("package.nw") or has("nw.dll"):
        return "NW.js"
    return None


def _detect_middleware(names: set[str], exts: Counter) -> list[str]:
    found = []
    if exts[".wem"] or exts[".bnk"]:
        found.append("Wwise")
    if exts[".bank"] or exts[".fsb"] or any(n.startswith("fmod") and n.endswith(".dll") for n in names):
        found.append("FMOD")
    if exts[".acb"] or exts[".awb"] or exts[".usm"] or exts[".cpk"]:
        found.append("CRIWARE")
    if exts[".bik"] or exts[".bk2"]:
        found.append("Bink Video")
    if "mss64.dll" in names or "mss32.dll" in names:
        found.append("Miles Sound System")
    return found


def analyze(root: str, max_files: int = 400_000) -> dict:
    started = time.time()
    total_size = 0
    file_count = 0
    loose_audio = 0  # отдельные звуковые файлы — признак, что игру можно озвучить без распаковки архивов
    truncated = False
    exts: Counter = Counter()
    names: set[str] = set()
    exes: list[tuple[int, str]] = []
    langs: dict[str, dict[str, int]] = {}
    evidence: dict[str, list[str]] = {}
    format_warnings = []
    probes = 0

    stack: list[tuple[str, tuple[str, ...]]] = [(root, ())]
    while stack:
        current, rel = stack.pop()
        try:
            entries = os.scandir(current)
        except OSError:
            continue
        with entries:
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        low = e.name.lower()
                        if low in SKIP_DIRS:
                            continue
                        if len(rel) < 5:
                            names.add(low)
                        if len(rel) < 16:
                            stack.append((e.path, rel + (e.name,)))
                        continue
                    if not e.is_file(follow_symlinks=False):
                        continue
                    size = e.stat(follow_symlinks=False).st_size
                except OSError:
                    continue

                file_count += 1
                total_size += size
                low = e.name.lower()
                stem, ext = os.path.splitext(low)
                exts[ext] += 1
                probe_exe = ext == '.exe' and len(rel) <= 3 and not stem.startswith(JUNK_EXE)
                if probes < 4096 and (ext in PROBE_EXT - {'.wav', '.wem', '.ogg', '.flac'} or probe_exe):
                    probes += 1
                    try:
                        with open(e.path, 'rb') as stream:
                            header = stream.read(64)
                            kind = signature(header)
                            if probe_exe and size >= 32 and kind is None:
                                stream.seek(size - 12)
                                footer = stream.read(12)
                                if footer[8:] == b'GDPC':
                                    offset = size - 12 - int.from_bytes(footer[:8], 'little')
                                    if 0 <= offset < size - 12:
                                        stream.seek(offset)
                                        if stream.read(4) == b'GDPC':
                                            kind = 'godot-embedded-pck'
                        if kind:
                            examples = evidence.setdefault(kind, [])
                            if len(examples) < 8:
                                examples.append('/'.join(rel + (e.name,)))
                            if kind == 'godot-pck' and len(format_warnings) < 8:
                                version = int.from_bytes(header[4:8], 'little')
                                if version not in (1, 2):
                                    format_warnings.append(f'{e.name}: Godot PCK v{version} требует отдельного читателя.')
                                elif version == 2 and int.from_bytes(header[20:24], 'little') & 1:
                                    format_warnings.append(f'{e.name}: каталог Godot PCK зашифрован.')
                    except OSError:
                        pass
                if ext in LOOSE_AUDIO_EXT and size > 8_000:
                    loose_audio += 1
                if len(rel) < 5:
                    names.add(low)
                if ext == ".exe" and len(rel) <= 3 and not stem.startswith(JUNK_EXE):
                    exes.append((size, "\\".join(rel + (e.name,))))

                code = _lang_for(rel, stem, ext)
                if code:
                    bucket = langs.setdefault(code, {"audio": 0, "text": 0, "video": 0, "other": 0})
                    bucket[_kind(tuple(p.lower() for p in rel), stem, ext)] += size

                if file_count >= max_files:
                    truncated = True
                    stack.clear()
                    break

    exes.sort(reverse=True)
    root_exe_stems = [os.path.splitext(p)[0].lower() for _, p in exes if "\\" not in p]
    engine_candidates = candidates(names, exts, evidence)
    middleware = _detect_middleware(names, exts)
    for label, kinds in [('Wwise', {'wwise-package', 'wwise-bank'}), ('FMOD', {'fmod-fsb5', 'fmod-bank'})]:
        if kinds & evidence.keys() and label not in middleware:
            middleware.append(label)
    return {
        'analysis_version': 2,
        'engine_candidates': engine_candidates,
        'formats': evidence,
        'format_warnings': format_warnings,
        'signature_probes': probes,
        "loose_audio": loose_audio,
        "engine": engine_candidates[0]['name'] if engine_candidates else _detect_engine(names, exts, root_exe_stems),
        "middleware": middleware,
        "size": total_size,
        "files": file_count,
        "truncated": truncated,
        "languages": langs,
        "exes": [p for _, p in exes[:8]],
        "analyzed_at": time.time(),
        "duration": round(time.time() - started, 2),
    }
