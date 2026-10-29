"""Unreal Engine 4/5 (Wwise): реплики из архивов .pak / .utoc, текст из .locres и словарей в архивах,
установка отдельным pak-модом в <Проект>\\Content\\Paks\\~mods\\RuVoice_P.pak (оригиналы не трогаются).

Привязки к играм нет: всё распознаётся по устройству файлов. Как сопоставляются звук, текст и персонаж
(по убыванию надёжности):
  1. Метка внутри .wem (LIST/adtl/labl) — ключ субтитра и имя персонажа.
  2. Таблица маркеров событий Wwise (файл FUStAkEventMarkerDesc.data, если он есть в архивах): событие →
     ID реплик → тексты из .locres, границы реплик внутри одного файла, имя говорящего из FUStDialogueDesc.
  3. Имя события из банка .bnk, ссылающегося на .wem: его хвост-хеш или оно само — ключ в .locres;
     персонаж — папка VO_<Имя> ассета события в .utoc или код в имени.
Словари AVAFDICT (Localization/…/*.bin) читаются по сигнатуре внутри файла. Если связь «звук ↔ строка»
найти не удалось, реплика получает текст на общем шаге подготовки: все пары EN/RU из .locres и словарей
сохраняются пулом, реплика сопоставляется по распознанной речи (subtitles.py).
"""
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import audio, tools
from ..languages import code_from_component, code_from_filename
from . import uelocres
from .ueiostore import IoStore
from .uepak import NeedsKey, Oodle, PakFile, parse_aes_key
from .wwisebnk import Bank

MOD_NAME = "RuVoice_P"
MIN_SEC, MAX_SEC = 0.3, 300.0
VOICE_TOKENS = {"vo", "voice", "voices", "dialog", "dialogue", "dialogues", "speech", "talk", "conv",
                "conversation", "dia", "lines", "npc", "barks"}
GENERIC_DIRS = {"media", "wwiseaudio", "wwise", "audio", "sound", "sounds", "windows", "content", "event",
                "events", "soundbank", "soundbanks", "generatedsoundbanks", "vo", "voice", "voices", "dialogue",
                "dialog", "sfx", "english", "en", "en-us"}
SKIP_LOCRES_DIRS = ("/engine/", "/plugins/")
HEX_SUFFIX_RE = re.compile(r"_([0-9A-Fa-f]{8})$")
DEC_SUFFIX_RE = re.compile(r"_(\d{5,})$")
ANTICHEAT = ("easyanticheat", "battleye", "eac_", "beclient", "beservice")

_lock = threading.Lock()
_cache: dict[str, tuple] = {}  # root → (подпись, UnrealFiles)


def _is_lang(path_component: str, code: str) -> bool:
    return code_from_component(path_component) == code


def _bin_lang(name: str) -> str | None:
    """Язык словаря Localization/WIN64/SUB-ruRU.bin → ru."""
    tag = Path(name).stem.split("-")[-1]
    tag = re.sub(r"([a-z])([A-Z])", r"\1-\2", tag)
    return code_from_component(tag) or code_from_filename(tag.lower())


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


# --- таблицы маркеров событий и говорящих (protobuf: FUStAkEventMarkerDesc / FUStDialogueDesc) ---

def _pb_varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while i < len(b):
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            break
    return v, i


def _pb_decode(b: bytes) -> list[tuple[int, int, object]]:
    i, out = 0, []
    while i < len(b):
        tag, i = _pb_varint(b, i)
        f, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _pb_varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = _pb_varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            break
        out.append((f, wt, v))
    return out


def _pb_str(v) -> str | None:
    try:
        return v.decode("utf-8") if isinstance(v, bytes) else None
    except UnicodeDecodeError:
        return None


def parse_event_markers(data: bytes) -> dict[str, dict[str, list[tuple[str, float, float | None]]]]:
    """FUStAkEventMarkerDesc.data → {событие(lower): {язык: [(id реплики, начало, конец|None), …]}}."""
    result = {}
    for f, wt, row in _pb_decode(data):
        if wt != 2:
            continue
        name, langs = None, {}
        for f2, wt2, v in _pb_decode(row):
            if f2 == 2 and wt2 == 2:
                name = _pb_str(v)
            elif f2 == 4 and wt2 == 2:
                lang, starts, ends = None, {}, {}
                for f3, wt3, v3 in _pb_decode(v):
                    if f3 == 1 and wt3 == 2:
                        lang = _pb_str(v3)
                    elif f3 == 2 and wt3 == 2:
                        mark, t = None, None
                        for f4, wt4, v4 in _pb_decode(v3):
                            if f4 == 1 and wt4 == 2:
                                mark = _pb_str(v4)
                            elif f4 == 2 and wt4 == 5:
                                t = struct.unpack("<f", v4)[0]
                        if mark and t is not None and "#" in mark:
                            did, kind = mark.split("#", 1)
                            (starts if kind == "start" else ends)[did] = t
                if lang:
                    items = sorted(starts.items(), key=lambda kv: kv[1])
                    langs[lang] = [(did, t, ends.get(did)) for did, t in items]
        if name and langs:
            result[name.lower()] = langs
    return result


def parse_dialogue_speakers(data: bytes) -> dict[str, str]:
    """FUStDialogueDesc-*.data → {id реплики: имя говорящего}."""
    out = {}
    for f, wt, row in _pb_decode(data):
        if wt != 2:
            continue
        did, speaker = None, None
        for f2, wt2, v in _pb_decode(row):
            if f2 == 1 and wt2 == 0:
                did = str(v)
            elif f2 == 2 and wt2 == 2:
                speaker = _pb_str(v)
        if did and speaker:
            out[did] = speaker
    return out


# --- словари AVAFDICT (сигнатура «AVAFDICT» в UTF-16 в начале файла) ---

def parse_avafdict(data: bytes) -> dict[str, str]:
    if not data.startswith("AVAFDICT".encode("utf-16-le")):
        return {}
    count, hdr, _, off_c, _ = struct.unpack_from("<5Q", data, 32)
    blob = data[off_c:]
    out = {}
    for i in range(count):
        ko, kl, vo, vl = struct.unpack_from("<QIQI", data, hdr + i * 24)
        out[blob[ko:ko + kl].decode("utf-8", "replace")] = blob[vo:vo + vl].decode("utf-8", "replace")
    return out


# --- метки внутри .wem ---

def wem_labels(data: bytes) -> list[str]:
    labels = []
    for cid, body in audio.read_riff_chunks_from_bytes(data):
        if cid != b"LIST" or body[:4] != b"adtl":
            continue
        pos = 4
        while pos + 8 <= len(body):
            sub, size = body[pos:pos + 4], struct.unpack_from("<I", body, pos + 4)[0]
            if sub == b"labl" and size >= 4:
                labels.append(body[pos + 12:pos + 8 + size].split(b"\0")[0].decode("utf-8", "replace"))
            pos += 8 + size + (size & 1)
    return labels


# --- файлы игры ---

class UnrealFiles:
    """Все архивы игры одним индексом: путь внутри игры → (pak, запись). Поздние/_P-архивы перекрывают ранние."""

    def __init__(self, root: Path, key: bytes | None):
        self.root = root
        self.key = key
        self.paks_dir = self.find_paks_dir(root)
        self.project = self.paks_dir.parent.parent.name if self.paks_dir and self.paks_dir.parent.parent != root \
            else None
        self.paks: list[PakFile] = []
        self.index: dict[str, tuple[PakFile, object]] = {}
        self.errors: list[str] = []
        self.needs_key = False
        self.encrypted = False
        self.max_version = 11
        self.skipped: list[str] = []
        self._iostore_names: set[str] | None = None
        self._open()

    @staticmethod
    def find_paks_dir(root: Path) -> Path | None:
        cands = [p for p in [root / "Content" / "Paks", *root.glob("*/Content/Paks")] if p.is_dir()]
        cands = [p for p in cands if p.parent.parent.name.lower() != "engine" and any(p.glob("*.pak"))]
        return max(cands, key=lambda p: sum(f.stat().st_size for f in p.glob("*.pak")), default=None)

    @staticmethod
    def signature(paks_dir: Path) -> tuple:
        return tuple(sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in paks_dir.glob("*.pak")))

    @staticmethod
    def pak_order(path: Path) -> tuple:
        stem = path.stem
        return (1 if re.search(r"_P(?:[-_]|$)", stem) else 0, stem.lower())

    def _open(self):
        if not self.paks_dir:
            self.errors.append("Папка Content\\Paks не найдена")
            return
        for p in sorted(self.paks_dir.glob("*.pak"), key=self.pak_order):
            stem = p.stem
            lang = code_from_filename(stem)
            if stem.startswith(MOD_NAME) or (lang and lang != "en"):
                self.skipped.append(p.name)  # наш мод и архивы других языков (фанатские озвучки) не читаем
                continue
            try:
                pak = PakFile(p, self.key, [self.root])
            except NeedsKey:
                self.needs_key = self.encrypted = True
                continue
            except Exception as e:
                self.errors.append(f"{p.name}: {e}")
                continue
            self.encrypted |= pak.encrypted_index
            self.max_version = max(self.max_version, pak.version) if self.paks else pak.version
            self.paks.append(pak)
            for name, entry in pak.files.items():
                self.index[name] = (pak, entry)
        if self.index and self.key and self.needs_key:
            self.errors.append("Часть архивов не удалось расшифровать этим ключом")

    def read(self, name: str) -> bytes:
        pak, entry = self.index[name]
        return pak.read_entry(entry)

    def names(self, suffix: str) -> list[str]:
        return [n for n in self.index if n.lower().endswith(suffix)]

    def uses_oodle(self) -> bool:
        return any("oodle" in m.lower() for pak in self.paks for m in pak.methods)

    def iostore_names(self) -> set[str]:
        """Имена ассетов из .utoc (только индекс каталогов, без чтения .ucas)."""
        if self._iostore_names is None:
            names = set()
            for utoc in sorted(self.paks_dir.glob("*.utoc")) if self.paks_dir else []:
                if utoc.stem.startswith(MOD_NAME) or utoc.stem == "global":
                    continue
                try:
                    names |= set(IoStore(utoc, self.key).files)
                except Exception:
                    continue
            self._iostore_names = names
        return self._iostore_names

    def close(self):
        for p in self.paks:
            p.close()


def _files_for(game: dict) -> UnrealFiles:
    root = Path(game["path"])
    key = parse_aes_key(game.get("aes_key"))
    paks_dir = UnrealFiles.find_paks_dir(root)
    sig = (key, UnrealFiles.signature(paks_dir) if paks_dir else ())
    with _lock:
        cached = _cache.get(str(root))
        if cached and cached[0] == sig:
            return cached[1]
        files = UnrealFiles(root, key)
        _cache[str(root)] = (sig, files)
        return files


# --- адаптер ---

class UnrealAdapter:
    id = "unreal"
    name = "Unreal Engine (pak-мод в Paks\\~mods)"

    @staticmethod
    def detect(game: dict) -> bool:
        root = Path(game["path"])
        if not root.is_dir():
            return False
        engine = (game.get("analysis") or {}).get("engine") or ""
        if engine.startswith("Unreal"):
            return UnrealFiles.find_paks_dir(root) is not None
        return UnrealFiles.find_paks_dir(root) is not None

    # --- описание для интерфейса ---

    def describe(self, game: dict) -> dict:
        root = Path(game["path"])
        info = {"adapter": self.id, "name": self.name, "supports_official_ru": False, "notes": [], "warnings": [],
                "needs_key": False, "key_set": bool((game.get("aes_key") or "").strip()), "ready": False}
        try:
            files = _files_for(game)
        except ValueError as e:  # кривой ключ
            info.update(needs_key=True, note=str(e), warnings=[str(e)])
            return info
        info["paks_dir"] = str(files.paks_dir) if files.paks_dir else None
        info["mod_path"] = str(self._mod_pak(files)) if files.paks_dir else None
        if files.needs_key:
            info["needs_key"] = True
            msg = ("AES-ключ не подходит: архивы не расшифровались" if files.key
                   else "Архивы игры зашифрованы — введите AES-ключ игры в карточке игры (ключи для популярных игр "
                        "публикуют сообщества моддеров; программа их не распространяет)")
            info["warnings"].append(msg)
            info["note"] = "нужен AES-ключ"
        info["warnings"] += files.errors
        if files.uses_oodle() and not Oodle.find([root]):
            info["warnings"].append("Нет библиотеки Oodle (tools\\oodle\\oo2core_9_win64.dll) — архивы сжаты Oodle")
            info["note"] = info.get("note") or "нет библиотеки Oodle"
        wems = files.names(".wem")
        voice = self._voice_wems(wems)
        info["voice_files"] = len(voice)
        info["all_wems"] = len(wems)
        if files.index and not wems:
            info["warnings"].append("В архивах нет звука Wwise (.wem) — другие звуковые форматы Unreal пока не "
                                    "разбираются")
            info["note"] = info.get("note") or "звук не в формате Wwise"
        en_loc, ru_loc = self._locres(files, "en"), self._locres(files, "ru")
        sub = [n for n in files.index if n.lower().endswith(".bin") and "/localization/" in n.lower()]
        info["supports_official_ru"] = bool(ru_loc) or any(_bin_lang(n) == "ru" for n in sub)
        if en_loc or sub:
            info["notes"].append(f"Текст: {len(en_loc)} файлов .locres (английский)" +
                                 (f", русские субтитры есть ({len(ru_loc)})" if ru_loc else ""))
        if voice:
            info["notes"].append(f"Голосовых файлов (.wem, английская озвучка): {len(voice)}")
        if files.encrypted and not files.needs_key:
            info["notes"].append("Архивы зашифрованы, ключ подошёл")
        if files.skipped:
            info["notes"].append("Пропущены архивы других языков/модов: " + ", ".join(files.skipped[:4]) +
                                 (" …" if len(files.skipped) > 4 else ""))
        ac = self._anticheat(root)
        if ac:
            info["warnings"].append(f"В игре есть античит ({ac}): pak-мод может быть заблокирован или привести к "
                                    "бану в сетевой игре")
        info["ready"] = bool(voice) and not files.needs_key and not (files.uses_oodle() and not Oodle.find([root]))
        if not info.get("note"):
            info["note"] = "поддерживается" if info["ready"] else "нет подходящих файлов"
        return info

    @staticmethod
    def _anticheat(root: Path) -> str | None:
        for p in list(root.glob("*")) + list(root.glob("*/*")) + list(root.glob("*/Binaries/*/*")):
            low = p.name.lower()
            if any(t in low for t in ANTICHEAT):
                return p.name
        return None

    # --- выбор файлов ---

    @staticmethod
    def _voice_wems(wems: list[str]) -> list[str]:
        """Локализованные .wem (папка английского языка) — это и есть озвучка."""
        return [n for n in wems if any(_is_lang(part, "en") for part in n.split("/")[:-1])]

    @staticmethod
    def _locres(files: UnrealFiles, code: str) -> list[str]:
        out = []
        for n in files.names(".locres"):
            low = "/" + n.lower()
            if any(s in low for s in SKIP_LOCRES_DIRS):
                continue
            parts = n.split("/")
            if len(parts) >= 2 and _is_lang(parts[-2], code):
                out.append(n)
        # Если есть и en, и en-US — берём обе (ключи одинаковые), дубли безвредны
        return out

    def _texts(self, files: UnrealFiles, code: str, progress) -> dict[str, str]:
        """Ключ (в нижнем регистре; и 'ns/key', и 'key') → текст из .locres и словарей AVAFDICT."""
        out: dict[str, str] = {}
        for n in self._locres(files, code):
            try:
                tables = uelocres.parse(files.read(n))
            except Exception:
                continue
            for ns, keys in tables.items():
                for k, v in keys.items():
                    if not v or not v.strip():
                        continue
                    out.setdefault(k.lower(), v)
                    if ns:
                        out.setdefault(f"{ns}/{k}".lower(), v)
        for n in files.index:
            low = n.lower()
            if low.endswith(".bin") and "/localization/" in low:
                if _bin_lang(n) == code:
                    try:
                        for k, v in parse_avafdict(files.read(n)).items():
                            if v.strip():
                                out.setdefault(k.lower(), v)
                    except Exception:
                        continue
        return out

    @staticmethod
    def _hex_index(texts: dict[str, str]) -> dict[str, str]:
        out = {}
        for k, v in texts.items():
            m = HEX_SUFFIX_RE.search(k)
            if m:
                out.setdefault(m.group(1).lower(), v)
        return out

    def _event_sources(self, files: UnrealFiles, progress) -> dict[int, list[tuple[str, bool]]]:
        """SourceID (имя .wem) → [(имя банка/события, банк из папки английского языка)]."""
        banks = files.names(".bnk")
        out: dict[int, list[tuple[str, bool]]] = {}
        for i, n in enumerate(banks, 1):
            if i % 500 == 0 or i == len(banks):
                progress(f"Чтение банков Wwise: {i}/{len(banks)}", i, len(banks))
            parts = n.split("/")
            lang = any(_is_lang(p, "en") for p in parts[:-1])
            if any(code_from_component(p) not in (None, "en") for p in parts[:-1]):
                continue  # банки других языков
            try:
                bank = Bank(files.read(n))
            except Exception:
                continue
            stem = parts[-1][:-4]
            for sid in bank.source_ids:
                out.setdefault(sid, []).append((stem, lang))
        for sid, lst in out.items():
            lst.sort(key=lambda t: (not t[1], t[0]))
        return out

    @staticmethod
    def _marker_tables(files: UnrealFiles):
        markers, speakers = {}, {}
        for n in files.index:
            low = n.lower()
            if low.endswith("fustakeventmarkerdesc.data"):
                markers = parse_event_markers(files.read(n))
            elif re.search(r"fustdialoguedesc[^/]*\.data$", low):
                speakers.update(parse_dialogue_speakers(files.read(n)))
        return markers, speakers

    # --- персонаж ---

    @staticmethod
    def _speaker_from_assets(event: str, assets: dict[str, list[str]]) -> str | None:
        """…/Events/VO/VO_Maria/CH05/VO_Ma_James_76233101.uasset → Maria (у ассета может быть несколько путей —
        анимация губ, событие — берём тот, где есть папка VO_<Имя>)."""
        paths = []
        for stem in (event, re.sub(r"^(Play_Event_|Play_|Stop_)", "", event)):
            paths += assets.get(stem.lower(), [])
        for path in paths:
            for p in reversed(path.split("/")[:-1]):
                m = re.fullmatch(r"(?:VO|Voice|Dialog(?:ue)?|NV)_([A-Za-z][A-Za-z0-9]+)", p)
                if m:
                    return m.group(1)
        return None

    @staticmethod
    def _speaker_from_event(event: str) -> str | None:
        m = re.search(r"(?:^|_)VO_([A-Za-z]+)_", event)
        if m:
            return m.group(1)
        m = re.search(r"(?:^|_)(?:enm|npc|chr|char|boss)_(.+?)_(?:voice|vo|dialog)", event, re.I)
        if m:
            return m.group(1)
        m = re.search(r"^(?:Play_|Stop_|EVT_|Event_)?(.+?)_(?:voice|vo|dialog)", event, re.I)
        if m and m.group(1).lower() not in ("evt", "play", "event", "seq", "voice"):
            return m.group(1)
        return None

    @staticmethod
    def _speaker_from_path(pak_path: str) -> str:
        # Первые две папки — <Проект>/Content, они не про персонажа
        for part in reversed(pak_path.split("/")[2:-1]):
            low = part.lower()
            if low not in GENERIC_DIRS and not re.fullmatch(r"\d+", low) and code_from_component(part) is None:
                return part
        return "Голос"

    # --- извлечение ---

    def extract(self, project, game: dict, progress):
        files = _files_for(game)
        if files.needs_key:
            raise NeedsKey("Архивы игры зашифрованы — введите AES-ключ в карточке игры")
        if not files.index:
            raise RuntimeError("Архивы игры не прочитались: " + "; ".join(files.errors or ["нет .pak"]))
        progress("Чтение субтитров…")
        en, ru = self._texts(files, "en", progress), self._texts(files, "ru", progress)
        hex_en, hex_ru = self._hex_index(en), self._hex_index(ru)
        markers, speakers = self._marker_tables(files)
        wems = files.names(".wem")
        voice = self._voice_wems(wems)
        if not voice:
            raise RuntimeError("Не найдены голосовые файлы (.wem в папке английского языка)")
        sources = self._event_sources(files, progress)
        progress("Чтение индекса ассетов (.utoc)…")
        assets: dict[str, list[str]] = {}
        for n in files.iostore_names():
            if n.endswith(".uasset"):
                assets.setdefault(Path(n).stem.lower(), []).append(n)

        # 1. Для каждого .wem — события, текст, персонаж, (сегменты)
        plan = []  # (pak_path, data, [segment-записи])
        total = len(voice)
        for i, pak_path in enumerate(sorted(voice), 1):
            if i % 200 == 0 or i == total:
                progress(f"Сопоставление реплик и субтитров: {i}/{total}", i, total)
            stem = pak_path.rsplit("/", 1)[-1][:-4]
            sid = int(stem) if stem.isdigit() else None
            try:
                data = files.read(pak_path)
            except Exception:
                continue
            events = [e for e, _ in sources.get(sid, [])] if sid is not None else [stem]
            labels = wem_labels(data)
            segs = self._match(pak_path, events, labels, en, ru, hex_en, hex_ru, markers, speakers, assets)
            plan.append((pak_path, data, segs))
        self._fix_unknown_speakers(plan)

        # 2. Декодирование пачками через vgmstream
        lines, chars = [], {}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            batch = 48
            for b in range(0, len(plan), batch):
                chunk = plan[b:b + batch]
                progress(f"Извлечение звука: {min(b + batch, len(plan))}/{len(plan)}", b + len(chunk), len(plan))
                decoded = _decode_batch(td, [(f"{b + j:06d}", data) for j, (_, data, _) in enumerate(chunk)])
                for j, (pak_path, data, segs) in enumerate(chunk):
                    wav = decoded.get(f"{b + j:06d}")
                    if not wav:
                        continue
                    try:
                        pcm, sr = sf.read(str(wav), dtype="float32", always_2d=True)
                    except Exception:
                        continue
                    full = len(pcm) / sr
                    if full < MIN_SEC:
                        continue
                    for seg in segs:
                        start, end = seg.get("segment") or (0.0, full)
                        end = min(end if end else full, full)
                        if end - start < MIN_SEC or end - start > MAX_SEC:
                            continue
                        line_id = pak_path if not seg.get("segment") else f"{pak_path}#{seg['did']}"
                        orig = project.orig_dir / (_safe(line_id) + ".wav")
                        orig.parent.mkdir(parents=True, exist_ok=True)
                        part = pcm[int(start * sr):int(end * sr)]
                        sf.write(str(orig), part, sr, subtype="PCM_16")
                        d = round(len(part) / sr, 3)
                        spk = seg["char"]
                        ch = chars.setdefault(spk, {"id": spk, "name": spk, "best": None, "best_d": 0})
                        if seg.get("en") and 3.5 <= d <= 12 and d > ch["best_d"]:
                            ch["best"], ch["best_d"] = line_id, d
                        lines.append({
                            "id": line_id, "char": spk, "orig": str(orig.relative_to(project.dir)), "duration": d,
                            "en": seg.get("en"), "ru_official": seg.get("ru"),
                            "src": {"pak_path": pak_path, "sr": int(sr), "channels": int(pcm.shape[1]),
                                    "full": round(full, 3), "segment": seg.get("segment"), "event": seg.get("event"),
                                    "key": seg.get("key")}})
        project.reset()
        project.add_lines(lines)
        project.set_characters([{"id": c["id"], "name": c["name"], "ref_line": c["best"]} for c in chars.values()])
        project.set("unreal", {"project": files.project, "paks_dir": str(files.paks_dir), "pak_version": files.max_version})
        # Пул субтитров для общего шага подготовки: реплики, которым связь «звук ↔ строка» найти не удалось,
        # получат русский текст по распознанной речи (subtitles.py) — в любой игре на Unreal
        keys = [k for k in en if k in ru]
        if any("/" in k for k in keys):  # у .locres каждый ключ есть и как «пространство/ключ», и как «ключ»
            keys = [k for k in keys if "/" in k]
        rows = [{"key": k, "en": en[k], "ru": ru[k], "group": k.split("/")[0] if "/" in k else None} for k in keys]
        from .. import subtitles
        subtitles.save_pool(project, rows)
        return len(lines)

    @staticmethod
    def _fix_unknown_speakers(plan):
        """Реплики без текста: если в имени события встречается имя уже известного персонажа — это он
        (Play_C039_James_Kill_Eddie_NV → James)."""
        known = {}
        for _, _, segs in plan:
            for s in segs:
                if s.get("key") and s["char"]:
                    known[s["char"].lower()] = s["char"]
        if not known:
            return
        pat = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in sorted(known, key=len, reverse=True))
                         + r")(?![a-z])", re.I)
        for _, _, segs in plan:
            for s in segs:
                if s.get("key") or not s.get("event") or s["char"].lower() in known:
                    continue
                m = pat.search(s["event"])
                if m:
                    s["char"] = known[m.group(1).lower()]

    def _match(self, pak_path, events, labels, en, ru, hex_en, hex_ru, markers, speakers, assets) -> list[dict]:
        """Реплики одного .wem: обычно одна; у катсцен с маркерами — несколько сегментов."""
        # Маркеры событий: событие → ID реплик с временем
        for ev in events:
            m = markers.get(ev.lower())
            if not m:
                continue
            lang = next((l for l in m if _is_lang(l, "en")), next(iter(m)))
            items = m[lang]
            segs = []
            for idx, (did, start, end) in enumerate(items):
                nxt = items[idx + 1][1] if idx + 1 < len(items) else None
                stop = end if end and (nxt is None or end <= nxt + 0.05) else nxt
                key = f"fustdialoguedesc.{did}.content"
                spk = speakers.get(did) or self._speaker_from_event(ev) or self._speaker_from_path(pak_path)
                segs.append({"did": did, "segment": [round(max(0.0, start - 0.05), 3), round(stop, 3) if stop else None]
                             if len(items) > 1 else None,
                             "en": en.get(key), "ru": ru.get(key), "char": spk, "event": ev, "key": key})
            if segs:
                return segs
        # Метка внутри .wem: ключ субтитра = метка без суффикса варианта
        for lab in labels:
            if lab.startswith(":"):
                continue
            for cand in (lab, lab.split("-")[0], re.sub(r"[-_]\d+$", "", lab)):
                t = en.get(cand.lower())
                if t is not None or cand.lower() in ru:
                    # Регистр меток в разных файлах разный (AbrahamRonen_… / abrahamronen_…) — приводим к одному
                    spk = cand.split("_")[0].lower() if "_" in cand else self._speaker_from_path(pak_path)
                    return [{"en": t, "ru": ru.get(cand.lower()), "char": spk, "event": events[0] if events else None,
                             "key": cand}]
        # Имя события: хвост-хеш, само имя, десятичный хвост
        for ev in events:
            cands = []
            m = HEX_SUFFIX_RE.search(ev)
            if m:
                cands.append(("hex", m.group(1).lower()))
            for s in (ev, re.sub(r"^(Play_Event_|Play_|Stop_)", "", ev)):
                cands.append(("key", s.lower()))
            m = DEC_SUFFIX_RE.search(ev)
            if m:
                cands.append(("key", m.group(1)))
            for kind, c in cands:
                t = (hex_en if kind == "hex" else en).get(c)
                r = (hex_ru if kind == "hex" else ru).get(c)
                if t is not None or r is not None:
                    spk = (self._speaker_from_assets(ev, assets) or self._speaker_from_event(ev)
                           or self._speaker_from_path(pak_path))
                    return [{"en": t, "ru": r, "char": spk, "event": ev, "key": c}]
        # Текста нет: персонаж по событию/папке; метка тоже может подсказать
        ev = events[0] if events else None
        spk = None
        if labels:
            lab = next((l for l in labels if not l.startswith(":")), None)
            if lab and "_" in lab:
                spk = lab.split("_")[0].lower()
        if not spk and ev:
            spk = self._speaker_from_assets(ev, assets) or self._speaker_from_event(ev)
        return [{"en": None, "ru": None, "char": spk or self._speaker_from_path(pak_path), "event": ev, "key": None}]

    # --- установка ---

    def _mod_pak(self, files: UnrealFiles) -> Path:
        return files.paks_dir / "~mods" / f"{MOD_NAME}.pak"

    def install(self, project, game: dict, progress) -> int:
        files = _files_for(game)
        if not files.paks_dir:
            raise RuntimeError("Папка Content\\Paks не найдена")
        repak = tools.repak()
        if not repak:
            raise RuntimeError("Нет tools\\repak\\repak.exe — он нужен для сборки pak-мода")
        approved = project.lines(status="approved")
        if not approved:
            return 0
        stage = project.dir / "stage"
        shutil.rmtree(stage, ignore_errors=True)
        work = project.dir / "stage_wav"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        by_file: dict[str, list[dict]] = {}
        for l in approved:
            by_file.setdefault(l["src"]["pak_path"], []).append(l)
        pairs = []
        for i, (pak_path, group) in enumerate(by_file.items(), 1):
            progress(f"Подготовка звука: {i}/{len(by_file)}", i, len(by_file))
            src = group[0]["src"]
            sr, channels = src["sr"], src.get("channels", 1)
            if any(l["src"].get("segment") for l in group) or channels > 1:
                # Катсцена или многоканальный файл: берём оригинал целиком и заменяем одобренные отрезки,
                # раскладывая русскую реплику по каналам с той же громкостью, что была у оригинала
                try:
                    pcm, osr = _decode_one(files.read(pak_path), work / f"{i:06d}.src")
                except Exception as e:
                    raise RuntimeError(f"Не удалось декодировать оригинал {pak_path}: {e}")
                for l in group:
                    seg = l["src"].get("segment") or [0, len(pcm) / osr]
                    a, b = int(seg[0] * osr), int((seg[1] if seg[1] else len(pcm) / osr) * osr)
                    gen, gsr = audio.read(project.dir / l["gen"])
                    gen = audio.resample(gen, gsr, osr)
                    n = b - a
                    if len(gen) > n:
                        gen = gen[:n]
                        fade = min(n, int(0.02 * osr))
                        gen[-fade:] *= np.linspace(1, 0, fade)
                    pcm[a:b] = _spread(pcm[a:b], gen)
                out, sr = pcm, osr
            else:
                gen, gsr = audio.read(project.dir / group[0]["gen"])
                out = audio.resample(gen, gsr, sr)
            wav = work / f"{i:06d}.wav"
            audio.write(wav, out, sr, "PCM_16")
            pairs.append((wav, stage / pak_path))
        progress("Конвертация в формат Wwise…")
        audio.encode_wem_batch(pairs)
        progress("Сборка pak-мода…")
        mod = self._mod_pak(files)
        mod.parent.mkdir(parents=True, exist_ok=True)
        self._remove_mod(mod)
        version = {8: "V8B", 9: "V9", 10: "V10"}.get(files.max_version, "V11")
        r = subprocess.run([str(repak), "pack", "--version", version, "--mount-point", "../../../", "-q",
                            str(stage), str(mod)], capture_output=True, text=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode != 0 or not mod.exists():
            raise RuntimeError("repak не собрал pak-мод: " + (r.stderr or r.stdout).strip()[-800:])
        shutil.rmtree(work, ignore_errors=True)
        (project.dir / "installed.json").write_text(json.dumps(
            {"mod": str(mod), "files": sorted(by_file), "lines": len(approved), "time": time.time()},
            ensure_ascii=False, indent=1), "utf-8")
        return len(approved)

    @staticmethod
    def _remove_mod(mod: Path):
        for ext in (".pak", ".utoc", ".ucas", ".sig"):
            try:
                mod.with_suffix(ext).unlink(missing_ok=True)
            except PermissionError:
                raise RuntimeError(f"Файл мода занят ({mod.with_suffix(ext).name}) — закройте игру и повторите")

    def uninstall(self, project, game: dict, progress):
        files = _files_for(game)
        if files.paks_dir:
            progress("Удаление pak-мода…")
            self._remove_mod(self._mod_pak(files))
        shutil.rmtree(project.dir / "stage", ignore_errors=True)
        (project.dir / "installed.json").unlink(missing_ok=True)

    def is_installed(self, project, game: dict) -> bool:
        paks_dir = UnrealFiles.find_paks_dir(Path(game["path"]))
        return bool(paks_dir) and (paks_dir / "~mods" / f"{MOD_NAME}.pak").exists()


# --- декодирование .wem через vgmstream ---

def _decode_batch(td: Path, items: list[tuple[str, bytes]]) -> dict[str, Path]:
    """[(имя, данные wem)] → {имя: путь к wav}. Один запуск vgmstream на всю пачку, отказавшие — по одному."""
    vgm = tools.vgmstream()
    if not vgm:
        raise RuntimeError("Нет tools\\vgmstream\\vgmstream-cli.exe")
    paths = []
    for name, data in items:
        p = td / f"{name}.wem"
        p.write_bytes(data)
        paths.append(p)
    subprocess.run([str(vgm), "-o", str(td / "?f.wav"), *map(str, paths)], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    out = {}
    for name, _ in items:
        wav = td / f"{name}.wav"
        if not wav.exists():
            subprocess.run([str(vgm), "-o", str(wav), str(td / f"{name}.wem")], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        if wav.exists() and wav.stat().st_size > 44:
            out[name] = wav
        (td / f"{name}.wem").unlink(missing_ok=True)
    return out


def _spread(orig: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """Заменить отрезок (n × каналы) репликой gen (моно), сохранив распределение громкости по каналам."""
    n, channels = orig.shape
    region = np.zeros_like(orig)
    mono = np.zeros(n, dtype=np.float32)
    mono[:len(gen)] = gen[:n]
    if channels == 1:
        region[:, 0] = mono
        return region
    mix_rms = np.sqrt(np.mean(orig.mean(axis=1) ** 2)) + 1e-6
    gains = np.clip(np.sqrt(np.mean(orig ** 2, axis=0)) / mix_rms, 0, 1.5)
    if gains.max() < 1e-3:
        gains[0] = 1.0
    region[:] = mono[:, None] * gains[None, :]
    return region


def _decode_one(data: bytes, tmp: Path) -> tuple[np.ndarray, int]:
    tmp = tmp.with_suffix(".wem")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    wav = tmp.with_suffix(".wav")
    r = subprocess.run([str(tools.vgmstream()), "-o", str(wav), str(tmp)], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode != 0 or not wav.exists():
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
    pcm, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    tmp.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    return pcm, sr
