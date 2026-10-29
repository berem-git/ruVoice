"""Любая игра со звуком FMOD в папке игры: банки FMOD Studio (.bank — RIFF «FEV» с FSB5 внутри) и .fsb.

Какие банки — озвучка: сначала по именам (VO, voice, dialog, speech… и языковые суффиксы — берём английские,
если банк есть в нескольких языках); если подсказок нет — по звуку: из каждого банка расшифровываются
несколько потоков, классификатор звуков (AudioSet) решает, речь ли это. Зашифрованные .fsb: ключ (свой у
каждой игры) восстанавливается по самому банку — fromsoft_fsb.recover_key.
Реплика — поток FSB5 (id «<банк>:<номер>», имя потока, если оно есть). Текст — общий шаг подготовки
(субтитры из файлов игры по распознанной речи, иначе перевод). Установка — банк пересобирается
(Vorbis под таблицу FMOD, иначе весь банк в PCM16) и при необходимости шифруется обратно; оригинал — в
резервной копии проекта."""
from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf

from .. import tools
from . import fromsoft_fsb as X
from .fromsoft import FROMSOFT_DIR, VORBIS_QUALITIES

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SKIP_DIRS = {"_commonredist", "redist", "directx", "__pycache__"}
VOICE_TOK = {"vo", "voice", "voices", "voiceover", "dialog", "dialogue", "dialogues", "speech", "vox", "lines",
             "talk", "barks", "narration", "narrator", "story", "conversation", "conversations"}
LANG_TOK = {"en": "en", "enu": "en", "eng": "en", "english": "en", "enus": "en", "us": "en", "de": "de", "ger": "de", "german": "de",
            "fr": "fr", "fra": "fr", "french": "fr", "es": "es", "spa": "es", "spanish": "es", "it": "it", "ita": "it",
            "italian": "it", "ru": "ru", "rus": "ru", "russian": "ru", "jp": "ja", "ja": "ja", "jpn": "ja",
            "japanese": "ja", "ko": "ko", "kor": "ko", "zh": "zh", "chs": "zh", "cht": "zh", "pt": "pt", "br": "pt",
            "pl": "pl", "mx": "es", "cn": "zh", "tw": "zh", "kr": "ko", "fre": "fr", "deu": "de", "esp": "es"}
MIN_SEC, MAX_SEC = 0.4, 90.0


# --- банки FMOD Studio (RIFF «FEV»): где лежит FSB5 и как пересобрать с новым звуком ---

def _bank_layout(raw: bytes) -> dict:
    """Где в банке FSB5 и чанки SND/SNDH (смещения), чтобы заменить звук и поправить размеры."""
    if raw[:4] != b"RIFF":
        raise ValueError("не RIFF-банк FMOD")
    snd = raw.find(b"SND ", 12)
    fsb = raw.find(b"FSB5", snd)
    sndh = raw.find(b"SNDH", 12, snd if snd > 0 else len(raw))
    if snd < 0 or fsb < 0:
        raise ValueError("в банке нет звука (SND/FSB5)")
    return {"snd": snd, "fsb": fsb, "sndh": sndh}


def rebuild_bank(raw: bytes, fsb_bytes: bytes) -> bytes:
    """Банк с новым FSB5: те же чанки до звука, новые размеры RIFF, SND и SNDH."""
    lay = _bank_layout(raw)
    snd, fsb = lay["snd"], lay["fsb"]
    old_snd_size = struct.unpack_from("<I", raw, snd + 4)[0]
    after = raw[snd + 8 + old_snd_size + (old_snd_size & 1):]
    pad = fsb - (snd + 8)
    out = bytearray(raw[:fsb]) + fsb_bytes
    struct.pack_into("<I", out, snd + 4, pad + len(fsb_bytes))
    if len(out) % 2:
        out += b"\x00"
    out += after
    struct.pack_into("<I", out, 4, len(out) - 8)
    if lay["sndh"] > 0:
        size = struct.unpack_from("<I", raw, lay["sndh"] + 4)[0]
        vals = list(struct.unpack_from("<" + "I" * (size // 4), raw, lay["sndh"] + 8))
        for k in range(len(vals) - 1):
            if vals[k] == fsb:  # (смещение FSB5, размер FSB5)
                vals[k + 1] = len(fsb_bytes)
        struct.pack_into("<" + "I" * len(vals), out, lay["sndh"] + 8, *vals)
    return bytes(out)


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _fsb_offset(raw: bytes) -> int | None:
    """Где в файле FSB5: .fsb — с начала, .bank — внутри чанка SND (RIFF FEV)."""
    if raw[:4] == b"FSB5":
        return 0
    if raw[:4] != b"RIFF" and raw[:3] != b"FSB":
        return None
    if raw[:4] == b"RIFF" and raw[8:12] == b"FEV ":
        try:
            return _bank_layout(raw)["fsb"]
        except ValueError:
            return None
    return None


_KEYS: dict[str, bytes | None] = {}


def _open(path: Path, raw: bytes | None = None) -> tuple[bytes, int, bytes | None] | None:
    """(данные, смещение FSB5, ключ шифрования) или None. Зашифрованный .fsb — ключ восстанавливается по самому
    банку (fromsoft_fsb.recover_key); у игры он обычно один на папку — запоминаем."""
    raw = raw if raw is not None else path.read_bytes()
    off = _fsb_offset(raw)
    if off is not None:
        return raw, off, None
    if path.suffix.lower() != ".fsb":
        return None
    folder = str(path.parent).lower()
    key = _KEYS.get(folder)
    if not key or X.fsb_decrypt(raw[:4], key)[:4] != b"FSB5":
        key = X.recover_key(raw)
        if key:
            _KEYS[folder] = key
    if not key:
        return None
    return X.fsb_decrypt(raw, key), 0, key


def _banks(root: Path) -> list[Path]:
    out = []
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x.lower() not in SKIP_DIRS]
        for n in names:
            if n.lower().endswith((".bank", ".fsb")) and not n.lower().endswith(".strings.bank"):
                p = Path(d) / n
                try:
                    if p.stat().st_size > 2048:
                        out.append(p)
                except OSError:
                    pass
    return out


def _lang(p: Path, root: Path) -> str | None:
    toks = _tokens(p.relative_to(root).as_posix())
    return next((LANG_TOK[t] for t in reversed(toks) if t in LANG_TOK), None)


class FmodLooseAdapter:
    id = "fmod"
    name = "FMOD в папке игры (любой движок)"

    @staticmethod
    def detect(game: dict) -> bool:
        root = Path(game["path"])
        if not root.is_dir():
            return False
        tried = 0
        for p in _banks(root)[:600]:
            try:
                with open(p, "rb") as f:
                    head = f.read(12)
            except OSError:
                continue
            if head[:4] == b"FSB5" or (head[:4] == b"RIFF" and head[8:12] == b"FEV "):
                return True
            if p.suffix.lower() == ".fsb" and tried < 3:  # зашифрованный банк — пробуем восстановить ключ
                tried += 1
                if _open(p):
                    return True
        return False

    def describe(self, game: dict) -> dict:
        return {"adapter": self.id, "name": self.name, "supports_official_ru": False,
                "notes": ["Озвучка — потоки FSB5 в банках FMOD из папки игры; голосовые банки определяются по именам, "
                          "а без подсказок — по звуку (речь или нет). Ключ зашифрованных .fsb программа восстанавливает сама.",
                          "Русский текст — из субтитров игры (по распознанной речи), если их нет — перевод нейросетью.",
                          "Установка пересобирает банки, оригиналы хранятся в проекте и возвращаются кнопкой «Удалить озвучку»."]}

    # --- какие банки — озвучка ---

    def _voice_banks(self, root: Path, progress) -> list[Path]:
        banks = []
        for p in _banks(root):
            try:
                with open(p, "rb") as f:
                    head = f.read(12)
            except OSError:
                continue
            if head[:4] == b"FSB5" or (head[:4] == b"RIFF" and head[8:12] == b"FEV ") or \
                    (p.suffix.lower() == ".fsb" and _open(p)):
                banks.append(p)
        langs = {p: _lang(p, root) for p in banks}
        named = [p for p in banks if set(_tokens(p.stem)) & VOICE_TOK]
        localized = [p for p in banks if langs[p]]
        pick = [p for p in (named or localized) if langs[p] in (None, "en")] if (named or localized) else []
        if pick:
            return pick
        progress("Голосовых банков по именам не видно — слушаю, где речь…")
        return [p for p in banks if self._sounds_like_speech(p)]

    def _sounds_like_speech(self, bank: Path) -> bool:
        from .. import judge
        try:
            opened = _open(bank)
            if not opened:
                return False
            data, off, key = opened
            fsb = X.FSB5.parse(data[off:])
        except Exception:
            return False
        n = len(fsb.samples)
        if n == 0:
            return False
        idx = sorted({int(i * n / 6) for i in range(6)})
        with tempfile.TemporaryDirectory(prefix="ruvoice_fm_") as td:
            src = bank
            if key:
                src = Path(td) / bank.name
                src.write_bytes(data)
            wavs = []
            for i in idx:
                out = Path(td) / f"{i}.wav"
                subprocess.run([str(tools.vgmstream()), "-s", str(i + 1), "-o", str(out), str(src)],
                               capture_output=True, creationflags=NO_WINDOW)
                if out.exists():
                    wavs.append(str(out))
            if not wavs:
                return False
            scores = judge.speech_scores(wavs)
        return sum(1 for s, _ in scores if s >= 0.5) >= max(2, int(0.6 * len(wavs)))

    # --- извлечение ---

    def extract(self, project, game: dict, progress):
        root = Path(game["path"])
        t0 = time.time()
        progress("Поиск банков FMOD…")
        banks = self._voice_banks(root, progress)
        if not banks:
            raise RuntimeError("Банков FMOD с озвучкой не найдено (или они зашифрованы)")
        project.reset()
        project.orig_dir.mkdir(parents=True, exist_ok=True)
        lines, chars = [], {}
        with tempfile.TemporaryDirectory(prefix="ruvoice_fm_") as td:
            for bi, bank in enumerate(banks, 1):
                rel = bank.relative_to(root).as_posix()
                progress(f"Банк {bank.name} ({bi}/{len(banks)})…", bi - 1, len(banks))
                opened = _open(bank)
                if not opened:
                    continue
                data, off, key = opened
                try:
                    fsb = X.FSB5.parse(data[off:])
                except Exception as e:
                    progress(f"Банк {bank.name} пропущен: {e}")
                    continue
                out_dir = Path(td) / f"b{bi}"
                out_dir.mkdir()
                if key:  # vgmstream получает расшифрованную копию
                    bank = out_dir / bank.name
                    bank.write_bytes(data)
                subprocess.run([str(tools.vgmstream()), "-S", "0", "-o", str(out_dir / "?s.wav"), str(bank)],
                               capture_output=True, creationflags=NO_WINDOW)
                group = re.sub(r"[_\-. ]*(en|eng|english|en[-_]?us)$", "", bank.stem, flags=re.I) or bank.stem
                for i, s in enumerate(fsb.samples):
                    wav = out_dir / f"{i + 1}.wav"
                    if not wav.exists():
                        continue
                    try:
                        d = sf.info(str(wav)).duration
                    except Exception:
                        continue
                    if not MIN_SEC <= d <= MAX_SEC:
                        continue
                    line_id = f"{rel}:{i}"
                    orig = project.orig_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "__", line_id) + ".wav")
                    shutil.move(str(wav), str(orig))
                    ch = chars.setdefault(group, {"id": group, "name": group, "best": None, "best_d": 0})
                    if 3.5 <= d <= 12 and d > ch["best_d"]:
                        ch["best"], ch["best_d"] = line_id, d
                    lines.append({"id": line_id, "char": group, "orig": str(orig.relative_to(project.dir)),
                                  "duration": round(d, 3), "en": None,
                                  "src": {"bank": rel, "index": i, "name": s.name, "sr": s.rate,
                                          "channels": s.channels, "samples": s.num_samples}})
        project.add_lines(lines)
        project.set_characters([{"id": c["id"], "name": c["name"], "ref_line": c["best"]} for c in chars.values()])
        project.set("fmod_loose", {"banks": len(banks), "voices": len(lines), "seconds": round(time.time() - t0)})
        return len(lines)

    # --- установка ---

    def _original(self, project, root: Path, rel: str) -> bytes:
        backup = project.backup_dir / rel
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, backup)
        return backup.read_bytes()

    def _build(self, project, rel: str, raw: bytes, items: list[dict], work: Path, known: set[int]) -> bytes:
        opened = _open(Path(rel), raw)
        if not opened:
            raise RuntimeError(f"{rel}: банк не читается (ключ шифрования не найден)")
        data, off, key = opened
        fsb = X.FSB5.parse(data[off:])
        for l in items:
            if fsb.samples[l["src"]["index"]].num_samples != l["src"]["samples"]:
                raise RuntimeError(f"{l['src']['bank']}: банк изменился после обновления игры — извлеките заново")
        done = False
        if fsb.codec == X.CODEC_VORBIS and X.ffmpeg_has_libvorbis():
            enc = []
            for l in items:
                s = fsb.samples[l["src"]["index"]]
                got = None
                for rate in dict.fromkeys([s.rate, 48000, 44100]):
                    for q in VORBIS_QUALITIES:
                        data, crc, ns = X.encode_vorbis_fsb(project.dir / l["gen"], rate, s.channels, q, work)
                        if not known or crc in known:
                            got = (s, data, ns, crc, rate)
                            break
                    if got:
                        break
                if not got:
                    enc = None
                    break
                enc.append(got)
            if enc is not None:
                for s, data, ns, crc, rate in enc:
                    s.data, s.num_samples, s.extras, s.rate = data, ns, [X.vorbis_extra(crc)], rate
                done = True
        if not done:  # другой кодек или неизвестный setup Vorbis: весь банк в PCM16 (FSB5 не смешивает кодеки)
            tmp = work / "orig.bank"
            tmp.write_bytes(data)
            pcm_dir = work / "pcm"
            shutil.rmtree(pcm_dir, ignore_errors=True)
            pcm_dir.mkdir()
            subprocess.run([str(tools.vgmstream()), "-S", "0", "-o", str(pcm_dir / "?s.wav"), str(tmp)],
                           capture_output=True, creationflags=NO_WINDOW)
            for i, s in enumerate(fsb.samples):
                s.data, s.num_samples = X.pcm16_data(pcm_dir / f"{i + 1}.wav", s.rate, s.channels)
                s.extras = []
            for l in items:
                s = fsb.samples[l["src"]["index"]]
                s.data, s.num_samples = X.pcm16_data(project.dir / l["gen"], s.rate, s.channels)
            fsb.codec = X.CODEC_PCM16
        built = fsb.build()
        out = built if off == 0 else rebuild_bank(data, built)
        return X.fsb_encrypt(out, key) if key else out

    def install(self, project, game: dict, progress) -> int:
        root = Path(game["path"])
        by_bank: dict[str, list[dict]] = {}
        for l in project.lines(status="approved"):
            by_bank.setdefault(l["src"]["bank"], []).append(l)
        known = X.known_setup_ids(FROMSOFT_DIR / "fsb_vorbis_setup_ids.txt")
        done = 0
        with tempfile.TemporaryDirectory(prefix="ruvoice_fm_") as td:
            for bi, (rel, items) in enumerate(sorted(by_bank.items()), 1):
                progress(f"Сборка банка {Path(rel).name} ({bi}/{len(by_bank)}, реплик: {len(items)})…", bi - 1, len(by_bank))
                work = Path(td) / f"b{bi}"
                work.mkdir()
                data = self._build(project, rel, self._original(project, root, rel), items, work, known)
                target = root / rel
                tmp = target.with_name(target.name + ".ruvoice.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, target)
                done += len(items)
        project.set("installed_count", done)
        return done

    def uninstall(self, project, game: dict, progress):
        root = Path(game["path"])
        if project.backup_dir.is_dir():
            for b in [p for p in project.backup_dir.rglob("*") if p.is_file()]:
                progress(f"Восстановление {b.name}…")
                target = root / b.relative_to(project.backup_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(b, target)
            shutil.rmtree(project.backup_dir, ignore_errors=True)
        project.set("installed_count", 0)

    def is_installed(self, project, game: dict) -> bool:
        return project.backup_dir.is_dir() and any(project.backup_dir.rglob("*.*"))
