"""Игры FromSoftware со звуком FMOD (Sekiro, Dark Souls III и другие на том же движке) — без таблиц под игры.

Всё нужное программа находит сама в файлах игры:
  • ключи архивов Data*.bhd: RSA-ключи (PEM) лежат открыто в exe игры — каждый .bhd пробуем всеми найденными
    ключами (с 32- и 64-битными хешами путей); какой открыл, тот и его. Для озвучки достаточно архива с
    субтитрами — остальные могут и не открыться;
  • ключ звуковых банков: восстанавливается по самому банку (fromsoft_fsb.recover_key);
  • субтитры: в открытых архивах — по содержимому: DCX → BND4 с FMG разговоров («会話»/TalkMsg — общее имя
    в движке); язык — по папке в пути записи (engUS, rusRU…);
  • голосовые банки: sound\\*.fsb, у которых имена потоков «v<число>» — это id строки FMG, так реплика
    получает текст точно; английские — по суффиксу файла (_enu, _eng, _en).
Группа реплик — по talk-id (id // 1000): это разговор, а не человек; кто говорит на самом деле, определяет
общий шаг подготовки (speakers.py) по голосу.
Установка: банк пересобирается (нетронутые потоки копируются как есть, новые — Vorbis, чьи setup-пакеты есть
в таблице FMOD, иначе весь банк в PCM16) и заменяет оригинал в sound\\; оригинал — в резервной копии проекта,
удаление возвращает его. Игра запускается как обычно."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf

from .. import tools
from . import fromsoft_formats as F
from . import fromsoft_fsb as X

FROMSOFT_DIR = tools.TOOLS_DIR / "fromsoft"   # таблица setup-пакетов Vorbis FMOD (общая для всех игр на FMOD)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
TALK_FMG = ("会話", "TalkMsg", "ムービー字幕", "MovieSubtitle")  # реплики и субтитры роликов (имена движка)
VORBIS_QUALITIES = (4.0, 4.6, 3.7, 5.0, 3.0, 6.0, 2.0, 7.0)
STREAM_RE = re.compile(r"^v(\d{6,10})$")
EN_SUFFIX = {"enu", "eng", "en", "english", "enus"}
LANG_DIRS = {"engus": "en", "enus": "en", "english": "en", "rusru": "ru", "russian": "ru", "ruru": "ru"}
PEM_RE = re.compile(rb"-----BEGIN RSA PUBLIC KEY-----[A-Za-z0-9+/=\r\n]+-----END RSA PUBLIC KEY-----")
_cache: dict[str, object] = {}


def _root(game: dict) -> Path | None:
    """Папка с архивами Data*.bhd/.bdt и sound\\*.fsb (у DS3 это <игра>\\Game)."""
    base = Path(game["path"])
    for cand in (base, base / "Game"):
        if any(cand.glob("*.bhd")) and (cand / "sound").is_dir() and any((cand / "sound").glob("*.fsb")):
            return cand
    return None


def _pems(root: Path) -> list[str]:
    """RSA-ключи архивов из exe игры (лежат открытым текстом)."""
    key = f"pem:{root}"
    if key not in _cache:
        found: list[str] = []
        for exe in root.glob("*.exe"):
            try:
                for m in PEM_RE.findall(exe.read_bytes()):
                    s = m.decode("ascii")
                    if s not in found:
                        found.append(s)
            except OSError:
                continue
        _cache[key] = found
    return _cache[key]


def _archives(root: Path) -> dict[str, F.Archive]:
    """Архивы, которые удалось открыть ключами из exe (или незашифрованные)."""
    out = {}
    for bhd in sorted(root.glob("*.bhd")):
        if not bhd.with_suffix(".bdt").exists():
            continue
        size = bhd.with_suffix(".bdt").stat().st_size
        for pem in [None] + _pems(root):
            got = None
            for wide in (False, True):
                try:
                    a = F.Archive(bhd, pem, wide)
                except Exception:
                    continue
                inside = sum(1 for f in a.files if 0 <= f.offset and f.offset + f.padded_size <= size)
                if a.files and inside >= 0.98 * len(a.files):
                    got = a
                    break
                a.close()
            if got:
                out[bhd.stem] = got
                break
    return out


def _fsb_key(root: Path) -> bytes | None:
    """Ключ звуковых банков игры — по первому зашифрованному банку (кэш на время работы)."""
    key = f"fsb:{root}"
    if key not in _cache:
        k = None
        for bank in sorted((root / "sound").glob("*.fsb")):
            head = bank.read_bytes()
            if X.is_plain_fsb(head):
                continue
            k = X.recover_key(head)
            if k:
                break
        _cache[key] = k
    return _cache[key]


def _lang_of(name: str) -> str | None:
    for part in re.split(r"[\\/]", name.lower()):
        if part in LANG_DIRS:
            return LANG_DIRS[part]
    return None


class FromsoftAdapter:
    id = "fromsoft"
    name = "FromSoftware (архивы Data*.bhd, звук FMOD)"

    @staticmethod
    def detect(game: dict) -> bool:
        root = _root(game)
        return bool(root and (_pems(root) or any(b.read_bytes()[:4] == b"BHD5" for b in root.glob("*.bhd"))))

    def describe(self, game: dict) -> dict:
        root = _root(game)
        if not root:
            return {"adapter": self.id, "name": self.name, "supports_official_ru": False}
        # Description must not decrypt or decode sound banks. Extraction validates their contents.
        banks = sorted((root / "sound").glob("*.fsb"))
        notes = [f"Звуковых банков: {len(banks)} ({', '.join(b.name for b in banks[:6])}{'…' if len(banks) > 6 else ''})",
                 "Ключи архивов и звуковых банков программа находит сама в файлах игры; субтитры — по содержимому архивов.",
                 "Установка заменяет банки в папке sound с резервной копией; «Удалить озвучку» возвращает оригиналы. "
                 "Игра запускается как обычно; в её настройках оставьте английскую озвучку и русский текст."]
        if not any(root.glob("oo2core_*_win64.dll")):
            notes.append("В папке игры нет oo2core_*_win64.dll — субтитры (DCX/Oodle) прочитать не получится.")
        return {"adapter": self.id, "name": self.name, "supports_official_ru": False, "notes": notes}

    # --- служебное ---

    def _voice_banks(self, root: Path) -> list[Path]:
        """Банки с репликами (потоки v<id>), английские — если язык обозначен суффиксом."""
        key = _fsb_key(root)
        found = []
        for bank in sorted((root / "sound").glob("*.fsb")):
            try:
                fsb = X.FSB5.parse(X.fsb_open(bank.read_bytes(), key))
            except Exception:
                continue
            if any(STREAM_RE.match(s.name) for s in fsb.samples):
                found.append(bank)
        english = [b for b in found if b.stem.lower().rsplit("_", 1)[-1] in EN_SUFFIX]
        return english or found

    def _texts(self, root: Path, progress) -> tuple[dict[int, str], dict[int, str]]:
        """Субтитры реплик {id: строка} на английском и русском — поиском по содержимому архивов."""
        en: dict[int, str] = {}
        ru: dict[int, str] = {}
        arcs = _archives(root)
        if not arcs:
            progress("Архивы с субтитрами не открылись (ключи в exe не подошли) — текст распознаем по речи")
            return en, ru
        for name, a in arcs.items():
            files = [f for f in a.files if f.padded_size <= 64 << 20]
            for i, f in enumerate(files, 1):
                if i % 300 == 0:
                    progress(f"Поиск субтитров в {name}: {i}/{len(files)}")
                try:
                    raw = a.read(f)
                    if F.is_dcx(raw):
                        raw = F.dcx_decompress(raw, root)
                    if raw[:4] != b"BND4":
                        continue
                    for e in F.bnd4_read(raw, root):
                        ename = e.name or ""
                        if not ename.lower().endswith(".fmg") or not any(t in Path(ename).stem for t in TALK_FMG):
                            continue
                        lang = _lang_of(ename)
                        target = en if lang == "en" else ru if lang == "ru" else None
                        if target is None:
                            continue
                        for k, v in F.fmg_read(e.data).items():
                            v = (v or "").replace("\r", "").strip()
                            if v and v not in ("-", "*") and k not in target:
                                target[k] = re.sub(r"\s*\n\s*", " ", v)
                except F.OodleMissing:
                    progress("Нет oo2core_*_win64.dll в папке игры — субтитры не прочитать")
                    return en, ru
                except Exception:
                    continue
            a.close()
        return en, ru

    # --- извлечение ---

    def extract(self, project, game: dict, progress):
        root = _root(game)
        if not root:
            raise RuntimeError("Не найдены архивы Data*.bhd и папка sound с банками .fsb")
        t0 = time.time()
        progress("Ключ звуковых банков…")
        key = _fsb_key(root)
        banks = self._voice_banks(root)
        if not banks:
            raise RuntimeError("В папке sound нет банков с репликами (или их не удалось расшифровать)")
        progress("Субтитры: открываю архивы ключами из exe игры…")
        en, ru = self._texts(root, progress)
        vg = tools.vgmstream()
        if not vg:
            raise RuntimeError("vgmstream не найден в tools\\vgmstream")
        project.reset()
        project.orig_dir.mkdir(parents=True, exist_ok=True)
        lines, chars = [], {}
        parsed = []
        for bank in banks:
            try:
                parsed.append((bank, X.FSB5.parse(X.fsb_open(bank.read_bytes(), key))))
            except Exception as e:
                progress(f"Банк {bank.name} пропущен: {e}")
        total, done = sum(len(f.samples) for _, f in parsed), 0
        with tempfile.TemporaryDirectory(prefix="ruvoice_fs_") as td:
            for bank, fsb in parsed:
                out_dir = Path(td) / bank.stem
                out_dir.mkdir()
                # vgmstream сам расшифровывает банки FromSoftware (знает их ключи); иначе — расшифрованная копия
                src = bank
                if key:
                    src = out_dir / bank.name
                    src.write_bytes(X.fsb_decrypt(bank.read_bytes(), key))
                subprocess.run([str(vg), "-S", "0", "-o", str(out_dir / "?s.wav"), str(src)],
                               capture_output=True, creationflags=NO_WINDOW)
                for i, s in enumerate(fsb.samples):
                    done += 1
                    if done % 100 == 0 or done == total:
                        progress(f"Извлечение реплик {bank.name}: {done}/{total}", done, total)
                    wav = out_dir / f"{i + 1}.wav"
                    if not wav.exists():
                        continue
                    m = STREAM_RE.match(s.name)
                    tid = int(m.group(1)) if m else None
                    line_id = f"{bank.stem}:{s.name}"
                    orig = project.orig_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "__", line_id) + ".wav")
                    shutil.move(str(wav), str(orig))
                    try:
                        d = round(sf.info(str(orig)).duration, 3)
                    except Exception:
                        continue
                    if d < 0.3:
                        orig.unlink(missing_ok=True)
                        continue
                    spk = f"Группа {tid // 1000}" if tid is not None else bank.stem
                    text = en.get(tid) if tid is not None else None
                    ch = chars.setdefault(spk, {"id": spk, "name": spk, "best": None, "best_d": 0})
                    if text and 3.5 <= d <= 12 and d > ch["best_d"]:
                        ch["best"], ch["best_d"] = line_id, d
                    lines.append({
                        "id": line_id, "char": spk, "orig": str(orig.relative_to(project.dir)), "duration": d,
                        "en": text, "ru_official": ru.get(tid) if tid is not None else None,
                        "src": {"bank": bank.name, "index": i, "name": s.name, "talk_id": tid,
                                "sr": s.rate, "channels": s.channels, "samples": s.num_samples}})
        project.add_lines(lines)
        project.set_characters([{"id": c["id"], "name": c["name"], "ref_line": c["best"]} for c in chars.values()])
        project.set("fromsoft_extract", {"banks": len(parsed), "texts_en": len(en), "texts_ru": len(ru),
                                         "encrypted": bool(key), "seconds": round(time.time() - t0, 1)})
        return len(lines)

    # --- установка ---

    def _original_bank(self, project, root: Path, bank: str) -> Path:
        """Нетронутый банк: резервная копия проекта, иначе файл игры (и тогда делаем копию)."""
        backup = project.backup_dir / "sound" / bank
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / "sound" / bank, backup)
        return backup

    def build_bank(self, project, root: Path, bank: str, approved: list[dict], work: Path, known_ids: set[int]) -> bytes:
        raw = self._original_bank(project, root, bank).read_bytes()
        key = _fsb_key(root)
        encrypted = not X.is_plain_fsb(raw)
        fsb = X.FSB5.parse(X.fsb_open(raw, key))
        for l in approved:
            s = fsb.samples[l["src"]["index"]]
            if s.name != l["src"]["name"]:
                raise RuntimeError(f"{bank}: поток #{l['src']['index']} теперь {s.name}, ожидался {l['src']['name']}")
        use_vorbis = fsb.codec == X.CODEC_VORBIS and X.ffmpeg_has_libvorbis()
        encoded = []
        if use_vorbis:
            # всё в Vorbis; если для какого-то потока setup-пакет libvorbis неизвестен FMOD — весь банк в PCM16
            for l in approved:
                s = fsb.samples[l["src"]["index"]]
                got = None
                for rate in dict.fromkeys([s.rate, 48000, 44100]):
                    for q in VORBIS_QUALITIES:
                        data, crc, ns = X.encode_vorbis_fsb(project.dir / l["gen"], rate, s.channels, q, work)
                        if not known_ids or crc in known_ids:
                            got = (s, data, ns, crc, rate)
                            break
                    if got:
                        break
                if not got:
                    use_vorbis = False
                    break
                encoded.append(got)
        if use_vorbis:
            for s, data, ns, crc, rate in encoded:
                s.data, s.num_samples, s.extras, s.rate = data, ns, [X.vorbis_extra(crc)], rate
        else:
            if fsb.codec != X.CODEC_PCM16:
                self._convert_bank_to_pcm16(fsb, raw, key, work)
            for l in approved:
                s = fsb.samples[l["src"]["index"]]
                s.data, s.num_samples = X.pcm16_data(project.dir / l["gen"], s.rate, s.channels)
                s.extras = []
        built = fsb.build()
        return X.fsb_encrypt(built, key) if encrypted and key else built

    def _convert_bank_to_pcm16(self, fsb: X.FSB5, raw: bytes, key: bytes | None, work: Path):
        """Запасной путь: все потоки банка → PCM16 (FSB5 не допускает смешивать кодеки)."""
        out_dir = work / "pcm"
        out_dir.mkdir(parents=True, exist_ok=True)
        src = work / "orig.fsb"
        src.write_bytes(X.fsb_open(raw, key))
        subprocess.run([str(tools.vgmstream()), "-S", "0", "-o", str(out_dir / "?s.wav"), str(src)], check=True,
                       capture_output=True, creationflags=NO_WINDOW)
        for i, s in enumerate(fsb.samples):
            s.data, s.num_samples = X.pcm16_data(out_dir / f"{i + 1}.wav", s.rate, s.channels)
            s.extras = []
        fsb.codec = X.CODEC_PCM16

    def install(self, project, game: dict, progress) -> int:
        root = _root(game)
        if not root:
            raise RuntimeError("Папка игры не найдена")
        by_bank: dict[str, list[dict]] = {}
        for l in project.lines(status="approved"):
            by_bank.setdefault(l["src"]["bank"], []).append(l)
        known_ids = X.known_setup_ids(FROMSOFT_DIR / "fsb_vorbis_setup_ids.txt")
        done = 0
        with tempfile.TemporaryDirectory(prefix="ruvoice_fs_") as td:
            for bi, (bank, items) in enumerate(sorted(by_bank.items()), 1):
                progress(f"Сборка банка {bank} ({bi}/{len(by_bank)}, реплик: {len(items)})…", bi - 1, len(by_bank))
                work = Path(td) / f"b{bi}"
                work.mkdir()
                data = self.build_bank(project, root, bank, items, work, known_ids)
                tmp = root / "sound" / (bank + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, root / "sound" / bank)
                done += len(items)
        project.set("fromsoft_installed", {"mode": "direct", "banks": sorted(by_bank), "when": time.time()})
        return done

    def uninstall(self, project, game: dict, progress):
        root = _root(game)
        if not root:
            return
        progress("Удаление озвучки…")
        # следы прежнего способа установки (загрузчик me3), если остались
        shutil.rmtree(root / "ruvoice", ignore_errors=True)
        for b in root.glob("RuVoice - запуск с озвучкой*.bat"):
            b.unlink(missing_ok=True)
        backups = project.backup_dir / "sound"
        if backups.is_dir():
            for b in backups.iterdir():
                progress(f"Восстановление {b.name}…")
                shutil.copy2(b, root / "sound" / b.name)
        project.set("fromsoft_installed", None)

    def is_installed(self, project, game: dict) -> bool:
        state = project.get("fromsoft_installed") or {}
        return bool(state.get("banks")) or (_root(game) and (_root(game) / "ruvoice" / "mod").is_dir())
