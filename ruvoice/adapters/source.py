"""Движок Source (любые игры и моды: папка мода с gameinfo.txt).

Реплики берутся из sound/vo внутри *_sound_vo_english_dir.vpk (или из папки sound/vo), текст — из
resource/closecaption_<язык>.txt. Если в одной папке несколько игр (как у Half-Life 2 и эпизодов), мод
выбирается по названию игры из его gameinfo.txt. Говорящий — папка vo/npc/<кто> или приставка файла;
кто это на самом деле, определяет общий шаг подготовки по голосу. Установка — в <мод>/custom/ruvoice: игра
читает custom раньше своих архивов, поэтому файлы игры не меняются, а удаление озвучки — удаление одной папки.
"""
import io
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import audio
from .vpk import VPK

CAPTION_RE = re.compile(r'^\s*"([^"]+)"\s+"(.*)"\s*$', re.M)
COLOR_RE = re.compile(r"<clr:(\d+),(\d+),(\d+)>")
SOUNDSCRIPT_ENTRY_RE = re.compile(r'^\s*"([^"]+)"\s*\{(.*?)^\s*\}', re.M | re.S)
WAVE_RE = re.compile(r'"wave"\s+"([^"]+)"', re.I)


def _read_text(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8", "replace")


def parse_captions(path: Path) -> dict[str, tuple[str, str | None]]:
    """Ключ → (чистый текст, цвет #rrggbb). Звуковые эффекты ([Взрыв], <sfx>) пропускаются."""
    result = {}
    if not path.exists():
        return result
    for key, raw in CAPTION_RE.findall(_read_text(path.read_bytes())):
        if "<sfx>" in raw.lower():
            continue
        m = COLOR_RE.search(raw)
        color = "#%02x%02x%02x" % tuple(int(x) for x in m.groups()) if m else None
        text = re.sub(r"<[^>]*>", " ", raw).replace("\\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        # [Взрыв], *ликование*, (кашель) — звуковые ремарки, а не речь
        if text and not (text[0] in "[*(" and text[-1] in "]*)"):
            result[key.lower()] = (text, color)
    return result


def _norm_wave(wave: str) -> str:
    wave = wave.lstrip("*#@)(^<>}!?$&").replace("\\", "/").lower()
    return "sound/" + wave if not wave.startswith("sound/") else wave


def speaker_of(wave: str) -> str:
    """sound/vo/npc/<кто>/x.wav → <кто>; sound/vo/<сцена>/al_x.wav → приставка «al»; иначе папка."""
    parts = wave.split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    if len(parts) >= 4 and parts[1] == "vo" and parts[2] == "npc":
        return parts[3]
    m = re.match(r"([a-z]+)\d*_", stem)
    if m:
        return m.group(1)
    return parts[-2] if len(parts) >= 2 else "unknown"


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _mod_title(mod: Path) -> str:
    """Название игры из gameinfo.txt мода (строка game "…")."""
    try:
        m = re.search(r'^\s*game\s+"([^"]+)"', (mod / "gameinfo.txt").read_text("utf-8", "replace"), re.M | re.I)
        return m.group(1) if m else mod.name
    except OSError:
        return mod.name


class SourceAdapter:
    id = "source"
    name = "Source (папка custom)"

    @staticmethod
    def detect(game: dict) -> bool:
        return bool(SourceAdapter.mods(game))

    @staticmethod
    def mods(game: dict) -> list[Path]:
        root = Path(game["path"])
        dirs = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
        # Реплики могут лежать в отдельном *_sound_vo_english_dir.vpk или внутри общего *_pak_dir.vpk,
        # либо отдельными файлами в sound/vo
        mods = [d for d in dirs if (d / "gameinfo.txt").exists()
                and (any(d.glob("*_dir.vpk")) or (d / "sound" / "vo").is_dir())]
        if len(mods) > 1 and game.get("name"):
            # несколько игр в одной папке: берём мод, чьё название из gameinfo.txt совпадает с названием игры
            want = _norm_title(game["name"])
            titles = {m: _norm_title(_mod_title(m)) for m in mods}
            exact = [m for m in mods if titles[m] == want]
            if exact:
                return exact
            # «Half-Life 2: Episode One» → моды с названием «HALF-LIFE 2» (начало названия), без Lost Coast
            prefix = [m for m in mods if titles[m] and want.startswith(titles[m])]
            if prefix:
                return prefix
        return mods

    def describe(self, game: dict) -> dict:
        return {"adapter": self.id, "name": self.name, "mods": [m.name for m in self.mods(game)],
                "supports_official_ru": any((m / "resource" / "closecaption_russian.txt").exists()
                                            for m in self.mods(game))}

    # --- извлечение ---

    def _soundscripts(self, mod: Path, vpks: list[VPK]) -> dict[str, str]:
        """Имя звука (citadel.al_gogordon) → путь к wav."""
        texts = [p.read_bytes() for p in (mod / "scripts").glob("*.txt")] if (mod / "scripts").is_dir() else []
        for v in vpks:
            texts += [v.read(f) for f in v.files("scripts/", ".txt")]
        mapping = {}
        for raw in texts:
            for name, body in SOUNDSCRIPT_ENTRY_RE.findall(_read_text(raw)):
                waves = WAVE_RE.findall(body)
                if waves:
                    mapping.setdefault(name.lower(), _norm_wave(waves[0]))
        return mapping

    def extract(self, project, game: dict, progress):
        lines, chars = [], {}
        seen: dict[str, dict] = {}  # путь реплики → реплика: одна и та же реплика в нескольких модах — одна строка
        for mod in self.mods(game):
            progress(f"Чтение архивов {mod.name}…")
            vpks = [VPK(p) for p in sorted(mod.glob("*_dir.vpk"))]
            vo_vpks = [v for v in vpks if "_sound_vo_english" in v.dir_path.name]
            waves: dict[str, tuple] = {}  # путь wav → (откуда читать)
            for v in vo_vpks + [v for v in vpks if v not in vo_vpks]:
                for f in v.files("sound/vo/", ".wav"):
                    waves.setdefault(f, ("vpk", v))
            loose = mod / "sound" / "vo"
            if loose.is_dir():
                for p in loose.rglob("*.wav"):
                    waves.setdefault("sound/" + p.relative_to(mod / "sound").as_posix().lower(), ("file", p))

            en = parse_captions(mod / "resource" / "closecaption_english.txt")
            ru = parse_captions(mod / "resource" / "closecaption_russian.txt")
            scripts = self._soundscripts(mod, vpks)
            by_name: dict[str, list[str]] = {}
            for w in waves:
                by_name.setdefault(w.rsplit("/", 1)[-1][:-4], []).append(w)

            # Субтитр → wav: звуковой скрипт, затем vo/<префикс>/<имя>.wav, затем уникальное имя файла
            wave_text: dict[str, tuple[str, str]] = {}
            for key in en:
                wave = scripts.get(key)
                if not wave or wave not in waves:
                    prefix, _, name = key.partition(".")
                    guess = f"sound/vo/{prefix}/{name}.wav"
                    if guess in waves:
                        wave = guess
                    elif name and len(by_name.get(name, [])) == 1:
                        wave = by_name[name][0]
                    else:
                        wave = None
                if wave and wave not in wave_text:
                    wave_text[wave] = (key, en[key][0])

            total = len(waves)
            for i, (wave, (kind, where)) in enumerate(sorted(waves.items()), 1):
                if i % 25 == 0 or i == total:
                    progress(f"Извлечение реплик {mod.name}: {i}/{total}", i, total)
                key, text = wave_text.get(wave, (None, None))
                if wave in seen:  # уже взята из другого мода — ставить нужно и сюда
                    src = seen[wave]["src"]
                    if mod.name not in src["mods"]:
                        src["mods"].append(mod.name)
                    if text and not seen[wave]["en"]:
                        seen[wave]["en"] = text
                        seen[wave]["ru_official"] = ru.get(key, (None,))[0]
                    continue
                raw = where.read(wave) if kind == "vpk" else where.read_bytes()
                line_id = f"{mod.name}:{wave[len('sound/'):-4]}"
                orig = project.orig_dir / (re.sub(r"[^a-z0-9_.-]+", "__", line_id) + ".wav")
                try:
                    info = _save_pcm(raw, orig)
                except Exception:
                    continue
                spk = speaker_of(wave)
                color = en.get(key, (None, None))[1] if key else None
                ch = chars.setdefault(spk, {"id": spk, "name": spk, "color": color, "best": None, "best_d": 0})
                ch["color"] = ch["color"] or color
                d = info["duration"]
                if text and 3.5 <= d <= 12 and d > ch["best_d"]:
                    ch["best"], ch["best_d"] = line_id, d
                line = {"id": line_id, "char": spk, "orig": str(orig.relative_to(project.dir)), "duration": d,
                        "en": text, "ru_official": ru.get(key, (None,))[0] if key else None,
                        "src": {"mod": mod.name, "mods": [mod.name], "wave": wave, "caption": key, **info}}
                seen[wave] = line
                lines.append(line)
        project.reset()
        project.add_lines(lines)
        project.set_characters([{"id": c["id"], "name": c["name"], "color": c["color"], "ref_line": c["best"]}
                                for c in chars.values()])
        return len(lines)

    # --- установка ---

    def _target(self, game: dict, mod: str) -> Path:
        return Path(game["path"]) / mod / "custom" / "ruvoice"

    def install(self, project, game: dict, progress) -> int:
        approved = project.lines(status="approved")
        mods_of = lambda l: l["src"].get("mods") or [l["src"]["mod"]]
        for mod in {m for l in approved for m in mods_of(l)}:
            shutil.rmtree(self._target(game, mod), ignore_errors=True)
        for i, l in enumerate(approved, 1):
            if i % 25 == 0 or i == len(approved):
                progress(f"Установка: {i}/{len(approved)}", i, len(approved))
            src = l["src"]
            data, sr = audio.read(project.dir / l["gen"])
            data = audio.resample(data, sr, src["sr"])
            for mod in mods_of(l):
                out = self._target(game, mod) / src["wave"]
                audio.write(out, data, src["sr"], "PCM_16")
                if src.get("vdat"):
                    # Данные движения губ из оригинала: время не совпадёт идеально, но рот будет двигаться
                    orig_chunks = audio.read_riff_chunks(project.dir / "orig_chunks" / (Path(l["orig"]).stem + ".vdat"))
                    audio.append_chunks(out, orig_chunks)
        (project.dir / "installed.txt").write_text("\n".join(sorted({m for l in approved for m in mods_of(l)})), "utf-8")
        return len(approved)

    def uninstall(self, project, game: dict, progress):
        for mod in self.mods(game):
            shutil.rmtree(self._target(game, mod.name), ignore_errors=True)
        (project.dir / "installed.txt").unlink(missing_ok=True)

    def is_installed(self, project, game: dict) -> bool:
        return any(self._target(game, m.name).is_dir() for m in self.mods(game))


def _save_pcm(raw: bytes, dst: Path) -> dict:
    """Оригинальный wav → PCM wav в orig/. Чанки липсинка (VDAT) сохраняются отдельно."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    chunks = audio.read_riff_chunks_from_bytes(raw)
    vdat = [(cid, d) for cid, d in chunks if cid in (b"VDAT", b"PHON")]
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        channels = data.shape[1]
        sf.write(str(dst), data, sr, subtype="PCM_16")
    except Exception:
        tmp = dst.with_suffix(".src.wav")
        tmp.write_bytes(raw)
        audio.decode_to_wav(tmp, dst)
        tmp.unlink(missing_ok=True)
        info = sf.info(str(dst))
        sr, channels = info.samplerate, info.channels
    if vdat:
        vdir = dst.parent.parent / "orig_chunks"
        vdir.mkdir(exist_ok=True)
        fake = b"RIFF" + b"\0" * 4 + b"WAVE" + b"".join(
            cid + len(d).to_bytes(4, "little") + d + (b"\0" if len(d) & 1 else b"") for cid, d in vdat)
        (vdir / (dst.stem + ".vdat")).write_bytes(fake)
    return {"sr": int(sr), "channels": int(channels), "duration": round(sf.info(str(dst)).duration, 3),
            "vdat": bool(vdat)}
