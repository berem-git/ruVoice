"""Озвучка Wwise независимо от того, где лежат файлы (архив движка или папка игры).

Английская озвучка в Wwise опознаётся по языку, без имён файлов:
  • пакеты .pck (AKPK): файлы с языком «english(us)» в таблице языков пакета — потоки;
  • банки .bnk (BKHD) с языком english(us): встроенные звуки (короткие реплики целиком в банке).
Поток важнее встроенного: у потока со «стартом в банке» (prefetch) в банке лежит лишь начало записи.
Замена: потоки — в пакетах (wwisepck.Pck.rebuild), встроенные — в банках (wwisebnk.patch_bank); у потоков со
стартом в банке банк переключается на «только поток». Звук пишется в Wwise PCM — его читает любая версия Wwise.
Контейнер (архив RE Engine, папка игры…) передаёт сюда пары (метка или ручка файла, данные)."""
from __future__ import annotations

import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import audio
from .unreal import _decode_batch
from .wwisebnk import Bank, bank_language, codec_sources, fnv1_32, patch_bank
from .wwisepck import Pck, is_english, is_russian

EN_BANK = fnv1_32("english(us)")
RU_BANK = fnv1_32("russian")
MIN_SEC, MAX_SEC = 0.3, 90.0


def is_sound(head: bytes) -> bool:
    return head[:4] in (b"AKPK", b"BKHD")


def scan(items, progress=lambda *a: None) -> tuple[dict, dict, bool]:
    """items — [(метка группы, данные файла)]. → ({sid: (wem, группа)} потоков, … встроенных, есть ли русская озвучка)."""
    streams, embedded, has_ru = {}, {}, False
    for n, (label, data) in enumerate(items, 1):
        if n % 200 == 0:
            progress(f"Поиск озвучки: просмотрено {n} файлов…")
        if data[:4] == b"AKPK":
            try:
                p = Pck.parse(data)
            except Exception:
                continue
            has_ru = has_ru or any(is_russian(v) for v in p.languages.values())
            for x in p.entries:
                # пустые записи — заглушки; «пакет-оглавление» (RE Engine) хранит только таблицу, а сами данные —
                # в копии пакета на пути потоков: берём файл только оттуда, где он есть целиком
                if x.table != "banks" and x.size > 0 and is_english(p.lang_name(x.lang)) and x.id not in streams:
                    w = p.read(x)
                    if len(w) == x.size:
                        streams[x.id] = (w, f"Пакет {label}")
        elif data[:4] == b"BKHD":
            lang = bank_language(data)
            has_ru = has_ru or lang == RU_BANK
            if lang != EN_BANK:
                continue
            try:
                b = Bank(data)
            except Exception:
                continue
            for sid, wem in b.embedded.items():
                embedded.setdefault(sid, (wem, f"Банк {label}"))
    return streams, embedded, has_ru


def to_lines(project, streams: dict, embedded: dict, progress) -> tuple[list[dict], dict]:
    """Расшифровать английскую озвучку в orig\\ и собрать реплики (id «wem:<sid>», группа — пакет/банк)."""
    voices = {sid: (w, g, "bank") for sid, (w, g) in embedded.items()}
    voices.update({sid: (w, g, "stream") for sid, (w, g) in streams.items()})
    project.orig_dir.mkdir(parents=True, exist_ok=True)
    lines, chars = [], {}
    items = sorted(voices.items())
    done = 0
    with tempfile.TemporaryDirectory(prefix="ruvoice_ww_") as td:
        td = Path(td)
        for i in range(0, len(items), 120):
            chunk = items[i:i + 120]
            wavs = _decode_batch(td, [(str(sid), w) for sid, (w, _, _) in chunk])
            for sid, (w, group, kind) in chunk:
                done += 1
                wav = wavs.get(str(sid))
                if not wav:
                    continue
                try:
                    info = sf.info(str(wav))
                except Exception:
                    continue
                if not MIN_SEC <= info.duration <= MAX_SEC:
                    wav.unlink(missing_ok=True)
                    continue
                line_id = f"wem:{sid}"
                orig = project.orig_dir / f"wem_{sid}.wav"
                shutil.move(str(wav), str(orig))
                ch = chars.setdefault(group, {"id": group, "name": group, "best": None, "best_d": 0})
                if 3.5 <= info.duration <= 12 and info.duration > ch["best_d"]:
                    ch["best"], ch["best_d"] = line_id, info.duration
                lines.append({"id": line_id, "char": group, "orig": str(orig.relative_to(project.dir)),
                              "duration": round(info.duration, 3), "en": None, "ru_official": None,
                              "src": {"sid": sid, "kind": kind, "sr": info.samplerate, "channels": info.channels}})
            progress(f"Извлечение реплик: {done}/{len(items)}", done, len(items))
    return lines, chars


def encode(project, approved: list[dict], work: Path, progress) -> dict[int, bytes]:
    """Одобренные русские реплики → .wem (Wwise PCM, частота и каналы как у оригинала)."""
    pairs, sids = [], []
    for i, l in enumerate(approved):
        gen, gsr = audio.read(project.dir / l["gen"])
        sr = int(l["src"].get("sr") or gsr)
        out = audio.resample(gen, gsr, sr)
        ch = int(l["src"].get("channels") or 1)
        if ch > 1:
            out = np.repeat(out[:, None], ch, axis=1)
        wav = work / f"{i:06d}.wav"
        audio.write(wav, out, sr, "PCM_16")
        pairs.append((wav, work / f"{i:06d}.wem"))
        sids.append(int(l["src"]["sid"]))
    progress("Конвертация в формат Wwise…")
    audio.encode_wem_batch(pairs, conversion="PCM As Input")
    return {sid: Path(dst).read_bytes() for sid, (_, dst) in zip(sids, pairs)}


def patch(items, new: dict[int, bytes], kinds: dict[int, str]):
    """items — [(ручка файла, данные)] → [(ручка, новые данные)] только для изменившихся файлов."""
    streamed = {sid for sid, k in kinds.items() if k == "stream"}
    in_bank = {sid: new[sid] for sid, k in kinds.items() if k == "bank"}
    layout: dict[tuple, tuple[int, int]] = {}   # (таблица, id, язык, старый блок, старый размер) → (размер, блок)
    headers: list[tuple] = []                   # пакеты-оглавления (данные в копии): правим после полных
    for handle, data in items:
        if data[:4] == b"AKPK":
            try:
                p = Pck.parse(data)
            except Exception:
                continue
            targets = [x for x in p.entries if x.table != "banks" and x.size > 0 and x.id in streamed
                       and is_english(p.lang_name(x.lang))]
            if not targets:
                continue
            if any(x.offset + x.size > len(data) for x in targets):
                headers.append((handle, p, targets))
                continue
            rebuilt = p.rebuild({(x.table, x.id, x.lang): new[x.id] for x in targets})
            q = Pck.parse(rebuilt)
            for old, cur in zip(p.entries, q.entries):
                if old in targets:
                    layout[(old.table, old.id, old.lang, old.start_block, old.size)] = (cur.size, cur.start_block)
            yield handle, rebuilt
        elif data[:4] == b"BKHD":
            srcs = {sid for sid, *_ in codec_sources(data)}
            try:
                emb = set(Bank(data).embedded)
            except Exception:
                emb = set()
            if not ((srcs | emb) & (streamed | set(in_bank))):
                continue
            patched = patch_bank(data, {k: v for k, v in in_bank.items() if k in emb or k in srcs}, streamed)
            if patched != data:
                yield handle, patched
    # оглавления: те же новые размеры и номера блоков, что в полной копии (данные игра берёт оттуда)
    for handle, p, targets in headers:
        out = bytearray(p.data)
        changed = False
        for x in targets:
            got = layout.get((x.table, x.id, x.lang, x.start_block, x.size))
            if got:
                off = 8 if x.table == "externals" or (x.table == "streams" and p.wide_ids) else 4
                struct.pack_into("<II", out, x.pos + off + 4, got[0], got[1])
                changed = True
        if changed:
            yield handle, bytes(out)
