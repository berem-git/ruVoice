"""Официальные русские субтитры для любой игры — без знания, где игра хранит связь «звук ↔ текст».

Многие игры не подписывают, какой файл озвучки к какой строке субтитров относится (имён нет — только номера
или хеши). Тогда работает общий путь: адаптер (или поиск по файлам игры) даёт пул пар «английская строка →
русская строка», английская речь каждой реплики распознаётся, и реплика получает строку, которая лучше всего
совпадает со сказанным. Совпадение проверяется строго (по буквам), чтобы не подставить чужой текст.

Пул лежит в папке проекта (subtitles.jsonl). Адаптер пишет его при извлечении; если не написал — шаг
подготовки сам ищет тексты в файлах игры (harvest): таблицы CSV/TSV со столбцами языков, пары файлов
«…/en/…» ↔ «…/ru/…» (JSON, XML, INI/TXT «ключ=значение»), русские .po (msgid — английский оригинал)."""
from __future__ import annotations

import csv
import ast
import difflib
import io
import json
import os
import re
from collections import Counter
from pathlib import Path

POOL_FILE = "subtitles.jsonl"
MIN_SCORE = 0.76  # ниже — уже чужие строки («Thank you.», который Whisper слышит в тишине, даёт 0.75)
_TAG = re.compile(r"<[^<>]{0,80}>|\{[^{}]{0,40}\}|\[[A-Za-z_]{1,20}\]")
_PLACEHOLDER = re.compile(r"#rejected#|^\s*(tbd|todo|placeholder|dummy|test)\b|(?-i:^[A-Z0-9_]{6,}$)", re.I)


def clean(t: str) -> str:
    """Строка субтитра без разметки игры (<COLOR…>, {0}) и переносов."""
    t = _TAG.sub("", t or "")
    t = t.replace("\r\n", " ").replace("\n", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", t).strip()


def norm(t: str) -> str:
    t = clean(t).lower().replace("’", "'").replace("ё", "е")
    return " ".join(re.sub(r"[^a-z0-9а-я' ]+", " ", t).split())


def _usable(en: str, ru: str) -> bool:
    return bool(en and ru and len(norm(en)) >= 2 and not _PLACEHOLDER.search(en) and not _PLACEHOLDER.search(ru)
                and re.search(r"[А-Яа-яЁё]", ru))


def _with_runs(rows: list[dict]) -> list[dict]:
    """Добавить склейки 2–3 соседних строк одной группы (одного файла субтитров): в роликах один звуковой
    файл часто покрывает несколько строк подряд («Дай руку! Держись! Дай другую руку…»)."""
    out = list(rows)
    groups: dict = {}
    for r in rows:
        if r.get("group") is not None:
            groups.setdefault(r["group"], []).append(r)
    for g in groups.values():
        if len(g) > 20000:
            continue  # огромная таблица (весь интерфейс игры) — склейки только раздуют пул
        for size in (2, 3):
            for i in range(len(g) - size + 1):
                run = g[i:i + size]
                if all(_usable(clean(x["en"]), clean(x["ru"])) for x in run):
                    out.append({"key": "+".join(x["key"] for x in run), "en": " ".join(clean(x["en"]) for x in run),
                                "ru": " ".join(clean(x["ru"]) for x in run)})
    return out


def save_pool(project, rows: list[dict]) -> int:
    """rows — [{"key", "en", "ru", "group"?}] по порядку; сохраняет пригодные (оба языка, без заглушек).
    group — откуда строка (файл субтитров): соседние строки одной группы склеиваются для роликов."""
    seen, out = set(), []
    for r in _with_runs(rows):
        en, ru = clean(r.get("en", "")), clean(r.get("ru", ""))
        if not _usable(en, ru):
            continue
        k = (norm(en), ru)
        if k in seen:
            continue
        seen.add(k)
        out.append({"key": r.get("key") or "", "en": en, "ru": ru})
    path = project.dir / POOL_FILE
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + ("\n" if out else ""), "utf-8")
    return len(out)


def load_pool(project) -> list[dict] | None:
    path = project.dir / POOL_FILE
    if not path.exists():
        return None
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def exact_asset_match(rows: list[dict], asset: str) -> dict | None:
    """Only an unambiguous localisation key may attach text without recognising the audio."""
    stem = Path(asset.replace("\\", "/")).stem.casefold()
    candidates = []
    for row in rows:
        key = str(row.get("key") or "").rsplit(":", 1)[-1]
        key = re.split(r"[./\\]", key)[-1].casefold()
        if key == stem and _usable(row.get("en", ""), row.get("ru", "")):
            candidates.append(row)
    distinct = {(r["en"], r["ru"]) for r in candidates}
    return candidates[0] if len(distinct) == 1 else None


class Matcher:
    """Поиск строки субтитров по распознанной речи: точное совпадение, иначе кандидаты по редким словам
    и сравнение по буквам."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.normed = [norm(r["en"]) for r in rows]
        self.exact: dict[str, int] = {}
        self.ambiguous: set[str] = set()
        for i, n in enumerate(self.normed):
            if n in self.exact and norm(rows[self.exact[n]]["ru"]) != norm(rows[i]["ru"]):
                self.ambiguous.add(n)
            self.exact.setdefault(n, i)
        self.index: dict[str, list[int]] = {}
        for i, n in enumerate(self.normed):
            for w in set(n.split()):
                self.index.setdefault(w, []).append(i)
        self.lens = [len(n) for n in self.normed]

    def match(self, text: str) -> tuple[dict | None, float]:
        n = norm(text)
        if not n:
            return None, 0.0
        if n in self.ambiguous:
            return None, 0.0
        if n in self.exact:
            return self.rows[self.exact[n]], 1.0
        # кандидаты — строки близкой длины, у которых больше всего общих слов (редкие слова весят больше);
        # самые частые слова («the», «you») с огромными списками не перебираем
        lo, hi = 0.6 * len(n), 1.7 * len(n)
        votes: Counter = Counter()
        for w in set(n.split()):
            post = self.index.get(w)
            if not post or len(post) > 4000:
                continue
            weight = 1.0 / (1.0 + len(post) ** 0.5)
            for i in post:
                if lo <= self.lens[i] <= hi:
                    votes[i] += weight
        best, best_r = None, 0.0
        for i, _ in votes.most_common(120):
            t = self.normed[i]
            sm = difflib.SequenceMatcher(None, n, t, autojunk=False)
            if sm.real_quick_ratio() < best_r or sm.quick_ratio() < best_r:
                continue
            r = sm.ratio()
            if r > best_r:
                best, best_r = i, r
        return (self.rows[best], best_r) if best is not None else (None, 0.0)


# --- поиск текстов в файлах игры ---

EN_TOK = {"en", "eng", "english", "en-us", "en_us", "enus", "en-gb", "en_gb", "us", "int"}
RU_TOK = {"ru", "rus", "russian", "ru-ru", "ru_ru", "ruru"}
TEXT_EXT = {".json", ".csv", ".tsv", ".txt", ".xml", ".po", ".ini", ".lang", ".strings", ".yml", ".yaml",
            ".resx", ".properties", ".srt", ".vtt", ".ass", ".ssa", '.rpy'}
SKIP_DIRS = {"_commonredist", "redist", "directx", "__pycache__", "binaries", "mono"}
SKIP_EXT = {".exe", ".dll", ".pak", ".utoc", ".ucas", ".bank", ".fsb", ".pck", ".bnk", ".wem", ".bik", ".bk2",
            ".mp4", ".webm", ".dds", ".png", ".jpg", ".ogg", ".wav", ".mp3", ".ttf", ".otf"}
MAX_FILE = 60 << 20
MAX_ZIP = 2 << 30


def _decode_text(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            if enc == "utf-16" and not (b[:2] in (b"\xff\xfe", b"\xfe\xff") or b[1:2] == b"\x00"):
                continue
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("cp1251", "replace")


def _read_text(p: Path) -> str:
    return _decode_text(p.read_bytes())


def _is_text(b: bytes) -> bool:
    head = b[:4096]
    if head[:2] in (b"\xff\xfe", b"\xfe\xff") or head[:3] == b"\xef\xbb\xbf":
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start < len(head) - 4:
            return False
    return head.count(b"\x00") < len(head) // 20


def _kv(ext: str, text: str) -> dict[str, str]:
    """Текстовый файл одного языка → {ключ: строка}."""
    out: dict[str, str] = {}
    if ext in {".srt", ".vtt"}:
        # Timestamps and cue identifiers are stable across language translations.
        # Never align translated cues just by line number: an omitted cue shifts every later match.
        for block in re.split(r"\n\s*\n", text.replace("\r", "")):
            lines = block.strip().splitlines()
            for i, line in enumerate(lines):
                if "-->" in line:
                    stamp = re.sub(r"\s+", " ", line.strip()).replace(",", ".")
                    value = clean(" ".join(lines[i + 1:]))
                    if value:
                        out[stamp] = value
                    break
        return out
    if ext in {".ass", ".ssa"}:
        for line in text.splitlines():
            if line.lower().startswith("dialogue:"):
                parts = line.split(":", 1)[1].split(",", 9)
                if len(parts) == 10:
                    value = clean(re.sub(r"\{[^}]*\}", "", parts[9]).replace(r"\N", " "))
                    out[parts[1].strip() + " --> " + parts[2].strip()] = value
        return out
    if ext == ".json":
        try:
            def walk(o, pre=""):
                if isinstance(o, dict):
                    for k, v in o.items():
                        walk(v, f"{pre}.{k}" if pre else str(k))
                elif isinstance(o, list):
                    for k, v in enumerate(o):
                        walk(v, f"{pre}[{k}]")
                elif isinstance(o, str):
                    out[pre] = o
            walk(json.loads(text))
        except ValueError:
            pass
    elif ext in (".xml", ".resx"):
        for m in re.finditer(r'<(\w+)[^>]*?\b(?:name|key|id)="([^"]+)"[^>]*>(.*?)</\1>', text, re.S):
            out[m.group(2)] = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", m.group(3), flags=re.S)
        if not out:
            for m in re.finditer(r'<(\w+)>([^<>]+)</\1>', text):
                out.setdefault(m.group(1) + f"#{len(out)}", m.group(2))
    else:
        for line in text.splitlines():
            m = re.match(r'\s*"?([^=:"\t#;]{1,120}?)"?\s*[=:\t]\s*"?(.*?)"?\s*$', line)
            if m and m.group(2):
                out[m.group(1).strip()] = m.group(2).replace("\\n", " ")
    return out


def _from_rpy(text: str, group: str) -> list[dict]:
    """Read Ren'Py old/new strings and commented-original dialogue without executing scripts."""
    pending, rows = None, []
    quoted = re.compile(r'^\s*(.*?)\s*("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')\s*$')
    excluded = {'voice', 'play', 'scene', 'show', 'hide', 'image', 'define', 'default', 'stop'}
    for number, raw in enumerate(text.splitlines()):
        line = raw.strip()
        comment = line.startswith('#')
        match = quoted.match(line[1:].strip() if comment else line)
        if not match:
            if line.startswith('translate '):
                pending = None
            continue
        prefix, literal = match.groups()
        token = prefix.split()[0] if prefix else ''
        if token in excluded:
            continue
        try:
            value = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, str):
            continue
        value = clean(value)
        if comment or token == 'old':
            pending = (value, number)
        elif pending and number - pending[1] <= 4:
            if _usable(pending[0], value):
                rows.append({'key': f'{group}:{number}', 'en': pending[0], 'ru': value, 'group': group})
            pending = None
    return rows


def _cstr(d: bytes, off: int, wide: bool) -> str | None:
    if not 0 <= off < len(d):
        return None
    if wide:
        j = off
        while j + 1 < len(d) and d[j:j + 2] != b"\x00\x00":
            j += 2
            if j - off > 8192:
                return None
        try:
            return d[off:j].decode("utf-16-le")
        except UnicodeDecodeError:
            return None
    j = d.find(b"\x00", off, off + 8192)
    if j < 0:
        return None
    try:
        return d[off:j].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _good(s: str | None) -> bool:
    return bool(s) and not re.search(r"[\x00-\x08\x0e-\x1f]", s) and bool(re.search(r"[A-Za-zА-Яа-я]", s))


def binary_pairs(a: bytes, b: bytes, name: str = "") -> list[dict]:
    """Двоичные таблицы строк без знания формата: английский и русский варианты одного файла. В обоих одна и та
    же таблица записей (ключ u32, смещение u32): ключи совпадают, смещения различаются — по этому признаку таблица
    находится; строки (UTF-16 или UTF-8, до нуля) читаются от конца таблицы, от её начала или от начала файла —
    какой вариант даёт осмысленный текст. Так устроены, например, .str у Forza Horizon."""
    import numpy as np
    n = min(len(a), len(b)) // 4 * 4
    best = None
    for al in range(4):
        m = (n - al) // 4
        if m < 16:
            continue
        A = np.frombuffer(a[al:al + 4 * m], "<u4")
        B = np.frombuffer(b[al:al + 4 * m], "<u4")
        for par in (0, 1):
            ka, kb = A[par::2], B[par::2]
            oa, ob = A[par + 1::2], B[par + 1::2]
            k = min(len(ka), len(oa))
            good = (ka[:k] == kb[:k]) & (ka[:k] != 0)
            # самый длинный отрезок подряд совпавших ключей
            idx = np.flatnonzero(np.diff(np.concatenate(([0], good.astype(np.int8), [0]))))
            for s0, s1 in zip(idx[::2], idx[1::2]):
                cnt = s1 - s0
                if cnt < 8 or (best and cnt <= best[3]):
                    continue
                if np.mean(oa[s0:s1] != ob[s0:s1]) < 0.5:
                    continue  # одинаковые и смещения — просто совпадающий кусок файла, не таблица
                best = (al, par, s0, cnt, oa[s0:s1].astype(np.int64), ob[s0:s1].astype(np.int64),
                        ka[s0:s1].astype(np.int64))
    if not best:
        return []
    al, par, s0, cnt, oa, ob, keys = best
    start = al + 4 * (par + 2 * s0)
    end = start + 8 * cnt
    out = []
    for base in (end, start, 0):
        for wide in (True, False):
            rows = []
            for kk, x, y in zip(keys, oa, ob):
                en, ru = _cstr(a, base + int(x), wide), _cstr(b, base + int(y), wide)
                rows.append((int(kk), en, ru))
            ok = sum(1 for _, en, ru in rows if _good(en) and _good(ru))
            if ok >= 0.6 * len(rows) and sum(1 for _, _, ru in rows if ru and re.search("[А-Яа-я]", ru)) >= 0.3 * len(rows):
                return [{"key": f"{name}:{kk:08x}", "en": en, "ru": ru} for kk, en, ru in rows if _good(en) and _good(ru)]
    return out


_U8 = re.compile(rb"(?:[\x20-\x7e]|[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}){2,}")
_U16 = re.compile(rb"(?:[\x20-\x7e]\x00|[\x00-\xff]\x04|[\x10-\x2f]\x20|[\xa0-\xff]\x00){2,}")


def _strings(d: bytes) -> list[str]:
    """Строки файла по порядку: UTF-8 или UTF-16 — какая кодировка даёт больше текста."""
    u8 = [m.group(0).decode("utf-8", "replace") for m in _U8.finditer(d)]
    u16 = [m.group(0).decode("utf-16-le", "replace") for m in _U16.finditer(d)]
    return u16 if sum(map(len, u16)) > sum(map(len, u8)) else u8


def anchor_pairs(a: bytes, b: bytes, name: str = "") -> list[dict]:
    """Пары из двух вариантов одного файла (английский и русский), когда ключи хранятся строками: одинаковые
    строки в обоих файлах — опоры (ключи, имена), а то, что стоит между одинаковыми опорами, — значения на двух
    языках. Ключ пары — предыдущая опора. Подходит для словарей вида «ключ, значение, ключ, значение…»
    в любом двоичном формате (например, у Hogwarts Legacy)."""
    A, B = _strings(a), _strings(b)
    if not A or not B or len(A) > 400000:
        return []
    out = []
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        block = []
        for k in range(i2 - i1):
            en, ru = A[i1 + k], B[j1 + k]
            fits = (re.search(r"[A-Za-z]{2}", en) and re.search(r"[А-Яа-яЁё]{2}", ru)
                    and 0.5 <= len(ru) / max(1, len(en)) <= 2.5)
            block.append((k, en, ru, bool(fits)))
        # блок берём, только если почти все пары правдоподобны (иначе порядок строк мог разъехаться)
        if not block or sum(1 for b in block if b[3]) < 0.8 * len(block):
            continue
        anchor = A[i1 - 1] if i1 > 0 else ""
        for k, en, ru, fits in block:
            if fits:
                out.append({"key": anchor if k == 0 else f"{anchor}#{k}", "en": en, "ru": ru})
    return out


def pair_files(a: bytes, b: bytes, name: str = "") -> list[dict]:
    """Английский и русский варианты одного файла неизвестного формата → пары строк."""
    rows = binary_pairs(a, b, name)
    return rows if rows else anchor_pairs(a, b, name)


def _lang_swap(rel: str) -> list[str]:
    """Пути, где может лежать русская пара английского файла."""
    out = []
    parts = re.split(r"([\\/._\-])", rel)
    for i, part in enumerate(parts):
        if part.lower() in EN_TOK:
            for ru in ("ru", "RU", "Ru", "russian", "Russian", "RUSSIAN", "ru-RU", "ru_RU", "rus", "RUS"):
                out.append("".join(parts[:i] + [ru] + parts[i + 1:]))
    return out


def _sources(root: Path):
    """Все файлы игры, где может быть текст: обычные файлы и содержимое zip-архивов («архив.zip/файл»).
    → {путь в нижнем регистре: (путь, размер, чтение)}."""
    import zipfile
    found = {}
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x.lower() not in SKIP_DIRS]
        for n in names:
            p = Path(d) / n
            ext = p.suffix.lower()
            if ext in {'.pck', '.rpa'}:
                from .adapters.open_packages import PCK, RPA
                import pickle, zlib
                try:
                    archive = (PCK if ext == '.pck' else RPA)(p)
                    for entry in archive.entries:
                        if not entry.flags and 16 < entry.size <= MAX_FILE:
                            rel = p.relative_to(root).as_posix() + '/' + entry.name
                            toks = set(re.split(r'[\\/._\-]', rel.lower()))
                            if toks & (EN_TOK | RU_TOK) or Path(entry.name).suffix.lower() in TEXT_EXT:
                                found[rel.lower()] = (rel, entry.size, lambda a=archive, e=entry: a.read(e, MAX_FILE))
                except (OSError, ValueError, EOFError, TypeError, pickle.UnpicklingError, zlib.error):
                    pass
                continue
            is_zip = ext == ".zip"
            if ext in {".dat", ".bin", ".pak", ".pk3", ".pk4", ".iwd", ".love", ".nw"}:
                try:
                    with p.open("rb") as f:
                        is_zip = f.read(4) == b"PK\x03\x04"
                except OSError:
                    continue
            if ext in SKIP_EXT and not is_zip:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            if is_zip and size <= MAX_ZIP:
                # архив открывается только на время чтения (у игр бывают тысячи zip — все открытыми не удержать);
                # внутри берём лишь то, где помечен язык или что похоже на таблицу строк
                try:
                    with zipfile.ZipFile(p) as z:
                        infos = z.infolist()
                except Exception:
                    continue
                for info in infos:
                    if info.is_dir() or not 16 < info.file_size <= MAX_FILE:
                        continue
                    r = f"{rel}/{info.filename}"
                    toks = set(re.split(r"[\\/._\-]", r.lower()))
                    if not (toks & (EN_TOK | RU_TOK)) and Path(info.filename).suffix.lower() not in (".csv", ".tsv", ".po"):
                        continue
                    found[r.lower()] = (r, info.file_size, (lambda p=p, nm=info.filename: _zip_read(p, nm)))
            elif 16 < size <= MAX_FILE:
                found[rel.lower()] = (rel, size, (lambda p=p: p.read_bytes()))
    return found


def _zip_read(path: Path, member: str) -> bytes:
    import zipfile
    with zipfile.ZipFile(path) as z:
        return z.read(member)


def harvest(root: Path, progress=lambda *a: None) -> list[dict]:
    """Пары «английский → русский» из файлов игры (без знания игры): таблицы CSV/TSV со столбцами языков,
    русские .po, пары файлов en/ru (текстовые — по ключам, двоичные — по общей таблице ключей), в том числе
    внутри zip-архивов."""
    src = _sources(root)
    rows: list[dict] = []
    done_pairs = set()
    for low, (rel, size, read) in src.items():
        ext = Path(rel).suffix.lower()
        toks = set(re.split(r"[\\/._\-]", low))
        try:
            if ext in (".csv", ".tsv"):
                rows += _from_table_text(read(), ext)
                continue
            if ext == ".po" and toks & RU_TOK:
                rows += _from_po(_decode_text(read()))
                continue
            if ext == '.rpy' and toks & RU_TOK:
                rows += _from_rpy(_decode_text(read()), rel)
                continue
            if not toks & EN_TOK:
                continue
            pair = next((c for c in _lang_swap(rel) if c.lower() in src and c.lower() != low), None)
            if pair is None or (low, pair.lower()) in done_pairs:
                continue
            done_pairs.add((low, pair.lower()))
            a, b = read(), src[pair.lower()][2]()
            if ext in TEXT_EXT or (_is_text(a) and _is_text(b)):
                en, ru = _kv(ext, _decode_text(a)), _kv(ext, _decode_text(b))
                rows += [{"key": f"{rel}:{k}", "en": v, "ru": ru[k], "group": rel} for k, v in en.items() if k in ru]
            else:
                got = pair_files(a, b, Path(rel).name)
                for r in got:
                    r["group"] = rel
                rows += got
        except Exception:
            continue
    progress(f"Найдено строк субтитров в файлах игры: {len(rows)}")
    return rows


def _from_table_text(data: bytes, ext: str) -> list[dict]:
    text = _decode_text(data)
    dialect = csv.excel_tab if ext == ".tsv" or text[:2000].count("\t") > text[:2000].count(",") else csv.excel
    rd = csv.reader(io.StringIO(text), dialect)
    head = next(rd, None)
    if not head:
        return []
    low = [h.strip().lower() for h in head]
    en_i = next((i for i, h in enumerate(low) if h in EN_TOK or h.startswith("english")), None)
    ru_i = next((i for i, h in enumerate(low) if h in RU_TOK or h.startswith("russian")), None)
    if en_i is None or ru_i is None:
        return []
    out = []
    for r in rd:
        if len(r) > max(en_i, ru_i):
            out.append({"key": r[0], "en": r[en_i], "ru": r[ru_i]})
    return out


def _from_po(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r'msgid\s+((?:"(?:[^"\\]|\\.)*"\s*)+)msgstr\s+((?:"(?:[^"\\]|\\.)*"\s*)+)', text):
        def val(s):
            return "".join(json.loads(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"', s))
        try:
            en, ru = val(m.group(1)), val(m.group(2))
        except ValueError:
            continue
        if en and ru:
            out.append({"key": en[:60], "en": en, "ru": ru})
    return out
