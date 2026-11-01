"""Кто говорит: разбор реплик по голосам для любой игры.

Файлы игр редко подписаны персонажами честно: часто «персонаж» — это номер разговора, где вперемешку
говорят NPC и главный герой (так в Sekiro); в инди и Unreal-играх подписей нет вовсе. Каждая реплика получает отпечаток голоса
(ECAPA-TDNN) и высоту голоса; группы игры служат опорой (propose): у группы — основной голос, голос, звучащий
во многих разговорах, — сквозной персонаж (главный герой), одинаковые основные голоса разных групп — один NPC.
Сплошная кластеризация (cluster) склеивала похожие женские голоса студии — Эмму с Божественным дитя.
Имена: из групп игры, если голос в основном оттуда; главный герой — по знанию игры; остальные — языковая
модель по репликам (с раздумьем; имя, к которому в репликах обращаются, не принимается).
Результат — предложение (project meta "speaker_proposal"), которое пользователь просматривает и применяет:
тогда реплики переносятся к найденным персонажам. Установка в игру от этого не зависит — она идёт по
идентификаторам реплик, а не по персонажам."""
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import audio, judge

MIN_SEC_EMBED = 0.8      # короче — отпечаток не считаем
MIN_SEC_CLUSTER = 1.6    # кластеризуем только достаточно длинные реплики, короткие присоединяем потом
DIST = 0.62              # порог косинусного расстояния (1 - cos) при объединении голосов
ATTACH_COS = 0.38        # короткая реплика присоединяется к голосу, если похожа хотя бы настолько
MIN_CLUSTER = 6          # меньше — «прочие», остаются у исходного персонажа


def _generic(name: str) -> bool:
    return not name or bool(re.match(r"^(группа|group|голос|voice|speaker|пакет|банк|bank|package)\b", name.strip(), re.I)) or "_" in name


def embed_all(p, report=lambda *a: None, cancel=lambda: False) -> dict:
    """Отпечатки ECAPA и высота голоса всех реплик (кэш в projects/<id>/speakers.npz)."""
    cache = p.dir / "speakers.npz"
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = {i: (e, f) for i, e, f in zip(z["ids"].tolist(), z["E"], z["F0"])}
    lines = [l for l in p.lines() if (l["duration"] or 0) >= MIN_SEC_EMBED and l["orig"]]
    todo = [l for l in lines if l["id"] not in have]
    t0 = time.time()
    for i in range(0, len(todo), 32):
        if cancel():
            break
        chunk = todo[i:i + 32]
        report(f"Кто говорит: слушаю голоса {i}/{len(todo)}", i, len(todo))
        wavs = [audio.read(p.dir / l["orig"]) for l in chunk]
        E = judge.ecapa_embed(wavs)
        for l, (w, sr), e in zip(chunk, wavs, E):
            have[l["id"]] = (e, audio.f0_median(w, sr))
        if i % 320 == 0 and have:
            _save(cache, have)
    if have:
        _save(cache, have)
    print(f"кто говорит: отпечатков {len(have)} (новых {len(todo)}) за {time.time() - t0:.0f} с")
    return have


def _save(path: Path, have: dict):
    ids = list(have)
    np.savez(path, ids=np.array(ids), E=np.stack([have[i][0] for i in ids]).astype(np.float32),
             F0=np.array([have[i][1] or 0 for i in ids], dtype=np.float32))


def cluster(p, have: dict, dist: float = DIST) -> list[dict]:
    """Группы реплик по голосу: [{"lines": [id…], "centroid": e, "f0": Гц}] — от больших к малым."""
    from sklearn.cluster import AgglomerativeClustering
    by_id = {l["id"]: l for l in p.lines()}
    long_ids = [i for i in have if (by_id.get(i, {}).get("duration") or 0) >= MIN_SEC_CLUSTER]
    if len(long_ids) < 4:
        return []
    E = np.stack([have[i][0] for i in long_ids])
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=dist).fit_predict(E)
    groups: dict[int, list[str]] = {}
    for i, lab in zip(long_ids, labels):
        groups.setdefault(int(lab), []).append(i)
    clusters = []
    for ids in groups.values():
        c = np.mean([have[i][0] for i in ids], axis=0)
        clusters.append({"lines": ids, "centroid": c / (np.linalg.norm(c) + 1e-9)})
    # короткие реплики — к ближайшему голосу
    big = [c for c in clusters if len(c["lines"]) >= MIN_CLUSTER]
    if big:
        C = np.stack([c["centroid"] for c in big])
        for i in have:
            if i in set(long_ids):
                continue
            sims = C @ have[i][0]
            k = int(np.argmax(sims))
            if sims[k] >= ATTACH_COS:
                big[k]["lines"].append(i)
    for c in big:
        f = np.array([have[i][1] for i in c["lines"] if have[i][1]], dtype=np.float32)
        c["f0"] = float(np.median(f)) if f.size else 0.0
    big.sort(key=lambda c: -len(c["lines"]))
    return big


OWN_COS = 0.45      # реплика принадлежит основному голосу своей группы, если похожа на него хотя бы настолько
REC_DIST = 0.55     # порог объединения «чужих» реплик в сквозные голоса (главный герой в разговорах с NPC)
REC_MIN = 8         # сквозной голос: не меньше реплик…
REC_GROUPS = 3      # …и встречается хотя бы в стольких разговорах
REC_COS = 0.55      # реплика уходит к сквозному голосу, если похожа на него настолько и заметно больше, чем на свою группу
REC_MERGE_COS = 0.65  # куски одного сквозного голоса (подача разная — голос тот же)
LINK_COS = 0.85     # основные голоса двух групп — один человек (тот же NPC в разных разговорах)
NAME_MIN = 10       # безымянные голоса меньше этого не предлагаем (нечего выбирать)


def _speech(l: dict) -> bool:
    """В реплике есть слова (а не «...», «(Спит)», «(Стонет)», пустые субтитры)."""
    t = " ".join(x for x in (l.get("en"), l.get("ru"), l.get("ru_official")) if x)
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]|\*[^*]*\*", " ", t)
    return bool(re.search(r"[A-Za-zА-Яа-яЁё]{2,}", t))


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _robust(E: np.ndarray, keep: float = 0.7, iters: int = 4) -> np.ndarray:
    """Центр основного голоса группы: среднее по самым похожим репликам (чужие голоса не тянут центр)."""
    c = _unit(E.mean(0))
    for _ in range(iters):
        s = E @ c
        thr = np.quantile(s, 1 - keep) if len(s) > 4 else -1
        c = _unit(E[s >= thr].mean(0))
    return c


def _f0(ids, have) -> float:
    f = [have[i][1] for i in ids if have[i][1]]
    return float(np.median(f)) if f else 0.0


def propose(p, have: dict, tr=None, game: str = "", dist: float = DIST) -> dict:
    """Предложение «кто говорит» от групп игры.

    Группы из файлов игры — это чаще разговоры, а не люди: в разговоре с Эммой звучат и Эмма, и Волк.
    1) у каждой группы находим основной голос (центр по самым похожим репликам);
    2) реплики, далёкие от основного голоса своей группы, со всех групп кластеризуем — голос, который
       встречается во многих разговорах, это сквозной персонаж (главный герой); его реплики уходят к нему;
    3) основные голоса разных групп, совпадающие по отпечатку и высоте, — один человек (NPC в разных разговорах);
    4) имена: языковая модель по репликам голоса и именам групп, где он звучит.
    Реплики без слов (сон, стоны, «...») не трогаем — их и озвучивать не нужно."""
    from sklearn.cluster import AgglomerativeClustering
    by_id = {l["id"]: l for l in p.lines()}
    chars = {c["id"]: c for c in p.characters()}
    cname = lambda c: chars.get(c, {}).get("name") or c
    groups: dict[str, list[str]] = {}
    for i in have:
        l = by_id.get(i)
        if l and l["char"] and _speech(l):
            groups.setdefault(l["char"], []).append(i)
    dom = {g: _robust(np.stack([have[i][0] for i in ids])) for g, ids in groups.items() if len(ids) >= 3}
    own = {i: float(have[i][0] @ dom[g]) if g in dom else -1.0 for g, ids in groups.items() for i in ids}
    # 2) сквозные голоса из «чужих» реплик
    pool = [i for i, s in own.items() if s < OWN_COS and (by_id[i]["duration"] or 0) >= MIN_SEC_EMBED]
    recurring = []
    if len(pool) >= REC_MIN:
        lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=REC_DIST).fit_predict(np.stack([have[i][0] for i in pool]))
        cl: dict[int, list[str]] = {}
        for i, k in zip(pool, lab):
            cl.setdefault(int(k), []).append(i)
        for ids in cl.values():
            src = Counter(by_id[i]["char"] for i in ids)
            if (len(ids) >= REC_MIN and sum(1 for n in src.values() if n >= 2) >= REC_GROUPS
                    and src.most_common(1)[0][1] / len(ids) < 0.5):
                recurring.append((ids, _robust(np.stack([have[i][0] for i in ids]), keep=0.8), _f0(ids, have)))
        # один и тот же герой с разной подачей (шёпот, крик, коротко) даёт несколько кусков — склеиваем
        # в больший, если отпечаток близок к его центру и высота та же
        recurring.sort(key=lambda r: -len(r[0]))
        merged: list[list] = []
        for ids, c, f in recurring:
            for m in merged:
                if float(c @ m[1]) >= REC_MERGE_COS and f and m[2] and abs(12 * np.log2(f / m[2])) <= 2.5:
                    m[0] = m[0] + ids
                    m[1] = _unit(m[1] * 0.75 + c * 0.25)
                    break
            else:
                merged.append([ids, c, f])
        recurring = [m[1] for m in merged]
    assign: dict[str, tuple] = {}   # id → ("rec", k) | ("dom", группа)
    R = np.stack(recurring) if recurring else None
    for g, ids in groups.items():
        for i in ids:
            sr = R @ have[i][0] if R is not None else None
            k = int(np.argmax(sr)) if sr is not None else -1
            if sr is not None and sr[k] >= REC_COS and sr[k] > own[i] + 0.08:
                assign[i] = ("rec", k)
            elif own[i] >= OWN_COS or (own[i] >= ATTACH_COS and (by_id[i]["duration"] or 0) < MIN_SEC_CLUSTER):
                assign[i] = ("dom", g)
    # 3) один человек в разных группах: основные голоса близки по отпечатку и высоте
    keys = [g for g in dom if sum(1 for i in groups[g] if assign.get(i) == ("dom", g)) >= 3]
    gf0 = {g: _f0([i for i in groups[g] if assign.get(i) == ("dom", g)], have) for g in keys}
    parent = {g: g for g in keys}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    if keys:
        D = np.stack([dom[g] for g in keys])
        S = D @ D.T
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                fa, fb = gf0[keys[a]], gf0[keys[b]]
                if S[a, b] >= LINK_COS and fa and fb and abs(12 * np.log2(fa / fb)) <= 3:
                    parent[find(keys[a])] = find(keys[b])
    comps: dict[str, list[str]] = {}
    for g in keys:
        comps.setdefault(find(g), []).append(g)
    raw = []
    for k in range(len(recurring)):
        raw.append({"kind": "recurring", "lines": [i for i, a in assign.items() if a == ("rec", k)]})
    for gs in comps.values():
        ids = [i for i, a in assign.items() if a[0] == "dom" and a[1] in set(gs)]
        raw.append({"kind": "group", "groups": gs, "lines": ids})
    out = []
    for v in sorted(raw, key=lambda v: -len(v["lines"])):
        ids = v["lines"]
        if not ids:
            continue
        src = Counter(by_id[i]["char"] for i in ids)
        named = [g for g in (v.get("groups") or []) if not _generic(cname(g))]
        moves = sum(n for c, n in src.items() if c != src.most_common(1)[0][0])
        name = None
        if v["kind"] == "group":
            if len(v["groups"]) == 1 and named:
                continue  # подписанная группа сама по себе — менять нечего
            # имя группы игры, если её реплики составляют основную часть голоса (Эмма + её реплики в разговоре с Куро)
            share = {g: src.get(g, 0) for g in named}
            if share and max(share.values()) / len(ids) >= 0.5:
                name = cname(max(share, key=share.get))
            if len(ids) < NAME_MIN and not named:
                continue
        f0 = _f0(ids, have)
        texts = sorted((by_id[i]["ru"] or by_id[i]["ru_official"] or by_id[i]["en"] or "" for i in ids), key=len, reverse=True)
        out.append({"key": f"v{len(out)}", "kind": v["kind"], "name": name, "auto_name": bool(name), "lines": ids,
                    "count": len(ids), "moves": moves, "groups": len(src), "f0": round(f0),
                    "voice_type": judge.voice_type(f0),
                    "from": [{"char": ch, "name": cname(ch), "n": n} for ch, n in src.most_common(4)],
                    "candidates": [cname(g) for g in named],
                    "texts": [t for t in texts if t][:14],
                    "pairs": [{"en": (by_id[i]["en"] or "")[:160], "ru": (by_id[i]["ru"] or by_id[i]["ru_official"] or "")[:160]}
                              for i in sorted(ids, key=lambda i: -len(by_id[i]["en"] or by_id[i]["ru"] or ""))[:10]],
                    "samples": sorted(ids, key=lambda i: -(by_id[i]["duration"] or 0))[:3]})
    if tr is not None:
        # имена персонажей из файлов игры заняты: «Куро» не достанется торговцу, даже если голос Куро не найден
        name_voices(tr, out, game, {cname(c) for c in chars if not _generic(cname(c))})
    used: Counter = Counter()
    for v in out:
        v["name"] = v["name"] or f"Голос {v['key'][1:]} ({v['voice_type']})"
        used[v["name"]] += 1
    seen: Counter = Counter()
    for v in out:  # одинаковые имена у разных голосов — с номером: сливать или нет, решает пользователь
        if used[v["name"]] > 1:
            seen[v["name"]] += 1
            v["name"] = f"{v['name']} {seen[v['name']]}"
    covered = sum(v["count"] for v in out)
    return {"at": time.time(), "voices": out, "covered": covered, "total": len(by_id),
            "recurring": len(recurring)}


SYSTEM = ("Ты знаток видеоигры {game} и её русской локализации. Тебе дают реплики одного голоса (английский оригинал "
          "и русские субтитры), его тип (по высоте), в скольких разных разговорах он звучит и как названы эти разговоры. "
          "Кто это говорит? Правила: 1) разговор назван по собеседнику, а не по говорящему; 2) имена, упомянутые в "
          "репликах, — почти всегда НЕ говорящий: к нему обращаются или о нём рассказывают (исключение — «меня зовут…»); "
          "3) персонажи из already_found уже найдены отдельно — этим именем другой голос не называй; "
          "4) если среди candidates есть подходящее имя — выбери его; 5) если уверенно назвать нельзя — роль строчными "
          "буквами: торговец, монах, стражник, умирающий солдат. "
          'Ответ строго JSON: {{"name": "<имя по-русски, как в локализации, или роль>", "confidence": 0..1}}. '
          'Если реплики явно разных людей — "name": null.')
HERO = ('Как зовут главного героя (протагониста, чьими глазами играет игрок) видеоигры {game} в русской локализации? '
        'Ответ строго JSON: {{"name": "<имя или прозвище, как его называют в игре>", "confidence": 0..1}}. '
        'Если у героя нет имени или голоса — "name": null.')


THINK_MIN = 10**9  # имена голосов пользователю не показываются: «с раздумьем» (~1 мин) — только главный герой


def _ask(tr, user: str, system: str, think: bool = True) -> dict:
    raw = tr._chat(user, system, think=think)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    return json.loads(m.group(0) if m else raw)


def _clean_name(n) -> str | None:
    n = re.sub(r"^(роль|роли|role)\s*:\s*", "", str(n or "").strip(), flags=re.I).strip(" .«»\"'")[:40]
    if not n or _generic(n) or re.search(r"(?i)\bгрупп|\bgroup", n):
        return None  # «группа 30000», «женщина из группы 29900» — не имя
    return n


def _mentioned_not_self(name: str, texts: list[str]) -> bool:
    """Имя встречается в репликах самого голоса, но не как «меня зовут X» — значит, X не говорящий."""
    base = name.split()[0].lower()
    joined = " ".join(texts).lower()
    if len(base) < 3 or base not in joined:
        return False
    return not re.search(r"(меня зовут|моё имя|мое имя|зовут меня|я\s*[-—–]?\s*)" + re.escape(base), joined)


def name_voices(tr, voices: list[dict], game: str, known: set[str] | None = None):
    """Имена голосам. Главный герой — по игре (сквозной голос, звучащий в самом большом числе разговоров);
    остальные — по репликам. Собственное имя достаётся только одному голосу, роли могут повторяться."""
    taken = {v["name"].lower() for v in voices if v["name"]} | {k.lower() for k in known or ()}
    rec = sorted((v for v in voices if v.get("kind") == "recurring" and not v["name"]), key=lambda v: -v.get("groups", 0))
    total_groups = len({f["char"] for v in voices for f in v["from"]}) or 1
    if rec and rec[0].get("groups", 0) >= max(8, 0.1 * total_groups):
        try:
            d = _ask(tr, f"Игра: {game}", HERO.format(game=game or "—"))
            hero = _clean_name(d.get("name")) if float(d.get("confidence") or 0) >= 0.6 else None
        except Exception as e:
            print(f"кто говорит: главный герой: {e}")
            hero = None
        rec[0]["name"] = hero or "Главный герой"
        rec[0]["llm"], rec[0]["hero"] = 1.0, True
        taken.add(rec[0]["name"].lower())
    for v in voices:
        if v["name"]:
            continue
        payload = {"voice_type": v["voice_type"], "conversations": v.get("groups"),
                   "conversation_names": [f["name"] for f in v["from"]], "candidates": v.get("candidates") or [],
                   "already_found": sorted(taken)[:40], "lines": v.get("pairs") or v["texts"][:12]}
        try:
            d = _ask(tr, json.dumps(payload, ensure_ascii=False), SYSTEM.format(game=game or "—"),
                     think=v["count"] >= THINK_MIN)
        except Exception as e:
            print(f"кто говорит: имя голоса: {e}")
            continue
        name = _clean_name(d.get("name"))
        conf = float(d.get("confidence") or 0)
        if not name or conf < 0.6:
            continue
        proper = name[:1].isupper()
        cand = {c.lower() for c in v.get("candidates") or []}
        if name.lower() not in cand and (name.lower() in taken or (proper and _mentioned_not_self(name, v["texts"]))):
            continue  # имя чужое: уже занято другим голосом или к нему обращаются в репликах
        v["name"], v["llm"] = name, round(conf, 2)
        print(f"кто говорит: {v['key']} ({v['count']} реплик) → {name}", flush=True)
        if proper:
            taken.add(name.lower())


def apply(p, proposal: dict, names: dict | None = None, skip: set | None = None) -> int:
    """Перенести реплики к найденным голосам. names — правки имён {key: имя}; skip — голоса, которые не трогать.
    Голоса с одинаковым именем сливаются в одного персонажа."""
    names = names or {}
    existing = {c["name"]: c["id"] for c in p.characters()}
    moved = 0
    new_chars = []
    with p.db() as c:
        for v in proposal["voices"]:
            if v["key"] in (skip or set()):
                continue
            name = (names.get(v["key"]) or v["name"]).strip()
            cid = existing.get(name)
            if not cid:
                cid = "spk_" + re.sub(r"[^A-Za-z0-9]+", "_", name.encode("ascii", "ignore").decode() or "x")[:24] + "_" + v["key"]
                new_chars.append({"id": cid, "name": name})
                existing[name] = cid
            c.executemany("UPDATE lines SET char=? WHERE id=?", [(cid, i) for i in v["lines"]])
            moved += len(v["lines"])
        for ch in new_chars:
            c.execute("INSERT OR REPLACE INTO characters (id, name, color, ref_line, include) VALUES (?, ?, NULL, NULL, 1)",
                      (ch["id"], ch["name"]))
        # пустые персонажи больше не нужны
        c.execute("DELETE FROM characters WHERE id NOT IN (SELECT DISTINCT char FROM lines WHERE char IS NOT NULL)")
    # профили, отпечатки, выбор и голоса были посчитаны для старой разметки
    for key in ("profiles", "voice_f0", "emotions", "custom_speakers", "pitch_shift", "review", "samples", "bench_lines",
                "ru_voice", "voice_anchors"):
        p.set(key, None)
    p.set("selected_chars", [])
    p.set("speakers_applied", {"at": time.time(), "voices": len(proposal["voices"]), "moved": moved})
    return moved
