"""Режиссёр озвучки: языковая модель (та же, что для перевода) смотрит на оригинальную реплику — английский
текст, эмоцию, громкость, темп, паузы, смех/вздохи, соседние реплики сцены — и готовит русскую строку
к произнесению: те же слова, но пунктуация под подачу («Убирайся!», «Волк… постой.», «Что?!»), плюс короткое
описание подачи для контролёра и пользователя.

Модель речи читает только текст, поэтому пунктуация — главный рычаг её интонации. Слова не меняются никогда:
ответ проверяется, и если слова разошлись с исходной строкой, остаётся исходная."""
import json
import re

SYSTEM = (
    "Ты режиссёр русского дубляжа видеоигр. Для каждой реплики тебе дают оригинал (английский текст и что слышно "
    "в записи: эмоция, громкость, темп, паузы, смех/вздохи) и русскую строку для озвучки. Подготовь русскую строку "
    "к произнесению так, чтобы актёр сыграл как в оригинале.\n"
    "Правила:\n"
    "- НЕЛЬЗЯ менять, добавлять, удалять или переставлять слова. Можно менять только знаки препинания: "
    "«!», «?», «?!», «…», «—», запятые, точки, и заменять «е» на «ё».\n"
    "- Восклицание — для крика, злости, приказа, сильного удивления. «…» — для колебания, усталости, грусти, "
    "недосказанности, паузы внутри фразы. «—» — для резкой паузы или перебивки.\n"
    "- Если в оригинале подача ровная — оставь строку как есть.\n"
    "- intensity: 0 — спокойно, 1 — живо, 2 — эмоционально, 3 — крик/рыдание/ярость.\n"
    "- note: 3–8 слов по-русски, как играть (например «тихо, с упрёком», «срываясь на крик»).\n"
    'Ответ строго JSON: {"r": [{"id": <id>, "text": "<строка>", "intensity": 0-3, "note": "<как играть>"}]} '
    "для каждого id."
)

FIX_SYSTEM = (
    "Ты режиссёр русского дубляжа. Нейросеть уже озвучила реплику, контролёр нашёл проблемы (даны измерения). "
    "Предложи новую подачу. Правила для текста те же: слова не менять, только пунктуация и «ё». "
    'Ответ строго JSON: {"text": "<строка>", "intensity": 0-3, "note": "<как играть, 3–8 слов>", '
    '"temperature": <0.6..1.0>, "speed": <0.9..1.1>}.'
)

_WORD = re.compile(r"[а-яёa-z0-9]+", re.I)


def _words(s: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(s or "")]


def same_words(a: str, b: str) -> bool:
    return _words(a) == _words(b)


def describe_orig(style: dict | None, events: list | None = None) -> str:
    """Что слышно в оригинале — коротко, для подсказки модели."""
    st = style or {}
    parts = []
    emo = st.get("emotion")
    if emo and emo != "neutral":
        parts.append(f"emotion {emo} ({st.get('emotion_conf', 0):.0%})")
    loud = st.get("loud")
    if loud is not None:
        if loud >= 6:
            parts.append("loud, almost shouting")
        elif loud <= -8:
            parts.append("very quiet, whisper")
    for t in st.get("tags") or []:
        parts.append({"fast": "fast", "slow": "slow", "pauses": "noticeable pauses", "expressive": "very expressive",
                      "flat": "flat, monotone"}.get(t, t))
    if events:
        parts.append("non-speech: " + ", ".join(events))
    return "; ".join(parts) or "neutral delivery"


def direct(tr, items: list[dict], game: str = "", cards: dict | None = None) -> dict[str, dict]:
    """items: [{"id", "speaker", "en", "ru", "heard", "prev"}] → {id: {"text", "intensity", "note"}}.
    tr — запущенный translate.Translator. Строки с изменёнными словами отбрасываются."""
    num = {str(i): it for i, it in enumerate(items, 1)}
    payload = {"game": game, "characters": cards or {},
               "lines": [{"id": int(k), "speaker": it.get("speaker") or "", "original_en": it.get("en") or "",
                          "heard_in_original": it.get("heard") or "", "previous_lines": it.get("prev") or [],
                          "russian": it["ru"]} for k, it in num.items()]}
    out: dict[str, dict] = {}
    try:
        raw = tr._chat(json.dumps(payload, ensure_ascii=False), SYSTEM)
        data = json.loads(re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip())
        for r in data.get("r", []):
            it = num.get(str(r.get("id")))
            text = (r.get("text") or "").strip()
            if not it or not text:
                continue
            if not same_words(text, it["ru"]):
                text = it["ru"]  # модель тронула слова — подачу берём, текст нет
            out[it["id"]] = {"text": text, "intensity": int(r.get("intensity") or 0),
                             "note": str(r.get("note") or "")[:80]}
    except Exception as e:
        print(f"режиссёр: {e}")
    return out


def fix(tr, line: dict, diagnosis: str, current: dict | None = None) -> dict | None:
    """Новая подача для проблемной реплики по диагнозу контролёра."""
    payload = {"original_en": line.get("en") or "", "russian": line["ru"],
               "current_delivery": current or {}, "problems": diagnosis}
    try:
        raw = tr._chat(json.dumps(payload, ensure_ascii=False), FIX_SYSTEM)
        d = json.loads(re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip())
        text = (d.get("text") or "").strip() or line["ru"]
        if not same_words(text, line["ru"]):
            text = (current or {}).get("text") or line["ru"]
        return {"text": text, "intensity": int(d.get("intensity") or 0), "note": str(d.get("note") or "")[:80],
                "temperature": min(1.0, max(0.55, float(d.get("temperature") or 0.8))),
                "speed": min(1.12, max(0.88, float(d.get("speed") or 1.0)))}
    except Exception as e:
        print(f"режиссёр (доработка): {e}")
        return None
