"""Перевод реплик локальной языковой моделью (llama.cpp, Vulkan — работает на NVIDIA и AMD).

Нужен только реплкам без официальных русских субтитров (если это включено в настройках) или в режиме
«перевести всё». Переводчику даются предыдущие реплики диалога и карточки персонажей (пол, возраст,
манера, ты/вы), чтобы согласовать род и обращение."""
import json
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from . import models
from .paths import PROJECT_DIR

LLAMA_DIR = PROJECT_DIR / "tools" / "llama.cpp"
# ключ: (модель, нужно ГБ видеопамяти) — по убыванию: берётся самая сильная, что скачана и помещается
MODELS = {"qwen3-32b-q6": ("llm-32b-q6", 29), "qwen3-32b": ("llm-32b", 21), "qwen3-14b": ("llm-14b", 13),
          "qwen3-8b": ("llm-8b", 7), "qwen3-4b": ("llm-4b", 0)}
SYSTEM = (
    "You are a professional video game localizer. Translate English voice-over lines into natural, "
    "spoken Russian for dubbing. Rules:\n"
    "- Keep meaning, tone, emotion and the speaker's manner of speech.\n"
    "- The Russian line will be spoken in the same time as the original, so keep it about as short as "
    "the English (similar number of syllables). Prefer concise wording.\n"
    "- Use the glossary for names and terms exactly. Transliterate other names consistently "
    "(Breen → Брин, Alyx → Аликс).\n"
    "- Keep model names, Roman numerals and acronyms as in the original (Mark V → Mark V, HEV → HEV).\n"
    "- \"characters\" describes each speaker: gender (verbs and adjectives in the past tense must agree with "
    "the speaker's gender), age, manner of speech and whether they address others with ты or вы. Follow it.\n"
    "- \"previous_lines\" are the earlier lines of the same conversation (with their Russian where known) — "
    "context only, do not translate them; keep the pronouns, ты/вы and terms consistent with them.\n"
    "- Be consistent with ты/вы inside a line; companions and friends use ты.\n"
    "- Keep interjections short (Oh → О, Hey → Эй). Do not add explanations.\n"
    'Return JSON: {"t": [{"id": <id>, "ru": "<translation>"}]} with every id from the input.'
)
CARDS_SYSTEM = (
    "You analyze video game characters from samples of their English voice lines. For each character "
    "return a short card for a Russian translator: gender (male/female/unknown), approximate age "
    "(child/young/adult/old), manner of speech in 3-6 words (e.g. 'formal, calm, military commands'), "
    "and how they address others in Russian (ты or вы). "
    'Return JSON: {"c": [{"name": "<name>", "gender": "...", "age": "...", "manner": "...", "address": "ты|вы"}]}.'
)


def _model_path(key: str) -> Path | None:
    return models.path(MODELS[key][0])


def pick_model(vram_gb: float | None) -> str:
    for key, (_, need) in MODELS.items():
        if (vram_gb or 0) >= need and _model_path(key):
            return key
    return next((k for k in MODELS if _model_path(k)), "qwen3-4b")


class Translator:
    def __init__(self, model_key: str, gpu: bool = True):
        self.model_key = model_key
        path = _model_path(model_key)
        if not path:
            raise RuntimeError(f"Модель перевода {model_key} не скачана")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        from . import power
        args = [str(LLAMA_DIR / "llama-server.exe"), "-m", str(path), "--port", str(self.port),
                "--host", "127.0.0.1", "-c", "8192", "--jinja", "-ngl", "999" if gpu else "0", "--no-webui",
                "-t", str(power.threads())]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        power.apply_priority(self.proc.pid)
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("Сервер перевода (llama.cpp) не запустился")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.5)
        self.close()
        raise RuntimeError("Сервер перевода не ответил за 3 минуты")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _chat(self, user: str, system: str = SYSTEM, think: bool = False) -> str:
        """think — дать модели подумать (вопросы на знание, а не на перевод): медленнее, точнее; JSON
        тогда не навязывается грамматикой (она не даёт думать), его достаём из ответа."""
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.3, "top_p": 0.9, "max_tokens": 6144 if think else 4096,
                "chat_template_kwargs": {"enable_thinking": think}}
        if not think:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def translate(self, items: list[dict], glossary: dict[str, str] | None = None, game: str = "",
                  context: list[dict] | None = None, cards: dict[str, str] | None = None) -> dict[str, str]:
        """items: [{"id", "speaker", "text"}] → {id: перевод}. context — предыдущие реплики диалога
        [{"speaker", "text", "ru"}], cards — карточки персонажей {имя: описание}.
        Недостающие переводит повторно по одному."""
        numbered = {str(i): it for i, it in enumerate(items, 1)}
        payload = {"game": game, "glossary": glossary or {}, "characters": cards or {},
                   "previous_lines": context or [],
                   "lines": [{"id": int(k), "speaker": it.get("speaker") or "", "text": it["text"]}
                             for k, it in numbered.items()]}
        result: dict[str, str] = {}
        try:
            data = json.loads(_strip_think(self._chat(json.dumps(payload, ensure_ascii=False))))
            for t in data.get("t", []):
                k = str(t.get("id"))
                if k in numbered and isinstance(t.get("ru"), str) and t["ru"].strip():
                    result[numbered[k]["id"]] = t["ru"].strip()
        except Exception:
            pass
        missing = [it for it in items if it["id"] not in result]
        if missing and len(items) > 1:
            for it in missing:
                result.update(self.translate([it], glossary, game, context, cards))
        return result

    def describe_characters(self, chars: list[dict], game: str = "") -> dict[str, str]:
        """chars: [{"name", "lines": [...]}] → {имя: карточка одной строкой}."""
        payload = {"game": game, "characters": [{"name": c["name"], "lines": c["lines"]} for c in chars]}
        out = {}
        try:
            data = json.loads(_strip_think(self._chat(json.dumps(payload, ensure_ascii=False), CARDS_SYSTEM)))
            for c in data.get("c", []):
                name = str(c.get("name") or "")
                if name:
                    out[name] = ", ".join(str(c.get(k) or "") for k in ("gender", "age", "manner") if c.get(k)) \
                        + (f"; addresses others with {c['address']}" if c.get("address") else "")
        except Exception as e:
            print(f"карточки персонажей: {e}")
        return out


def _strip_think(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


def parse_glossary(text: str) -> dict[str, str]:
    """'Alyx = Аликс' по строкам."""
    out = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out
