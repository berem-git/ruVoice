"""Конвейер озвучки: извлечь → русский текст → обучить голоса → пробы → озвучить → проверить → (одобрить) → установить.

Все шаги выполняются в одном фоновом потоке по очереди: нейросети делят одну видеокарту.
Ничего не записывается в игру, пока пользователь не одобрил реплики и не нажал «Установить».

Один способ озвучки: речь читает русское дообучение Qwen3-TTS — голос снимается отпечатком прямо с записей
персонажа (без подражания английской речи, чтобы не было акцента), под эмоцию оригинала берётся образец из
его же реплик с такой эмоцией, ударения расставляет accent.py. По желанию поверх — RVC-модель, обученная на
записях персонажа (voice_transfer). Русский текст — из официальных субтитров;
реплики без субтитров остаются в оригинале (перевод нейросетью — только если включить в настройках).
Контролёр (judge.py) оценивает каждую реплику; плохие переозвучиваются с большим числом вариантов.
"""
import os
import queue
import random
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import adapters, audio
from .engines import available_tts, rvc_available
from .director import same_words as director_same
from .judge import DRIFT_MARGIN
from .paths import DATA_DIR
from .project import Project

DEFAULT_SETTINGS = {
    "speech_engine": "auto",    # capability-based backend selection; no game-dependent branches
    "preserve_performance": True,
    "effects_mode": "unknown",  # unknown / engine / dry / baked; explicit acoustic routing
    "text_source": "official",   # official — официальные русские субтитры; llm — перевести всё нейросетью
    "translate_missing": False,  # реплики без русских субтитров: False — оставить оригинальный голос, True — перевести
    "glossary": "",
    "translator": "auto",        # auto | qwen3-14b | qwen3-8b | qwen3-4b
    "quality": "normal",         # fast — 1 вариант; normal — 3; thorough — 5 (контролёр выбирает лучший)
    "stable_voice": True,        # один отпечаток на персонажа, строгий тембр, выбор варианта по сходству с его репликами
    "director": True,            # режиссёр (языковая модель) готовит подачу каждой реплики по оригиналу
    "emotion_refs": False,       # образец голоса под эмоцию из реплик персонажа (живее, но голос «плавает»)
    "judge": True,               # проверять реплики контролёром
    "autofix": True,             # после контролёра переозвучить проблемные реплики с большим числом вариантов
    "max_speedup": 1.1,          # русская фраза длиннее — ускоряем не более чем на 10 %, остальное остаётся длиннее
    "train_limit": 8,            # главных персонажей — полное обучение
    "train_minutes": 12,         # минут записей на главного персонажа
    "train_epochs": 300,         # эпох обучения RVC на главного персонажа
    "train_min_seconds": 120,    # остальные персонажи с таким запасом записей — короткое обучение
    "minor_minutes": 5,
    "minor_epochs": 150,
    "auto_install": False,       # после подготовки сразу озвучить всю игру и установить без прослушивания
    "voice_transfer": False,     # дополнительно обучать RVC-голоса и переносить тембр (редко нужно)
    "ft_level": "medium",        # дообучение модели речи под голос главных персонажей: off / light / medium / deep
                                 # (FT_LEVELS). Дообученный голос проверяется на репликах вне обучения и отключается,
                                 # если звучит хуже отпечатка (текст, высота голоса, похожесть) — хуже не станет
    "tts_finetune": True,        # выводится из ft_level (оставлено для старых проектов)
    "ft_min_seconds": 150,       # дообучаются персонажи с таким запасом речи
    "ft_minutes": 15,            # выводится из ft_level
    "ft_epochs": 4,              # выводится из ft_level
    "ft_method": "auto",         # auto — полное при 14 ГБ видеопамяти и больше, иначе LoRA; full; lora
    "event_gap": 0.3,            # пауза (с) из оригинала после смеха/вздоха перед русской речью
}
SETTINGS_VERSION = 2             # 2: перевод реплик без субтитров по умолчанию выключен
# Уровни дообучения: минут записей и эпох на персонажа, сколько главных персонажей обучать
FT_LEVELS = {
    "off": None,
    "light": {"name": "Быстрое", "minutes": 8, "epochs": 3, "limit": 4},
    "medium": {"name": "Среднее", "minutes": 15, "epochs": 4, "limit": 8},
    "deep": {"name": "Долгое", "minutes": 25, "epochs": 6, "limit": 12},
}
FT_SEC_PER_MIN_EPOCH = 8.0       # секунд обучения на минуту записей за эпоху (RTX 5070 Ti, полное, пачка 1)
FT_OVERHEAD_SEC = 120            # подбор записей, коды речи, проверка на контрольных репликах
FT_DISK_GB = {"full": 4.0, "lora": 0.05}
CANDIDATES = {"fast": 1, "normal": 3, "thorough": 5}
FIX_EXTRA = 3                    # сколько вариантов добавлять при переозвучке проблемной реплики
VOICE_VARIANTS = ["voxcpm", "ru", "custom"]
# Эмоция оригинала → инструкция для CustomVoice (модель понимает естественный язык)
EMOTION_INSTRUCT = {
    "angry": "Speak angrily and sharply, with force and tension in the voice.",
    "happy": "Speak cheerfully and warmly, with a smile in the voice.",
    "sad": "Speak sadly and quietly, with a heavy heart.",
    "fearful": "Speak anxiously, with a trembling, hurried voice.",
    "surprised": "Speak with surprise and excitement.",
    "disgust": "Speak with contempt and disgust.",
    "calm": "Speak calmly and softly.",
    "neutral": "",
}
SAMPLE_CHARS, SAMPLE_LINES = 5, 3
TR_BATCH = 16
TR_CONTEXT = 4                   # предыдущих реплик диалога в подсказке переводчику
PROFILE_LONG = 30.0


JOB_FILE = DATA_DIR / "job.json"   # идущая задача; остаётся, только если процесс умер посреди работы
MAX_RESUMES = 3                    # столько раз подряд продолжаем одну и ту же прерванную задачу


class DubJobs:
    def __init__(self, lib):
        self.lib = lib
        self.q: queue.Queue = queue.Queue()
        self.state: dict[str, dict] = {}   # game_id → {task, message, done, total, error, queued}
        self.cancel_flag = threading.Event()
        self.current: str | None = None
        self._projects: dict[str, Project] = {}
        self._translator = None
        self._resumes = 0
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self.collect_garbage, daemon=True).start()  # уборка при запуске
        threading.Thread(target=self._watch_resources, daemon=True).start()

    # --- прерванная задача: продолжить после падения или выключения ---

    def _mark_job(self, gid: str, task: str, kw: dict):
        import json
        try:
            JOB_FILE.write_text(json.dumps({"gid": gid, "task": task, "kw": kw, "at": time.time(), "pid": os.getpid(),
                                            "resumes": self._resumes}, ensure_ascii=False), "utf-8")
        except (OSError, TypeError) as e:
            print(f"job.json: {e}")

    def clear_job(self):
        try:
            JOB_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    def resume_interrupted(self) -> str | None:
        """При запуске: если прошлый процесс умер посреди задачи (остался job.json) — продолжить её.
        Готовое не теряется: озвученные реплики, обученные голоса и подача режиссёра уже сохранены,
        задача продолжает с того места. Больше MAX_RESUMES раз подряд не продолжаем (значит, падает всегда)."""
        import json
        if not JOB_FILE.exists():
            return None
        try:
            job = json.loads(JOB_FILE.read_text("utf-8"))
        except Exception:
            self.clear_job()
            return None
        if job.get("pid") == os.getpid():
            return None
        gid, task = job.get("gid"), job.get("task")
        if not gid or not task or not hasattr(self, f"_do_{task}") or not self.lib.get(gid):
            self.clear_job()
            return None
        n = int(job.get("resumes") or 0) + 1
        if n > MAX_RESUMES:
            print(f"прерванная задача {task} ({gid}) падала {n - 1} раз подряд — больше не продолжаю")
            self.clear_job()
            return None
        self._resumes = n
        when = time.strftime("%H:%M", time.localtime(job.get("at") or time.time()))
        print(f"продолжаю прерванную задачу {task} ({gid}), начатую в {when}; попытка {n}")
        self.enqueue(gid, task, **(job.get("kw") or {}))
        self.state[gid]["message"] = f"Продолжаю прерванную работу (началась в {when})…"
        return task

    def _watch_resources(self):
        """Во время работы — раз в 10 минут память процесса и видеопамять в журнал: если процесс снова
        исчезнет, по последним строкам видно, не кончалась ли память."""
        import sys as _sys
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            return
        while True:
            time.sleep(600)
            if not self.current:
                continue
            try:
                vm = psutil.virtual_memory()
                line = (f"ресурсы: задача {self.state.get(self.current, {}).get('task')}, ОЗУ процесса "
                        f"{proc.memory_info().rss / 2**30:.1f} ГБ, свободно ОЗУ {vm.available / 2**30:.1f} из {vm.total / 2**30:.0f} ГБ")
                torch = _sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    line += f", видеопамять занята {(total - free) / 2**30:.1f} из {total / 2**30:.0f} ГБ"
                print(line, flush=True)
            except Exception:
                pass

    def collect_garbage(self, dry: bool = False) -> dict:
        """Убрать мусор во всех проектах (кроме того, что сейчас в работе) и в кэше; см. cleanup.py."""
        from . import cleanup
        from .project import project_id
        busy = {project_id(self.current)} if self.current else set()
        try:
            res = cleanup.all_garbage(dry=dry, skip=busy)
        except Exception as e:
            print(f"уборка мусора: {e}")
            return {"bytes": 0, "projects": {}, "fixed_lines": 0}
        if not dry and (res["bytes"] or res["fixed_lines"]):
            print(f"уборка мусора: освобождено {res['bytes'] / 2**30:.2f} ГБ, исправлено реплик без файла: {res['fixed_lines']}")
        return res

    # --- публичное ---

    def project(self, gid: str) -> Project:
        if gid not in self._projects:
            self._projects[gid] = Project(gid)
        return self._projects[gid]

    def settings(self, gid: str) -> dict:
        stored = dict(self.project(gid).get("settings") or {})
        if stored.get("_v", 0) < SETTINGS_VERSION:  # проект старой версии: убрать настройки, чей смысл изменился
            stored = {k: v for k, v in stored.items() if k in DEFAULT_SETTINGS and k != "translate_missing"}
            stored["_v"] = SETTINGS_VERSION
            self.project(gid).set("settings", stored)
        base = dict(DEFAULT_SETTINGS)
        if "quality" not in stored and self._big_gpu():
            base["quality"] = "thorough"  # 24 ГБ и больше: 5 вариантов реплики помещаются одной пачкой
        s = {**base, **{k: v for k, v in stored.items() if k in DEFAULT_SETTINGS}}
        # уровень дообучения задаёт минуты, эпохи и число персонажей (старое «tts_finetune» не учитываем:
        # раньше оно было выключено по умолчанию, а не выбрано пользователем)
        lvl = FT_LEVELS.get(s.get("ft_level")) if s.get("ft_level") in FT_LEVELS else FT_LEVELS["medium"]
        s["tts_finetune"] = lvl is not None
        if lvl:
            s["ft_minutes"], s["ft_epochs"], s["ft_limit"] = lvl["minutes"], lvl["epochs"], lvl["limit"]
        return s

    _big: bool | None = None

    def _big_gpu(self) -> bool:
        """Видеокарта от 24 ГБ (один раз за запуск: accel.info() опрашивает драйвер и импортирует torch)."""
        if DubJobs._big is None:
            from . import accel, models
            try:
                DubJobs._big = (accel.info().get("vram_gb") or 0) >= models.TIERS["ultra"]["vram"]
            except Exception:
                DubJobs._big = False
        return DubJobs._big

    def save_settings(self, gid: str, values: dict):
        s = self.settings(gid)
        s.update({k: v for k, v in values.items() if k in DEFAULT_SETTINGS})
        self.project(gid).set("settings", {**s, "_v": SETTINGS_VERSION})
        return s

    def enqueue(self, gid: str, task: str, **kw):
        st = self.state.setdefault(gid, {"task": None, "message": "", "done": 0, "total": 0, "error": None,
                                         "queued": []})
        st["queued"].append(task)
        st["error"] = None
        self.q.put((gid, task, kw))

    def cancel(self, gid: str):
        if self.current == gid:
            self.cancel_flag.set()
            # Не ждать конца долгого шага: убить процесс RVC (обучение) и переводчика —
            # текущая операция завершится ошибкой «остановлено», а уже готовые результаты сохранены
            from .engines import client
            child = getattr(self, "_child", None)
            if child and child.poll() is None:
                try:
                    child.kill()
                except Exception:
                    pass
            threading.Thread(target=lambda: (client.stop(force=True), self._release_translator()), daemon=True).start()
        kept = []
        while not self.q.empty():
            item = self.q.get_nowait()
            if item[0] != gid:
                kept.append(item)
        for item in kept:
            self.q.put(item)
        if gid in self.state:
            self.state[gid]["queued"] = []

    def status(self, gid: str) -> dict:
        st = self.state.get(gid) or {"task": None, "message": "", "done": 0, "total": 0, "error": None,
                                     "queued": []}
        return {**st, "running": self.current == gid}

    # --- поток ---

    def _run(self):
        while True:
            gid, task, kw = self.q.get()
            st = self.state[gid]
            if task in st["queued"]:
                st["queued"].remove(task)
            self.current, st["task"], st["done"], st["total"] = gid, task, 0, 0
            self.cancel_flag.clear()
            self._mark_job(gid, task, kw)
            try:
                from . import power
                power.apply_priority()
                getattr(self, f"_do_{task}")(gid, **kw)
                if st.get("message") in ("", None) or st["task"] == task and not st.get("_keep_message"):
                    st["message"] = "Остановлено" if self.cancel_flag.is_set() else st.get("message") or "Готово"
            except Exception as e:
                if self.cancel_flag.is_set():
                    st["error"] = None
                    st["message"] = "Остановлено"
                else:
                    st["error"] = f"{e}"
                    st["message"] = "Ошибка"
                    traceback.print_exc()
            finally:
                st.pop("_keep_message", None)
                st["task"] = None
                self.current = None
                self._resumes = 0
                self.clear_job()  # задача закончилась сама, с ошибкой или остановлена — продолжать нечего
                if self.q.empty():
                    self._release_translator()

    def _progress(self, gid: str):
        st = self.state[gid]

        def report(message: str, done: int | None = None, total: int | None = None):
            st["message"] = message
            if done is not None:
                st["done"], st["total"] = done, total or 0
        return report

    def _finish(self, gid: str, message: str):
        self.state[gid]["message"] = message
        self.state[gid]["_keep_message"] = True

    def _game(self, gid: str) -> dict:
        g = self.lib.get(gid)
        if not g:
            raise RuntimeError("Игра не найдена в библиотеке")
        return g

    def _adapter(self, gid: str):
        p = self.project(gid)
        a = adapters.by_id(p.get("adapter")) if p.get("adapter") else adapters.for_game(self._game(gid))
        if not a:
            raise RuntimeError("Для этой игры пока нет способа извлечь озвучку")
        return a

    # --- выбор персонажей ---

    def selected(self, p: Project) -> set[str]:
        """Кого озвучивать — всех: единый порядок для всех игр, озвучивается всё, у чего есть русский текст
        (выбор персонажей из интерфейса убран). Старые проекты с выбором или флагом include — сбрасываем."""
        chars = p.characters()
        if p.get("selected_chars") or any(not c["include"] for c in chars):
            p.set("selected_chars", [])
            with p.db() as c:
                c.execute("UPDATE characters SET include=1")
        return {c["id"] for c in chars}

    def set_selected(self, p: Project, chars: list[str]) -> list[str]:
        ids = {c["id"] for c in p.characters()}
        sel = [c for c in dict.fromkeys(chars) if c in ids]
        p.set("selected_chars", sel)
        return sel

    def _in_selection(self, p: Project, lines: list[dict]) -> list[dict]:
        sel = self.selected(p)
        return [l for l in lines if l["char"] in sel]

    # --- профили голоса ---

    def character_profile(self, p: Project, char: str, max_sec: float = PROFILE_LONG,
                          emotion: str | None = None) -> dict | None:
        """Склейка чистых реплик персонажа (~30 с) — по ней считается отпечаток голоса: образец для речи,
        похожесть у контролёра, обычная громкость персонажа. emotion — образец только из реплик с этой
        эмоцией (голос персонажа «в гневе», «в печали»…); None, если таких реплик мало.
        Кэшируется в projects/<id>/profiles."""
        key = f"{char}_{int(max_sec)}" + (f"_{emotion}" if emotion else "")
        cached = (p.get("profiles") or {}).get(key)
        if cached is not None and cached.get("audio") is None:
            return None
        if cached and (p.dir / cached["audio"]).exists():
            return {"audio": str(p.dir / cached["audio"]), "text": cached["text"], "seconds": cached["seconds"]}
        lines = [l for l in p.lines(char=char) if 2.0 <= (l["duration"] or 0) <= 9 and l["status"] != "skipped"]
        if not lines:
            lines = [l for l in p.lines(char=char) if 1.5 <= (l["duration"] or 0) <= 12]
        if emotion:
            emos = self._emotions_for_char(p, char)
            lines = [l for l in lines if emos.get(l["id"], ("neutral", 0))[0] == emotion
                     and emos[l["id"]][1] >= 0.45]
            if len(lines) < 3 or sum(l["duration"] for l in lines) < 8:
                profiles = p.get("profiles") or {}
                profiles[key] = {"audio": None}
                p.set("profiles", profiles)
                return None
            max_sec = min(max_sec, 20.0)
        if not lines:
            return None
        random.Random(42).shuffle(lines)
        lines = lines[:60]
        lines.sort(key=lambda l: -min(l["duration"], 7))  # предпочитаем средние/длинные, но не монологи
        lines = self._most_typical(p, lines)
        parts, texts, total, sr = [], [], 0.0, None
        for l in lines:
            w, lsr = audio.read(p.dir / l["orig"])
            if sr is None:
                sr = lsr
            w = audio.resample(audio.trim_silence(w, lsr), lsr, sr)
            if np.max(np.abs(w)) < 0.02:
                continue
            parts += [w, np.zeros(int(sr * 0.3), dtype=np.float32)]
            texts.append((l["en"] or "").strip())
            total += len(w) / sr + 0.3
            if total >= max_sec:
                break
        if not parts:
            return None
        out = p.dir / "profiles" / f"{_safe(key)}.wav"
        audio.write(out, np.concatenate(parts), sr)
        text = " ".join(t for t in texts if t)
        profiles = p.get("profiles") or {}
        profiles[key] = {"audio": str(out.relative_to(p.dir)), "text": text, "seconds": round(total, 1)}
        p.set("profiles", profiles)
        return {"audio": str(out), "text": text, "seconds": round(total, 1)}

    def _emotions_for_char(self, p: Project, char: str) -> dict[str, tuple[str, float]]:
        """Эмоция каждой реплики персонажа (классификатор на оригинале), кэш в проекте."""
        if not self._judge_ready_quiet():
            return {}
        cache = p.get("emotions") or {}
        lines = [l for l in p.lines(char=char) if 1.0 <= (l["duration"] or 0) <= 12 and l["status"] != "skipped"]
        todo = [l for l in lines if l["id"] not in cache]
        if todo:
            from . import judge
            for l in todo[:600]:
                try:
                    emo, conf = judge.emotion(str(p.dir / l["orig"]))
                except Exception:
                    emo, conf = "neutral", 0.0
                cache[l["id"]] = [emo, round(conf, 2)]
            p.set("emotions", cache)
        return {k: (v[0], v[1]) for k, v in cache.items()}

    @staticmethod
    def _most_typical(p: Project, lines: list[dict]) -> list[dict]:
        """Отсортировать реплики по близости к «среднему» голосу персонажа: так в профиль и в обучение
        не попадают реплики через рацию, крики и записи с шумом. Без модели похожести — порядок не меняется."""
        from . import judge
        if not judge.VE_WEIGHTS.exists() or len(lines) < 4:
            return lines
        try:
            import torch
            from .thirdparty.voice_encoder import VoiceEncoder
            wavs = []
            for l in lines:
                w, sr = audio.read(p.dir / l["orig"])
                wavs.append(audio.resample(w, sr, 16000))
            with judge._lock, torch.no_grad():
                embs = judge._load_ve().embeds_from_wavs(wavs, sample_rate=16000, as_spk=False)
            centroid = VoiceEncoder.utt_to_spk_embed(embs)
            scored = sorted(zip(lines, embs), key=lambda x: -float(np.dot(x[1], centroid)))
            return [l for l, _ in scored]
        except Exception as e:
            print(f"профиль голоса: не удалось оценить реплики ({e})")
            return lines

    def _refs_for(self, p: Project, engine: str, chars: set[str]) -> dict[str, dict]:
        """Образец голоса для движка: ru — профиль персонажа (отпечаток снимается с его записей);
        custom — ближайший готовый диктор."""
        if engine == "voxcpm":
            refs = {}
            for char in chars:
                prof = self.character_profile(p, char)
                if prof:
                    refs[char] = {"audio": prof["audio"], "text": prof["text"], "speaker": None}
            return refs
        if engine == "ru":
            refs = {}
            use_ft = {**DEFAULT_SETTINGS, **(p.get("settings") or {})}["tts_finetune"]
            for c in chars:
                tv = self._tts_voice(p, c) if use_ft else None
                if tv and tv.get("kind") == "full":  # своя дообученная модель: диктор = персонаж
                    refs[c] = {"audio": tv["ref"], "text": None, "speaker": tv["speaker"], "ckpt": tv["dir"], "name": _safe(c)}
                    continue
                if tv:  # LoRA-адаптер поверх базовой модели: тот же образец, что при обучении
                    refs[c] = {"audio": tv["ref"], "text": None, "speaker": None, "voice": tv["dir"], "name": _safe(c)}
                    continue
                prof = self.character_profile(p, c)
                if prof:
                    refs[c] = {"audio": prof["audio"], "text": prof["text"], "speaker": None}
            return refs
        return {c: {"speaker": self._custom_speaker_for(p, c)} for c in chars}

    def _ref_for_line(self, p: Project, line: dict, engine: str, ref: dict, style: dict) -> dict:
        """Для ru: если у оригинала выраженная эмоция и у персонажа есть реплики с такой же — образец из них
        (голос «в гневе» звучит иначе, чем спокойный)."""
        emo = (style or {}).get("emotion")
        if not {**DEFAULT_SETTINGS, **(p.get("settings") or {})}["emotion_refs"]:
            return ref  # один отпечаток на персонажа: голос не «плавает» от реплики к реплике
        if engine == "ru" and not ref.get("voice") and not ref.get("ckpt") and emo and emo not in ("neutral", "calm"):
            ep = self.character_profile(p, line["char"], emotion=emo)
            if ep:
                return {"audio": ep["audio"], "text": ep["text"], "speaker": None}
        return ref

    SPEAKER_PICK_VERSION = 3  # 3: высота голоса через pyin; 2: сначала высота голоса, потом отпечаток (раньше мужчинам доставался женский диктор)

    def _rvc_params(self, p: Project, char: str, engine: str, speaker: str) -> dict:
        """Параметры переноса тембра: постоянный сдвиг высоты от диктора к персонажу (полутоны), без
        автоподгонки каждой реплики к 155 Гц — она ломала женские голоса и стирала разницу между спокойной
        и взволнованной подачей."""
        if engine == "ru":  # речь уже в высоте голоса персонажа (образец — его записи)
            return {"pitch": 0, "auto_pitch": False, "index_rate": 0.0, "protect": 0.5}
        cache = p.get("pitch_shift") or {}
        key = f"{engine}:{speaker}:{char}"
        if key not in cache:
            from . import basevoices
            prof = self.character_profile(p, char)
            clip = (basevoices.CLIPS_DIR / "custom" / f"{speaker}.wav") if engine == "custom" \
                else Path(basevoices.reference_clip(speaker, False)["audio"])
            shift = 0
            try:
                tw, tsr = audio.read(prof["audio"])
                sw, ssr = audio.read(clip)
                tf, sf_ = audio.f0_median(tw, tsr), audio.f0_median(sw, ssr)
                if tf and sf_:
                    shift = int(round(max(-12, min(12, 12 * np.log2(tf / sf_)))))
                print(f"сдвиг высоты для «{char}»: {shift} пт (персонаж {tf:.0f} Гц, диктор {speaker} {sf_:.0f} Гц)")
            except Exception as e:
                print(f"сдвиг высоты для «{char}» не вычислен: {e}")
            cache[key] = shift
            p.set("pitch_shift", cache)
        return {"pitch": cache[key], "auto_pitch": False, "index_rate": 0.0, "protect": 0.5}

    def _custom_speaker_for(self, p: Project, char: str) -> str:
        """Готовый диктор CustomVoice, ближайший к голосу персонажа: сначала по высоте голоса, затем по отпечатку."""
        from . import basevoices, judge, tts
        if p.get("speaker_pick_v") != self.SPEAKER_PICK_VERSION:  # выбор старой логикой — пересобрать
            p.set("custom_speakers", None)
            p.set("pitch_shift", None)
            p.set("speaker_pick_v", self.SPEAKER_PICK_VERSION)
        chosen = (p.get("custom_speakers") or {}).get(char)
        spks = tts.custom_speakers()
        if chosen in spks:
            return chosen
        clips_dir = basevoices.CLIPS_DIR / "custom"
        clips_dir.mkdir(parents=True, exist_ok=True)
        embs = {}
        for spk in spks:
            path = clips_dir / f"{spk}.wav"
            if not path.exists():
                (wav, sr), = tts.synthesize("custom", [basevoices.TEXT_SHORT], [{"speaker": spk}], [20.0], {})
                audio.write(path, audio.resample(wav, sr, 24000), 24000)
            embs[spk] = judge.speaker_embed([str(path)]) if judge.VE_WEIGHTS.exists() else None
        prof = self.character_profile(p, char)
        if prof and all(e is not None for e in embs.values()):
            # Сначала высота голоса: отпечаток VoiceEncoder почти не различает мужской и женский голос
            # Поэтому берём дикторов в пределах ±35 %
            # по высоте (если таких нет — трёх ближайших) и среди них самого похожего по отпечатку
            pe = judge.speaker_embed([prof["audio"]])
            pw, psr = audio.read(prof["audio"])
            target_f0 = audio.f0_median(pw, psr)
            f0s = {}
            for spk in spks:
                cw, csr = audio.read(clips_dir / f"{spk}.wav")
                f0s[spk] = audio.f0_median(cw, csr)
            pool = spks
            if target_f0 and all(f0s.values()):
                near = [sp for sp in spks if 0.65 <= f0s[sp] / target_f0 <= 1.35]
                pool = near or sorted(spks, key=lambda sp: abs(np.log(f0s[sp] / target_f0)))[:3]
            chosen = max(pool, key=lambda sp: float(np.dot(embs[sp], pe)))
            print(f"диктор для «{char}»: {chosen} (голос {target_f0:.0f} Гц, диктор {f0s.get(chosen, 0):.0f} Гц)")
        else:
            chosen = spks[0]
        cs = p.get("custom_speakers") or {}
        cs[char] = chosen
        p.set("custom_speakers", cs)
        return chosen

    def _engine_for(self, gid: str) -> str:
        avail = available_tts()
        chosen = self.settings(gid).get("speech_engine", "auto")
        if chosen != "auto":
            if chosen not in avail:
                raise RuntimeError(f"Движок {chosen} не готов: скачайте модель и установите его среду")
            return chosen
        for v in VOICE_VARIANTS:
            if v in avail:
                return v
        raise RuntimeError("Нет модели речи: скачайте CustomVoice в «Система и нейросети»")

    def _take(self, gid: str) -> int:
        """Номер прогона проб («переделать пробы» = другой прогон с другими случайными зёрнами)."""
        return int(self.project(gid).get("voice_variant") or 0)

    def variant_info(self, gid: str) -> dict:
        return {"index": self._take(gid) + 1, "engine": self._engine_for(gid) if available_tts() else None}

    def train_plan(self, gid: str) -> dict:
        """Кого обучаем автоматически: главные персонажи — полное обучение, остальные с достаточным
        числом записей — короткое; совсем короткие роли говорят голосом подобранного диктора."""
        p, s = self.project(gid), self.settings(gid)
        from . import accel
        if self._engine_for(gid) != "ru" and not s["voice_transfer"]:
            return {"planned": [], "trained": 0, "total": 0, "main": 0, "cpu": False,
                    "kind": "reference", "off": True, "skipped": []}
        if s["tts_finetune"] and not s["voice_transfer"]:
            min_sec = float(s["ft_min_seconds"])
            sel = self.selected(p)
            chars = [c for c in p.characters() if c["id"] in sel and (c["seconds"] or 0) >= min_sec]
            chars.sort(key=lambda c: -(c["lines"] or 0))
            chars = chars[: int(s.get("ft_limit") or 8)]  # главные персонажи; остальные — по отпечатку голоса
            planned = [{"id": c["id"], "name": c["name"], "minutes": round((c["seconds"] or 0) / 60, 1),
                        "trained": c["id"] in (p.get("tts_voices") or {}), "tier": "tts",
                        "epochs": int(s["ft_epochs"]), "max_seconds": float(s["ft_minutes"]) * 60} for c in chars]
            return {"planned": planned, "trained": len([c for c in planned if c["trained"]]), "total": len(planned),
                    "main": len(planned), "cpu": accel.device() != "cuda", "kind": "tts",
                    "skipped": [c["name"] for c in p.characters() if c["id"] in sel and (c["seconds"] or 0) < min_sec][:12]}
        if not s["voice_transfer"]:
            return {"planned": [], "trained": 0, "total": 0, "main": 0, "cpu": False, "skipped": [], "off": True}
        min_sec = float(s["train_min_seconds"])
        sel = self.selected(p)
        chars = [c for c in p.characters() if c["id"] in sel and (c["seconds"] or 0) >= min_sec]
        chars.sort(key=lambda c: -(c["lines"] or 0))
        planned = []
        for i, c in enumerate(chars):
            main = i < int(s["train_limit"])
            planned.append({"id": c["id"], "name": c["name"], "minutes": round((c["seconds"] or 0) / 60, 1),
                            "trained": self._voice(p, c["id"]) is not None, "tier": "main" if main else "minor",
                            "epochs": int(s["train_epochs"] if main else s["minor_epochs"]),
                            "max_seconds": float(s["train_minutes"] if main else s["minor_minutes"]) * 60})
        return {"planned": planned, "trained": len([c for c in planned if c["trained"]]), "total": len(planned),
                "main": len([c for c in planned if c["tier"] == "main"]), "cpu": accel.device() != "cuda", "kind": "rvc",
                "skipped": [c["name"] for c in p.characters() if c["id"] in sel and (c["seconds"] or 0) < min_sec][:12]}

    # --- уровни дообучения: оценка времени и места ---

    def ft_setup(self) -> dict:
        """Как дообучать на этой видеокарте: полное (от 14 ГБ) или LoRA, пачка 2 на картах от 24 ГБ
        (накопление градиента подстраивается — итоговый шаг тот же, качество не меняется, быстрее)."""
        from . import accel
        try:
            vram = accel.info().get("vram_gb") or 0
        except Exception:
            vram = 0
        method = "full" if vram >= 14 else "lora"
        batch = 2 if vram >= 22 else 1
        return {"method": method, "batch": batch, "accum": max(1, 8 // batch), "vram": vram}

    def _ft_rate(self) -> float:
        """Секунд обучения на минуту записей за эпоху: по прошлым обучениям на этом компьютере, иначе по замеру."""
        import json
        f = DATA_DIR / "train_speed.json"
        try:
            hist = json.loads(f.read_text("utf-8"))
            vals = sorted(x["rate"] for x in hist[-10:] if x.get("rate"))
            if vals:
                return vals[len(vals) // 2]
        except Exception:
            pass
        return FT_SEC_PER_MIN_EPOCH

    def _ft_record(self, rate: float, method: str, batch: int):
        import json
        f = DATA_DIR / "train_speed.json"
        try:
            hist = json.loads(f.read_text("utf-8")) if f.exists() else []
        except Exception:
            hist = []
        hist = (hist + [{"rate": round(rate, 2), "method": method, "batch": batch, "at": time.time()}])[-20:]
        f.write_text(json.dumps(hist), "utf-8")

    def train_levels(self, gid: str) -> dict:
        """Для экрана студии: сколько персонажей, времени и места займёт каждый уровень дообучения."""
        p, s = self.project(gid), self.settings(gid)
        setup = getattr(self, "_ft_setup_cache", None) or self.ft_setup()
        self._ft_setup_cache = setup
        rate = self._ft_rate() * (0.65 if setup["batch"] > 1 else 1.0)
        min_sec = float(s["ft_min_seconds"])
        chars = sorted((c for c in p.characters() if (c["seconds"] or 0) >= min_sec), key=lambda c: -(c["lines"] or 0))
        trained = p.get("tts_voices") or {}
        out = {}
        for key, lvl in FT_LEVELS.items():
            if not lvl:
                out[key] = {"name": "Без обучения", "chars": 0, "minutes": 0, "disk_gb": 0}
                continue
            pick = chars[: lvl["limit"]]
            todo = [c for c in pick if c["id"] not in trained]
            sec = sum(min((c["seconds"] or 0) / 60, lvl["minutes"]) * lvl["epochs"] * rate + FT_OVERHEAD_SEC for c in todo)
            out[key] = {"name": lvl["name"], "chars": len(pick), "todo": len(todo), "minutes": round(sec / 60),
                        "per_char": round((lvl["minutes"] * lvl["epochs"] * rate + FT_OVERHEAD_SEC) / 60, 1),
                        "disk_gb": round(len(pick) * FT_DISK_GB[setup["method"]], 1),
                        "names": [c["name"] for c in pick][:6]}
        return {"levels": out, "current": s["ft_level"], "method": setup["method"], "batch": setup["batch"],
                "vram": setup["vram"]}

    # --- шаги ---

    def _do_extract(self, gid: str):
        game, p = self._game(gid), self.project(gid)
        a = adapters.for_game(game)
        if not a:
            raise RuntimeError("Для этой игры пока нет способа извлечь озвучку")
        p.set("adapter", a.id)
        p.set("game_name", game["name"])
        p.set("profiles", {})
        n = a.extract(p, game, self._progress(gid))
        if n == 0:
            p.set("extracted_at", None)
            raise RuntimeError("Реплик не найдено. Похоже, звук этой игры упакован в архивы движка — "
                               "такой формат пока не поддерживается")
        p.set("extracted_at", time.time())
        self._finish(gid, f"Найдено реплик: {n}")

    def _needs_llm(self, gid: str) -> bool:
        """Нужна ли языковая модель для текста: по настройке — или когда у игры нет русского текста вовсе
        (иначе озвучивать было бы нечего: «локализуем всё, что получится»)."""
        s = self.settings(gid)
        if s["text_source"] == "llm" or bool(s["translate_missing"]):
            return True
        st = self.project(gid).stats()
        return bool(st["total"]) and not st["with_official"]

    def _do_subtitles(self, gid: str):
        """Официальные русские субтитры к репликам без текста — для любой игры (subtitles.py).

        Пул пар «английский → русский» даёт адаптер (RE Engine — из .msg) или находит сам поиск по текстовым
        файлам игры. Английская речь реплик распознаётся (Whisper), и каждая получает строку, совпадающую со
        сказанным; распознанное сохраняется в info.sub.heard."""
        from . import asr, subtitles, tts
        p, report = self.project(gid), self._progress(gid)
        rows = subtitles.load_pool(p)
        if rows is None and not p.get("subtitles_scanned"):
            game = self.lib.get(gid) or {}
            root = Path(game.get("path") or "")
            if root.is_dir():
                report("Ищу субтитры в файлах игры…")
                found = subtitles.harvest(root, report)
                kept = subtitles.save_pool(p, found) if found else 0
                p.set("subtitles_scanned", {"at": time.time(), "rows": kept})
                rows = subtitles.load_pool(p) if kept else []
        if not rows:
            return
        todo = [l for l in p.lines() if not l["ru_official"] and l["status"] != "skipped"]
        if not todo:
            return
        need = [l for l in todo if not l["en"]]
        if need:
            tts.unload()
            from . import power
            for i in range(0, len(need), 16):
                if self.cancel_flag.is_set():
                    return
                chunk = need[i:i + 16]
                report(f"Субтитры: слушаю английскую речь {i}/{len(need)}", i, len(need))
                for l, t in zip(chunk, asr.transcribe([str(p.dir / l["orig"]) for l in chunk], language="english",
                                                      batch_size=power.batch(8))):
                    if len(t.strip(" .,!?-…")) < 2:
                        p.update_line(l["id"], status="skipped", en="", error="Нет речи")
                    else:
                        p.update_line(l["id"], en=t.strip())
            asr.unload()
        report("Субтитры: сопоставляю реплики со строками игры…")
        m = subtitles.Matcher(rows)
        matched = 0
        for l in p.lines(ids=[l["id"] for l in todo]):
            if not l["en"] or l["status"] == "skipped":
                continue
            row, score = m.match(l["en"])
            if row and score >= subtitles.MIN_SCORE:
                info = dict(l["info"] or {})
                info["sub"] = {"key": row["key"], "score": round(score, 3), "heard": l["en"]}
                p.update_line(l["id"], en=row["en"], ru_official=row["ru"], info=info)
                matched += 1
        print(f"субтитры: сопоставлено {matched} из {len(todo)} реплик (строк в пуле {len(rows)})")
        report(f"Субтитры: русский текст найден для {matched} из {len(todo)} реплик")

    def _do_transcribe(self, gid: str, force: bool = False):
        """Распознать оригинальную речь — нужно только переводчику. Когда реплики без субтитров остаются
        в оригинале, распознавание пропускается (на большой игре это часы работы)."""
        from . import asr, tts
        p, report = self.project(gid), self._progress(gid)
        if not force and not self._needs_llm(gid):
            return
        todo = self._in_selection(p, [l for l in p.lines() if not l["en"] and l["status"] != "skipped"])
        if not self.settings(gid)["text_source"] == "llm":
            todo = [l for l in todo if not l["ru_official"]]
        if not todo:
            return
        tts.unload()
        from . import power
        report("Загрузка модели распознавания речи…")
        for i in range(0, len(todo), 16):
            if self.cancel_flag.is_set():
                return
            chunk = todo[i:i + 16]
            report(f"Распознавание речи: {i}/{len(todo)}", i, len(todo))
            t0 = time.time()
            texts = asr.transcribe([str(p.dir / l["orig"]) for l in chunk], language=None,
                                   batch_size=power.batch(8))
            power.pause(time.time() - t0)
            for l, t in zip(chunk, texts):
                if len(t.strip(" .,!?-…")) < 2:
                    p.update_line(l["id"], status="skipped", en="", error="Нет речи")
                else:
                    p.update_line(l["id"], en=t)
        asr.unload()

    def _translator_for(self, gid: str):
        from . import accel, translate
        key = self.settings(gid)["translator"]
        if key == "auto":
            info = accel.info()
            key = translate.pick_model(info["vram_gb"] if info["device"] == "cuda" else 0)
        if self._translator and self._translator.model_key == key:
            return self._translator
        self._release_translator()
        self._translator = translate.Translator(key, gpu=True)
        return self._translator

    def _release_translator(self):
        if self._translator:
            self._translator.close()
            self._translator = None

    def _do_translate(self, gid: str, ids: list[str] | None = None, force: bool = False):
        from . import translate, tts
        p, report, s = self.project(gid), self._progress(gid), self.settings(gid)
        lines = p.lines(ids=ids) if ids else self._in_selection(p, p.lines())
        lines = [l for l in lines if l["status"] != "skipped" and (force or l["ru_source"] != "user")
                 and (force or ids or not l["ru"])]
        need_llm = []
        for l in lines:
            if _is_sfx_text(l["en"]) or (not l["en"] and _is_sfx_text(l["ru_official"])):
                p.update_line(l["id"], status="skipped", error="Ремарка/крик, а не речь — остаётся оригинал")
            elif s["text_source"] == "official" and l["ru_official"]:
                p.update_line(l["id"], ru=_clean_ru(l["ru_official"]), ru_source="official",
                              status="translated" if l["status"] in ("new", "translated", "error") else l["status"])
            elif l["en"] and self._needs_llm(gid):
                need_llm.append(l)
            elif not l["ru"]:
                p.update_line(l["id"], status="skipped",
                              error="Нет русских субтитров — остаётся оригинальный голос")
        if not need_llm or self.cancel_flag.is_set():
            return
        tts.unload()
        report("Запуск модели перевода…")
        tr = self._translator_for(gid)
        names = {c["id"]: c["name"] for c in p.characters()}
        glossary = translate.parse_glossary(s["glossary"])
        cards = self._character_cards(p, tr, {l["char"] for l in need_llm}, names, report)
        # Реплики в порядке файлов игры ≈ порядок диалога: переводчик видит предыдущие реплики
        order = sorted((l for l in p.lines() if l["status"] != "skipped"), key=lambda l: l["id"])
        pos = {l["id"]: i for i, l in enumerate(order)}
        ru_now = {l["id"]: l["ru"] for l in order if l["ru"]}
        need_llm.sort(key=lambda l: l["id"])
        for i in range(0, len(need_llm), TR_BATCH):
            if self.cancel_flag.is_set():
                return
            chunk = need_llm[i:i + TR_BATCH]
            report(f"Перевод: {i}/{len(need_llm)}", i, len(need_llm))
            first = pos.get(chunk[0]["id"], 0)
            context = [{"speaker": names.get(l["char"], l["char"]), "text": l["en"] or "", "ru": ru_now.get(l["id"], "")}
                       for l in order[max(0, first - TR_CONTEXT):first] if l["en"] or ru_now.get(l["id"])]
            t0 = time.time()
            result = tr.translate([{"id": l["id"], "speaker": names.get(l["char"], l["char"]), "text": l["en"]}
                                   for l in chunk], glossary, p.get("game_name", ""), context=context,
                                  cards={names.get(l["char"], l["char"]): cards[l["char"]]
                                         for l in chunk if cards.get(l["char"])})
            from . import power
            power.pause(time.time() - t0)
            for l in chunk:
                if l["id"] in result:
                    ru_now[l["id"]] = result[l["id"]]
                    p.update_line(l["id"], ru=result[l["id"]], ru_source="llm",
                                  status="translated" if l["status"] in ("new", "error", "translated")
                                  else l["status"])
        report(f"Перевод: {len(need_llm)}/{len(need_llm)}", len(need_llm), len(need_llm))

    def _character_cards(self, p: Project, tr, chars: set[str], names: dict, report) -> dict[str, str]:
        """Карточки персонажей (пол, возраст, манера речи, ты/вы) — переводчик согласует род глаголов
        и обращение. Составляет сама языковая модель по типичным репликам; кэш в проекте."""
        cards = dict(p.get("cards") or {})
        todo = [c for c in chars if c and c not in cards]
        if not todo:
            return cards
        report("Переводчик знакомится с персонажами…")
        items = []
        for c in todo:
            lines = [l for l in p.lines(char=c) if l["en"] and l["status"] != "skipped"]
            if len(lines) < 2:
                cards[c] = ""
                continue
            random.Random(7).shuffle(lines)
            items.append({"name": names.get(c, c), "key": c, "lines": [l["en"] for l in lines[:12]]})
        for i in range(0, len(items), 8):
            if self.cancel_flag.is_set():
                return cards
            got = tr.describe_characters(items[i:i + 8], p.get("game_name", ""))
            for it in items[i:i + 8]:
                cards[it["key"]] = got.get(it["name"], "")
        p.set("cards", cards)
        return cards

    def _do_screen(self, gid: str):
        """Отсев по звуку: реплики с текстом, где на самом деле смех, кашель, крик, дыхание, а не речь,
        остаются в оригинале (нейросеть их не воспроизводит, а текстовая ремарка есть не всегда)."""
        from . import judge, models, tts
        if not models.path("judge-events"):
            return
        p, report = self.project(gid), self._progress(gid)
        lines = self._in_selection(p, [l for l in p.lines(status="translated") if l["ru"] and "speech" not in (l["info"] or {})])
        if not lines:
            return
        tts.unload()
        from . import power
        report("Проверка: где речь, а где смех и крики…")
        for i in range(0, len(lines), 16):
            if self.cancel_flag.is_set():
                return
            chunk = lines[i:i + 16]
            report(f"Отсев неречевых реплик: {i}/{len(lines)}", i, len(lines))
            t0 = time.time()
            try:
                scores = judge.speech_scores([str(p.dir / l["orig"]) for l in chunk], batch_size=power.batch(8))
            except Exception as e:
                print(f"отсев по звуку не удался: {e}")
                return
            power.pause(time.time() - t0)
            for l, (speech, ev) in zip(chunk, scores):
                info = dict(l["info"] or {})
                info["speech"] = round(speech, 2)
                letters = len(re.sub(r"[^А-Яа-яЁёA-Za-z]", "", l["ru"] or ""))
                # речь в оригинале AST оценивает 0.6–0.9, крики и стоны — около 0; длинную реплику
                # с настоящим текстом («Мелодия… она зовёт меня…» шёпотом) по звуку не отбрасываем
                if speech < 0.1 and ((l["duration"] or 0) < 4 or letters <= 8):
                    what = ev[0] if ev else "не речь (крик, стон, дыхание)"
                    p.update_line(l["id"], info=info, status="skipped",
                                  error=f"По звуку это {what} — остаётся оригинал")
                else:
                    p.update_line(l["id"], info=info)
        judge.unload()

    # --- озвучка ---

    # --- стабильность голоса: «русский голос» персонажа ---

    def _ru_voice(self, p: Project, char: str) -> tuple[np.ndarray | None, float | None]:
        """Центр отпечатков (ECAPA) уже озвученных реплик персонажа и нижняя граница «своего» голоса.
        Пока реплик мало — центр стоит на оригинальном голосе персонажа."""
        from . import judge
        if not judge.ecapa_ready():
            return None, None
        rv = (p.get("ru_voice") or {}).get(char)
        if rv and rv.get("e"):
            e = np.asarray(rv["e"], dtype=np.float32)
            sims = rv.get("sims") or []
            floor = (float(np.median(sims)) - DRIFT_MARGIN) if len(sims) >= 6 else None
            return e / (np.linalg.norm(e) + 1e-9), floor
        prof = self.character_profile(p, char)
        if not prof:
            return None, None
        e = judge.ecapa_embed([prof["audio"]])[0]
        all_rv = p.get("ru_voice") or {}
        all_rv[char] = {"e": e.tolist(), "n": 0, "sims": [], "seed": "original"}
        p.set("ru_voice", all_rv)
        return e, None

    def _ru_voice_add(self, p: Project, char: str, emb: np.ndarray, sim: float):
        all_rv = p.get("ru_voice") or {}
        rv = all_rv.get(char) or {"e": emb.tolist(), "n": 0, "sims": []}
        n = min(int(rv.get("n") or 0), 60)
        e = (np.asarray(rv["e"], dtype=np.float32) * (n + 1) + emb) / (n + 2)  # оригинал остаётся якорем с весом 1
        rv["e"] = (e / (np.linalg.norm(e) + 1e-9)).tolist()
        rv["n"] = n + 1
        rv["sims"] = (rv.get("sims") or [])[-80:] + [round(float(sim), 3)]
        all_rv[char] = rv
        p.set("ru_voice", all_rv)

    # --- режиссёр подачи ---

    def _director_ready(self, gid: str) -> bool:
        from . import translate
        return bool(self.settings(gid)["director"]) and any(translate._model_path(k) for k in translate.MODELS)

    def _do_direct(self, gid: str, lines: list[dict], force: bool = False):
        """Режиссёр: по оригинальной реплике (английский текст, эмоция, громкость, темп, паузы, смех) готовит
        русскую строку к произнесению (пунктуация) и короткое указание, как играть. Слова не меняются."""
        from . import asr, director, judge, tts
        p, report = self.project(gid), self._progress(gid)
        todo = [l for l in lines if force or not (l["info"] or {}).get("delivery")]
        if not todo:
            return
        tts.unload()
        need_en = [l for l in todo if not l["en"]]
        if need_en:
            report(f"Режиссёр: слушаю оригинал ({len(need_en)} реплик)…")
            from . import power
            for i in range(0, len(need_en), 32):
                if self.cancel_flag.is_set():
                    return
                chunk = need_en[i:i + 32]
                report(f"Режиссёр: слушаю оригинал {i}/{len(need_en)}", i, len(need_en))
                for l, t in zip(chunk, asr.transcribe([str(p.dir / l["orig"]) for l in chunk], language=None,
                                                      batch_size=power.batch(8))):
                    l["en"] = t.strip()
                    p.update_line(l["id"], en=l["en"])
            asr.unload()
        # что слышно в оригинале — для всех реплик сразу (модели оценки звука, потом языковая модель: вместе они
        # в видеопамять не помещаются). Храним здесь, а не только в _style_cache: тот держит 2000 последних,
        # и в большой игре режиссёр остался бы без описания оригинала у большинства реплик
        styles: dict[str, dict] = {}
        for i, l in enumerate(todo):
            if self.cancel_flag.is_set():
                return
            if i % 100 == 0:
                report(f"Режиссёр: что слышно в оригинале {i}/{len(todo)}", i, len(todo))
            if l["id"] not in self._style_cache:
                self._style_params(p, l, "ru")
            styles[l["id"]] = self._style_cache.get(l["id"])
        judge.unload()
        names = {c["id"]: c["name"] for c in p.characters()}
        order = sorted(p.lines(), key=lambda x: x["id"])
        pos = {x["id"]: k for k, x in enumerate(order)}
        tr = self._translator_for(gid)
        done = 0
        todo.sort(key=lambda l: l["id"])
        for i in range(0, len(todo), 12):
            if self.cancel_flag.is_set():
                return
            chunk = todo[i:i + 12]
            report(f"Режиссёр: подача реплик {i}/{len(todo)}", i, len(todo))
            items = []
            for l in chunk:
                k = pos.get(l["id"], 0)
                prev = [f"{names.get(x['char'], x['char'])}: {x['ru'] or ''}" for x in order[max(0, k - 2):k] if x["ru"]]
                edges = (p.get("edges") or {}).get(l["id"]) or {}
                items.append({"id": l["id"], "speaker": names.get(l["char"], l["char"]), "en": l["en"], "ru": l["ru"],
                              "heard": director.describe_orig(styles.get(l["id"]), edges.get("events")),
                              "prev": prev})
            got = director.direct(tr, items, p.get("game_name", ""), p.get("cards") or {})
            for l in chunk:
                d = got.get(l["id"])
                if d:
                    info = dict(l["info"] or {})
                    info["delivery"] = d
                    l["info"] = info
                    p.update_line(l["id"], info=info)
                    done += 1
        self._release_translator()
        report(f"Режиссёр: подача готова для {done} реплик")

    # --- контролёр → план доработки ---

    def _fix_plan(self, line: dict) -> dict:
        """По диагнозу контролёра — конкретные изменения для переозвучки и человеческое описание плана."""
        j = (line["info"] or {}).get("judge") or {}
        codes = [x["code"] for x in j.get("problems") or []]
        prev = (line["info"] or {}).get("fix") or {}
        rnd = int(prev.get("round") or 0) + 1
        plan = {"round": rnd, "problems": codes, "extra": 3, "temperature": None, "subtalker": None,
                "max_speedup": None, "steps": []}
        if "words" in codes or "short" in codes or "empty" in codes:
            plan.update(temperature=0.6, extra=4)
            plan["steps"].append("чётче и без пропусков: меньше случайности, больше вариантов")
        if "drift" in codes or "voice" in codes or "pitch" in codes:
            plan.update(temperature=min(plan["temperature"] or 0.65, 0.65), subtalker=0.35, extra=5)
            plan["steps"].append("держать голос персонажа: строже тембр, выбор по сходству с его репликами")
        if "accent" in codes:
            from . import pron
            plan.update(temperature=min(plan["temperature"] or 0.6, 0.6), extra=5, director=True)
            words = pron.describe({"words": j.get("pron_words")})
            plan["steps"].append("чистое русское произношение" + (f" ({words})" if words else "")
                                 + ": меньше случайности, выбор варианта по чистоте произношения")
        if "flat" in codes:
            plan.update(temperature=0.9, extra=max(plan["extra"], 4), director=True)
            plan["steps"].append("сыграть живее, как в оригинале")
        if "over" in codes:
            plan.update(temperature=0.65, extra=max(plan["extra"], 3), director=True)
            plan["steps"].append("сыграть спокойнее, как в оригинале")
        if "fast" in codes:
            plan["max_speedup"] = 1.18
            plan["steps"].append("уложиться во время: допустимое ускорение до ×1.18")
        if not plan["steps"]:
            plan["steps"].append("больше вариантов на выбор")
        plan["plan"] = f"Доработка {rnd}: " + "; ".join(plan["steps"])
        return plan

    def _judge_ready(self, gid: str) -> bool:
        from . import judge
        return self.settings(gid)["judge"] and judge.VE_WEIGHTS.exists()

    def _do_generate(self, gid: str, ids: list[str] | None = None, fix: bool = False):
        """Озвучить реплики (ids — только эти). fix — переозвучка проблемных по плану контролёра (info["fix"]):
        больше вариантов, другие параметры, старая запись участвует в выборе наравне с новыми.

        Стабильный голос: один отпечаток на персонажа, строгий тембр (sub-talker), и из нескольких вариантов
        выбирается тот, что ближе к «русскому голосу» персонажа (центр отпечатков ECAPA уже озвученных реплик)
        и чище звучит по-русски (распознаватель без словаря, pron.py — слышит акцент). Подачу задаёт режиссёр
        (пунктуация + указание) по оригиналу."""
        from . import accel, asr, judge, pron, tts
        accel.require_gpu()
        p, report, s = self.project(gid), self._progress(gid), self.settings(gid)
        if ids:
            lines = [l for l in p.lines(ids=ids) if l["ru"] and l["status"] != "skipped"]
        else:
            lines = self._in_selection(p, [l for l in p.lines(status="translated") if l["ru"]])
        if not lines:
            report("Нет реплик для озвучки: сначала нужен русский текст")
            return
        self._release_translator()
        asr.unload()
        engine = self._engine_for(gid)
        base_cand = CANDIDATES.get(s["quality"], 3)
        use_judge = self._judge_ready(gid)
        info = accel.info()
        inline_asr = use_judge and info["device"] == "cuda" and (info["vram_gb"] or 0) >= 10
        stable = bool(s["stable_voice"]) and engine == "ru"
        use_ecapa = stable and judge.ecapa_ready()
        use_pron = use_judge and inline_asr and pron.ready()
        take = self._take(gid)
        if not fix and self._director_ready(gid):
            self._do_direct(gid, lines)
            lines = p.lines(ids=[l["id"] for l in lines])
            if self.cancel_flag.is_set():
                return

        report("Профили голосов персонажей…")
        chars = {l["char"] for l in lines}
        refs = self._refs_for(p, engine, chars)
        voices = {c: self._voice(p, c) for c in chars} if s["voice_transfer"] else {}
        profiles = {}
        if use_judge:
            for c in chars:
                prof = self.character_profile(p, c)
                if prof:
                    profiles[c] = judge.speaker_embed([prof["audio"]])
        report(f"Загрузка модели речи ({engine})…")
        if engine == "custom":
            tts.load_custom()
        elif engine == "ru":
            tts.load_ru()
        # Персонажи подряд (своя модель грузится один раз), главные — первыми: если озвучку остановить
        # на полпути, в игре уже будут главные герои, а не безымянные группы
        weight = {c["id"]: -(c["lines"] or 0) for c in p.characters()}
        lines.sort(key=lambda l: (weight.get(l["char"], 0), l["char"] or "", l["id"]))
        p.gen_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        from . import power
        batch = power.batch(2)
        last_pos = {}
        for pos, l in enumerate(lines):
            last_pos[l["char"]] = pos
        notes = p.get("char_notes") or {}
        for i in range(0, len(lines), batch):
            batch_started = time.time()
            if self.cancel_flag.is_set():
                return
            chunk = lines[i:i + batch]
            eta = ""
            if i:
                left = (time.time() - started) / i * (len(lines) - i)
                eta = f", осталось ~{int(left // 60)} мин" if left >= 60 else f", осталось ~{int(left)} с"
            report(f"{'Доработка' if fix else 'Озвучка'} ({engine}, вариантов: {base_cand}{'+' if fix else ''}): {i}/{len(lines)}{eta}", i, len(lines))
            candidates: list[list[tuple[np.ndarray, int]]] = [[] for _ in chunk]
            plans, texts = [], []
            t_line = time.time()
            try:
                for j, l in enumerate(chunk):
                    li = l["info"] or {}
                    plan = li.get("fix") if fix else None
                    deliv = (plan or {}).get("delivery") or li.get("delivery") or {}
                    text = deliv.get("text") or l["ru"]
                    if not director_same(text, l["ru"]):
                        text = l["ru"]
                    plans.append(plan)
                    texts.append(text)
                    maxlen = max((l["duration"] or 2) * 2.0, 3.0) + len(l["ru"]) * 0.03
                    params = {"speaker": refs[l["char"]].get("speaker"), **self._style_params(p, l, engine)}
                    if stable:
                        params["temperature"] = 0.75 + 0.05 * min(3, int(deliv.get("intensity") or 0))
                        params["subtalker_temperature"] = 0.5
                    if plan:
                        if plan.get("temperature"):
                            params["temperature"] = plan["temperature"]
                        if plan.get("subtalker"):
                            params["subtalker_temperature"] = plan["subtalker"]
                        if deliv.get("temperature"):
                            params["temperature"] = deliv["temperature"]
                    note_t = _parse_note(notes.get(l["char"])).get("temperature")
                    if note_t:
                        params["temperature"] = note_t
                    ref = self._ref_for_line(p, l, engine, refs[l["char"]], params)
                    n_cand = base_cand + ((plan or {}).get("extra", FIX_EXTRA) if fix else 0)
                    if engine in ("ru", "custom") and n_cand > 1:  # все варианты одной пачкой — быстрее по одному
                        seed = None if not (take or fix) else 100003 * (take + 1) + 7919 * int((plan or {}).get("round") or 1)
                        candidates[j] += tts.synthesize(engine, [text] * n_cand, [ref] * n_cand, [maxlen] * n_cand, params, seed=seed)
                    else:
                        for k in range(n_cand):
                            seed = None if not (k or take or fix) else 100003 * (take + 1) + 977 * k + (7919 if fix else 0)
                            candidates[j] += tts.synthesize(engine, [text], [ref], [maxlen], params, seed=seed)
            except Exception as e:
                for l in chunk:
                    p.update_line(l["id"], status="error", error=str(e)[:300])
                continue
            for l, cands, plan, text in zip(chunk, candidates, plans, texts):
                orig, orig_sr = audio.read(p.dir / l["orig"])
                raw = []  # (wav, sr) до подгонки — по ним меряем голос и акцент (без вклеенного оригинала)
                scored = []  # (fitted, finfo)
                speedup = (plan or {}).get("max_speedup") or s["max_speedup"]
                for wav, sr in cands:
                    wav, sr = self._apply_note(wav, sr, notes.get(l["char"]))
                    if voices.get(l["char"]):
                        try:
                            wav, sr = tts.convert_timbre(wav, sr, voices[l["char"]],
                                                         self._rvc_params(p, l["char"], engine, refs[l["char"]]["speaker"]))
                        except Exception as e:
                            print(f"RVC: {e}")
                    fitted, finfo = self._fit_line(p, l, wav, sr, orig, orig_sr, speedup)
                    raw.append((wav, sr))
                    scored.append((fitted, finfo))
                if fix and l["gen"] and (p.dir / l["gen"]).exists():  # старая запись соревнуется с новыми
                    old, osr = audio.read(p.dir / l["gen"])
                    old = audio.resample(old, osr, orig_sr)
                    raw.append((old, orig_sr))
                    scored.append((old, dict((l["info"] or {}).get("fit") or {"ratio": 1.0})))
                best, best_info, best_judge, best_k = None, None, None, 0
                tmps = []
                for ci, (fitted, finfo) in enumerate(scored):
                    tmp = p.gen_dir / f"_cand_{_safe(l['id'])}_{ci}.wav"
                    audio.write(tmp, fitted, orig_sr)
                    tmps.append(tmp)
                judging = use_judge and (len(scored) > 1 or inline_asr)
                heard_all = asr.transcribe([str(t) for t in tmps], language="russian") if judging and inline_asr else [None] * len(tmps)
                prons = pron.check(raw, [l["ru"]] * len(raw)) if judging and use_pron else [None] * len(raw)
                centroid, floor = self._ru_voice(p, l["char"]) if use_ecapa else (None, None)
                embs = judge.ecapa_embed(raw) if use_ecapa and centroid is not None else None
                cons = [float(e @ centroid) for e in embs] if embs is not None else [None] * len(raw)
                orig_emo, orig_ev = self._orig_judge(p, l) if judging else (None, None)
                ostats = self._orig_stats(p, l) if judging else None
                for k, ((fitted, finfo), tmp, heard) in enumerate(zip(scored, tmps, heard_all)):
                    jr = None
                    if judging:
                        gstats = audio.f0_stats(*raw[k])
                        jr = judge.judge_line(str(tmp), l["ru"], str(p.dir / l["orig"]), profiles.get(l["char"]),
                                              heard=heard, orig_emotion=orig_emo, orig_events=orig_ev,
                                              full=self._judge_ready_quiet(), profile_f0=self.profile_f0(p, l["char"]),
                                              consistency=cons[k], cons_floor=floor, pron=prons[k],
                                              gen_stats=gstats, orig_stats=ostats, events_kept=finfo.get("kept"))
                        jr["problems"] = judge._problems(jr)
                    # оценка контролёра + сходство с «русским голосом» персонажа + чистота произношения
                    score = ((jr["score"] if jr else 0) + (30 * cons[k] if cons[k] is not None else 0)
                             - (60 * prons[k]["cer"] if prons[k] else 0))
                    if best is None or score > best_score:
                        best, best_info, best_judge, best_k, best_score = fitted, finfo, jr or {"score": 0}, k, score
                    tmp.unlink(missing_ok=True)
                if embs is not None and cons[best_k] is not None and (best_judge.get("verdict") != "rework" or not fix):
                    self._ru_voice_add(p, l["char"], embs[best_k], cons[best_k])
                if i < 12 or i % 200 == 0:
                    print(f"озвучка: реплика {i} ({l['char']}, {len(scored)} вар.) за {time.time() - t_line:.1f} с")
                t_line = time.time()
                out = p.gen_dir / Path(l["orig"]).name
                audio.write(out, best, orig_sr)
                ratio = best_info.get("ratio", 1.0)
                old_info = l["info"] or {}
                new_info = {"fit": best_info, "flag": "long" if ratio > 1.35 else ("short" if ratio < 0.45 else None),
                            "engine": engine, "trained": bool(voices.get(l["char"])), "candidates": len(scored),
                            "speech": old_info.get("speech"), "style": self._style_cache.get(l["id"]),
                            "delivery": old_info.get("delivery"), "performance": old_info.get("performance"),
                            "spoken": text if text != l["ru"] else None}
                if plan:
                    new_info["fix"] = plan
                    new_info["fix_history"] = (old_info.get("fix_history") or [])[-4:] + [
                        {"round": plan.get("round"), "problems": plan.get("problems"), "score_before": (old_info.get("judge") or {}).get("score")}]
                if use_judge and best_judge.get("verdict"):
                    new_info["judge"] = best_judge
                p.update_line(l["id"], gen=str(out.relative_to(p.dir)), gen_duration=round(len(best) / orig_sr, 3),
                              status="generated", info=new_info, error=None)
            power.pause(time.time() - batch_started)
            if ids is None:  # персонаж озвучен целиком — предложить послушать, пока идут остальные
                for c in {l["char"] for l in chunk}:
                    if last_pos.get(c, -1) < i + batch:
                        self._mark_review(p, c)
        report(f"Озвучка: {len(lines)}/{len(lines)}", len(lines), len(lines))
        if use_judge:
            self._do_judge(gid, ids=[l["id"] for l in lines], quick=False, autofix=not fix)
        if ids is None and not fix and s["auto_install"] and not self.cancel_flag.is_set():
            # Automatic installation only accepts checked recordings without unresolved warnings.
            with p.db() as c:
                c.execute("""UPDATE lines SET status='approved' WHERE status='generated'
                    AND json_extract(info, '$.judge.verdict') = 'ok'
                    AND COALESCE(json_extract(info, '$.fit.requires_review'), 0) = 0""")
            self.enqueue(gid, "install")
            self._finish(gid, "Озвучка готова — устанавливаем в игру")

    def _orig_stats(self, p: Project, line: dict) -> tuple[float, float]:
        cache = p.get("orig_stats") or {}
        if line["id"] not in cache:
            w, sr = audio.read(p.dir / line["orig"])
            cache[line["id"]] = list(audio.f0_stats(w, sr))
            if len(cache) % 25 == 0:
                p.set("orig_stats", cache)
            self._orig_stats_cache = cache
        return tuple(cache[line["id"]])

    def _do_judge(self, gid: str, ids: list[str] | None = None, quick: bool = False, autofix: bool = True, rounds: int = 2):
        """Контролёр: распознать русскую речь, проверить текст, голос (сходство с персонажем и с его остальными
        репликами, высоту), акцент, живость подачи, эмоцию и события оригинала. Каждая проблема получает код
        и конкретный план доработки (_fix_plan); проблемные реплики переозвучиваются по плану, до rounds раз.
        Для проблем подачи режиссёр (языковая модель) получает диагноз и пишет новое указание."""
        from . import asr, director, judge, pron, tts
        p, report, s = self.project(gid), self._progress(gid), self.settings(gid)
        lines = [l for l in (p.lines(ids=ids) if ids else p.lines(status="generated,approved,rejected")) if l["gen"]]
        use_pron = not quick and pron.ready()
        if not lines:
            return
        tts.unload()
        report("Контролёр: загрузка моделей…")
        profiles = {}
        for c in {l["char"] for l in lines}:
            r = self.character_profile(p, c)
            if r:
                profiles[c] = judge.speaker_embed([r["audio"]])
        use_ecapa = judge.ecapa_ready()
        for i in range(0, len(lines), 8):
            if self.cancel_flag.is_set():
                return
            chunk = lines[i:i + 8]
            report(f"Контролёр: проверка {i}/{len(lines)}", i, len(lines))
            files = [str(p.dir / l["gen"]) for l in chunk]
            heard = asr.transcribe(files, language="russian")
            prons = pron.check(files, [l["ru"] for l in chunk]) if use_pron else [None] * len(files)
            embs = judge.ecapa_embed(files) if use_ecapa else None
            for k, (l, h) in enumerate(zip(chunk, heard)):
                try:
                    orig_emo, orig_ev = self._orig_judge(p, l) if not quick else (None, None)
                    cons, floor = None, None
                    if embs is not None:
                        centroid, floor = self._ru_voice(p, l["char"])
                        cons = float(embs[k] @ centroid) if centroid is not None else None
                    g, gsr = audio.read(p.dir / l["gen"])
                    res = judge.judge_line(str(p.dir / l["gen"]), l["ru"], str(p.dir / l["orig"]),
                                           profiles.get(l["char"]), heard=h, orig_emotion=orig_emo, orig_events=orig_ev,
                                           full=not quick, profile_f0=self.profile_f0(p, l["char"]),
                                           consistency=cons, cons_floor=floor, pron=prons[k],
                                           gen_stats=audio.f0_stats(g, gsr), orig_stats=self._orig_stats(p, l),
                                           events_kept=((l["info"] or {}).get("fit") or {}).get("kept"))
                    res["problems"] = judge._problems(res)
                except Exception as e:
                    res = {"score": 0, "verdict": "rework", "reasons": [f"проверка не удалась: {e}"], "problems": []}
                info = dict(l["info"] or {})
                info["judge"] = res
                p.update_line(l["id"], info=info)
        p.set("orig_stats", getattr(self, "_orig_stats_cache", None) or p.get("orig_stats"))
        asr.unload()
        judge.unload()
        pron.unload()
        checked = p.lines(ids=[l["id"] for l in lines])
        rework = [l for l in checked if (l["info"].get("judge") or {}).get("verdict") == "rework" and l["status"] == "generated"]
        if rework and autofix and s["autofix"] and rounds > 0 and not self.cancel_flag.is_set():
            # план доработки на каждую реплику; проблемы подачи — режиссёру с диагнозом
            need_dir = []
            for l in rework:
                plan = self._fix_plan(l)
                info = dict(l["info"] or {})
                info["fix"] = plan
                l["info"] = info
                p.update_line(l["id"], info=info)
                if plan.get("director"):
                    need_dir.append(l)
            if need_dir and self._director_ready(gid):
                report(f"Режиссёр: новая подача для {len(need_dir)} реплик…")
                tr = self._translator_for(gid)
                for l in need_dir:
                    if self.cancel_flag.is_set():
                        break
                    j = l["info"].get("judge") or {}
                    diag = "; ".join(j.get("reasons") or []) + f". В оригинале: {director.describe_orig((l['info'] or {}).get('style'))}"
                    d = director.fix(tr, l, diag, (l["info"] or {}).get("delivery"))
                    if d:
                        info = dict(l["info"])
                        info["fix"] = {**info["fix"], "delivery": d, "plan": info["fix"]["plan"] + f"; режиссёр: «{d['note']}»"}
                        p.update_line(l["id"], info=info)
                self._release_translator()
            report(f"Контролёр: доработка {len(rework)} реплик по плану…")
            self._do_generate(gid, ids=[l["id"] for l in rework], fix=True)
            self._do_judge(gid, ids=[l["id"] for l in rework], quick=quick, autofix=True, rounds=rounds - 1)
            still = sum(1 for l in p.lines(ids=[l["id"] for l in rework]) if (l["info"].get("judge") or {}).get("verdict") == "rework")
            self._finish(gid, f"Контролёр: проверено {len(lines)}, доработано {len(rework)}, "
                              f"осталось проверить на слух {still}")
            return
        self._finish(gid, f"Контролёр: проверено {len(lines)}, требуют доработки {len(rework)}")

    _style_cache: dict = {}

    def _style_params(self, p: Project, line: dict, engine: str) -> dict:
        """Подача под оригинал: эмоция (классификатор), громкость относительно обычной речи персонажа,
        темп, паузы и динамика (по огибающей оригинала). CustomVoice получает инструкцию словами,
запасной CustomVoice — инструкцию словами."""
        emo, conf = "neutral", 0.0
        if self._judge_ready_quiet():
            try:
                from . import judge
                emo, conf = judge.emotion(str(p.dir / line["orig"]))
            except Exception:
                pass
        if conf < 0.4:
            emo = "neutral"
        loud = self._loudness_delta(p, line)  # дБ относительно обычной громкости персонажа
        w, sr = audio.read(p.dir / line["orig"])
        feat = audio.speech_features(w, sr)
        base = self._speech_baseline(p, line["char"])  # обычные темп/паузы/динамика этого персонажа
        fast = feat["dur"] > 1 and feat["rate"] > base["rate"] * 1.5
        slow = feat["dur"] > 1.5 and feat["rate"] < base["rate"] * 0.7
        pauses = feat["dur"] > 2.5 and feat["pause_ratio"] > base["pause_ratio"] + 0.22
        expressive = feat["dyn_db"] > base["dyn_db"] + 3
        flat = feat["dur"] > 2 and feat["dyn_db"] < base["dyn_db"] - 3
        style = {"emotion": emo, "emotion_conf": round(conf, 2), "loud": round(loud, 1), **feat,
                 "tags": [t for t, on in (("fast", fast), ("slow", slow), ("pauses", pauses),
                                          ("expressive", expressive), ("flat", flat)) if on]}
        self._style_cache[line["id"]] = style
        if len(self._style_cache) > 2000:
            self._style_cache.pop(next(iter(self._style_cache)))
        note = _parse_note((p.get("char_notes") or {}).get(line["char"]))
        if engine == "voxcpm":
            from .performance import instruction
            plan = self._source_performance(p, line)
            return {"emotion": emo, "instruct": instruction(style, (line.get("info") or {}).get("delivery"),
                                                           plan.get("vocal_style")),
                    "temperature": 0.8}
        if engine == "ru":
            temp = 0.9 if emo not in ("neutral", "calm") else 0.8
            return {"emotion": emo, "temperature": note.get("temperature", temp)}
        if engine != "custom":
            return {"emotion": emo, "rate": "fast" if fast else "slow" if slow else None}
        parts = [EMOTION_INSTRUCT.get(emo, "")]
        if not fast:  # модель склонна тараторить по-русски — просим спокойный темп, если оригинал не спешит
            parts.append("Natural, unhurried pace, clear articulation.")
        if loud >= 6:
            parts.append("Loud, almost shouting.")
        elif loud <= -8:
            parts.append("Very quietly, almost whispering.")
        if fast:
            parts.append("Speak fast, in a hurry.")
        elif slow:
            parts.append("Speak slowly and deliberately.")
        if pauses:
            parts.append("With noticeable pauses, as if hesitating.")
        if expressive:
            parts.append("Very expressive, with strong rises and falls.")
        elif flat:
            parts.append("Even, flat intonation.")
        if note.get("text"):
            parts.append(note["text"])  # комментарий пользователя — прямо в инструкцию модели
        ins = " ".join(x for x in parts if x).strip()
        return {"instruct": ins, "emotion": emo, "temperature": note.get("temperature", 0.85 if emo not in ("neutral", "calm") else 0.75)}

    def _speech_baseline(self, p: Project, char: str) -> dict:
        """Медианные темп, доля пауз и динамика по репликам персонажа (до 120 штук): подача каждой реплики
        описывается относительно его обычной манеры, а не абсолютными порогами."""
        cache = getattr(self, "_base_cache", None)
        if cache is None:
            cache = self._base_cache = {}
        key = (p.id, char)
        if key not in cache:
            lines = [l for l in p.lines(char=char) if 1.0 <= (l["duration"] or 0) <= 10][:120]
            feats = []
            for l in lines:
                try:
                    w, sr = audio.read(p.dir / l["orig"])
                    feats.append(audio.speech_features(w, sr))
                except Exception:
                    pass
            if len(feats) >= 5:
                cache[key] = {k: float(np.median([f[k] for f in feats])) for k in ("rate", "pause_ratio", "dyn_db")}
            else:
                cache[key] = {"rate": 4.0, "pause_ratio": 0.3, "dyn_db": 13.0}
        return cache[key]

    def profile_f0(self, p: Project, char: str) -> float:
        """Обычная высота голоса персонажа по его профилю (Гц), кэш в проекте: по ней подбирается диктор,
        проверяется высота каждой русской реплики и дообученной модели (чтобы старик не стал ребёнком)."""
        cache = p.get("voice_f0") or {}
        if char in cache:
            return float(cache[char])
        prof = self.character_profile(p, char)
        f0 = 0.0
        if prof:
            try:
                w, sr = audio.read(prof["audio"])
                f0 = audio.f0_median(w, sr)
            except Exception:
                f0 = 0.0
        cache[char] = round(f0, 1)
        p.set("voice_f0", cache)
        return f0

    def _mark_review(self, p: Project, char: str):
        """Персонаж озвучен — показать в студии «послушать и одобрить» (пока озвучиваются остальные)."""
        review = p.get("review") or {}
        n = p.count(char=char, status="generated")
        if n:
            review[char] = {"at": time.time(), "lines": n, "state": "ready"}
            p.set("review", review)

    def review_approve(self, p: Project, char: str) -> int:
        with p.db() as c:
            n = c.execute("UPDATE lines SET status='approved' WHERE status='generated' AND char=?", (char,)).rowcount
        review = p.get("review") or {}
        review[char] = {**review.get(char, {}), "state": "approved", "at": time.time()}
        p.set("review", review)
        return n

    def review_reset(self, p: Project, char: str) -> list[str]:
        """Сбросить озвучку персонажа (кроме реплик, одобренных вручную раньше) — для переозвучки с учётом комментария."""
        ids = [l["id"] for l in p.lines(char=char) if l["status"] in ("generated", "approved", "rejected")]
        with p.db() as c:
            c.execute("UPDATE lines SET gen=NULL, gen_duration=NULL, info=NULL, status='translated' "
                      "WHERE char=? AND status IN ('generated','approved','rejected')", (char,))
        review = p.get("review") or {}
        review.pop(char, None)
        p.set("review", review)
        return ids

    @staticmethod
    def _apply_note(wav: np.ndarray, sr: int, note: str | None) -> tuple[np.ndarray, int]:
        """Темп и высота по комментарию к персонажу («медленнее», «ниже»…) — после синтеза."""
        prm = _parse_note(note)
        if prm.get("speed") and abs(prm["speed"] - 1) > 0.02:
            wav = audio.time_stretch(wav, sr, prm["speed"])
        if prm.get("pitch"):
            try:
                import librosa
                wav = librosa.effects.pitch_shift(wav.astype(np.float32), sr=sr, n_steps=float(prm["pitch"]))
            except Exception as e:
                print(f"сдвиг высоты по комментарию не удался: {e}")
        return wav, sr

    def _source_performance(self, p: Project, line: dict) -> dict:
        from . import asr, judge, performance
        path = p.dir / line["orig"]
        info = dict(line.get("info") or {})
        cached = info.get("performance") or {}
        effects = (line.get("src") or {}).get("effects",
            {**DEFAULT_SETTINGS, **(p.get("settings") or {})}.get("effects_mode", "unknown"))
        if (cached.get("source") == performance.fingerprint(path) and cached.get("effects") == effects
                and not any(x.startswith("Анализ оригинала:") for x in cached.get("warnings", []))):
            return cached
        try:
            words = asr.words(str(path), language="english")
            plan = performance.analyse(path, words, classify=judge.classify_segment, effects=effects)
        except Exception as error:
            plan = performance.analyse(path, [], effects=effects)
            plan["warnings"].append(f"Анализ оригинала: {error}")
        finally:
            asr.unload()
        info["performance"] = plan
        line["info"] = info
        p.update_line(line["id"], info=info)
        return plan

    def _edges(self, p: Project, line: dict) -> dict:
        """Границы смеха/вздохов по краям оригинала (кэш в проекте): эти куски остаются из оригинала."""
        from . import judge, models
        cache = p.get("edges") or {}
        if line["id"] in cache and (cache[line["id"]] or {}).get("v") == judge.EDGES_VERSION:
            return cache[line["id"]]
        res = {"lead": 0.0, "trail": None, "dur": None, "events": []}
        if models.path("judge-events"):
            try:
                res = judge.event_edges(str(p.dir / line["orig"]))
            except Exception as e:
                print(f"края реплики: {e}")
        cache[line["id"]] = res
        if len(cache) % 50 == 0 or res["events"]:
            p.set("edges", cache)
        return res

    def _fit_line(self, p: Project, line: dict, wav: np.ndarray, sr: int, orig: np.ndarray, orig_sr: int,
                  max_speedup: float) -> tuple[np.ndarray, dict]:
        """Подогнать русскую речь под оригинал; смех/вздох/кашель по краям оригинала сохраняются и приклеиваются."""
        if {**DEFAULT_SETTINGS, **(p.get("settings") or {})}.get("preserve_performance", True):
            from . import performance
            plan = self._source_performance(p, line)
            fitted, info = performance.compose(wav, sr, orig, orig_sr, plan, max_speedup)
            # Optional speech-free stems/IR are project assets, never arbitrary external paths.
            effects = (line.get("src") or {}).get("effect_stems") or {}
            stems = {}
            for key in ("background", "impulse"):
                if effects.get(key):
                    source = (p.dir / effects[key]).resolve()
                    if not source.is_relative_to(p.dir.resolve()):
                        raise ValueError("Дорожка эффекта вне проекта")
                    data, rate = audio.read(source)
                    stems[key] = audio.resample(data, rate, orig_sr)
            if stems:
                fitted = performance.mix_stems(fitted, orig_sr, **stems, wet=effects.get("wet", .2))
            return fitted, info
        e = self._edges(p, line)
        lead, trail = float(e.get("lead") or 0.0), e.get("trail")
        dur = len(orig) / orig_sr
        if trail is None or trail > dur:
            trail = dur
        if not e.get("events") or trail - lead < 0.3:
            fitted, info = audio.fit_to_original(wav, sr, orig, orig_sr, max_speedup=max_speedup)
            return fitted, info
        # Паузы между смехом/вздохом и речью — как у актёра: ищем в оригинале, где речь реально начинается
        # (и где заканчивается перед концовкой), и оставляем эти куски оригинала; запасное значение — настройка
        default_gap = float({**DEFAULT_SETTINGS, **(p.get("settings") or {})}.get("event_gap", 0.3))
        lead_cut, trail_cut = lead, trail
        if lead > 0:
            onset = audio.speech_onset(orig, orig_sr, lead, min(trail, lead + 1.2))
            gap = (onset - lead) if onset is not None else default_gap
            lead_cut = lead + max(0.12, min(0.6, gap))
        if trail < dur:
            offset = audio.speech_offset(orig, orig_sr, max(lead_cut, trail - 1.2), trail)
            gap2 = (trail - offset) if offset is not None else 0.15
            trail_cut = trail - max(0.05, min(0.6, gap2))
        if trail_cut - lead_cut < 0.4:  # речи должно остаться хоть что-то
            lead_cut, trail_cut = lead, trail
        speech = orig[int(lead_cut * orig_sr): int(trail_cut * orig_sr)]
        fitted, info = audio.fit_to_original(wav, sr, speech, orig_sr, max_speedup=max_speedup)
        out = audio.splice_edges(orig, orig_sr, fitted, lead_cut, trail_cut)
        info = {**info, "kept": e["events"], "lead": lead, "trail": round(dur - trail, 2),
                "gap": round(lead_cut - lead, 2), "gap2": round(trail - trail_cut, 2)}
        return out, info

    def _loudness_delta(self, p: Project, line: dict) -> float:
        cache = getattr(self, "_rms_cache", None)
        if cache is None:
            cache = self._rms_cache = {}
        key = (p.id, line["char"])
        if key not in cache:
            prof = self.character_profile(p, line["char"])
            if prof:
                w, sr = audio.read(prof["audio"])
                cache[key] = audio.rms_db(w)
            else:
                cache[key] = None
        base = cache[key]
        if base is None:
            return 0.0
        w, sr = audio.read(p.dir / line["orig"])
        return audio.rms_db(w) - base

    def _orig_judge(self, p: Project, line: dict):
        """Эмоция и события оригинала — один раз на реплику (кэш в проекте), для выбора варианта и контролёра."""
        from . import judge, models
        cache = getattr(self, "_orig_cache", None)
        if cache is None:
            cache = self._orig_cache = {}
        key = (p.id, line["id"])
        if key not in cache:
            emo = None
            if self._judge_ready_quiet():
                st = self._style_cache.get(line["id"])
                if st and st.get("emotion_conf") is not None:
                    emo = (st["emotion"], st["emotion_conf"])
                else:
                    try:
                        emo = judge.emotion(str(p.dir / line["orig"]))
                    except Exception:
                        emo = None
            ev = None
            if models.path("judge-events"):
                try:
                    ev = judge.events(str(p.dir / line["orig"]))
                except Exception:
                    ev = None
            cache[key] = (emo, ev)
            if len(cache) > 20000:
                cache.pop(next(iter(cache)))
        return cache[key]

    def _judge_ready_quiet(self) -> bool:
        from . import judge, models
        return judge.VE_WEIGHTS.exists() and models.path("judge-emotion8") is not None

    def _render(self, gid: str, p: Project, char: str, lines: list[dict], engine: str, voice: dict | None,
                out_dir: Path, seed: int | None = None) -> tuple[list[str], float | None]:
        """Озвучить реплики в out_dir; вернуть файлы и среднюю похожесть на персонажа."""
        from . import judge, tts
        prof_long = self.character_profile(p, char)
        profile_emb = judge.speaker_embed([prof_long["audio"]]) if prof_long and judge.VE_WEIGHTS.exists() else None
        ref = self._refs_for(p, engine, {char}).get(char)
        if not ref:
            raise RuntimeError("Не удалось собрать образец голоса персонажа")
        if engine == "custom":
            tts.load_custom()
        elif engine == "ru":
            tts.load_ru()
        outs = []
        stable = bool(self.settings(gid)["stable_voice"]) and engine == "ru"
        for l in lines:  # по одной: у каждой реплики своя подача под эмоцию оригинала
            params = {"speaker": ref.get("speaker"), **self._style_params(p, l, engine)}
            deliv = (l["info"] or {}).get("delivery") or {}
            text = deliv.get("text") if deliv.get("text") and director_same(deliv["text"], l["ru"]) else l["ru"]
            if stable:
                params["temperature"] = 0.75 + 0.05 * min(3, int(deliv.get("intensity") or 0))
                params["subtalker_temperature"] = 0.5
            outs += tts.synthesize(engine, [text], [self._ref_for_line(p, l, engine, ref, params)],
                                   [max((l["duration"] or 2) * 2, 3) + len(l["ru"]) * 0.03], params, seed=seed)
        files, sims = [], []
        note = (p.get("char_notes") or {}).get(char)
        for l, (wav, sr) in zip(lines, outs):
            wav, sr = self._apply_note(wav, sr, note)
            wav, sr = tts.convert_timbre(wav, sr, voice, self._rvc_params(p, char, engine, ref["speaker"]) if voice else None)
            orig, osr = audio.read(p.dir / l["orig"])
            fitted, _ = self._fit_line(p, l, wav, sr, orig, osr, self.settings(gid)["max_speedup"])
            f = out_dir / Path(l["orig"]).name
            audio.write(f, fitted, osr)
            files.append(str(f.relative_to(p.dir)))
            if profile_emb is not None:
                sims.append(judge.voice_similarity(str(f), profile_emb))
        return files, (round(float(np.mean(sims)), 3) if sims else None)

    # --- обучение голоса персонажа ---

    def _tts_voice(self, p: Project, char: str) -> dict | None:
        """Дообученная под персонажа модель речи (LoRA-адаптер + образец) или None."""
        from . import ttsft
        v = (p.get("tts_voices") or {}).get(char)
        if not v or v.get("disabled") or not (p.dir / v["ref"]).exists():
            return None
        need = "model.safetensors" if v.get("kind") == "full" else ttsft.VOICE_FILE
        if (p.dir / v["dir"] / need).exists():
            return {"dir": str(p.dir / v["dir"]), "ref": str(p.dir / v["ref"]), **{k: v[k] for k in v if k not in ("dir", "ref")}}
        return None

    def _run_child(self, gid: str, job: dict, on_progress) -> dict:
        """Запустить ruvoice.ttsft отдельным процессом (видеопамять освобождается по завершении), читать прогресс."""
        import json
        import os
        import subprocess
        import sys
        from . import power
        from .paths import DATA_DIR, PROJECT_DIR
        jf = Path(job["out_dir"]).parent / "job.json"
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(job, ensure_ascii=False), "utf-8")
        env = power.env_for_children(dict(os.environ))
        env["PYTHONUTF8"] = "1"
        (PROJECT_DIR / "data" / "engines").mkdir(parents=True, exist_ok=True)
        log = open(PROJECT_DIR / "data" / "engines" / "ttsft.log", "a", encoding="utf-8")
        exe = str(Path(sys.executable).with_name("python.exe"))
        self._child = subprocess.Popen([exe, "-m", "ruvoice.ttsft", str(jf)], cwd=str(PROJECT_DIR), stdout=subprocess.PIPE,
                                       stderr=log, text=True, encoding="utf-8", env=env,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        power.apply_priority(self._child.pid)
        last = {}
        for line in self._child.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                info = json.loads(line)
            except ValueError:
                continue
            last = info
            if info.get("error"):
                break
            on_progress(info)
        self._child.wait()
        log.close()
        if self.cancel_flag.is_set():
            raise RuntimeError("остановлено")
        if self._child.returncode or last.get("error"):
            raise RuntimeError(last.get("error") or f"процесс дообучения завершился с кодом {self._child.returncode}")
        return last

    def _do_train_tts(self, gid: str, char: str, max_seconds: float = 900, epochs: int = 4):
        """Дообучить модель речи под голос персонажа: нарезка записей → расшифровка (Whisper) → коды речи →
        LoRA. Результат в voices/<char>/tts (адаптер ~60 МБ + образец)."""
        import json
        import shutil
        from . import asr, models, tts, ttsft
        p, report = self.project(gid), self._progress(gid)
        cid, char_name = char, next((c["name"] for c in p.characters() if c["id"] == char), char)
        if not models.path("tts-ru") or not models.path(ttsft.TOKENIZER_KEY):
            raise RuntimeError("Для дообучения нужны модель речи и токенайзер речи — «Система и нейросети»")
        cands = [l for l in p.lines(char=char) if 1.5 <= (l["duration"] or 0) <= 12 and l["status"] != "skipped"]
        if len(cands) < 8:
            raise RuntimeError("У персонажа слишком мало реплик для дообучения (нужно хотя бы 8)")
        report(f"Голос «{char_name}»: подбор записей…")
        cands = self._most_typical(p, cands[:400])
        # Контрольные реплики (не участвуют в обучении): по ним потом сверяем дообученный голос с отпечатком
        holdout = [l for l in cands[8:] if l["ru"] and 2.0 <= (l["duration"] or 0) <= 7][:6]
        hold_ids = {l["id"] for l in holdout}
        cands = [l for l in cands if l["id"] not in hold_ids]
        vdir = p.dir / "voices" / _safe(char)
        ddir = vdir / "tts_data"
        shutil.rmtree(ddir, ignore_errors=True)
        ddir.mkdir(parents=True)
        items, total = [], 0.0
        for i, l in enumerate(cands):
            w, sr = audio.read(p.dir / l["orig"])
            w = audio.trim_silence(w, sr)
            if len(w) < sr * 1.0:
                continue
            f = ddir / f"{i:04d}.wav"
            audio.write(f, audio.resample(w, sr, 24000), 24000)
            items.append({"audio": str(f), "line": l, "sec": len(w) / sr})
            total += len(w) / sr
            if total >= max_seconds:
                break
        if total < 45:
            raise RuntimeError("Слишком мало чистой речи персонажа для дообучения (нужно от 45 с)")
        self._release_translator()
        tts.unload()
        need = [it for it in items if not it["line"]["en"]]
        if need:
            report(f"Голос «{char_name}»: расшифровка {len(need)} записей…")
            from . import power
            texts = asr.transcribe([it["audio"] for it in need], language=None, batch_size=power.batch(8))
            for it, t in zip(need, texts):
                it["line"] = {**it["line"], "en": t}
                if t.strip():
                    p.update_line(it["line"]["id"], en=t)
            asr.unload()
        ref = next((it for it in items if 4 <= it["sec"] <= 9), items[0])
        ref_path = vdir / "tts_ref.wav"
        shutil.copy2(ref["audio"], ref_path)
        rows = [{"audio": it["audio"], "text": (it["line"]["en"] or "").strip(), "ref_audio": str(ref_path)}
                for it in items if len((it["line"]["en"] or "").strip()) > 3]
        raw = ddir / "train_raw.jsonl"
        raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
        codes = ddir / "train_with_codes.jsonl"
        report(f"Голос «{char_name}»: коды речи для {len(rows)} записей…")
        self._run_child(gid, {"op": "codes", "input_jsonl": str(raw), "output_jsonl": str(codes),
                              "tokenizer_dir": str(models.path(ttsft.TOKENIZER_KEY)), "out_dir": str(vdir / "tts")},
                        lambda info: None)
        started = time.time()

        def on_progress(info):
            if info.get("message"):
                report(f"Голос «{char_name}»: {info['message']}")
            elif info.get("step"):
                st, tot = info["step"], max(1, info.get("steps") or 1)
                left = (time.time() - started) / st * (tot - st)
                report(f"Дообучение голоса «{char_name}»: эпоха {info.get('epoch')}/{info.get('total')}, шаг {st}/{tot}, "
                       f"осталось ~{int(left // 60)} мин · loss {info.get('loss')}", st, tot)
        setup = self.ft_setup()
        method = self.settings(gid)["ft_method"]
        method = setup["method"] if method == "auto" else method
        if method == "full" and setup["vram"] and setup["vram"] < 14:
            method = "lora"  # полное дообучение в такую видеопамять не помещается
        out = vdir / ("tts_full" if method == "full" else "tts")
        shutil.rmtree(out, ignore_errors=True)
        from . import judge
        judge.unload()
        report(f"Дообучение голоса «{char_name}»: {total / 60:.1f} мин записей, {epochs} эпох…")
        if method == "full":
            res = self._run_child(gid, {"op": "train_full", "dataset_jsonl": str(codes), "out_dir": str(out), "epochs": epochs,
                                        "lr": 1e-5, "base": "tts-ru", "speaker": _safe(char), "accum": setup["accum"],
                                        "batch": setup["batch"]}, on_progress)
            final = Path(res["ckpt"])
            for d in out.glob("checkpoint-epoch-*"):  # промежуточные эпохи — по ~4 ГБ каждая, нужна только последняя
                if d.resolve() != final.resolve():
                    shutil.rmtree(d, ignore_errors=True)
            ttsft.share_tokenizer(final, models.path("tts-ru"))  # общий токенайзер речи — ссылкой, не копией (−0,65 ГБ)
            entry = {"kind": "full", "dir": str(final.relative_to(p.dir)), "speaker": _safe(char)}
        else:
            self._run_child(gid, {"dataset_jsonl": str(codes), "out_dir": str(out), "epochs": epochs, "lr": 1e-4,
                                  "base": "tts-ru", "rank": 16, "alpha": 32, "accum": 8}, on_progress)
            entry = {"kind": "lora", "dir": str(out.relative_to(p.dir))}
        shutil.rmtree(ddir, ignore_errors=True)
        took = time.time() - started
        if total and epochs:
            self._ft_record(took / (total / 60 * epochs) / (0.65 if setup["batch"] > 1 else 1.0), method, setup["batch"])
        voices = p.get("tts_voices") or {}
        voices[char] = {**entry, "ref": str(ref_path.relative_to(p.dir)), "epochs": epochs,
                        "dataset_seconds": round(total), "train_seconds": round(time.time() - started),
                        "trained_at": time.time(), "holdout": [l["id"] for l in holdout]}
        p.set("tts_voices", voices)
        verdict = self._validate_tts_voice(gid, p, char, holdout)
        self._finish(gid, f"Голос «{char_name}» дообучен за {voices[char]['train_seconds'] // 60} мин{verdict}")

    def _validate_tts_voice(self, gid: str, p: Project, char: str, holdout: list[dict]) -> str:
        """Контроль по репликам вне обучения: дообученный голос против отпечатка. Если дообученная модель
        читает хуже (срыв, пропуски слов), она отключается — персонаж озвучивается по отпечатку."""
        if not holdout:
            return ""
        from . import asr, judge, tts
        report = self._progress(gid)
        report(f"Голос «{char}»: проверка на контрольных репликах…")
        voices = p.get("tts_voices") or {}
        entry = voices.get(char) or {}
        try:
            prof = self.character_profile(p, char)
            pe = judge.speaker_embed([prof["audio"]]) if prof and judge.VE_WEIGHTS.exists() else None
            out_dir = p.dir / "voices" / _safe(char) / "val"
            res = {}
            for tag, use_voice in (("trained", True), ("base", False)):
                if not use_voice:
                    entry["disabled"] = True  # временно: _refs_for даст отпечаток
                    p.set("tts_voices", {**voices, char: entry})
                files, sim = self._render(gid, p, char, holdout, "ru", None, out_dir / tag)
                heard = asr.transcribe([str(p.dir / f) for f in files], language="russian")
                cer = float(np.mean([judge.cer(l["ru"], h) for l, h in zip(holdout, heard)]))
                res[tag] = {"cer": round(cer, 3), "sim": sim}
            entry.pop("disabled", None)
            # высота голоса: дообученная модель не должна уводить персонажа в другой тип голоса
            pf0 = self.profile_f0(p, char)
            for tag in ("trained", "base"):
                f0s = []
                for f in (out_dir / tag).glob("*.wav"):
                    w, sr = audio.read(f)
                    v = audio.f0_median(w, sr)
                    if v:
                        f0s.append(v)
                res[tag]["f0"] = round(float(np.median(f0s))) if f0s else None
            shift = abs(12 * np.log2(res["trained"]["f0"] / pf0)) if res["trained"].get("f0") and pf0 else 0
            res["trained"]["f0_shift"] = round(float(shift), 1)
            # хуже отпечатка — отключаем: срывы текста, уход в другой тип голоса или голос заметно менее похож
            less_similar = (res["trained"].get("sim") is not None and res["base"].get("sim") is not None
                            and res["trained"]["sim"] < res["base"]["sim"] - 0.03)
            bad = (res["trained"]["cer"] > 0.2 and res["trained"]["cer"] > res["base"]["cer"] + 0.1) or shift > 4 or less_similar
            entry["val"] = res
            if bad:
                entry["disabled"] = True
            voices[char] = entry
            p.set("tts_voices", voices)
            asr.unload()
            tts.unload()
            note = (f" · контроль: ошибок текста {res['trained']['cer']:.0%} против {res['base']['cer']:.0%} у отпечатка, "
                    f"похожесть {res['trained']['sim']} против {res['base']['sim']}")
            if bad:
                note += (" — дообученная модель увела высоту голоса" if shift > 4 else " — дообученная модель читает хуже") \
                        + ", персонаж озвучивается по отпечатку"
            print(f"контроль голоса «{char}»: {res}{' ОТКЛЮЧЕН' if bad else ''}")
            return note
        except Exception as e:
            entry.pop("disabled", None)
            voices[char] = entry
            p.set("tts_voices", voices)
            print(f"контроль голоса «{char}» не удался: {e}")
            return ""

    def _voice(self, p: Project, char: str) -> dict | None:
        """Обученный голос персонажа с абсолютными путями или None."""
        v = (p.get("voices") or {}).get(char)
        if v and v.get("pth") and (p.dir / v["pth"]).exists():
            return {"method": "rvc", "pth": str(p.dir / v["pth"]),
                    "index": str(p.dir / v["index"]) if v.get("index") and (p.dir / v["index"]).exists() else None}
        return None

    def _do_prepare(self, gid: str):
        """Автоматическая подготовка: реплики → русский текст → отсев неречевых → обучение голосов → пробы."""
        from . import accel
        accel.require_gpu()
        p, report = self.project(gid), self._progress(gid)
        if not p.get("extracted_at") or not p.stats()["total"]:
            self._do_extract(gid)
        for step in (self._do_subtitles, self._do_transcribe, self._do_translate, self._do_screen, self._auto_speakers):
            if self.cancel_flag.is_set():
                return
            step(gid)
        from . import accel
        s = self.settings(gid)
        if s["tts_finetune"] and not s["voice_transfer"] and self._engine_for(gid) == "ru":
            plan = self.train_plan(gid)
            todo = [c for c in plan["planned"] if not c["trained"]]
            sample_ids = {c["id"] for c in self._sample_chars(p)}
            for i, c in enumerate(todo, 1):
                if self.cancel_flag.is_set():
                    return
                report(f"Дообучение голосов: {c['name']} ({i}/{len(todo)}), {plan['total'] - len(todo) + i - 1} уже готовы")
                try:
                    self._do_train_tts(gid, c["id"], max_seconds=c["max_seconds"], epochs=c["epochs"])
                    if c["id"] in sample_ids:  # проба сразу после обучения — слушать, не дожидаясь остальных
                        self._do_samples(gid, only_char=c["id"])
                except Exception as e:
                    if self.cancel_flag.is_set():
                        return
                    print(f"дообучение голоса {c['name']} не удалось: {e}")
        elif not s["voice_transfer"]:
            pass
        elif rvc_available():
            plan = self.train_plan(gid)
            todo = [c for c in plan["planned"] if not c["trained"]]
            sample_ids = {c["id"] for c in self._sample_chars(p)}
            for i, c in enumerate(todo, 1):
                if self.cancel_flag.is_set():
                    return
                report(f"Обучение голосов: {c['name']} ({i}/{len(todo)}), {plan['total'] - len(todo) + i - 1} уже готовы")
                try:
                    self._do_train_voice(gid, c["id"], steps=c["epochs"], max_seconds=c["max_seconds"], measure=False)
                    if c["id"] in sample_ids:  # проба сразу после обучения — слушать, не дожидаясь остальных
                        self._do_samples(gid, only_char=c["id"])
                except Exception as e:
                    if self.cancel_flag.is_set():
                        return
                    print(f"обучение голоса {c['name']} не удалось: {e}")
        else:
            report("RVC не установлен — перенос тембра пропущен")
        if self.cancel_flag.is_set():
            return
        self._do_samples(gid)
        p.set("prepared_at", time.time())
        from . import cleanup
        cleanup.project_garbage(p.dir, dry=False)
        if self.settings(gid)["auto_install"]:  # режим «на ночь»: дальше без участия пользователя
            self.enqueue(gid, "generate")
            self._finish(gid, "Подготовка завершена — озвучиваем всю игру и ставим без прослушивания")
            return
        self._finish(gid, "Подготовка завершена — послушайте, как звучат персонажи")

    def _sample_chars(self, p: Project) -> list[dict]:
        sel = self.selected(p)
        chars = [c for c in p.characters() if c["id"] in sel and (c["lines"] or 0) >= 3]
        chars.sort(key=lambda c: -(c["lines"] or 0))
        return chars[:SAMPLE_CHARS]

    def _do_samples(self, gid: str, next_variant: bool = False, only_char: str | None = None):
        """Пробы: по 3 реплики главных персонажей. next_variant — другой прогон (другие случайные зёрна).
        only_char — обновить пробу одного персонажа (сразу после его обучения), остальные оставить."""
        from . import asr, tts
        p, report = self.project(gid), self._progress(gid)
        if next_variant:
            p.set("voice_variant", self._take(gid) + 1)
        engine = self._engine_for(gid)
        take = self._take(gid)
        chars = self._sample_chars(p)
        if only_char:
            chars = [c for c in chars if c["id"] == only_char] or [c for c in p.characters() if c["id"] == only_char]
        self._release_translator()
        asr.unload()
        prev = p.get("samples") if only_char else None
        result = {"engine": engine, "variant": self.variant_info(gid), "at": time.time(), "partial": bool(only_char),
                  "chars": [c for c in (prev or {}).get("chars", []) if c["id"] != only_char] if prev else []}
        for i, c in enumerate(chars, 1):
            if self.cancel_flag.is_set():
                return
            report(f"Пробы голосов: {c['name']} ({i}/{len(chars)})", i - 1, len(chars))
            cands = [l for l in p.lines(char=c["id"]) if l["ru"] and 2.0 <= (l["duration"] or 0) <= 8
                     and l["status"] != "skipped"]
            if len(cands) < SAMPLE_LINES:
                cands = [l for l in p.lines(char=c["id"]) if l["ru"] and l["status"] != "skipped"]
            if not cands:
                continue
            step = max(1, len(cands) // SAMPLE_LINES)
            lines = cands[::step][:SAMPLE_LINES]
            try:
                voice = self._voice(p, c["id"]) if self.settings(gid)["voice_transfer"] else None
                files, sim = self._render(gid, p, c["id"], lines, engine, voice,
                                          p.dir / "samples" / _safe(c["id"]), seed=100003 * take if take else None)
            except Exception as e:
                result["chars"].append({"id": c["id"], "name": c["name"], "error": str(e)[:200], "items": []})
                continue
            result["chars"].append({"id": c["id"], "name": c["name"], "similarity": sim,
                                    "trained": bool(voice) or self._tts_voice(p, c["id"]) is not None,
                                    "items": [{"id": l["id"], "ru": l["ru"], "en": l["en"], "file": f}
                                              for l, f in zip(lines, files)]})
        if only_char:  # сохранить порядок главных персонажей
            order = {c["id"]: i for i, c in enumerate(sorted(p.characters(), key=lambda c: -(c["lines"] or 0)))}
            result["chars"].sort(key=lambda c: order.get(c["id"], 999))
        p.set("samples", result)
        if not only_char:
            self._finish(gid, "Пробы готовы — послушайте")

    def _do_train_voice(self, gid: str, char: str, steps: int = 300, max_seconds: float = 720,
                        measure: bool = True, **_):
        """Обучить RVC-голос персонажа на его записях (до max_seconds речи) и измерить похожесть до/после."""
        from . import asr, tts
        p, report = self.project(gid), self._progress(gid)
        if not rvc_available():
            raise RuntimeError("RVC не установлен (runtime\\engines\\rvc)")
        cands = [l for l in p.lines(char=char) if 1.5 <= (l["duration"] or 0) <= 12 and l["status"] != "skipped"]
        if len(cands) < 3:
            raise RuntimeError("У персонажа слишком мало реплик для обучения (нужно хотя бы 3)")
        report("Подбор записей персонажа…")
        cands = self._most_typical(p, cands[:400])
        voice_dir = p.dir / "voices" / _safe(char)
        ds = voice_dir / "dataset"
        import shutil
        shutil.rmtree(voice_dir, ignore_errors=True)
        ds.mkdir(parents=True)
        total = 0.0
        for i, l in enumerate(cands):
            w, sr = audio.read(p.dir / l["orig"])
            w = audio.trim_silence(w, sr)
            if len(w) < sr * 1.0:
                continue
            audio.write(ds / f"{i:04d}.wav", audio.resample(w, sr, 22050), 22050)
            total += len(w) / sr
            if total >= max_seconds:
                break
        if total < 20:
            raise RuntimeError("Слишком мало чистой речи персонажа для обучения")
        self._release_translator()
        asr.unload()
        tts.unload()
        started = time.time()
        train_start = [None]

        def on_progress(info):
            step, tot = info.get("epoch", 0) or 0, info.get("total", steps)
            if info.get("message"):
                msg = f"Голос «{char}»: {info['message']}"
            else:
                msg = f"Обучение голоса «{char}»: эпоха {step}/{tot}"
                if step and train_start[0]:
                    left = (time.time() - train_start[0]) / step * (tot - step)
                    msg += f", осталось ~{int(left // 60)} мин {int(left % 60)} с"
            if step == 0 and info.get("phase") == "train" and train_start[0] is None:
                train_start[0] = time.time()
            if info.get("loss") is not None:
                msg += f" · loss {info['loss']}"
            report(msg, step, tot)

        voices = p.get("voices") or {}
        report(f"Обучение голоса «{char}»: {total / 60:.1f} мин записей, {steps} эпох…")
        res = tts.train_voice_rvc(str(ds), str(voice_dir), f"{p.id}__{_safe(char)}", steps, on_progress)
        entry = {"method": "rvc", "pth": str(Path(res["pth"]).relative_to(p.dir)),
                 "index": str(Path(res["index"]).relative_to(p.dir)) if res.get("index") else None,
                 "epochs": steps, "trained_at": time.time(), "dataset_seconds": round(total),
                 "train_seconds": round(time.time() - started)}
        shutil.rmtree(ds, ignore_errors=True)
        voices[char] = entry
        p.set("voices", voices)

        # Замер: те же реплики голосом диктора и обученным голосом
        test = [l for l in p.lines(char=char) if l["ru"] and 1.5 <= (l["duration"] or 0) <= 8][:3]
        if test and measure:
            report("Проверка: похожесть до и после обучения…")
            engine = self._engine_for(gid)
            try:
                _, before = self._render(gid, p, char, test, engine, None, voice_dir / "before")
                _, after = self._render(gid, p, char, test, engine, self._voice(p, char), voice_dir / "after")
                voices[char].update(sim_before=before, sim_after=after)
                p.set("voices", voices)
            except Exception as e:
                print(f"замер после обучения не удался: {e}")
        v = voices[char]
        gain = f", похожесть {v.get('sim_before')} → {v.get('sim_after')}" if v.get("sim_after") is not None else ""
        self._finish(gid, f"Голос «{char}» обучен за {v['train_seconds'] // 60} мин{gain}")

    def _do_events_pass(self, gid: str):
        """Для уже озвученных реплик: найти смех/вздохи/кашель по краям оригинала и переозвучить такие реплики
        (чтобы они вошли в русскую реплику из оригинала)."""
        p, report = self.project(gid), self._progress(gid)
        lines = [l for l in p.lines(status="generated,approved") if l["gen"] and not ((l["info"] or {}).get("fit") or {}).get("kept")]
        todo = []
        for i, l in enumerate(lines):
            if self.cancel_flag.is_set():
                return
            if i % 50 == 0:
                report(f"Поиск смеха и вздохов в оригинале: {i}/{len(lines)}", i, len(lines))
            e = self._edges(p, l)
            if e.get("events"):
                todo.append(l["id"])
        p.set("edges", p.get("edges") or {})
        if not todo:
            self._finish(gid, "Смеха и вздохов по краям реплик не найдено")
            return
        with p.db() as c:
            c.executemany("UPDATE lines SET status='translated' WHERE id=?", [(i,) for i in todo])
        report(f"Переозвучка {len(todo)} реплик со смехом/вздохами из оригинала…")
        self._do_generate(gid, ids=todo)
        if self.settings(gid)["auto_install"]:  # без прослушивания — сразу в одобренные
            with p.db() as c:
                c.executemany("""UPDATE lines SET status='approved' WHERE id=? AND status='generated'
                    AND json_extract(info, '$.judge.verdict') = 'ok'
                    AND COALESCE(json_extract(info, '$.fit.requires_review'), 0) = 0""", [(i,) for i in todo])
        self._finish(gid, f"Вклеены смех/вздохи из оригинала: {len(todo)} реплик")

    # --- контрольная сцена ---

    @staticmethod
    def _safe_name(s: str) -> str:
        return _safe(s)

    def _do_preview(self, gid: str, count: int = 2, engine: str | None = None,
                    ids: list[str] | None = None, label: str | None = None):
        from . import preview
        result = preview.run(self, gid, count, engine, ids, label)
        self._finish(gid, f"Пробы готовы: {result['label']} ({len(result['rows'])})")

    def _do_bench(self, gid: str, label: str | None = None):
        """Озвучить постоянный контрольный набор реплик текущим способом и оценить (bench.py)."""
        from . import bench
        self._release_translator()
        res = bench.run(self, gid, label)
        if res:
            self._finish(gid, f"Контрольная сцена готова: ошибок текста {res['cer']:.1%}, длина {res['ratio']}, "
                              f"похожесть {res['sim']} — projects\\…\\bench\\{res['label']}\\index.html")

    # --- установка ---

    def _do_install(self, gid: str):
        p = self.project(gid)
        if not p.count(status="approved"):
            raise RuntimeError("Нет одобренных реплик — сначала прослушайте и одобрите их")
        n = self._adapter(gid).install(p, self._game(gid), self._progress(gid))
        p.set("installed_at", time.time())
        p.set("installed_count", n)
        from . import cleanup
        cleanup.project_garbage(p.dir, dry=False)
        self._finish(gid, f"Установлено реплик: {n}")

    # --- кто говорит ---

    def _do_speakers(self, gid: str):
        """Разобрать реплики по голосам (ECAPA + кластеры + имена от языковой модели) и сохранить предложение."""
        from . import judge, speakers, tts
        p, report = self.project(gid), self._progress(gid)
        if not judge.ecapa_ready():
            raise RuntimeError("Нужна модель отпечатка голоса ECAPA — «Система и нейросети»")
        self._release_translator()
        tts.unload()
        have = speakers.embed_all(p, report, self.cancel_flag.is_set)
        judge.unload()
        if self.cancel_flag.is_set():
            return
        report("Кто говорит: группирую голоса и подбираю имена…")
        tr = self._translator_for(gid) if self._director_ready(gid) or any(
            __import__("ruvoice.translate", fromlist=["x"])._model_path(k) for k in ("qwen3-14b", "qwen3-8b", "qwen3-4b")) else None
        prop = speakers.propose(p, have, tr, p.get("game_name", ""))
        self._release_translator()
        p.set("speaker_proposal", prop)
        self._finish(gid, f"Найдено голосов: {len(prop['voices'])} (реплик в них {prop['covered']} из {prop['total']}) — "
                          "проверьте имена и примените")

    def _auto_speakers(self, gid: str):
        """Кто говорит — сам, без участия пользователя (часть подготовки, один раз на проект).

        Группы из файлов многих игр — это разговоры, а не люди (в Sekiro в разговоре с Эммой звучит и Волк),
        поэтому голос «плавал»: профиль Эммы собирался и из реплик Волка. Здесь реплики раскладываются по
        голосам (speakers.propose), и применяется надёжная часть: главный герой (сквозной голос, который звучит
        во многих разговорах) и объединение групп с одним и тем же голосом (NPC в разных разговорах). Прочие
        сквозные голоса (часто — смесь похожих голосов, например стоны раненых) не трогаются.
        Если адаптер знает говорящих точно (project meta speakers_reliable), шаг пропускается."""
        from . import judge, speakers, translate, tts
        p, report = self.project(gid), self._progress(gid)
        if p.get("speakers_applied") or p.get("speakers_reliable") or not judge.ecapa_ready():
            return
        if len(p.characters()) < 2:
            p.set("speakers_applied", {"at": time.time(), "voices": 0, "moved": 0, "skipped": "один персонаж"})
            return
        self._release_translator()
        tts.unload()
        have = speakers.embed_all(p, report, self.cancel_flag.is_set)
        judge.unload()
        if self.cancel_flag.is_set():
            return
        report("Кто говорит: раскладываю реплики по голосам…")
        tr = None
        if any(translate._model_path(k) for k in translate.MODELS):
            try:
                tr = self._translator_for(gid)
            except Exception as e:
                print(f"кто говорит: без имён от языковой модели ({e})")
        prop = speakers.propose(p, have, tr, p.get("game_name", ""))
        self._release_translator()
        skip = {v["key"] for v in prop["voices"] if v.get("kind") == "recurring" and not v.get("hero")}
        moved = speakers.apply(p, prop, {}, skip)
        hero = next((v["name"] for v in prop["voices"] if v.get("hero")), None)
        print(f"кто говорит: перенесено {moved} реплик, голосов {len(prop['voices']) - len(skip)}"
              + (f", главный герой — {hero}" if hero else ""))
        report(f"Кто говорит: реплики разложены по голосам" + (f" (главный герой — {hero})" if hero else ""))

    # --- указание к реплике от пользователя ---

    def _do_redirect(self, gid: str, ids: list[str]):
        """Переозвучить реплики по указанию пользователя (info.user_note): режиссёр переводит указание в подачу
        (пунктуация, выразительность, темп), затем озвучка с планом и проверка контролёром."""
        from . import director
        p, report = self.project(gid), self._progress(gid)
        lines = p.lines(ids=ids)
        tr = self._translator_for(gid) if self._director_ready(gid) else None
        for l in lines:
            info = dict(l["info"] or {})
            note = info.get("user_note") or ""
            plan = {"round": int((info.get("fix") or {}).get("round") or 0) + 1, "problems": ["user"], "extra": 3,
                    "plan": f"По вашему указанию: «{note}»", "steps": []}
            if tr is not None:
                d = director.fix(tr, l, f"Указание режиссёра-человека: {note}", info.get("delivery"))
                if d:
                    plan["delivery"] = d
                    plan["temperature"] = d.get("temperature")
                    plan["plan"] += f"; подача: «{d['note']}»"
            info["fix"] = plan
            p.update_line(l["id"], info=info, status="translated" if l["status"] in ("generated", "approved", "rejected", "error") else l["status"])
        self._release_translator()
        self._do_generate(gid, ids=ids, fix=True)

    # --- экспорт / импорт ---

    def _do_export(self, gid: str, only_approved: bool = False):
        from . import package
        from .paths import DATA_DIR
        p = self.project(gid)
        path = package.export_project(p, self._game(gid), self._adapter(gid).id, DATA_DIR / "exports",
                                      self._progress(gid), only_approved=only_approved)
        p.set("last_export", {"path": str(path), "at": time.time(), "mb": round(path.stat().st_size / 2**20, 1)})
        self._finish(gid, f"Озвучка сохранена в пакет: {path.name} ({path.stat().st_size / 2**20:.0f} МБ)")

    def _do_import(self, gid: str, path: str):
        """Импорт пакета: при необходимости сначала извлечь реплики из игры, затем положить русскую озвучку."""
        from . import package
        p = self.project(gid)
        if not p.get("extracted_at") or not p.stats()["total"]:
            self._do_extract(gid)
            self._do_translate(gid)
        if self.cancel_flag.is_set():
            return
        res = package.import_package(p, Path(path), self._progress(gid))
        p.set("prepared_at", p.get("prepared_at") or time.time())
        p.set("last_import", {**res, "at": time.time(), "file": Path(path).name})
        miss = f", не нашлось в этой версии игры: {res['missing']}" if res["missing"] else ""
        self._finish(gid, f"Импортировано реплик с озвучкой: {res['audio']} из {res['matched']}{miss}. "
                          "Послушайте и нажмите «Установить в игру»")

    def _do_uninstall(self, gid: str):
        p = self.project(gid)
        self._adapter(gid).uninstall(p, self._game(gid), self._progress(gid))
        p.set("installed_at", None)
        p.set("installed_count", 0)
        self._finish(gid, "Русская озвучка удалена, игра вернулась к оригиналу")


def _clean_ru(text: str) -> str:
    """Убрать ремарки в скобках «(Усмехается) Вот подожди» и служебные теги; если после этого пусто — оставить как было."""
    import re
    t = re.sub(r"\s*[\(\[<][^\)\]>]{0,60}[\)\]>]\s*", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t if re.search(r"[А-Яа-яЁёA-Za-z]", t) else (text or "").strip()


NOTE_RULES = [  # ключевые слова комментария → параметры (темп ×, высота в полутонах, температура)
    (("медленн", "неторопл", "спокойн", "размерен"), {"speed": 0.92, "temperature": 0.7}),
    (("быстр", "торопл", "энергичн"), {"speed": 1.08}),
    (("ниже", "груб", "басов", "старше", "хрипл"), {"pitch": -2}),
    (("выше", "звонч", "моложе", "детск"), {"pitch": 2}),
    (("эмоциональн", "живее", "ярче", "выразительн"), {"temperature": 0.95}),
    (("ровн", "монотон", "сдержан", "холодн"), {"temperature": 0.65}),
]


def _parse_note(note: str | None) -> dict:
    """Комментарий пользователя к персонажу → что можно применить к текущему движку речи.
    Для CustomVoice текст уходит в инструкцию целиком; для основной модели (ru) и Silero инструкций нет,
    поэтому понимаются ключевые слова: темп, высота, выразительность."""
    out: dict = {}
    t = (note or "").strip().lower()
    if not t:
        return out
    out["text"] = note.strip()
    for keys, prm in NOTE_RULES:
        if any(k in t for k in keys):
            for k, v in prm.items():
                if k == "pitch":
                    out["pitch"] = out.get("pitch", 0) + v
                elif k == "speed":
                    out["speed"] = out.get("speed", 1.0) * v
                else:
                    out[k] = v
    import re as _re
    m = _re.search(r"(?:на|на )?\s*([+-]?\d+)\s*(?:полутон|пт)", t)  # «ниже на 3 полутона»
    if m:
        v = int(m.group(1))
        out["pitch"] = -abs(v) if "ниж" in t else abs(v)
    return out


def _is_sfx_text(text: str | None) -> bool:
    """«(Cough, cough...)», «[Explosion]», «*cheers*», «AHHHH…» — ремарки и крики, а не речь: такие не озвучиваем."""
    import re
    t = (text or "").strip()
    if not t:
        return False
    if t[0] in "([*<" and t[-1] in ")]*>":
        return True
    letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", t)
    return len(letters) >= 6 and len(set(letters.lower())) <= 2  # «AHHHHH», «Mmmm»


def _safe(s: str) -> str:
    """Имя файла из произвольной строки: латиница как есть, остальное (кириллица) — через короткий хэш,
    чтобы «Аликс» и «Барни» не превращались в одно и то же имя."""
    import hashlib
    import re
    ascii_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    if ascii_part == re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") and all(ord(ch) < 128 for ch in s):
        return ascii_part or "x"
    return f"{ascii_part or 'x'}-{hashlib.sha1(s.encode('utf-8')).hexdigest()[:8]}"
