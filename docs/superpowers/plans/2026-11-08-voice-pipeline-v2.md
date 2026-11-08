# Голосовой конвейер v2 — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Русская речь без акцента и один тембр на персонажа: источник речи — русский донор, тембр — обученная RVC-модель; эпизодические — быстрым клонированием; стенд для выбора движков.

**Architecture:** Три ступени (`источник речи → тембр → подгонка`) поверх существующего конвейера `dub.DubJobs`: реестр бэкендов в `engines/__init__.py` + диспетчер `tts.synthesize`, донорский банк `donors.py`, обучение/контроль тембра `timbre.py`, стенд `enginebench.py`. Полное дообучение Qwen и закреплённые эталоны удаляются.

**Tech Stack:** Python 3.12 (`runtime/python/python.exe`, torch 2.11+cu128, qwen-tts 0.1.1, transformers 4.57, speechbrain, librosa, soxr), FastAPI, vanilla JS; среды движков в `runtime/engines/<env>` (Applio/RVC, OmniVoice, VoxCPM) ставятся `tools/uv/uv.exe`; тесты — `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-08-voice-pipeline-design.md`

## Global Constraints

- Команда тестов (из `G:\ruVoice`): `set PYTHONPATH=G:\ruVoice && runtime\python\python.exe -m unittest discover -s tests -v`; тесты не ходят в сеть и не грузят модели (моки `tts.synthesize`, `tts.convert_timbre`, `judge.*`, `asr.*`, `subprocess`).
- PyTorch в `runtime/python` не переустанавливать; новые пакеты только `tools/uv/uv.exe pip install --python runtime/python/python.exe …`; среды движков — `--target runtime/engines/<env> --no-deps -r requirements-<env>.txt --link-mode copy`.
- Внутренние ключи бэкендов: `ru` (Qwen-RU), `custom` (Qwen CustomVoice), `voxcpm`, `silero`, `omnivoice`. В интерфейсе показываются `BACKENDS[key]["short"]`.
- Настройки проекта: `SETTINGS_VERSION = 3`; новые ключи `speech_source` (`auto`|ключ), `clone_engine` (`auto`|ключ), `episodic_mode` (`clone`|`donor`), `timbre_level` (`off`|`fast`|`normal`|`deep`), `timbre_min_seconds` (120); удалены `ft_*`, `tts_finetune`, `train_limit`, `train_minutes`, `train_epochs`, `train_min_seconds`, `minor_minutes`, `minor_epochs`, `voice_transfer`, `emotion_refs`, `stable_voice`, `speech_engine`.
- Уровни тембра (`timbre.TIMBRE_LEVELS`): `fast` = 150 эпох / 8 мин, `normal` = 300 / 15, `deep` = 500 / 25, `off` = None. Главный персонаж: речи ≥ `timbre_min_seconds` и реплик ≥ 12.
- Зерно варианта `k` реплики `id` при прогоне `take`: `(zlib.crc32(id.encode()) ^ (100003 * (take + 1) + 977 * k)) & 0x7FFFFFFF`.
- Порог дрейфа: `floor = median(sims) - judge.DRIFT_MARGIN` (0.12).
- Донор: кандидаты в пределах ±4 полутонов по f0 (иначе 3 ближайших), среди них — максимум косинуса ECAPA к профилю; `pitch = round(12 * log2(f0_char / f0_donor))`, в пределах ±12.
- Контроль RVC-модели: принята, если `sim_rvc > sim_base` и (`pron` недоступен или `pron_rvc <= pron_base + 0.03`) и (`profile_f0 == 0` или `|12*log2(f0_rvc/profile_f0)| <= 3`).
- Все тексты интерфейса и сообщений — по-русски, в стиле существующих (`studio.js`, `dub.py`).
- Коммит после каждой задачи; сообщение заканчивается строкой `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. У персонажа нет профиля или `profile_f0 == 0` → подбор донора не падает: без f0 — только по косинусу, без профиля — первый донор нужного пола по умолчанию `male`; тест в задаче 3.
2. Среда RVC не установлена при `timbre_level != off` → подготовка идёт дальше, главные персонажи озвучиваются быстрым путём, в `voices` видна причина `«перенос тембра не установлен»`; тест в задаче 9.
3. Старые проекты 1.0.0: `voices[char]` без поля `val` и папки `voices/<char>/tts_full` → модель не считается обученной, папки — мусор; тесты в задачах 7 и 11.
4. Отмена во время обучения RVC → `meta.json` не записан, `voices[char]` не появляется, незавершённая папка `voices/<char>/rvc` убирается уборкой; тест в задаче 7.
5. «Переделать пробы» (`take` + 1) и повторная озвучка той же реплики: другой `take` даёт другие зёрна, тот же — те же; тест в задаче 9.

---

### Task 1: Настройки v3 и уровни персонажей

**Files:**
- Create: `ruvoice/timbre.py`
- Modify: `ruvoice/dub.py:31-75` (DEFAULT_SETTINGS, SETTINGS_VERSION, FT_*), `ruvoice/dub.py:205-239` (`settings`, `save_settings`)
- Test: `tests/test_pipeline_v2.py`

**Interfaces:**
- Consumes: `Project.get/set("settings")`.
- Produces: `dub.DEFAULT_SETTINGS` (ключи из Global Constraints), `dub.SETTINGS_VERSION = 3`, `dub.DEFAULT_SOURCE = "ru"`, `dub.DEFAULT_CLONE = "ru"`, модульная функция `dub.line_seed(line_id: str, take: int, k: int) -> int` (формула из Global Constraints); `timbre.TIMBRE_LEVELS: dict`, `timbre.tier(char: dict, settings: dict, override: str | None) -> str` (`"main"` | `"episodic"`; `char` — строка `Project.characters()` с `seconds`, `lines`).

- [ ] **Step 1: Write the failing tests**

```python
class SettingsTests(unittest.TestCase):  # tests/test_pipeline_v2.py, проект во временной папке как в test_voice_identity
    def test_v2_settings_migrate_to_v3(self):
        self.p.set("settings", {"_v": 2, "ft_level": "deep", "voice_transfer": True, "quality": "fast", "stable_voice": False})
        s = self.jobs.settings("test:game")
        self.assertEqual(s["quality"], "fast")
        self.assertEqual(s["timbre_level"], "normal")
        self.assertNotIn("ft_level", s); self.assertNotIn("voice_transfer", s); self.assertNotIn("stable_voice", s)
        self.assertEqual(self.p.get("settings")["_v"], 3)

    def test_unknown_keys_are_not_saved(self):
        s = self.jobs.save_settings("test:game", {"timbre_level": "deep", "ft_level": "light"})
        self.assertEqual(s["timbre_level"], "deep"); self.assertNotIn("ft_level", s)

class TierTests(unittest.TestCase):
    def test_main_needs_seconds_and_lines(self):
        s = {"timbre_level": "normal", "timbre_min_seconds": 120}
        self.assertEqual(timbre.tier({"seconds": 130, "lines": 12}, s, None), "main")
        self.assertEqual(timbre.tier({"seconds": 130, "lines": 11}, s, None), "episodic")
        self.assertEqual(timbre.tier({"seconds": 119, "lines": 50}, s, None), "episodic")
    def test_override_and_off(self):
        s = {"timbre_level": "normal", "timbre_min_seconds": 120}
        self.assertEqual(timbre.tier({"seconds": 10, "lines": 2}, s, "train"), "main")
        self.assertEqual(timbre.tier({"seconds": 999, "lines": 99}, s, "episodic"), "episodic")
        self.assertEqual(timbre.tier({"seconds": 999, "lines": 99}, {**s, "timbre_level": "off"}, "train"), "episodic")

class SeedTests(unittest.TestCase):
    def test_line_seed_formula(self):
        self.assertEqual(dub.line_seed("abc", 0, 0), (zlib.crc32(b"abc") ^ 100003) & 0x7FFFFFFF)
        self.assertNotEqual(dub.line_seed("abc", 0, 1), dub.line_seed("abc", 1, 1))
```

`self.jobs = DubJobs.__new__(DubJobs); self.jobs._projects = {"test:game": self.p}; self.jobs.lib = Mock()`; `_big_gpu` подменить на `lambda: False`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `runtime\python\python.exe -m unittest tests.test_pipeline_v2 -v` → FAIL (`ImportError: timbre`, `KeyError: timbre_level`).

- [ ] **Step 3: Implement**

`timbre.py`: `TIMBRE_LEVELS = {"off": None, "fast": {"name": "Быстро", "epochs": 150, "minutes": 8}, "normal": {"name": "Обычно", "epochs": 300, "minutes": 15}, "deep": {"name": "Долго", "epochs": 500, "minutes": 25}}` и `tier()` по правилам Global Constraints (`override == "train"` → `main`, `override == "episodic"` → `episodic`, `timbre_level == "off"` → всегда `episodic`).

`dub.py`: новый `DEFAULT_SETTINGS` (старые ключи удалить, комментарии по-русски), `SETTINGS_VERSION = 3`, `DEFAULT_SOURCE`, `DEFAULT_CLONE`; в `settings()` при `_v < 3` оставить только ключи из `DEFAULT_SETTINGS` (и, как раньше, при `_v < 2` ещё убрать `translate_missing`), удалить блок `FT_LEVELS`/`tts_finetune`/`ft_limit`. Константы `FT_LEVELS`, `FT_SEC_PER_MIN_EPOCH`, `FT_OVERHEAD_SEC`, `FT_DISK_GB` удалить; ссылки на них в `dub.py` временно оставить компилируемыми (задача 9 их уберёт) — проще: удалить вместе с `train_plan`, `train_levels`, `ft_setup`, `_ft_rate`, `_ft_record` уже здесь, а в `api_dub.project_info` временно отдавать `train_plan={"planned": [], "off": True, "kind": "rvc"}`, `train_levels=None`.

- [ ] **Step 4: Run all tests** → `tests.test_pipeline_v2` PASS, остальные без регрессий (`test_voice_identity` ещё существует и должен проходить).

- [ ] **Step 5: Commit** — `feat: settings v3 and character timbre tiers`

---

### Task 2: Реестр бэкендов, Silero v5.6, записи моделей

**Files:**
- Modify: `ruvoice/engines/__init__.py`, `ruvoice/tts.py`, `ruvoice/models.py`, `ruvoice/basevoices.py`, `ruvoice/preview.py:19,70`
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces: `engines.BACKENDS[key] = {"name", "short", "desc", "env": str|None, "models": list[Path], "capabilities": {"stress": "acute"|"plus"|"none", "style": "punct"|"instruct"|"tags", "nonverbal_tags": bool, "clone": bool, "cross_lingual": bool, "commercial": bool}}` для `ru`, `custom`, `voxcpm`, `silero`, `omnivoice`; `engines.TTS_ENGINES = BACKENDS`; `engines.ready(key) -> bool`; `engines.available_tts() -> list[str]`; `engines.sources() -> list[str]` (готовые из `ru`, `custom`, `silero`, `omnivoice`); `engines.cloners() -> list[str]` (готовые с `cross_lingual`: `ru`, `voxcpm`, `omnivoice`); `engines.describe()` добавляет `capabilities`.
- `tts.synthesize("silero", texts, refs, max_seconds, params, seed)`: `refs[i] = {"speaker": "aidar"|"eugene"|"baya"|"kseniya"|"xenia"}`, возвращает `(wav float32, 48000)`; текст — `accent.mark(text, "silero")` → `basevoices.clean_text`.
- `basevoices.SPEAKERS`, `basevoices.MODEL = models.local_dir("silero-ru") / "v5_6_ru.pt"`, `basevoices.synth(text, speaker)`, `basevoices.pick_for(profile_emb, gender: str | None) -> str` (`gender` ограничивает выбор полом).
- `models.ALL["silero-ru"]` = `_m("Silero TTS v5.6 — русский диктор (некоммерческая лицензия CC BY-NC)", repo=None, 0.07, "tts", url="https://models.silero.ai/models/tts/ru/v5_6_ru.pt", file="v5_6_ru.pt")`; `models.ALL["omnivoice"]` = `_m("OmniVoice — русский донор и теги смеха/вздохов", "k2-fsa/OmniVoice", 2.5, "tts")` (размер уточнить при первом скачивании); `models.download()` для записей с `url` качает `urllib.request.urlretrieve` в `.part` и переименовывает; `tts-tok` удалена из `ALL` и `TIERS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_has_five_backends_with_capabilities(self):
    self.assertEqual(set(engines.BACKENDS), {"ru", "custom", "voxcpm", "silero", "omnivoice"})
    self.assertTrue(engines.BACKENDS["omnivoice"]["capabilities"]["nonverbal_tags"])
    self.assertFalse(engines.BACKENDS["silero"]["capabilities"]["commercial"])
def test_sources_and_cloners_follow_readiness(self):
    with patch("ruvoice.engines.ready", side_effect=lambda k: k in ("ru", "silero")):
        self.assertEqual(engines.sources(), ["ru", "silero"]); self.assertEqual(engines.cloners(), ["ru"])
def test_silero_synthesize_marks_stress_and_returns_48k(self):
    model = Mock(); model.apply_tts.return_value = np.zeros(4800)
    with patch("ruvoice.basevoices._load", return_value=model), patch("ruvoice.accent.plus", return_value="з+амок"), \
         patch("ruvoice.engines.available_tts", return_value=["silero"]):
        (wav, sr), = tts.synthesize("silero", ["замок"], [{"speaker": "aidar"}], [3.0], {})
    self.assertEqual(sr, 48000); self.assertEqual(model.apply_tts.call_args.kwargs["text"], "з+амок")
    self.assertEqual(model.apply_tts.call_args.kwargs["speaker"], "aidar")
def test_models_tok_removed_and_silero_url(self):
    self.assertNotIn("tts-tok", models.ALL); self.assertTrue(models.ALL["silero-ru"]["url"].endswith("v5_6_ru.pt"))
    self.assertTrue(all("tts-tok" not in t["models"] for t in models.TIERS.values()))
```

- [ ] **Step 2: Run** `tests.test_backends` → FAIL.
- [ ] **Step 3: Implement** реестр и функции по Interfaces; `tts.py`: ветка `silero` (`basevoices.synth` в цикле), ветки `ru`/`custom`/`voxcpm` без изменений кроме `_use_voice`/`_full_generate`/`_attached`, которые удаляются (ветка `ru` всегда `lang="Russian"`, промпт ICL когда `ref.get("text")` и `ref.get("language") == "ru"`, иначе x-vector); `basevoices.py`: модель v5_6 из `models`, `pick_for(emb, gender)`; `models.py` по Interfaces; `preview.py`: `TTS_ENGINES` → `BACKENDS`.
- [ ] **Step 4: Run all tests** → PASS.
- [ ] **Step 5: Commit** — `feat: backend registry with capabilities, Silero v5.6 source, model entries`

---

### Task 3: Донорский банк

**Files:**
- Create: `ruvoice/donors.py`, `tools/make_donors.py`, `ruvoice/donors/manifest.json` (+ `*.flac` после генерации)
- Modify: `.gitignore` (строка `!ruvoice/donors/*.flac` не нужна — `*.flac` не игнорируется; убедиться)
- Test: `tests/test_donors.py`

**Interfaces:**
- `donors.BANK_DIR = Path(__file__).parent / "donors"`; `donors.load(bank_dir=BANK_DIR) -> list[dict]`; запись донора: `{"id": "m_low_adult_1", "gender": "male"|"female", "age": "young"|"adult"|"old", "f0": float, "source": str, "license": str, "clips": {"calm": {"file": "m_low_adult_1_calm.flac", "text": "…"}, "angry": {...}, "sad": {...}}}`.
- `donors.style_for(emotion: str | None) -> str`: `angry`/`disgust` → `angry`; `sad`/`fearful` → `sad`; иначе `calm`.
- `donors.pick(bank: list[dict], f0: float, emb: np.ndarray | None, embed_fn) -> str` (`embed_fn(path) -> np.ndarray` единичной нормы; правила из Global Constraints; `f0 == 0` → все кандидаты; `emb is None` → ближайший по f0, а при `f0 == 0` — первый `male`).
- `donors.pitch_shift(char_f0: float, donor_f0: float) -> int`.
- `donors.reference(donor: dict, style: str, bank_dir=BANK_DIR) -> dict` = `{"audio": str(path), "text": clip text, "language": "ru", "speaker": None, "donor": id, "style": style}`; если стиля нет — `calm`.
- `donors.assigned(jobs, p, char: str) -> dict`: читает `p.get("donors")[char]`, иначе подбирает через `jobs.profile_f0(p, char)` и ECAPA профиля (`judge.ecapa_embed([prof["audio"]])[0]`, при отсутствии профиля `None`), сохраняет `{"id", "pitch", "picked_at", "manual": False}`.
- `donors.set_manual(jobs, p, char, donor_id) -> dict` (пересчитывает `pitch`, `manual: True`).
- `tools/make_donors.py`: `python tools/make_donors.py --out ruvoice/donors [--design | --wavs DIR] [--count 16] [--styles]`. `--design`: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (`qwen_tts.Qwen3TTSModel.generate_voice_design(text, language="Russian", instruct=описание)`), 16 описаний в скрипте (8 мужских, 8 женских: низкий/средний/высокий × молодой/взрослый/пожилой), текст `basevoices.TEXT_SHORT`; `--wavs`: клипы человеческой речи с рядом лежащими `.txt`; обе ветки: обрезка тишины, 6–10 с, 24 кГц моно FLAC, `f0` через `audio.f0_median`, `--styles` — `angry`/`sad` клипы той же фразой через `tts.synthesize("ru", …)` с `calm` как референс (ICL) и пунктуацией `!`/`…`, `temperature` 0.9/0.7.

- [ ] **Step 1: Write the failing tests** (банк из 3 фиктивных доноров во временной папке; `embed_fn` возвращает заданные векторы)

```python
def test_pick_prefers_pitch_band_then_similarity(self):   # доноры m110, m150, f220 Гц; профиль 145 Гц → в ±4 пт только m150 (110 → 4,8 пт, 220 → 7,2 пт), выбирается m150 даже когда вектор ближе к m110
def test_pick_falls_back_to_three_closest_when_no_band(self)
def test_pick_without_f0_uses_similarity_only(self)
def test_pick_without_profile_returns_first_male(self)
def test_style_for_and_missing_style_falls_back_to_calm(self)
def test_pitch_shift_rounding_and_clip(self)              # 220/110 → 12; 110/220 → -12; 0 → 0
def test_assigned_is_sticky_and_manual_wins(self)         # второй вызов не пересчитывает; set_manual меняет id и pitch
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `donors.py` и скрипт. **Step 4: Run** → PASS.
- [ ] **Step 5: Сгенерировать банк** (видеокарта, ~4 ГБ модель VoiceDesign в `models/`): `runtime\python\python.exe tools\make_donors.py --out ruvoice\donors --design --styles`. Ожидание: 16 доноров × 3 клипа, `manifest.json`, суммарно < 10 МБ; прослушать 3–4 клипа.
- [ ] **Step 6: Commit** — `feat: donor voice bank and per-character donor assignment`

---

### Task 4: Установщик сред движков

**Files:**
- Create: `ruvoice/engines/install.py`, `requirements-omnivoice.txt`, `requirements-rvc.txt`
- Modify: `ruvoice/api_dub.py` (`/system` + `POST /system/engines/install`), `ruvoice/web/system.js` (кнопки «Установить» у сред)
- Test: `tests/test_install.py`

**Interfaces:**
- `install.ENVS = {"voxcpm": {"requirements": "requirements-voxcpm.txt", "models": ["tts-voxcpm"]}, "omnivoice": {"requirements": "requirements-omnivoice.txt", "models": ["omnivoice"]}, "rvc": {"requirements": "requirements-rvc.txt", "applio": {"url": "<zip релиза Applio, тег закрепить>", "sha256": "<хэш>"}, "pretrained": [(url, "applio/rvc/models/pretraineds/hifi-gan/f0G40k.pth"), (url, ".../f0D40k.pth"), (url, "applio/rvc/models/predictors/rmvpe.pt"), (url, "applio/rvc/models/embedders/contentvec/pytorch_model.bin"), (url, ".../contentvec/config.json")]}}` — адреса взять из `rvc/lib/tools/prerequisites_download.py` скачанного Applio.
- `install.status() -> dict[str, {"installed": bool, "running": bool, "message": str, "error": str|None}]`; `install.install_async(engine) -> dict`; `install.install(engine, progress=lambda m: None)`.
- Шаги `install("rvc")`: `_pip_target(env, requirements, progress)` → `_download(url, dest, progress, sha256)` + `_extract_zip_strip(zip, ENGINES_DIR/"rvc"/"applio")` (срезать верхнюю папку архива) → претрейны → `models.download_async(ENVS[engine]["models"])`.
- Файлы требований: списки закреплённых пакетов без `torch`, `torchaudio`, `numpy`, `transformers` основной среды… для `rvc` — с `transformers==5.13.1` (своя версия, папка движка первая в `sys.path`). Содержимое получить из `uv pip install --dry-run` при реализации.
- API: `GET /system` → поле `engine_envs: install.status()`; `POST /system/engines/install {"engine"}` → `install_async`.

- [ ] **Step 1: Write the failing tests** (подменить `subprocess.run`, `urllib.request.urlretrieve`, `models.download_async`; архив — `zipfile` в памяти с верхней папкой `Applio-3.x/core.py`)

```python
def test_rvc_install_runs_uv_without_deps_then_extracts_and_downloads(self)   # uv args содержат "--no-deps", "--target"; после распаковки существует applio/core.py; претрейны по путям
def test_status_reports_running_and_error(self)
def test_api_install_route_starts_install(self)
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Реальная установка** `runtime\python\python.exe -c "from ruvoice.engines import install; install.install('rvc', print)"` → `runtime/engines/rvc/applio/core.py` и претрейны на месте; затем `install('omnivoice', print)` → `import omnivoice` из среды проходит (`python -c "import sys; sys.path.insert(0,'runtime/engines/omnivoice'); import omnivoice"`).
- [ ] **Step 6: Commit** — `feat: engine environment installer (RVC/Applio, OmniVoice, VoxCPM)`

---

### Task 5: Бэкенд OmniVoice

**Files:**
- Modify: `ruvoice/engines/worker.py` (класс `OmniVoiceEngine`, `ENGINES`), `ruvoice/tts.py` (ветка `omnivoice`)
- Test: `tests/test_backends.py`

**Interfaces:**
- `OmniVoiceEngine.load()`: `_setup("omnivoice")`; `from omnivoice import OmniVoice`; `self.model = OmniVoice.from_pretrained(str(MODELS / "OmniVoice"), device_map="cuda:0"|"cpu", dtype=torch.float16)`.
- `OmniVoiceEngine.generate(req)`: `req = {"text", "ref_audio", "ref_text", "seed", "params": {"steps": int}}`; при `seed` — `torch.manual_seed`; `self.model.generate(text=..., ref_audio=..., ref_text=..., num_step=params.get("steps", 32))` → `(np.float32, 24000)`; `ref_text` обязателен (`ValueError` иначе).
- `tts.synthesize("omnivoice", …)`: `client.get("omnivoice").request({"op": "generate", "text": accent.mark(text, "custom"), "ref_audio": ref["audio"], "ref_text": ref["text"], "params": params, "seed": seed, "out": tmp})` → `audio.read(out)`; английский референс допустим (`ref["text"]` — английская расшифровка профиля).

- [ ] **Step 1: Tests** (`engine.model = Mock()` как в `test_voxcpm_continuation_is_russian_only_and_opt_in`): `test_omnivoice_requires_ref_text_and_seeds`, `test_tts_omnivoice_sends_ref_text`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Smoke (GPU):** `echo {"op":"generate","text":"Привет, как дела?","ref_audio":"ruvoice/donors/m_low_adult_1_calm.flac","ref_text":"<текст клипа>","out":"cache/omni.wav"} | runtime\python\python.exe -u -m ruvoice.engines.worker omnivoice` → строка `{"ok": true, "sr": 24000, …}`.
- [ ] **Step 6: Commit** — `feat: OmniVoice backend`

---

### Task 6: RVC-работник под закреплённый Applio

**Files:**
- Modify: `ruvoice/engines/worker.py:38-148` (`RVC._env`, `_run`, `train`, `convert`), `ruvoice/tts.py` (`train_voice_rvc` принимает `batch_size`, `save_only_latest`)
- Test: `tests/test_backends.py`

**Interfaces:**
- `RVC.train(req, progress)`: `req = {"model_name", "dataset_dir", "out_dir", "epochs", "batch_size"}` → `{"pth", "index"}`; позиционные аргументы `preprocess.py`, `extract.py`, `train.py`, `extract_index.py` — в порядке, в котором их формирует `core.py` закреплённого Applio (`run_preprocess_script` и т. д.): sample rate 40000, `cut-preprocess Automatic`, `chunk 3.0`, `overlap 0.3`, `normalization none`, `f0 rmvpe`, `embedder contentvec`, `vocoder HiFi-GAN`, `pretrained` HiFi-GAN f0G40k/f0D40k, `save-only-latest`, `index Auto`.
- `RVC.convert(req)`: без изменений протокола (`pitch`, `index_rate`, `protect`, `auto_pitch`).
- `_env()`: `PYTHONPATH` = `runtime/engines/rvc`, `runtime/engines/rvc/applio`, `PROJECT_DIR` (папку `runtime/accel/cuda` добавлять только если существует).

- [ ] **Step 1: Test:** `test_rvc_train_command_lines_match_pinned_applio` — подменить `subprocess.Popen`, проверить, что для `preprocess` аргументы `[python, "rvc/train/preprocess/preprocess.py", logs, dataset, "40000", cores, "Automatic", …]` совпадают с порядком из `core.py` (значения перечислить в тесте после чтения `core.py`).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Smoke (GPU, ~5 мин):** временный скрипт в scratchpad: 20 WAV реплик Аликс (Half-Life 2, из `data/projects/<hl2>/orig` после извлечения в студии) → `tts.train_voice_rvc(dataset_dir, out_dir, "smoke", epochs=2, on_progress=print)`; ожидание: `out_dir/model.pth` создан, `tts.convert_timbre` на клипе донора возвращает звук той же длины ±10 %. Записать секунды на эпоху на минуту записей — это `timbre.RVC_SEC_PER_MIN_EPOCH` (задача 7).
- [ ] **Step 6: Commit** — `feat: RVC worker aligned with pinned Applio scripts`

---

### Task 7: `timbre.py` — обучение и контроль тембра

**Files:**
- Modify: `ruvoice/timbre.py`, `ruvoice/dub.py` (`_do_train_voice` → `timbre.train`; `_most_typical` остаётся)
- Test: `tests/test_timbre.py`

**Interfaces:**
- `timbre.RVC_SEC_PER_MIN_EPOCH: float` (замер задачи 6), `timbre.speed_rate() -> float` (медиана `data/train_speed.json` записей с `"method": "rvc"`, иначе константа), `timbre.record_speed(rate, batch)`.
- `timbre.plan(jobs, gid) -> dict` = `{"level", "ready": engines.rvc_available(), "planned": [{"id", "name", "minutes", "trained", "weak", "epochs", "max_seconds", "est_minutes"}], "total_minutes", "skipped": [names]}` — персонажи с `tier == "main"`, по убыванию `lines`.
- `timbre.estimate(jobs, gid) -> dict` = `{"levels": {key: {"name", "chars", "todo", "minutes", "disk_gb"}}, "current", "ready"}`; `disk_gb = 0.06 * chars`.
- `timbre.dataset(jobs, p, char, max_seconds) -> (files: list[Path], holdout: list[dict], total_seconds: float)`: папка `voices/<safe>/rvc/dataset`, WAV 40 кГц; правила: `1.0 с` после `audio.trim_silence`, `status != skipped`, `info.speech is None or >= 0.5`, порядок `jobs._most_typical`, holdout — 6 реплик с `ru`, 2–7 с, из кандидатов после первых 8.
- `timbre.train(jobs, gid, char) -> dict`: уровень из настроек; `tts.train_voice_rvc(dataset_dir, out_dir=voices/<safe>/rvc, model_name=f"{p.id}__{safe}", epochs, on_progress, batch_size=8 if vram>=12 else 4)`; `record_speed`; `meta.json` в `voices/<safe>/rvc`; запись `voices[char] = {"method": "rvc", "pth", "index", "epochs", "dataset_seconds", "train_seconds", "trained_at", "holdout": [ids], "val": None, "weak": True, "reason": "ждёт контроля"}` (контроль — `timbre.calibrate`, задача 9); dataset удаляется; при `jobs.cancel_flag` — `RuntimeError("остановлено")`, записи нет.
- `timbre.voice(p, char) -> dict | None`: `{"method": "rvc", "pth": abs, "index": abs|None}` только если `val` непустой, не `weak` и файл существует.

- [ ] **Step 1: Tests** (подменить `tts.train_voice_rvc`, `judge.*`, `accel.info`):
`test_dataset_filters_and_holdout`, `test_train_writes_entry_waiting_for_calibration`, `test_cancel_leaves_no_voice_entry`, `test_old_voice_without_val_is_not_trained`, `test_estimate_uses_speed_history`, `test_plan_orders_mains_by_lines`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** — `feat: timbre training, calibration and estimates`

---

### Task 8: Стенд движков

**Files:**
- Create: `ruvoice/enginebench.py`
- Delete: `ruvoice/bench.py`
- Modify: `ruvoice/dub.py` (`_do_bench` → `_do_enginebench(gid, variants, label=None)`), `ruvoice/api_dub.py` (`POST /projects/{gid}/enginebench`, `GET /projects/{gid}/bench?label=` → `FileResponse` index.html), `ruvoice/web/studio.js` (`benchPanel`: чекбоксы готовых вариантов + кнопка «Сравнить движки», ссылки на прогоны)
- Test: `tests/test_enginebench.py`

**Interfaces:**
- `enginebench.parse_variant(s: str) -> dict` = `{"backend", "english": bool, "timbre": bool}` для грамматики `backend[@en][+rvc]`; `ValueError` для неизвестного бэкенда.
- `enginebench.variants_available(p, jobs) -> list[str]`: все готовые `sources()` (+`+rvc`, если у хотя бы одного персонажа стенда есть `timbre.voice`), все готовые `cloners()` с `@en`.
- `enginebench.pick_lines(jobs, p, chars=3, per_char=10) -> list[dict]` (перенос `bench.pick_lines`).
- `enginebench.run(jobs, gid, variants: list[str], label=None, chars=3, per_char=10) -> dict`: для каждой реплики и варианта — референс: `@en` → `{"audio": профиль персонажа, "text": None}` (для `omnivoice@en` — `text` = английский текст профиля), иначе донор по `donors.assigned` и стилю эмоции; `tts.synthesize(backend, …, seed=line_seed(id, 0, 0))`; `+rvc` → `tts.convert_timbre(...)` с `timbre.voice`; метрики на сыром звуке: `pron_cer` (`pron.check`), `whisper_cer` (`asr.transcribe` + `judge.cer`), `ecapa_sim` (к `judge.ecapa_embed([профиль])`), `f0_range` (`audio.f0_stats`), `ratio`, `seconds`; результат `bench/<label>/rows.json`, `index.html` (таблица реплика × вариант с плеерами, сводка средних по вариантам), запись в `p.bench` (`{"label", "variants", "at", "summary": {variant: {metrics}}}`).
- CLI: `python -m ruvoice.enginebench --game <gid> --variants ru,ru+rvc,silero+rvc,ru@en,voxcpm@en [--label]`.

- [ ] **Step 1: Tests:** `test_parse_variant`, `test_run_builds_rows_for_each_variant_and_summary` (подменённые `tts.synthesize`, `tts.convert_timbre`, `pron.check`, `asr.transcribe`, `judge.*`; 2 реплики × 2 варианта → 4 строки, `index.html` содержит имена вариантов), `test_english_variant_uses_profile_not_donor`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** — `feat: engine bench replacing the fixed control scene`

---

### Task 9: Конвейер озвучки на планах персонажей

**Files:**
- Modify: `ruvoice/dub.py` (`voice_plan`, `line_ref`, `_refs_for`/`_ref_for_line` удалить, `_do_generate`, `_render`, `_do_samples`, `_do_prepare`, `_ru_voice`, `_ru_voice_note`, `_rvc_params` удалить, `_do_train_tts`/`_validate_tts_voice`/`_tts_voice`/`_run_child` удалить), `ruvoice/timbre.py` (`calibrate`, вызов из `train`), `ruvoice/preview.py` (вызов `_render`)
- Test: `tests/test_pipeline_v2.py`, `tests/test_timbre.py`

**Interfaces:**
- `timbre.calibrate(jobs, gid, char, holdout: list[dict]) -> dict` = `{"accept": bool, "reason": str, "sim_base", "sim_rvc", "pron_base", "pron_rvc", "f0_rvc", "f0_shift"}`: рендер holdout через `jobs._render(gid, p, char, holdout, voices/<safe>/rvc/calib/<tag>, timbre=flag)` для `tag` `base` (без тембра) и `rvc`; `sim_*` — `judge.voice_similarity` к VoiceEncoder-профилю, `pron_*` — средний `pron.check(...)["cer"]` (None, если `pron.ready()` ложно), `f0_rvc` — медиана `audio.f0_median`; правило принятия из Global Constraints; при `accept` → `voices[char].val = {...}`, `weak = False`, `ru_voice[char] = {"e": ECAPA-центр записей `rvc`, "fixed": True, "floor": median(sims) - DRIFT_MARGIN, "n": 6}`; иначе `weak = True`, `reason`. `timbre.train` вызывает `calibrate` последним шагом.
- `DubJobs.voice_plan(p, char) -> dict` = `{"tier", "source", "clone", "donor": {"id","pitch"}|None, "rvc": timbre.voice|None, "timbre": bool, "reason": str|None, "speaker": str|None}`: `tier = timbre.tier(...)`; `main` и `rvc` есть → `source = settings.speech_source` (`auto` → `DEFAULT_SOURCE`, но только из `engines.sources()`, иначе первый готовый), `timbre = True`; иначе: `episodic_mode == "donor"` → донор без тембра; `clone` → `settings.clone_engine` (`auto` → `DEFAULT_CLONE` из `cloners()`), референс — профиль оригинала; `reason` объясняет быстрый путь (`"перенос тембра не установлен"`, `"модель тембра не прошла контроль"`, `"эпизодический персонаж"`). Для `source in ("custom", "silero")` — `speaker` (`_custom_speaker_for` / `basevoices.pick_for(emb, gender)`).
- `DubJobs.line_ref(p, line, plan) -> dict`: донор → `donors.reference(donor, donors.style_for(emotion))`; клон → `{"audio": профиль, "text": профиль.text if source=="omnivoice" else None, "language": "en"}`; `custom`/`silero` → `{"speaker"}`.
- `DubJobs.line_seed(line_id, take, k) -> int` (формула из Global Constraints).
- `DubJobs._render(gid, p, char, lines, out_dir, seed=None, plan=None, timbre=True) -> (files, sim)`.
- `_do_generate`: план на персонажа один раз; `tts.synthesize(plan["source"], …)`; при `plan["timbre"]` — `tts.convert_timbre(wav, sr, plan["rvc"], {"pitch": plan["donor"]["pitch"], "auto_pitch": False, "index_rate": 0.0, "protect": 0.5})`; центр из `_ru_voice` (фиксированный, если `ru_voice[char]["fixed"]`); `_ru_voice_note(p, char, sim)` только копит `sims`; `info.plan = {"tier","source","donor","style","timbre"}`.
- `_do_prepare`: после `_auto_speakers` — `timbre.plan` → для каждого непройденного `timbre.train` (ошибки одного персонажа не останавливают остальных; отмена — останавливает), проба персонажа после его обучения (`_do_samples(only_char=...)`); при `not engines.rvc_available()` — пропуск с `report("Перенос тембра не установлен — главные персонажи пойдут быстрым путём")`.

- [ ] **Step 1: Tests** (`tts.synthesize`, `tts.convert_timbre`, `judge.*`, `pron.*`, `asr.*`, `accel.*` подменены; проект с 2 персонажами: 200 с/20 реплик и 10 с/2 реплики):
`test_plan_main_with_rvc_uses_donor_and_timbre`, `test_plan_without_rvc_env_falls_back_with_reason`, `test_plan_weak_model_is_episodic`, `test_generate_converts_timbre_only_for_main`, `test_generate_is_deterministic_per_take`, `test_fixed_centroid_not_moved_by_takes`, `test_episodic_donor_mode_has_no_timbre`; в `tests/test_timbre.py` — `test_accept_rule_matrix` (4 случая: похожесть выросла → accept; не выросла → reject «похожесть не выросла»; pron хуже на 0.05 → reject; f0 на 4 пт → reject) и `test_calibrate_sets_fixed_centroid`.
- [ ] **Step 2–4:** FAIL → implement → PASS (и `test_voice_identity` удалить в задаче 11; до неё падающие его тесты, завязанные на удалённые функции, переписать или пометить `skip` с причиной).
- [ ] **Step 5: Commit** — `feat: per-character voice plans: donor speech, RVC timbre, deterministic seeds`

---

### Task 10: «Кто говорит» — дораскладка, API «Голоса», минимальный экран

**Files:**
- Modify: `ruvoice/speakers.py` (`attach_rest`), `ruvoice/dub.py` (`_auto_speakers`), `ruvoice/api_dub.py` (`GET /projects/{gid}/voices`, `POST …/voices/donor`, `POST …/voices/tier`, `GET /donors`, `GET /projects/{gid}/audio?kind=donor&id=&style=`), `ruvoice/web/studio.js` (`voicesPanel`, `timbreLevelPicker` вместо `trainLevelPicker`, удалить кнопки эталонов и «Пробы постоянного голоса»)
- Test: `tests/test_speakers_attach.py`, `tests/test_api_voices.py`

**Interfaces:**
- `speakers.attach_rest(p, have: dict) -> int`: центры — `_robust` по отпечаткам реплик каждого персонажа с ≥ 3 отпечатками; кандидаты — реплики с отпечатком, чей `own` к своему центру < `OWN_COS` (0.45); перенос, если лучший косинус ≥ `ATTACH_COS` (0.38) и превышает второй на ≥ 0.08; `UPDATE lines SET char`; возвращает число перенесённых. Вызывается в `_auto_speakers` после `speakers.apply`, число пишется в `speakers_applied.attached`.
- `GET /projects/{gid}/voices` → `[{"id","name","lines","seconds","f0","voice_type","tier","override","donor": {"id","gender","age","pitch","manual"}|None,"timbre": {"state": "trained"|"weak"|"none"|"training","reason"},"samples": [3 id]}]`.
- `POST /voices/donor {"char","donor"}` → `donors.set_manual`; `POST /voices/tier {"char","tier": "train"|"episodic"|"auto"}` → `timbre_override` (`auto` удаляет); оба 409 при `dub.current == gid`.
- `GET /donors` → `donors.load()` с полями без путей + `url` клипов.
- `studio.js`: `voicesPanel(info)` — таблица по `GET /voices` (имя, реплик, тип голоса, донор `<select>`, уровень `<select>`, тембр-статус, 3 кнопки ▶ оригинал, ▶ донор), `timbreLevelPicker` по `timbre.estimate` (из `project_info.timbre`), кнопка «Обучить сейчас».

- [ ] **Step 1: Tests:** `test_attach_moves_clear_outsiders_only` (синтетические векторы: реплика группы A ближе к центру B на 0.3 → переносится; реплика с разницей 0.05 — нет), `test_voices_api_lists_tier_donor_and_timbre`, `test_voices_api_busy_guard`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Ручная проверка в браузере:** `runtime\python\python.exe -m ruvoice.main --browser` → студия Half-Life 2 → панель «Голоса» рисуется, смена донора сохраняется после перезагрузки страницы.
- [ ] **Step 6: Commit** — `feat: attach stray lines to voices, voices API and panel`

---

### Task 11: Удаление старого, уборка, модели, документы

**Files:**
- Delete: `ruvoice/ttsft.py`, `ruvoice/identity.py`, `ruvoice/voice_trials.py`, `tools/qwen3-tts-finetune/`, `tests/test_voice_identity.py`
- Modify: `ruvoice/api_dub.py` (удалить `voice-identities`, `voice-trials`, `train-voice` только RVC), `ruvoice/cleanup.py`, `ruvoice/models.py` (TIERS без `tts-tok`; `tts-custom` в наборы не входит), `ruvoice/web/studio.js` (остатки эталонов), `README.md`, `KNOWN_ISSUES.md`, `THIRD_PARTY_NOTICES.md` (Applio MIT, OmniVoice Apache-2.0, Silero CC BY-NC, донорский банк — источник и лицензия), `.gitignore` (`tools/qwen3-tts-finetune` строки)
- Test: `tests/test_cleanup_v2.py`

**Interfaces:**
- `cleanup.project_garbage`: мусор — `voices/*/tts`, `voices/*/tts_full`, `voices/*/tts_data`, `voices/*/val`, `voices/*/dataset`, `voices/*/before`, `voices/*/after`, `voices/*/rvc/dataset`, `voices/*/rvc` без `meta.json`, `voice_identities/`; не мусор — `voices/*/rvc/{model.pth,model.index,meta.json,calib/}` при записи в `meta.voices`.

- [ ] **Step 1: Tests:** `test_old_finetune_and_anchor_dirs_are_garbage`, `test_rvc_model_with_meta_is_kept`, `test_unfinished_rvc_dir_is_garbage`.
- [ ] **Step 2–4:** FAIL → implement → PASS; полный прогон тестов зелёный; `grep -rn "ttsft\|identity\|voice_trials\|tts_voices\|ft_level" ruvoice tests` пусто.
- [ ] **Step 5: Commit** — `refactor: remove Qwen fine-tuning and voice anchors; cleanup rules for v2`

---

### Task 12: Приёмка на видеокарте и решения стенда

**Files:**
- Modify: `docs/superpowers/specs/2026-11-08-voice-pipeline-design.md` (раздел 8), `ruvoice/dub.py` (`DEFAULT_SOURCE`, `DEFAULT_CLONE`), `ruvoice/timbre.py` (`RVC_SEC_PER_MIN_EPOCH`)

- [ ] **Step 1:** Подготовка Half-Life 2 в студии (`--browser`): извлечение, субтитры, «кто говорит»; в «Голосах» 3 главных персонажа с донорами.
- [ ] **Step 2:** Обучить одного персонажа уровнем `fast`; записать секунды/эпоху/минуту в `RVC_SEC_PER_MIN_EPOCH`; контроль принят (`sim_rvc > sim_base`).
- [ ] **Step 3:** Стенд: `python -m ruvoice.enginebench --game <gid> --variants ru,ru+rvc,custom,custom+rvc,silero+rvc,omnivoice,omnivoice+rvc,ru@en,voxcpm@en,omnivoice@en` → `index.html`; совместное прослушивание с пользователем; критерии: `pron_cer` ниже у источников с донором, `ecapa_sim` выше у `+rvc`.
- [ ] **Step 4:** Зафиксировать `DEFAULT_SOURCE`/`DEFAULT_CLONE`, заполнить раздел 8 спецификации, решить про `omnivoice` в релизе.
- [ ] **Step 5:** Озвучка 30 реплик (`generate` по ids) → контролёр → установка в Half-Life 2 → запуск игры пользователем → «Удалить озвучку» возвращает оригинал.
- [ ] **Step 6: Commit** — `chore: bench decisions and measured training speed`
