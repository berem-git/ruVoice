import math
import tempfile
import threading
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ruvoice import audio, dub, timbre
from ruvoice.dub import DubJobs
from ruvoice.project import Project


class ProjectCase(unittest.TestCase):
    """Проект во временной папке и DubJobs без фоновых потоков."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch("ruvoice.project.PROJECTS_DIR", Path(self.tmp.name))
        self.patch.start()
        self.gid = "test:game"
        self.p = Project(self.gid)
        self.jobs = DubJobs.__new__(DubJobs)
        self.jobs._projects = {self.gid: self.p}
        self.jobs.lib = Mock()
        self.jobs._big_gpu = lambda: False

    def tearDown(self):
        self.p.close()
        self.patch.stop()
        self.tmp.cleanup()


class SettingsTests(ProjectCase):
    def test_v2_settings_migrate_to_v3(self):
        self.p.set("settings", {"_v": 2, "ft_level": "deep", "voice_transfer": True, "quality": "fast", "stable_voice": False})
        s = self.jobs.settings(self.gid)
        self.assertEqual(s["quality"], "fast")
        self.assertEqual(s["timbre_level"], "normal")
        self.assertEqual(s["timbre_min_seconds"], 120)
        self.assertEqual(s["episodic_mode"], "clone")
        self.assertNotIn("ft_level", s)
        self.assertNotIn("voice_transfer", s)
        self.assertNotIn("stable_voice", s)
        self.assertEqual(self.p.get("settings")["_v"], 3)

    def test_unknown_keys_are_not_saved(self):
        s = self.jobs.save_settings(self.gid, {"timbre_level": "deep", "ft_level": "light"})
        self.assertEqual(s["timbre_level"], "deep")
        self.assertNotIn("ft_level", s)
        self.assertNotIn("ft_level", self.p.get("settings"))


class SpeechSecondsTests(ProjectCase):
    def test_characters_report_speech_seconds(self):
        self.p.set_characters([{"id": "a", "name": "A"}])
        self.p.add_lines([{"id": "x1", "char": "a", "orig": "orig/x1.wav", "duration": 10.0, "en": "Hi", "src": {}},
                          {"id": "x2", "char": "a", "orig": "orig/x2.wav", "duration": 30.0, "en": "", "src": {}}])
        self.p.update_line("x2", status="skipped")
        c = self.p.characters()[0]
        self.assertEqual((c["seconds"], c["lines"], c["speech_seconds"], c["speech_lines"]), (40.0, 2, 10.0, 1))


class TierTests(unittest.TestCase):
    def test_main_needs_seconds_and_lines(self):
        s = {"timbre_level": "normal", "timbre_min_seconds": 120}
        self.assertEqual(timbre.tier({"seconds": 130, "lines": 12}, s, None), "main")
        self.assertEqual(timbre.tier({"seconds": 130, "lines": 11}, s, None), "episodic")
        self.assertEqual(timbre.tier({"seconds": 119, "lines": 50}, s, None), "episodic")

    def test_tier_counts_speech_seconds_not_all_audio(self):
        s = {"timbre_level": "normal", "timbre_min_seconds": 120}
        # 8 минут звука, но речи с текстом 13 секунд (крики и стоны без субтитров) — не главный
        self.assertEqual(timbre.tier({"seconds": 497, "lines": 40, "speech_seconds": 13, "speech_lines": 9}, s, None), "episodic")
        self.assertEqual(timbre.tier({"seconds": 497, "lines": 40, "speech_seconds": 130, "speech_lines": 12}, s, None), "main")

    def test_override_and_off(self):
        s = {"timbre_level": "normal", "timbre_min_seconds": 120}
        self.assertEqual(timbre.tier({"seconds": 10, "lines": 2}, s, "train"), "main")
        self.assertEqual(timbre.tier({"seconds": 999, "lines": 99}, s, "episodic"), "episodic")
        self.assertEqual(timbre.tier({"seconds": 999, "lines": 99}, {**s, "timbre_level": "off"}, "train"), "episodic")

    def test_levels_table(self):
        self.assertIsNone(timbre.TIMBRE_LEVELS["off"])
        self.assertEqual((timbre.TIMBRE_LEVELS["fast"]["epochs"], timbre.TIMBRE_LEVELS["fast"]["minutes"]), (150, 8))
        self.assertEqual((timbre.TIMBRE_LEVELS["normal"]["epochs"], timbre.TIMBRE_LEVELS["normal"]["minutes"]), (300, 15))
        self.assertEqual((timbre.TIMBRE_LEVELS["deep"]["epochs"], timbre.TIMBRE_LEVELS["deep"]["minutes"]), (500, 25))


def tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class PlanCase(ProjectCase):
    """Герой (20 реплик, 200 с) и прохожий (2 реплики, 10 с); донорский банк из одного мужского голоса."""

    def setUp(self):
        super().setUp()
        root = Path(self.tmp.name)
        self.p.set_characters([{"id": "hero", "name": "Герой"}, {"id": "npc", "name": "Прохожий"}])
        self.p.orig_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(20):
            lid = f"h{i:02d}"
            audio.write(self.p.orig_dir / f"{lid}.wav", tone(3.0), 16000)
            rows.append({"id": lid, "char": "hero", "orig": f"orig/{lid}.wav", "duration": 10.0, "en": "Hello there", "src": {}})
        for i in range(2):
            lid = f"n{i:02d}"
            audio.write(self.p.orig_dir / f"{lid}.wav", tone(3.0), 16000)
            rows.append({"id": lid, "char": "npc", "orig": f"orig/{lid}.wav", "duration": 5.0, "en": "Hi", "src": {}})
        self.p.add_lines(rows)
        for r in rows:
            self.p.update_line(r["id"], ru="Привет тебе", status="translated")
        self.profile = root / "profile.wav"
        audio.write(self.profile, tone(3.0), 16000)
        (self.p.dir / "voices" / "hero" / "rvc").mkdir(parents=True)
        (self.p.dir / "voices" / "hero" / "rvc" / "model.pth").write_bytes(b"pth")
        self.bank = [{"id": "m110", "gender": "male", "age": "adult", "f0": 110.0, "source": "t", "license": "t", "dir": str(root),
                      "clips": {"calm": {"file": "m110_calm.flac", "text": "Донорский текст."},
                                "angry": {"file": "m110_angry.flac", "text": "Злой текст!"}}}]
        self.jobs.cancel_flag = threading.Event()
        self.jobs._translator = None
        self.jobs.state = {self.gid: {"task": "generate", "message": "", "done": 0, "total": 0, "error": None, "queued": []}}
        self.jobs.character_profile = lambda p, char, **kw: {"audio": str(self.profile), "text": "Hello there", "seconds": 3.0}
        self.jobs.profile_f0 = lambda p, char: 120.0
        self.jobs._style_cache = {}
        self.jobs._emotions_for_char = lambda p, char: {}
        self.synth_calls = []

        def synth(engine, texts, refs, max_seconds, params=None, seed=None):
            self.synth_calls.append({"engine": engine, "texts": list(texts), "refs": list(refs), "seed": seed, "params": params})
            return [(tone(1.0, 24000), 24000) for _ in texts]
        self.convert = Mock(side_effect=lambda wav, sr, voice, params: (wav * 0.5, sr))
        self.more = [
            patch("ruvoice.donors.load", return_value=self.bank),
            patch("ruvoice.tts.synthesize", side_effect=synth),
            patch("ruvoice.tts.convert_timbre", self.convert),
            patch("ruvoice.tts.unload"), patch("ruvoice.asr.unload"), patch("ruvoice.judge.unload"), patch("ruvoice.pron.unload"),
            patch("ruvoice.asr.words", return_value=[]),
            patch("ruvoice.accel.require_gpu"), patch("ruvoice.accel.info", return_value={"device": "cpu", "vram_gb": 0}),
            patch("ruvoice.judge.ecapa_ready", return_value=False),
            patch("ruvoice.judge.VE_WEIGHTS", root / "no-ve.pt"),  # контролёр выключен даже при скачанных моделях
            patch("ruvoice.translate._model_path", return_value=None),  # режиссёр выключен даже при скачанном переводчике
            patch("ruvoice.engines.sources", return_value=["ru"]),
            patch("ruvoice.engines.cloners", return_value=["ru", "voxcpm"]),
            patch("ruvoice.engines.rvc_available", return_value=True),
        ]
        for p in self.more:
            p.start()

    def tearDown(self):
        for p in self.more:
            p.stop()
        super().tearDown()

    def accept_hero(self):
        self.p.set("voices", {"hero": {"method": "rvc", "pth": "voices/hero/rvc/model.pth", "index": None,
                                       "val": {"accept": True}, "weak": False, "epochs": 300}})


class VoicePlanTests(PlanCase):
    def test_plan_main_with_rvc_uses_donor_and_timbre(self):
        self.accept_hero()
        plan = self.jobs.voice_plan(self.p, "hero")
        self.assertEqual((plan["tier"], plan["source"], plan["timbre"], plan["reason"]), ("main", "ru", True, None))
        self.assertEqual(plan["donor"], {"id": "m110", "pitch": round(12 * math.log2(120 / 110))})
        self.assertEqual(plan["pitch"], plan["donor"]["pitch"])
        self.assertTrue(plan["rvc"]["pth"].endswith("model.pth"))
        npc = self.jobs.voice_plan(self.p, "npc")
        self.assertEqual((npc["tier"], npc["source"], npc["timbre"], npc["donor"], npc["reason"]),
                         ("episodic", "ru", False, None, "эпизодический персонаж"))
        ref = self.jobs.line_ref(self.p, self.p.line("n00"), npc)
        self.assertEqual((ref["audio"], ref["language"], ref["text"]), (str(self.profile), "en", None))

    def test_plan_without_rvc_env_falls_back_with_reason(self):
        self.accept_hero()
        with patch("ruvoice.engines.rvc_available", return_value=False):
            plan = self.jobs.voice_plan(self.p, "hero")
        self.assertEqual((plan["tier"], plan["timbre"], plan["source"], plan["reason"]),
                         ("main", False, "ru", "перенос тембра не установлен"))
        self.assertIsNone(plan["donor"])

    def test_plan_weak_model_is_episodic(self):
        self.p.set("voices", {"hero": {"method": "rvc", "pth": "voices/hero/rvc/model.pth", "val": {"accept": False}, "weak": True,
                                       "reason": "похожесть не выросла"}})
        plan = self.jobs.voice_plan(self.p, "hero")
        self.assertEqual((plan["timbre"], plan["reason"]), (False, "модель тембра не прошла контроль"))
        self.p.set("voices", {})
        self.assertEqual(self.jobs.voice_plan(self.p, "hero")["reason"], "тембр ещё не обучен")

    def test_episodic_donor_mode_has_no_timbre(self):
        self.jobs.save_settings(self.gid, {"episodic_mode": "donor"})
        plan = self.jobs.voice_plan(self.p, "npc")
        self.assertEqual((plan["source"], plan["timbre"], plan["donor"]["id"], plan["reason"]), ("ru", False, "m110", "эпизодический персонаж"))
        ref = self.jobs.line_ref(self.p, self.p.line("n00"), plan)
        self.assertEqual((ref["language"], ref["donor"], ref["style"]), ("ru", "m110", "calm"))
        with patch.object(self.jobs, "_emotions_for_char", side_effect=AssertionError("emotion must not switch the reference")):
            self.assertEqual(self.jobs.line_ref(self.p, self.p.line("n00"), plan), ref)


class GenerateTests(PlanCase):
    def test_generate_converts_timbre_only_for_main(self):
        self.accept_hero()
        self.jobs._do_generate(self.gid)
        hero = self.p.line("h00")
        self.assertEqual(hero["status"], "generated")
        self.assertTrue((self.p.dir / hero["gen"]).is_file())
        self.assertEqual(hero["info"]["plan"], {"tier": "main", "source": "ru", "donor": "m110", "style": "calm", "timbre": True})
        npc = self.p.line("n00")
        self.assertEqual(npc["info"]["plan"], {"tier": "episodic", "source": "ru", "donor": None, "style": None, "timbre": False})
        self.assertEqual(self.convert.call_count, 20 * 3)  # 20 реплик героя × 3 варианта; прохожий без тембра
        params = self.convert.call_args.args[3]
        self.assertEqual((params["pitch"], params["auto_pitch"], params["index_rate"], params["protect"]),
                         (round(12 * math.log2(120 / 110)), False, 0.0, 0.5))
        hero_calls = [c for c in self.synth_calls if c["refs"][0].get("donor")]
        npc_calls = [c for c in self.synth_calls if c["refs"][0].get("language") == "en"]
        self.assertEqual((len(hero_calls), len(npc_calls)), (20, 2))
        self.assertEqual(len(hero_calls[0]["texts"]), 3)

    def test_generate_is_deterministic_per_take(self):
        self.accept_hero()
        self.jobs._do_generate(self.gid, ids=["h00"])
        seed0 = self.synth_calls[-1]["seed"]
        self.assertEqual(seed0, dub.line_seed("h00", 0, 0))
        self.jobs._do_generate(self.gid, ids=["h00"])
        self.assertEqual(self.synth_calls[-1]["seed"], seed0)
        self.p.set("voice_variant", 1)
        self.jobs._do_generate(self.gid, ids=["h00"])
        self.assertEqual(self.synth_calls[-1]["seed"], dub.line_seed("h00", 1, 0))
        self.assertNotEqual(self.synth_calls[-1]["seed"], seed0)

    def test_fixed_centroid_not_moved_by_takes(self):
        self.accept_hero()
        self.p.set("ru_voice", {"hero": {"e": [1.0, 0.0], "fixed": True, "floor": 0.5, "n": 6, "sims": []}})
        with patch("ruvoice.judge.ecapa_ready", return_value=True), \
             patch("ruvoice.judge.ecapa_embed", side_effect=lambda items: np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(items), 1))):
            self.jobs._do_generate(self.gid, ids=["h00"])
        rv = self.p.get("ru_voice")["hero"]
        self.assertEqual(rv["e"], [1.0, 0.0])
        self.assertTrue(rv["fixed"])
        self.assertEqual(rv["sims"], [1.0])
        self.assertEqual(self.p.line("h00")["info"]["consistency"], 1.0)


class ReviewFixTests(PlanCase):
    """Замечания рецензии: пробы меняются от прогона к прогону, сбой плана одного персонажа не валит озвучку,
    сбой переноса тембра не выдаёт чужой голос за персонажа, устаревший фиксированный центр не применяется."""

    def test_samples_next_variant_changes_seeds_and_matches_generation(self):
        self.accept_hero()
        with patch("ruvoice.judge.VE_WEIGHTS", Path(self.tmp.name) / "no-ve.pt"):
            self.jobs._do_samples(self.gid)
            first = [c["seed"] for c in self.synth_calls]
            self.synth_calls.clear()
            self.jobs._do_samples(self.gid, next_variant=True)
            second = [c["seed"] for c in self.synth_calls]
        self.assertEqual(self.p.get("voice_variant"), 1)
        self.assertNotEqual(first, second)
        ids = [c["id"] for c in self.p.get("samples")["chars"][0]["items"]]
        self.assertEqual(second[:len(ids)], [dub.line_seed(i, 1, 0) for i in ids])  # те же зёрна, что у полной озвучки

    def test_samples_api_sends_next_variant(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ruvoice.api_dub import create_router
        jobs, lib = Mock(), Mock()
        lib.get.return_value = {"name": "x"}
        jobs.project.return_value = self.p
        jobs.status.return_value = {}
        with patch("ruvoice.accel.info", return_value={}):
            app = FastAPI()
            app.include_router(create_router(lib, jobs))
            with TestClient(app) as client:
                client.post(f"/api/projects/{self.gid}/samples", json={"next": True})
        jobs.enqueue.assert_called_with(self.gid, "samples", next_variant=True)

    def test_failed_plan_of_one_character_does_not_abort_others(self):
        self.accept_hero()
        real = self.jobs.voice_plan

        def plan(p, char, rvc=None):
            if char == "npc":
                raise RuntimeError("Донорский банк пуст")
            return real(p, char, rvc)
        self.jobs.voice_plan = plan
        self.jobs._do_generate(self.gid)
        self.assertEqual(self.p.line("h00")["status"], "generated")
        npc = self.p.line("n00")
        self.assertEqual(npc["status"], "error")
        self.assertIn("Донорский банк пуст", npc["error"])

    def test_failed_timbre_conversion_never_ships_donor_voice(self):
        self.accept_hero()
        self.convert.side_effect = RuntimeError("Процесс движка rvc завершился")
        self.jobs._do_generate(self.gid, ids=["h00", "n00"])
        hero = self.p.line("h00")
        self.assertEqual(hero["status"], "error")
        self.assertIsNone(hero["gen"])
        self.assertIn("тембр", hero["error"].lower())
        self.assertEqual(self.p.line("n00")["status"], "generated")  # быстрый путь без тембра не пострадал

    def test_fixed_centre_ignored_when_plan_has_no_timbre(self):
        self.accept_hero()
        self.p.set("ru_voice", {"hero": {"e": [1.0, 0.0], "fixed": True, "floor": 0.9, "n": 6, "sims": []}})
        with patch("ruvoice.judge.ecapa_ready", return_value=True), patch("ruvoice.engines.rvc_available", return_value=False), \
             patch("ruvoice.judge.ecapa_embed", side_effect=lambda items: np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(items), 1))):
            self.jobs._do_generate(self.gid, ids=["h00"])
        hero = self.p.line("h00")
        self.assertEqual(hero["status"], "generated")
        self.assertNotEqual((hero["info"].get("judge") or {}).get("verdict"), "rework")  # не судим клон по центру тембра
        self.assertEqual(self.p.get("ru_voice")["hero"]["e"], [1.0, 0.0])  # сам центр не тронут


class SeedTests(unittest.TestCase):
    def test_line_seed_formula(self):
        self.assertEqual(dub.line_seed("abc", 0, 0), (zlib.crc32(b"abc") ^ 100003) & 0x7FFFFFFF)
        self.assertEqual(dub.line_seed("abc", 1, 2), (zlib.crc32(b"abc") ^ (100003 * 2 + 977 * 2)) & 0x7FFFFFFF)
        self.assertNotEqual(dub.line_seed("abc", 0, 1), dub.line_seed("abc", 1, 1))
        self.assertEqual(dub.line_seed("abc", 0, 1), dub.line_seed("abc", 0, 1))


if __name__ == "__main__":
    unittest.main()
