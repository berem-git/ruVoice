import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ruvoice import audio, enginebench
from ruvoice.dub import DubJobs
from ruvoice.project import Project


def tone(seconds: float, sr: int = 24000, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class VariantTests(unittest.TestCase):
    def test_parse_variant(self):
        self.assertEqual(enginebench.parse_variant("ru"), {"backend": "ru", "english": False, "timbre": False})
        self.assertEqual(enginebench.parse_variant("ru@en+rvc"), {"backend": "ru", "english": True, "timbre": True})
        self.assertEqual(enginebench.parse_variant("silero+rvc"), {"backend": "silero", "english": False, "timbre": True})
        with self.assertRaises(ValueError):
            enginebench.parse_variant("nope")
        with self.assertRaises(ValueError):
            enginebench.parse_variant("silero@en")  # Silero не клонирует с английской записи

    def test_variants_available(self):
        with patch("ruvoice.engines.sources", return_value=["ru", "silero"]), patch("ruvoice.engines.cloners", return_value=["ru"]):
            self.assertEqual(enginebench.variants_available(timbre=False), ["ru", "silero", "ru@en"])
            self.assertEqual(enginebench.variants_available(timbre=True), ["ru", "ru+rvc", "silero", "silero+rvc", "ru@en"])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [patch("ruvoice.project.PROJECTS_DIR", root / "projects")]
        for p in self.patches:
            p.start()
        self.gid = "test:game"
        self.p = Project(self.gid)
        self.p.set_characters([{"id": "hero", "name": "Герой"}, {"id": "npc", "name": "Прохожий"}])
        self.p.orig_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for char, lid in (("hero", "h1"), ("hero", "h2"), ("npc", "n1")):
            audio.write(self.p.orig_dir / f"{lid}.wav", tone(2.0, 16000), 16000)
            rows.append({"id": lid, "char": char, "orig": f"orig/{lid}.wav", "duration": 2.0, "en": "Hello there", "src": {}})
        self.p.add_lines(rows)
        for lid in ("h1", "h2", "n1"):
            self.p.update_line(lid, ru="Привет тебе", status="translated")
        self.profile = root / "profile.wav"
        audio.write(self.profile, tone(3.0, 16000), 16000)
        self.jobs = DubJobs.__new__(DubJobs)
        self.jobs._projects = {self.gid: self.p}
        self.jobs.lib = Mock()
        self.jobs._big_gpu = lambda: False
        self.jobs.cancel_flag = threading.Event()
        self.jobs.state = {self.gid: {"task": "enginebench", "message": "", "done": 0, "total": 0, "error": None, "queued": []}}
        self.jobs.character_profile = lambda p, char, **kw: {"audio": str(self.profile), "text": "Hello there", "seconds": 3.0}
        self.jobs.profile_f0 = lambda p, char: 120.0
        self.jobs._style_cache = {}
        self.jobs._emotions_for_char = lambda p, char: {}
        bank = [{"id": "m110", "gender": "male", "age": "adult", "f0": 110.0, "source": "t", "license": "t", "dir": str(root),
                 "clips": {"calm": {"file": "m110_calm.flac", "text": "Донорский текст."}}}]
        self.synth_calls = []

        def synth(engine, texts, refs, max_seconds, params=None, seed=None):
            self.synth_calls.append({"engine": engine, "refs": refs, "seed": seed, "params": params})
            return [(tone(1.0, 24000, 300.0 if engine == "ru" else 200.0), 24000) for _ in texts]
        self.more = [
            patch("ruvoice.donors.load", return_value=bank),
            patch("ruvoice.tts.synthesize", side_effect=synth),
            patch("ruvoice.tts.convert_timbre", side_effect=lambda wav, sr, voice, params: (wav * 0.5, sr)),
            patch("ruvoice.tts.unload"),
            patch("ruvoice.asr.unload"),
            patch("ruvoice.judge.unload"),
            patch("ruvoice.pron.unload"),
            patch("ruvoice.pron.ready", return_value=True),
            patch("ruvoice.pron.check", side_effect=lambda items, texts: [{"cer": 0.05, "heard": "привет тебе", "words": []} for _ in items]),
            patch("ruvoice.asr.transcribe", side_effect=lambda paths, **kw: ["Привет тебе" for _ in paths]),
            patch("ruvoice.judge.ecapa_ready", return_value=True),
            patch("ruvoice.judge.ecapa_embed", side_effect=lambda paths: np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(paths), 1))),
            patch("ruvoice.timbre.voice", side_effect=lambda p, char: {"method": "rvc", "pth": "m.pth", "index": None} if char == "hero" else None),
        ]
        for p in self.more:
            p.start()

    def tearDown(self):
        self.p.close()
        for p in self.more + self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_run_builds_rows_for_each_variant_and_summary(self):
        res = enginebench.run(self.jobs, self.gid, ["ru", "ru+rvc"], label="t1", chars=2, per_char=1)
        self.assertEqual(res["label"], "t1")
        self.assertEqual(res["variants"], ["ru", "ru+rvc"])
        self.assertEqual(len(res["rows"]), 4)  # 2 реплики × 2 варианта
        ok = [r for r in res["rows"] if not r.get("error")]
        self.assertEqual(len(ok), 3)  # у прохожего нет модели тембра → ru+rvc с ошибкой
        bad = next(r for r in res["rows"] if r.get("error"))
        self.assertEqual((bad["variant"], bad["char"]), ("ru+rvc", "npc"))
        row = next(r for r in ok if r["variant"] == "ru+rvc")
        for key in ("pron_cer", "whisper_cer", "ecapa_sim", "f0_range", "ratio", "seconds", "generated"):
            self.assertIn(key, row)
        self.assertEqual(row["pron_cer"], 0.05)
        self.assertEqual(row["whisper_cer"], 0.0)
        self.assertAlmostEqual(row["ecapa_sim"], 1.0, places=3)
        self.assertTrue((self.p.dir / row["generated"]).is_file())
        self.assertIn("ru+rvc", res["summary"])
        self.assertEqual(res["summary"]["ru+rvc"]["rows"], 1)
        self.assertEqual(res["summary"]["ru"]["rows"], 2)
        out = self.p.dir / "bench" / "t1"
        self.assertTrue((out / "rows.json").is_file())
        html = (out / "index.html").read_text("utf-8")
        self.assertIn("ru+rvc", html)
        self.assertIn("Герой", html)
        self.assertEqual(self.p.get("bench")[-1]["label"], "t1")
        self.assertEqual(self.p.get("bench")[-1]["variants"], ["ru", "ru+rvc"])
        donor_calls = [c for c in self.synth_calls if c["engine"] == "ru"]
        self.assertTrue(all(c["refs"][0]["language"] == "ru" and c["refs"][0]["donor"] == "m110" for c in donor_calls))
        self.assertTrue(all(isinstance(c["seed"], int) for c in self.synth_calls))

    def test_english_variant_uses_profile_not_donor(self):
        with patch("ruvoice.donors.assigned") as assigned:
            res = enginebench.run(self.jobs, self.gid, ["ru@en"], label="t2", chars=1, per_char=1)
        assigned.assert_not_called()
        ref = self.synth_calls[0]["refs"][0]
        self.assertEqual((ref["audio"], ref["language"], ref["text"]), (str(self.profile), "en", None))
        self.assertEqual(len(res["rows"]), 1)

    def test_pick_lines_is_fixed_in_project(self):
        first = enginebench.pick_lines(self.jobs, self.p, chars=2, per_char=1)
        self.assertEqual([l["char"] for l in first], ["hero", "npc"])
        self.assertEqual(self.p.get("bench_lines"), [l["id"] for l in first])
        self.assertEqual([l["id"] for l in enginebench.pick_lines(self.jobs, self.p, chars=2, per_char=1)], [l["id"] for l in first])


if __name__ == "__main__":
    unittest.main()
