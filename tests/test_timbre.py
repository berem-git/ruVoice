import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ruvoice import audio, timbre
from ruvoice.dub import DubJobs
from ruvoice.project import Project


def tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TimbreCase(unittest.TestCase):
    """Проект с главным персонажем (20 реплик, 200 с) и эпизодическим (2 реплики, 10 с)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [patch("ruvoice.project.PROJECTS_DIR", root / "projects"),
                        patch("ruvoice.timbre.SPEED_FILE", root / "train_speed.json")]
        for p in self.patches:
            p.start()
        self.gid = "test:game"
        self.p = Project(self.gid)
        self.p.set_characters([{"id": "hero", "name": "Герой"}, {"id": "npc", "name": "Прохожий"}])
        self.p.orig_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(20):
            lid = f"h{i:02d}"
            seconds = 0.5 if lid == "h05" else 7.0
            audio.write(self.p.orig_dir / f"{lid}.wav", tone(seconds), 16000)
            rows.append({"id": lid, "char": "hero", "orig": f"orig/{lid}.wav", "duration": seconds, "en": "Hello there", "src": {}})
        for i in range(2):
            lid = f"n{i:02d}"
            audio.write(self.p.orig_dir / f"{lid}.wav", tone(5.0), 16000)
            rows.append({"id": lid, "char": "npc", "orig": f"orig/{lid}.wav", "duration": 5.0, "en": "Hi", "src": {}})
        self.p.add_lines(rows)
        for lid in [f"h{i:02d}" for i in range(20)]:
            self.p.update_line(lid, ru="Привет тебе", status="translated")
        self.p.update_line("h03", info={"speech": 0.2})
        self.p.update_line("h04", status="skipped")
        self.jobs = DubJobs.__new__(DubJobs)
        self.jobs._projects = {self.gid: self.p}
        self.jobs.lib = Mock()
        self.jobs._big_gpu = lambda: False
        self.jobs.cancel_flag = threading.Event()
        self.jobs.state = {self.gid: {"task": "train_voice", "message": "", "done": 0, "total": 0, "error": None, "queued": []}}
        self.jobs._most_typical = lambda p, lines: lines

    def tearDown(self):
        self.p.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()


class DatasetTests(TimbreCase):
    def test_dataset_filters_and_holdout(self):
        files, holdout, total = timbre.dataset(self.jobs, self.p, "hero", max_seconds=60)
        names = {f.stem for f in files}
        self.assertEqual(len(holdout), 6)
        hold_ids = {l["id"] for l in holdout}
        self.assertTrue(hold_ids.isdisjoint(names))
        self.assertTrue(hold_ids.isdisjoint({"h00", "h01", "h02", "h03", "h04", "h05", "h06", "h07"}))  # после первых восьми
        for excluded in ("h03", "h04", "h05"):
            self.assertNotIn(excluded, names)
            self.assertNotIn(excluded, hold_ids)
        self.assertLessEqual(total, 60 + 7)
        self.assertGreaterEqual(total, 45)
        self.assertTrue(all(f.parent == self.p.dir / "voices" / "hero" / "rvc" / "dataset" for f in files))
        import soundfile as sf
        self.assertEqual(sf.info(str(files[0])).samplerate, 40000)


class TrainTests(TimbreCase):
    def fake_train(self, dataset_dir, out_dir, model_name, epochs, on_progress=None, batch_size=8):
        self.train_call = {"dataset_dir": dataset_dir, "out_dir": out_dir, "model_name": model_name, "epochs": epochs,
                           "batch_size": batch_size}
        on_progress({"phase": "train", "epoch": 1, "total": epochs})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "model.pth").write_bytes(b"pth")
        return {"pth": str(Path(out_dir) / "model.pth"), "index": None}

    def test_train_writes_entry_waiting_for_calibration(self):
        with patch("ruvoice.tts.train_voice_rvc", side_effect=self.fake_train), \
             patch("ruvoice.accel.info", return_value={"vram_gb": 16}):
            res = timbre.train(self.jobs, self.gid, "hero")
        entry = self.p.get("voices")["hero"]
        self.assertEqual((entry["method"], entry["val"], entry["weak"], entry["reason"]), ("rvc", None, True, "ждёт контроля"))
        self.assertEqual(entry["epochs"], 300)
        self.assertEqual(len(entry["holdout"]), 6)
        self.assertGreater(entry["dataset_seconds"], 0)
        self.assertEqual(entry["pth"], "voices/hero/rvc/model.pth")
        self.assertEqual(self.train_call["batch_size"], 8)
        self.assertEqual(self.train_call["model_name"], f"{self.p.id}__hero")
        self.assertTrue((self.p.dir / "voices" / "hero" / "rvc" / "meta.json").is_file())
        self.assertFalse((self.p.dir / "voices" / "hero" / "rvc" / "dataset").exists())
        self.assertEqual(res["epochs"], 300)
        hist = json.loads(timbre.SPEED_FILE.read_text("utf-8"))
        self.assertEqual(hist[-1]["method"], "rvc")

    def test_small_vram_uses_batch_4(self):
        with patch("ruvoice.tts.train_voice_rvc", side_effect=self.fake_train), \
             patch("ruvoice.accel.info", return_value={"vram_gb": 8}):
            timbre.train(self.jobs, self.gid, "hero")
        self.assertEqual(self.train_call["batch_size"], 4)

    def test_cancel_leaves_no_voice_entry(self):
        def cancelled(*a, **k):
            self.jobs.cancel_flag.set()
            raise RuntimeError("остановлено")
        with patch("ruvoice.tts.train_voice_rvc", side_effect=cancelled), patch("ruvoice.accel.info", return_value={"vram_gb": 16}):
            with self.assertRaises(RuntimeError):
                timbre.train(self.jobs, self.gid, "hero")
        self.assertFalse((self.p.get("voices") or {}).get("hero"))
        self.assertFalse((self.p.dir / "voices" / "hero" / "rvc" / "meta.json").exists())

    def test_old_voice_without_val_is_not_trained(self):
        (self.p.dir / "voices" / "hero" / "rvc").mkdir(parents=True)
        (self.p.dir / "voices" / "hero" / "rvc" / "model.pth").write_bytes(b"pth")
        self.p.set("voices", {"hero": {"method": "rvc", "pth": "voices/hero/rvc/model.pth"}})
        self.assertIsNone(timbre.voice(self.p, "hero"))
        self.p.set("voices", {"hero": {"method": "rvc", "pth": "voices/hero/rvc/model.pth", "index": None,
                                       "val": {"accept": True}, "weak": False}})
        v = timbre.voice(self.p, "hero")
        self.assertEqual(v["pth"], str(self.p.dir / "voices" / "hero" / "rvc" / "model.pth"))
        self.assertIsNone(v["index"])
        self.p.set("voices", {"hero": {"method": "rvc", "pth": "voices/hero/rvc/model.pth", "val": {"accept": False}, "weak": True}})
        self.assertIsNone(timbre.voice(self.p, "hero"))


class PlanTests(TimbreCase):
    def test_estimate_uses_speed_history(self):
        timbre.SPEED_FILE.write_text(json.dumps([{"rate": 4.0, "method": "rvc"}, {"rate": 6.0, "method": "rvc"},
                                                 {"rate": 5.0, "method": "rvc"}]), "utf-8")
        self.assertEqual(timbre.speed_rate(), 5.0)
        with patch("ruvoice.engines.rvc_available", return_value=True):
            est = timbre.estimate(self.jobs, self.gid)
        normal = est["levels"]["normal"]
        self.assertEqual((normal["chars"], normal["todo"]), (1, 1))
        hero_seconds = next(c["seconds"] for c in self.p.characters() if c["id"] == "hero")
        self.assertGreaterEqual(hero_seconds, 120)
        self.assertEqual(normal["minutes"], round(min(hero_seconds / 60, 15) * 300 * 5.0 / 60))
        self.assertEqual(normal["disk_gb"], 0.06)
        self.assertEqual(est["current"], "normal")
        self.assertTrue(est["ready"])
        self.assertEqual(est["levels"]["off"]["chars"], 0)

    def test_plan_orders_mains_by_lines(self):
        self.p.set_characters([{"id": "boss", "name": "Босс"}])
        self.p.add_lines([{"id": f"b{i:02d}", "char": "boss", "orig": "orig/h00.wav", "duration": 9.0, "en": "x", "src": {}} for i in range(15)])
        self.p.set("timbre_override", {"npc": "train"})
        with patch("ruvoice.engines.rvc_available", return_value=False):
            plan = timbre.plan(self.jobs, self.gid)
        self.assertEqual([c["id"] for c in plan["planned"]], ["hero", "boss", "npc"])
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["level"], "normal")
        self.assertTrue(all(c["epochs"] == 300 and c["max_seconds"] == 900 and "est_minutes" in c and not c["trained"]
                            for c in plan["planned"]))


if __name__ == "__main__":
    unittest.main()
