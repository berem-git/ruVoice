import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ruvoice import donors
from ruvoice.project import Project


def make_bank(folder: Path) -> list[dict]:
    rows = [
        {"id": "m110", "gender": "male", "age": "adult", "f0": 110.0, "source": "test", "license": "test",
         "clips": {"calm": {"file": "m110_calm.flac", "text": "Спокойный текст."},
                   "angry": {"file": "m110_angry.flac", "text": "Злой текст!"}}},
        {"id": "m150", "gender": "male", "age": "young", "f0": 150.0, "source": "test", "license": "test",
         "clips": {"calm": {"file": "m150_calm.flac", "text": "Спокойный текст."}}},
        {"id": "f220", "gender": "female", "age": "adult", "f0": 220.0, "source": "test", "license": "test",
         "clips": {"calm": {"file": "f220_calm.flac", "text": "Спокойный текст."}}},
    ]
    (folder / "manifest.json").write_text(json.dumps({"donors": rows}, ensure_ascii=False), "utf-8")
    for d in rows:
        for c in d["clips"].values():
            (folder / c["file"]).write_bytes(b"flac")
    return donors.load(folder)


VECTORS = {"m110": [1.0, 0.0], "m150": [0.0, 1.0], "f220": [0.7, 0.7]}


def embed_by_name(path) -> np.ndarray:
    return np.array(VECTORS[Path(path).name.split("_")[0]], dtype=np.float32)


class PickTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bank = make_bank(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_pick_prefers_pitch_band_then_similarity(self):
        # профиль 145 Гц: в ±4 пт только m150 (110 → 4,8 пт, 220 → 7,2 пт) — выбирается он, хотя вектор ближе к m110
        self.assertEqual(donors.pick(self.bank, 145.0, np.array([1.0, 0.0]), embed_by_name), "m150")

    def test_pick_falls_back_to_three_closest_when_no_band(self):
        self.assertEqual(donors.pick(self.bank, 400.0, np.array([1.0, 0.0]), embed_by_name), "m110")

    def test_pick_without_f0_uses_similarity_only(self):
        self.assertEqual(donors.pick(self.bank, 0.0, np.array([0.0, 1.0]), embed_by_name), "m150")

    def test_pick_without_profile_returns_first_male_or_closest_pitch(self):
        self.assertEqual(donors.pick(self.bank, 0.0, None, embed_by_name), "m110")
        self.assertEqual(donors.pick(self.bank, 215.0, None, embed_by_name), "f220")
        self.assertIsNone(donors.pick([], 100.0, None, embed_by_name))

    def test_style_for_and_missing_style_falls_back_to_calm(self):
        self.assertEqual([donors.style_for(e) for e in ("angry", "disgust", "sad", "fearful", "happy", None)],
                         ["angry", "angry", "sad", "sad", "calm", "calm"])
        m150 = donors.by_id(self.bank, "m150")
        ref = donors.reference(m150, "angry")
        self.assertTrue(ref["audio"].endswith("m150_calm.flac"))
        self.assertEqual(ref["style"], "calm")
        m110 = donors.by_id(self.bank, "m110")
        ref = donors.reference(m110, "angry")
        self.assertTrue(ref["audio"].endswith("m110_angry.flac"))
        self.assertEqual((ref["text"], ref["language"], ref["donor"], ref["style"], ref["speaker"]),
                         ("Злой текст!", "ru", "m110", "angry", None))

    def test_pitch_shift_rounding_and_clip(self):
        self.assertEqual(donors.pitch_shift(220.0, 110.0), 12)
        self.assertEqual(donors.pitch_shift(110.0, 220.0), -12)
        self.assertEqual(donors.pitch_shift(0.0, 110.0), 0)
        self.assertEqual(donors.pitch_shift(110.0, 0.0), 0)
        self.assertEqual(donors.pitch_shift(146.0, 110.0), round(12 * math.log2(146 / 110)))
        self.assertEqual(donors.pitch_shift(440.0, 110.0), 12)


class AssignedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bank = make_bank(Path(self.tmp.name))
        self.patches = [patch("ruvoice.project.PROJECTS_DIR", Path(self.tmp.name) / "projects"),
                        patch("ruvoice.donors.load", return_value=self.bank),
                        patch("ruvoice.judge.ecapa_embed", side_effect=lambda paths: np.stack([embed_by_name(p) for p in paths])),
                        patch("ruvoice.judge.ecapa_ready", return_value=True)]
        for p in self.patches:
            p.start()
        self.p = Project("test:donors")
        self.jobs = Mock()
        self.jobs.profile_f0.return_value = 145.0
        self.jobs.character_profile.return_value = {"audio": str(Path(self.tmp.name) / "m110_profile.wav"), "text": "x", "seconds": 5}

    def tearDown(self):
        self.p.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_assigned_is_sticky_and_manual_wins(self):
        first = donors.assigned(self.jobs, self.p, "hero")
        self.assertEqual(first["id"], "m150")
        self.assertEqual(first["pitch"], round(12 * math.log2(145 / 150)))
        self.assertFalse(first["manual"])
        self.jobs.profile_f0.return_value = 300.0
        self.assertEqual(donors.assigned(self.jobs, self.p, "hero")["id"], "m150")
        manual = donors.set_manual(self.jobs, self.p, "hero", "f220")
        self.assertEqual((manual["id"], manual["manual"], manual["pitch"]), ("f220", True, round(12 * math.log2(300 / 220))))
        self.assertEqual(donors.assigned(self.jobs, self.p, "hero")["id"], "f220")
        with self.assertRaises(ValueError):
            donors.set_manual(self.jobs, self.p, "hero", "nope")

    def test_assigned_without_profile_uses_first_male(self):
        self.jobs.profile_f0.return_value = 0.0
        self.jobs.character_profile.return_value = None
        self.assertEqual(donors.assigned(self.jobs, self.p, "npc")["id"], "m110")


if __name__ == "__main__":
    unittest.main()
