"""Эхо и фон оригинала на русской речи, стыки вклеенных событий (спецификация 2026-11-08, часть B)."""
import unittest
from unittest.mock import patch

import numpy as np

from ruvoice import acoustics, performance

try:
    from tests.test_pipeline_v2 import PlanCase
except ImportError:  # discover -s tests
    from test_pipeline_v2 import PlanCase

SR = 16000


def speechlike(seconds: float = 1.6, sr: int = SR, seed: int = 1) -> np.ndarray:
    """Похожий на речь сигнал: шум, промодулированный «слогами» по 120 мс, тишина −80 дБ в конце."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = rng.standard_normal(n).astype(np.float32) * 0.2
    t = np.arange(n) / sr
    x *= (0.55 + 0.45 * np.sin(2 * np.pi * 4.2 * t)).astype(np.float32)
    x[int(1.3 * sr):] = rng.standard_normal(n - int(1.3 * sr)).astype(np.float32) * 1e-4
    return x


class RoomTests(unittest.TestCase):
    def test_dry_signal_has_no_room(self):
        self.assertIsNone(acoustics.analyse_room(speechlike(), SR))

    def test_reverberant_signal_estimates_decay(self):
        h = acoustics.room_impulse(0.6, SR, seed=3)
        wet = acoustics.apply_room(speechlike(2.4), SR, {"rt60": 0.6, "wet": 0.35}, impulse=h)  # после речи есть место хвосту
        room = acoustics.analyse_room(wet, SR)
        self.assertIsNotNone(room)
        self.assertGreaterEqual(room["rt60"], 0.35)
        self.assertLessEqual(room["rt60"], 1.0)
        self.assertGreater(room["wet"], 0.05)
        self.assertLess(room["noise_db"], -50)

    def test_apply_room_adds_a_tail(self):
        dry = speechlike()
        wet = acoustics.apply_room(dry, SR, {"rt60": 0.6, "wet": 0.35})
        tail = slice(int(1.35 * SR), int(1.6 * SR))
        self.assertGreater(acoustics.rms_db(wet[tail]), acoustics.rms_db(dry[tail]) + 20)
        self.assertEqual(len(wet), len(dry), "длина реплики не меняется — хвост обрезается по оригиналу")

    def test_mode_gating(self):
        self.assertTrue(acoustics.wants_room("auto"))
        self.assertTrue(acoustics.wants_room("unknown"))
        self.assertTrue(acoustics.wants_room("baked"))
        self.assertFalse(acoustics.wants_room("engine"))
        self.assertFalse(acoustics.wants_room("dry"))

    def test_noise_bed_matches_level(self):
        dry = speechlike()
        out = acoustics.noise_bed(dry, SR, -48.0, seed=2)
        tail = slice(int(1.35 * SR), int(1.6 * SR))
        self.assertAlmostEqual(acoustics.rms_db(out[tail]), -48.0, delta=3.0)


class ComposeTests(unittest.TestCase):
    def test_internal_insert_has_no_clicks(self):
        rng = np.random.default_rng(5)
        t = np.arange(int(3.0 * SR)) / SR
        orig = np.zeros(int(3.0 * SR), dtype=np.float32)
        orig[int(0.2 * SR):int(0.8 * SR)] = 0.3 * np.sin(2 * np.pi * 220 * t[int(0.2 * SR):int(0.8 * SR)])
        orig[int(1.0 * SR):int(1.3 * SR)] = 0.1 * np.sin(2 * np.pi * 60 * t[int(1.0 * SR):int(1.3 * SR)])
        orig[int(1.5 * SR):int(2.1 * SR)] = 0.3 * np.sin(2 * np.pi * 220 * t[int(1.5 * SR):int(2.1 * SR)])
        gen = np.concatenate([0.3 * np.sin(2 * np.pi * 220 * t[:int(0.8 * SR)]), np.zeros(int(0.2 * SR), dtype=np.float32),
                              0.3 * np.sin(2 * np.pi * 220 * t[:int(0.8 * SR)])]).astype(np.float32)
        plan = {"events": [{"start": 1.0, "end": 1.3, "label": "Laughter", "confidence": 0.9, "before_words": 1}],
                "words": [{"start": 0.2, "end": 0.8, "text": "a"}, {"start": 1.5, "end": 2.1, "text": "b"}], "warnings": []}
        out, info = performance.compose(gen, SR, orig, SR, plan)
        self.assertEqual(info["kept"], ["Laughter"])
        self.assertLess(float(np.max(np.abs(np.diff(out)))), 0.2, "на стыках не должно быть скачков")


class FitLineTests(PlanCase):
    def test_room_is_applied_by_default_and_reported(self):
        room = {"rt60": 0.5, "wet": 0.3, "noise_db": -70.0}
        with patch("ruvoice.acoustics.analyse_room", return_value=room), \
             patch("ruvoice.acoustics.apply_room", side_effect=lambda w, sr, r, **k: w) as ap:
            self.jobs._do_generate(self.gid, ids=["h00"])
        self.assertGreaterEqual(ap.call_count, 1)
        self.assertEqual(self.p.line("h00")["info"]["fit"]["room"], {"rt60": 0.5, "wet": 0.3, "noise_db": -70.0})

    def test_engine_mode_skips_room(self):
        self.jobs.save_settings(self.gid, {"effects_mode": "engine"})
        with patch("ruvoice.acoustics.analyse_room") as an:
            self.jobs._do_generate(self.gid, ids=["h00"])
        an.assert_not_called()
        self.assertNotIn("room", self.p.line("h00")["info"]["fit"])


if __name__ == "__main__":
    unittest.main()
