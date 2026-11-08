import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ruvoice import basevoices, engines, models, tts


class RegistryTests(unittest.TestCase):
    def test_registry_has_five_backends_with_capabilities(self):
        self.assertEqual(set(engines.BACKENDS), {"ru", "custom", "voxcpm", "silero", "omnivoice"})
        self.assertTrue(engines.BACKENDS["omnivoice"]["capabilities"]["nonverbal_tags"])
        self.assertFalse(engines.BACKENDS["silero"]["capabilities"]["commercial"])
        self.assertIs(engines.TTS_ENGINES, engines.BACKENDS)
        for key, spec in engines.BACKENDS.items():
            self.assertEqual(set(spec["capabilities"]),
                             {"stress", "style", "nonverbal_tags", "clone", "cross_lingual", "commercial"}, key)
            for field in ("name", "short", "desc", "env"):
                self.assertIn(field, spec, key)

    def test_sources_and_cloners_follow_readiness(self):
        with patch("ruvoice.engines.ready", side_effect=lambda k: k in ("ru", "silero")):
            self.assertEqual(engines.sources(), ["ru", "silero"])
            self.assertEqual(engines.cloners(), ["ru"])
        with patch("ruvoice.engines.ready", side_effect=lambda k: k in ("voxcpm", "omnivoice")):
            self.assertEqual(engines.sources(), ["omnivoice"])
            self.assertEqual(engines.cloners(), ["voxcpm", "omnivoice"])

    def test_describe_includes_capabilities_and_readiness(self):
        with patch("ruvoice.engines.ready", return_value=False):
            d = engines.describe()
        self.assertIn("capabilities", d["tts"]["silero"])
        self.assertFalse(d["tts"]["silero"]["ready"])


class SileroTests(unittest.TestCase):
    def test_silero_synthesize_marks_stress_and_returns_48k(self):
        model = Mock()
        model.apply_tts.return_value = np.zeros(4800)
        with patch("ruvoice.basevoices._load", return_value=model), patch("ruvoice.accent.plus", return_value="з+амок"), \
             patch("ruvoice.tts.available_tts", return_value=["silero"]):
            (wav, sr), = tts.synthesize("silero", ["замок"], [{"speaker": "aidar"}], [3.0], {})
        self.assertEqual(sr, 48000)
        self.assertEqual(model.apply_tts.call_args.kwargs["text"], "з+амок")
        self.assertEqual(model.apply_tts.call_args.kwargs["speaker"], "aidar")
        self.assertEqual(wav.dtype, np.float32)

    def test_silero_model_path_comes_from_models_registry(self):
        self.assertEqual(basevoices.MODEL, models.local_dir("silero-ru") / "v5_6_ru.pt")

    def test_pick_for_restricts_by_gender(self):
        embs = {k: np.array([1.0, 0.0]) for k in basevoices.SPEAKERS}
        embs["xenia"] = np.array([0.0, 1.0])
        with patch("ruvoice.basevoices.embeddings", return_value=embs):
            self.assertEqual(basevoices.pick_for(np.array([0.0, 1.0]), None), "xenia")
            self.assertIn(basevoices.pick_for(np.array([0.0, 1.0]), "male"), ("aidar", "eugene"))
        self.assertEqual(basevoices.pick_for(None, "male"), "aidar")
        self.assertEqual(basevoices.pick_for(None, None), "xenia")


class ModelsTests(unittest.TestCase):
    def test_models_tok_removed_and_new_entries(self):
        self.assertNotIn("tts-tok", models.ALL)
        self.assertTrue(models.ALL["silero-ru"]["url"].endswith("v5_6_ru.pt"))
        self.assertTrue(all("tts-tok" not in t["models"] for t in models.TIERS.values()))
        self.assertEqual(models.ALL["omnivoice"]["repo"], "k2-fsa/OmniVoice")

    def test_url_model_path_and_download(self):
        with tempfile.TemporaryDirectory() as td, patch("ruvoice.models.MODELS_DIR", Path(td)):
            self.assertIsNone(models.path("silero-ru"))

            def fake_retrieve(url, dest):
                Path(dest).write_bytes(b"pt")
            with patch("urllib.request.urlretrieve", side_effect=fake_retrieve) as ret:
                models.download("silero-ru")
            self.assertTrue(ret.call_args.args[0].endswith("v5_6_ru.pt"))
            self.assertEqual(models.path("silero-ru"), Path(td) / "silero-ru" / "v5_6_ru.pt")


if __name__ == "__main__":
    unittest.main()
