import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ruvoice import audio, tts


class OmniVoiceWorkerTests(unittest.TestCase):
    def test_omnivoice_requires_ref_text_and_seeds(self):
        from ruvoice.engines.worker import OmniVoiceEngine
        engine = OmniVoiceEngine()
        engine.model = Mock()
        engine.model.generate.return_value = [np.zeros(2400)]
        torch = Mock()
        with patch.dict("sys.modules", {"torch": torch}):
            wav, sr = engine.generate({"text": "Привет [laughter]", "ref_audio": "d.flac", "ref_text": "Текст донора",
                                       "seed": 7, "params": {"steps": 16}})
        self.assertEqual(sr, 24000)
        self.assertEqual(wav.dtype, np.float32)
        torch.manual_seed.assert_called_with(7)
        kw = engine.model.generate.call_args.kwargs
        self.assertEqual((kw["text"], kw["ref_audio"], kw["ref_text"], kw["num_step"]),
                         ("Привет [laughter]", "d.flac", "Текст донора", 16))
        with self.assertRaises(ValueError):
            engine.generate({"text": "x", "ref_audio": "d.flac"})

    def test_worker_registry_has_omnivoice(self):
        from ruvoice.engines import worker
        self.assertIn("omnivoice", worker.ENGINES)


class OmniVoiceDispatchTests(unittest.TestCase):
    def test_tts_omnivoice_sends_ref_text(self):
        proc = Mock()

        def request(msg):
            audio.write(msg["out"], np.zeros(2400, dtype=np.float32), 24000)
            return {"ok": True, "sr": 24000}
        proc.request.side_effect = request
        with patch("ruvoice.tts.available_tts", return_value=["omnivoice"]), patch("ruvoice.engines.client.get", return_value=proc), \
             patch("ruvoice.accent.plus", side_effect=lambda t: t):
            (wav, sr), = tts.synthesize("omnivoice", ["Привет"], [{"audio": "d.flac", "text": "Текст донора", "language": "ru"}],
                                        [5.0], {"steps": 16}, seed=3)
        msg = proc.request.call_args.args[0]
        self.assertEqual((msg["op"], msg["ref_audio"], msg["ref_text"], msg["seed"], msg["params"]["steps"]),
                         ("generate", "d.flac", "Текст донора", 3, 16))
        self.assertEqual(sr, 24000)
        self.assertEqual(len(wav), 2400)


if __name__ == "__main__":
    unittest.main()
