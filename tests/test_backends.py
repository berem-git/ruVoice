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


class RVCWorkerTests(unittest.TestCase):
    """Процесс-работник RVC вызывает скрипты Applio 3.7.1 с позиционными аргументами в порядке core.py."""

    def setUp(self):
        from ruvoice.engines.worker import RVC
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.engine = RVC()
        self.engine.APPLIO = self.root / "applio"
        (self.engine.APPLIO / "rvc" / "models" / "pretraineds" / "hifi-gan").mkdir(parents=True)
        self.engine.torch = Mock()
        self.engine.torch.cuda.is_available.return_value = False
        self.engine.vc = None
        self.calls = []
        logs = self.engine.APPLIO / "logs" / "game__hero"

        def popen(args, **kw):
            args = [str(a) for a in args]
            self.calls.append(args)
            logs.mkdir(parents=True, exist_ok=True)
            if args[1].endswith("train.py"):
                (logs / "game__hero_2e_10s.pth").write_bytes(b"pth")
                (logs / "G_10.pth").write_bytes(b"g")
            if args[1].endswith("extract_index.py"):
                (logs / "added_IVF1_Flat_nprobe_1_game__hero_v2.index").write_bytes(b"idx")
            proc = Mock()
            proc.stdout = iter(["Epoch 1\n", "Epoch 2\n"])
            proc.wait.return_value = 0
            proc.pid = 1
            return proc
        self.engine.subprocess = Mock(Popen=popen, PIPE=-1, STDOUT=-2, CREATE_NO_WINDOW=0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rvc_train_command_lines_match_pinned_applio(self):
        out = self.root / "out"
        res = self.engine.train({"model_name": "game__hero", "dataset_dir": str(self.root / "ds"), "out_dir": str(out),
                                 "epochs": 2, "batch_size": 4}, lambda info: None)
        self.assertEqual(res["pth"], str(out / "model.pth"))
        self.assertEqual(res["index"], str(out / "model.index"))
        self.assertTrue((out / "model.pth").is_file() and (out / "model.index").is_file())
        logs = str(self.engine.APPLIO / "logs" / "game__hero")
        pre, ext, trn, idx = self.calls
        self.assertEqual(pre[1:], ["rvc/train/preprocess/preprocess.py", logs, str(self.root / "ds"), "40000", pre[5],
                                   "Automatic", "False", "3.0", "0.3", "none"])
        self.assertTrue(pre[5].isdigit())
        self.assertEqual(ext[1:], ["rvc/train/extract/extract.py", logs, "rmvpe", ext[4], "0", "40000", "contentvec", "None", "2"])
        hifi = self.engine.APPLIO / "rvc" / "models" / "pretraineds" / "hifi-gan"
        self.assertEqual(trn[1:], ["rvc/train/train.py", "game__hero", "2", "2", str(hifi / "f0G40k.pth"), str(hifi / "f0D40k.pth"),
                                   "0", trn[8], "40000", "True", "True", "False", "False", "HiFi-GAN", "False"])
        self.assertIn(trn[8], ("2", "4"))
        self.assertEqual(idx[1:], ["rvc/train/process/extract_index.py", logs, "Auto"])

    def test_rvc_env_puts_engine_first_and_skips_missing_accel_dir(self):
        env = self.engine._env()
        parts = env["PYTHONPATH"].split(";")
        self.assertTrue(parts[0].endswith("rvc"), parts)
        self.assertTrue(parts[1].endswith("applio"), parts)
        self.assertFalse(any("accel" in p and not Path(p).is_dir() for p in parts), parts)
        self.assertEqual(env["PYTHONUTF8"], "1")

    def test_rvc_convert_defaults_fixed_pitch_no_index(self):
        import soundfile as sf
        self.engine.vc = Mock()

        def convert_audio(**kw):
            sf.write(kw["audio_output_path"], np.zeros(400, dtype=np.float32), 40000)
        self.engine.vc.convert_audio.side_effect = convert_audio
        out = self.root / "out.wav"
        wav, sr = self.engine.convert({"source": str(self.root / "src.wav"), "out": str(out), "pth": "m.pth",
                                       "params": {"pitch": 3}})
        kw = self.engine.vc.convert_audio.call_args.kwargs
        self.assertEqual((kw["pitch"], kw["index_rate"], kw["protect"], kw["proposed_pitch"], kw["f0_method"], kw["index_path"]),
                         (3, 0.0, 0.5, False, "rmvpe", ""))
        self.assertEqual((sr, len(wav)), (40000, 400))


if __name__ == "__main__":
    unittest.main()
