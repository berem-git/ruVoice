import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ruvoice import cleanup
from ruvoice.project import Project


class CleanupV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch("ruvoice.project.PROJECTS_DIR", Path(self.tmp.name))
        self.patch.start()
        self.p = Project("test:clean")
        d = self.p.dir
        for rel in ("voices/hero/tts/voice.safetensors", "voices/hero/tts_full/checkpoint-epoch-3/model.safetensors",
                    "voices/hero/tts_data/0001.wav", "voices/hero/rvc/model.pth", "voices/hero/rvc/model.index",
                    "voices/hero/rvc/calib/rvc/a.wav", "voices/hero/rvc/dataset/x.wav", "voices/npc/rvc/model.pth",
                    "voices/old/model.pth", "voices/old/model.index", "voice_identities/abc.wav"):
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(b"x" * 1024)
        (d / "voices/hero/rvc/meta.json").write_text(json.dumps({"method": "rvc"}), "utf-8")
        self.p.set("voices", {"hero": {"method": "rvc", "pth": "voices/hero/rvc/model.pth", "index": "voices/hero/rvc/model.index",
                                       "val": {"accept": True}, "weak": False},
                              "old": {"method": "rvc", "pth": "voices/old/model.pth"},
                              "gone": {"method": "rvc", "pth": "voices/gone/rvc/model.pth", "val": {"accept": True}}})
        self.p.set("tts_voices", {"hero": {"kind": "full", "dir": "voices/hero/tts_full/checkpoint-epoch-3"}})
        self.p.close()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_old_finetune_and_anchor_dirs_are_garbage(self):
        d = self.p.dir
        res = cleanup.project_garbage(d, dry=True)
        text = " ".join(res["items"])
        for marker in ("voices\\hero\\tts ", "tts_full", "tts_data", "voice_identities"):
            self.assertIn(marker.strip(), text.replace("/", "\\"))
        self.assertTrue((d / "voices/hero/tts").exists())  # dry — ничего не удалено
        cleanup.project_garbage(d, dry=False)
        for rel in ("voices/hero/tts", "voices/hero/tts_full", "voices/hero/tts_data", "voice_identities"):
            self.assertFalse((d / rel).exists(), rel)

    def test_rvc_model_with_meta_is_kept(self):
        d = self.p.dir
        cleanup.project_garbage(d, dry=False)
        for rel in ("voices/hero/rvc/model.pth", "voices/hero/rvc/model.index", "voices/hero/rvc/meta.json", "voices/hero/rvc/calib/rvc/a.wav"):
            self.assertTrue((d / rel).exists(), rel)
        self.assertFalse((d / "voices/hero/rvc/dataset").exists())

    def test_unfinished_and_legacy_rvc_dirs_are_garbage(self):
        d = self.p.dir
        cleanup.project_garbage(d, dry=False)
        self.assertFalse((d / "voices/npc/rvc").exists())      # без meta.json — незавершённое обучение
        self.assertFalse((d / "voices/old/model.pth").exists())  # модель 1.0.0 без контроля
        p = Project("test:clean")
        try:
            voices = p.get("voices")
            self.assertIn("hero", voices)
            self.assertNotIn("gone", voices)   # файла нет
            self.assertNotIn("old", voices)    # старая раскладка без контроля
            self.assertIsNone(p.get("tts_voices"))
        finally:
            p.close()


if __name__ == "__main__":
    unittest.main()
