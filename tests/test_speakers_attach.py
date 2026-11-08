import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ruvoice import speakers
from ruvoice.project import Project


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class AttachTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch("ruvoice.project.PROJECTS_DIR", Path(self.tmp.name))
        self.patch.start()
        self.p = Project("test:attach")
        self.p.set_characters([{"id": "A", "name": "Алекс"}, {"id": "B", "name": "Борис"}])
        rows, self.have = [], {}
        rng = np.random.default_rng(1)
        for i in range(5):
            rows.append({"id": f"a{i}", "char": "A", "orig": f"orig/a{i}.wav", "duration": 3.0, "en": "Hello there", "src": {}})
            self.have[f"a{i}"] = (unit([1.0, 0.0, 0.0] + rng.normal(0, 0.02, 3)), 120.0)
            rows.append({"id": f"b{i}", "char": "B", "orig": f"orig/b{i}.wav", "duration": 3.0, "en": "Hello there", "src": {}})
            self.have[f"b{i}"] = (unit([0.0, 1.0, 0.0] + rng.normal(0, 0.02, 3)), 200.0)
        # чужая реплика у A: по голосу — явно B; спорная реплика у A: одинаково далека от обоих
        rows.append({"id": "a_stray", "char": "A", "orig": "orig/a_stray.wav", "duration": 3.0, "en": "Hello there", "src": {}})
        self.have["a_stray"] = (unit([0.1, 0.99, 0.0]), 200.0)
        rows.append({"id": "a_mid", "char": "A", "orig": "orig/a_mid.wav", "duration": 3.0, "en": "Hello there", "src": {}})
        self.have["a_mid"] = (unit([0.4, 0.4, 0.82]), 150.0)
        rows.append({"id": "a_quiet", "char": "A", "orig": "orig/a_quiet.wav", "duration": 3.0, "en": "...", "src": {}})
        self.have["a_quiet"] = (unit([0.1, 0.99, 0.0]), 200.0)
        self.p.add_lines(rows)

    def tearDown(self):
        self.p.close()
        self.patch.stop()
        self.tmp.cleanup()

    def test_attach_moves_clear_outsiders_only(self):
        moved = speakers.attach_rest(self.p, self.have)
        self.assertEqual(moved, 1)
        self.assertEqual(self.p.line("a_stray")["char"], "B")
        self.assertEqual(self.p.line("a_mid")["char"], "A")      # лучший голос не лучше второго на 0.08 — остаётся
        self.assertEqual(self.p.line("a_quiet")["char"], "A")    # без слов не трогаем
        self.assertTrue(all(self.p.line(f"a{i}")["char"] == "A" for i in range(5)))
        self.assertEqual(speakers.attach_rest(self.p, self.have), 0)


if __name__ == "__main__":
    unittest.main()
