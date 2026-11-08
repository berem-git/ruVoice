import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

from ruvoice import dub, timbre
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


class TierTests(unittest.TestCase):
    def test_main_needs_seconds_and_lines(self):
        s = {"timbre_level": "normal", "timbre_min_seconds": 120}
        self.assertEqual(timbre.tier({"seconds": 130, "lines": 12}, s, None), "main")
        self.assertEqual(timbre.tier({"seconds": 130, "lines": 11}, s, None), "episodic")
        self.assertEqual(timbre.tier({"seconds": 119, "lines": 50}, s, None), "episodic")

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


class SeedTests(unittest.TestCase):
    def test_line_seed_formula(self):
        self.assertEqual(dub.line_seed("abc", 0, 0), (zlib.crc32(b"abc") ^ 100003) & 0x7FFFFFFF)
        self.assertEqual(dub.line_seed("abc", 1, 2), (zlib.crc32(b"abc") ^ (100003 * 2 + 977 * 2)) & 0x7FFFFFFF)
        self.assertNotEqual(dub.line_seed("abc", 0, 1), dub.line_seed("abc", 1, 1))
        self.assertEqual(dub.line_seed("abc", 0, 1), dub.line_seed("abc", 0, 1))


if __name__ == "__main__":
    unittest.main()
