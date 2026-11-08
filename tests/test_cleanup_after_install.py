"""Уборка после установки в игру — режим «минимум» (спецификация 2026-11-08, часть C)."""
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ruvoice import audio, cleanup
from ruvoice.dub import DEFAULT_SETTINGS
from ruvoice.project import Project

try:
    from tests.test_pipeline_v2 import PlanCase, tone
except ImportError:  # discover -s tests
    from test_pipeline_v2 import PlanCase, tone


def _fill(pdir: Path):
    """Папка проекта после озвучки: всё, что может быть в проекте."""
    for d in ("orig", "orig_chunks", "samples/hero", "bench/run1", "gen", "backup/sound", "voices/hero/rvc", "profiles"):
        (pdir / d).mkdir(parents=True, exist_ok=True)
    (pdir / "orig" / "h00.wav").write_bytes(b"\0" * 4000)
    (pdir / "orig_chunks" / "h00.vdat").write_bytes(b"\0" * 100)
    (pdir / "samples" / "hero" / "h00.wav").write_bytes(b"\0" * 300)
    (pdir / "bench" / "run1" / "index.html").write_bytes(b"\0" * 200)
    (pdir / "gen" / "h00.wav").write_bytes(b"\0" * 500)
    (pdir / "backup" / "sound" / "bank.fsb").write_bytes(b"\0" * 700)
    (pdir / "voices" / "hero" / "rvc" / "model.pth").write_bytes(b"pth")
    (pdir / "voices" / "hero" / "rvc" / "meta.json").write_text("{}", "utf-8")
    (pdir / "profiles" / "hero.wav").write_bytes(b"\0" * 50)


class PurgeTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.pdir = Path(self.tmp.name) / "proj"
        _fill(self.pdir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_purge_size_counts_originals_samples_and_bench(self):
        self.assertEqual(cleanup.purge_size(self.pdir), 4000 + 300 + 200)

    def test_purge_originals_removes_only_regenerable_folders(self):
        freed = cleanup.purge_originals(self.pdir)
        self.assertEqual(freed, 4000 + 300 + 200)
        for gone in ("orig", "samples", "bench"):
            self.assertFalse((self.pdir / gone).exists(), gone)
        for kept in ("orig_chunks/h00.vdat", "gen/h00.wav", "backup/sound/bank.fsb", "voices/hero/rvc/model.pth",
                     "voices/hero/rvc/meta.json", "profiles/hero.wav"):
            self.assertTrue((self.pdir / kept).exists(), kept)
        self.assertEqual(cleanup.purge_originals(self.pdir), 0)  # повторно — нечего


class InstallCleanupTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.adapter = Mock()
        self.adapter.install.return_value = 3
        self.jobs._adapter = lambda gid: self.adapter
        self.jobs._game = lambda gid: {"name": "Игра", "path": self.tmp.name}
        self.jobs.state[self.gid]["task"] = "install"
        for lid in ("h00", "h01", "h02"):
            self.p.gen_dir.mkdir(parents=True, exist_ok=True)
            audio.write(self.p.gen_dir / f"{lid}.wav", tone(1.0), 16000)
            self.p.update_line(lid, gen=f"gen/{lid}.wav", status="approved")
        (self.p.dir / "samples" / "hero").mkdir(parents=True, exist_ok=True)
        (self.p.dir / "samples" / "hero" / "h00.wav").write_bytes(b"\0" * 100)

    def test_default_after_install_is_minimum(self):
        self.assertEqual(DEFAULT_SETTINGS["after_install"], "minimum")

    def test_install_minimum_purges_originals_and_marks_project(self):
        self.jobs._do_install(self.gid)
        self.adapter.install.assert_called_once()
        self.assertFalse(self.p.orig_dir.exists())
        self.assertFalse((self.p.dir / "samples").exists())
        purged = self.p.get("originals_purged")
        self.assertTrue(purged and purged["bytes"] > 0 and purged["at"] > 0)
        self.assertTrue((self.p.gen_dir / "h00.wav").exists())
        self.assertIn("освобождено", self.jobs.state[self.gid]["message"])

    def test_install_keep_mode_leaves_originals(self):
        self.jobs.save_settings(self.gid, {"after_install": "keep"})
        self.jobs._do_install(self.gid)
        self.assertTrue((self.p.orig_dir / "h00.wav").exists())
        self.assertIsNone(self.p.get("originals_purged"))

    def test_failed_install_does_not_purge(self):
        self.adapter.install.side_effect = RuntimeError("нет доступа")
        with self.assertRaises(RuntimeError):
            self.jobs._do_install(self.gid)
        self.assertTrue((self.p.orig_dir / "h00.wav").exists())


class RestoreOriginalsTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.jobs._game = lambda gid: {"name": "Игра", "path": self.tmp.name}

    def _purge(self):
        cleanup.purge_originals(self.p.dir)
        self.p.set("originals_purged", {"at": 1.0, "bytes": 123})

    def test_restore_copies_audio_from_a_temporary_extraction(self):
        self._purge()
        adapter = Mock()

        def extract(tmp, game, progress):
            tmp.orig_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for lid in ("h00", "h01", "n00", "zz99"):  # zz99 — реплики нет в проекте, её файл не нужен
                audio.write(tmp.orig_dir / f"{lid}.wav", tone(0.5), 16000)
                rows.append({"id": lid, "char": "x", "orig": f"orig/{lid}.wav", "duration": 1.0, "en": "x", "src": {}})
            tmp.add_lines(rows)
            return len(rows)
        adapter.extract.side_effect = extract
        self.jobs._adapter = lambda gid: adapter
        self.jobs._do_restore_originals(self.gid)
        for lid in ("h00", "h01", "n00"):
            self.assertTrue((self.p.orig_dir / f"{lid}.wav").exists(), lid)
        self.assertFalse((self.p.orig_dir / "zz99.wav").exists())
        self.assertIsNone(self.p.get("originals_purged"))
        self.assertEqual(self.p.line("h00")["status"], "translated", "переводы и статусы не сбрасываются")
        self.assertFalse(any(d.name.endswith("_restore") for d in self.p.dir.parent.iterdir()), "временный проект удалён")

    def test_generate_restores_originals_first(self):
        self._purge()
        restored = []

        def restore(gid):
            restored.append(gid)
            self.p.orig_dir.mkdir(parents=True, exist_ok=True)
            for l in self.p.lines():
                audio.write(self.p.dir / l["orig"], tone(3.0), 16000)
            self.p.set("originals_purged", None)
        with patch.object(self.jobs, "_do_restore_originals", side_effect=restore):
            self.jobs._do_generate(self.gid, ids=["h00"])
        self.assertEqual(restored, [self.gid])
        self.assertEqual(self.p.line("h00")["status"], "generated")

    def test_generate_without_purge_does_not_restore(self):
        with patch.object(self.jobs, "_do_restore_originals") as restore:
            self.jobs._do_generate(self.gid, ids=["h00"])
        restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
