"""Пауза и «остановить и удалить озвучку» (спецификация 2026-11-08, часть D)."""
import threading
import unittest
from unittest.mock import patch

from ruvoice import audio

try:
    from tests.test_pipeline_v2 import PlanCase, tone
except ImportError:  # discover -s tests
    from test_pipeline_v2 import PlanCase, tone


class PauseTests(PlanCase):
    def test_pause_blocks_checkpoint_until_resume_and_cancel_wins(self):
        jobs = self.jobs
        jobs.current = self.gid
        jobs.pause(self.gid)
        self.assertTrue(jobs.status(self.gid)["paused"])
        done = []
        t = threading.Thread(target=lambda: done.append(jobs._checkpoint()), daemon=True)
        t.start()
        t.join(0.4)
        self.assertTrue(t.is_alive(), "на паузе контрольная точка должна ждать")
        jobs.resume(self.gid)
        t.join(3)
        self.assertEqual(done, [False])
        self.assertFalse(jobs.status(self.gid)["paused"])
        jobs.pause(self.gid)
        jobs.cancel_flag.set()
        self.assertTrue(jobs._checkpoint(), "отмена не ждёт конца паузы")
        self.assertFalse(jobs.status(self.gid)["paused"], "отмена снимает паузу")

    def test_pause_only_applies_to_the_running_project(self):
        self.jobs.current = "other:game"
        self.jobs.pause(self.gid)
        self.assertFalse(self.jobs.status(self.gid)["paused"])

    def test_generate_stops_at_checkpoint(self):
        calls = []

        def checkpoint():
            calls.append(1)
            return len(calls) > 2  # после двух пачек — остановка
        with patch.object(self.jobs, "_checkpoint", side_effect=checkpoint):
            self.jobs._do_generate(self.gid)
        generated = self.p.count(status="generated")
        self.assertGreater(generated, 0)
        self.assertLess(generated, 22, "озвучка должна останавливаться на контрольной точке между пачками")


class StopDeleteTests(PlanCase):
    def setUp(self):
        import queue
        super().setUp()
        self.jobs.q = queue.Queue()  # cancel() чистит очередь

    def test_stop_delete_reverts_only_lines_of_this_run(self):
        old = self.p.line("h00")
        self.p.gen_dir.mkdir(parents=True, exist_ok=True)
        audio.write(self.p.gen_dir / "h00.wav", tone(1.0), 16000)
        self.p.update_line("h00", gen="gen/h00.wav", status="generated")
        self.jobs._do_generate(self.gid, ids=["h01", "h02"])
        self.assertEqual(sorted(self.jobs.state[self.gid]["produced"]), ["h01", "h02"])
        self.jobs.current = self.gid
        self.jobs.stop_delete(self.gid)
        self.assertTrue(self.jobs.cancel_flag.is_set())
        self.assertTrue(self.jobs.state[self.gid]["purge"])
        n = self.jobs._purge_run(self.gid)
        self.assertEqual(n, 2)
        for lid in ("h01", "h02"):
            l = self.p.line(lid)
            self.assertEqual((l["status"], l["gen"]), ("translated", None))
            self.assertFalse((self.p.gen_dir / f"{lid}.wav").exists())
        kept = self.p.line("h00")
        self.assertEqual((kept["status"], kept["gen"]), ("generated", "gen/h00.wav"))
        self.assertTrue((self.p.gen_dir / "h00.wav").exists())
        self.assertIn("удалена", self.jobs.state[self.gid]["message"])
        self.assertEqual(self.jobs.state[self.gid]["produced"], [])

    def test_plain_cancel_keeps_produced_lines(self):
        self.jobs._do_generate(self.gid, ids=["h01"])
        self.jobs.current = self.gid
        self.jobs.cancel(self.gid)
        self.assertFalse(self.jobs.state[self.gid].get("purge"))
        self.assertEqual(self.jobs._purge_run(self.gid), 0)
        self.assertEqual(self.p.line("h01")["status"], "generated")


if __name__ == "__main__":
    unittest.main()
