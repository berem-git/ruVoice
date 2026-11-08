"""Перезапуск программы из неё самой: новая копия не должна увидеть мьютекс старой (главный модуль)."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ruvoice import accel, main


class ReleaseInstanceTests(unittest.TestCase):
    def test_release_instance_drops_mutex_and_file(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "instance.json"
            f.write_text('{"pid": %d, "port": 1, "url": "x"}' % main.os.getpid(), "utf-8")
            with patch.object(main, "INSTANCE_FILE", f), patch.object(main, "_mutex", None):
                main.release_instance()
                self.assertIsNone(main._mutex)
            self.assertFalse(f.exists())

    def test_restart_releases_instance_before_spawning(self):
        order, real_sleep = [], time.sleep  # time.sleep ниже подменяется глобально — ждать настоящим
        with patch("ruvoice.main.release_instance", side_effect=lambda: order.append("release")), \
             patch("ruvoice.accel.subprocess.Popen", side_effect=lambda *a, **k: order.append("spawn")), \
             patch("ruvoice.accel.os._exit", side_effect=lambda code: order.append("exit")), \
             patch("ruvoice.accel.time.sleep"):
            accel.restart_app(delay=0)
            for _ in range(100):
                if "exit" in order:
                    break
                real_sleep(0.05)
        self.assertEqual(order, ["release", "spawn", "exit"])


if __name__ == "__main__":
    unittest.main()
