"""API паузы и «остановить и удалить» (спецификация 2026-11-08, часть D)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruvoice.api_dub import create_router
from ruvoice.project import Project


class ControlsApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [patch("ruvoice.project.PROJECTS_DIR", root / "projects"), patch("ruvoice.accel.info", return_value={})]
        for p in self.patches:
            p.start()
        self.gid = "test:controls"
        self.p = Project(self.gid)
        self.jobs, lib = Mock(), Mock()
        lib.get.return_value = {"name": "Игра"}
        self.jobs.project.return_value = self.p
        self.jobs.current = self.gid
        self.jobs.status.return_value = {"task": "generate", "message": "", "done": 1, "total": 5, "error": None,
                                         "queued": [], "running": True, "paused": True}
        app = FastAPI()
        app.include_router(create_router(lib, self.jobs))
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.p.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_pause_resume_and_stop_delete_routes(self):
        res = self.client.post(f"/api/projects/{self.gid}/pause")
        self.assertEqual(res.status_code, 200, res.text)
        self.jobs.pause.assert_called_once_with(self.gid)
        self.assertTrue(res.json()["paused"])
        self.assertEqual(self.client.post(f"/api/projects/{self.gid}/resume").status_code, 200)
        self.jobs.resume.assert_called_once_with(self.gid)
        self.assertEqual(self.client.post(f"/api/projects/{self.gid}/stop-delete").status_code, 200)
        self.jobs.stop_delete.assert_called_once_with(self.gid)


if __name__ == "__main__":
    unittest.main()
