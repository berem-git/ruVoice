"""Сводка проекта (GET /api/projects/{gid}) на настоящем проекте: студия строится по ней целиком."""
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruvoice import cleanup
from ruvoice.api_dub import create_router

try:
    from tests.test_pipeline_v2 import PlanCase
except ImportError:  # discover -s tests
    from test_pipeline_v2 import PlanCase


class ProjectInfoTests(PlanCase):
    def setUp(self):
        super().setUp()
        lib = Mock()
        lib.get.return_value = {"id": self.gid, "name": "Игра", "path": self.tmp.name}
        self.jobs.lib = lib
        self.jobs.current = None
        app = FastAPI()
        app.include_router(create_router(lib, self.jobs))
        self.client = TestClient(app)
        self.client.__enter__()
        self.patch_adapter = patch("ruvoice.adapters.for_game", return_value=None)
        self.patch_adapter.start()

    def tearDown(self):
        self.patch_adapter.stop()
        self.client.__exit__(None, None, None)
        super().tearDown()

    def test_info_reports_purge_estimate_and_state(self):
        res = self.client.get(f"/api/projects/{self.gid}")
        self.assertEqual(res.status_code, 200, res.text)
        info = res.json()
        self.assertTrue(info["exists"])
        self.assertIsNone(info["originals_purged"])
        self.assertGreater(info["purge_estimate_mb"], 0)
        self.assertEqual(info["settings"]["after_install"], "minimum")
        cleanup.purge_originals(self.p.dir)
        self.p.set("originals_purged", {"at": 1.0, "bytes": 2 ** 20})
        info = self.client.get(f"/api/projects/{self.gid}").json()
        self.assertEqual(info["originals_purged"]["bytes"], 2 ** 20)
        self.assertEqual(info["purge_estimate_mb"], 0)


if __name__ == "__main__":
    unittest.main()
