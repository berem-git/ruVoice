import io
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from ruvoice.engines import install


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.engines_dir = self.root / "runtime" / "engines"
        for env in ("rvc", "omnivoice", "voxcpm"):
            (self.root / f"requirements-{env}.txt").write_text("somepkg==1.0\n", "utf-8")
        self.patches = [patch("ruvoice.engines.install.ENGINES_DIR", self.engines_dir),
                        patch("ruvoice.engines.install.PROJECT_DIR", self.root),
                        patch("ruvoice.models.download_async")]
        for p in self.patches:
            p.start()
        install._state.update(running=False, engine=None, message="", error=None, thread=None)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_rvc_install_runs_uv_without_deps_then_extracts_and_downloads(self):
        calls = []

        def fake_run(args, **kw):
            calls.append([str(a) for a in args])
            return Mock(returncode=0, stdout="", stderr="")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Applio-3.4.0/core.py", "print()")
            z.writestr("Applio-3.4.0/rvc/__init__.py", "")
        downloads = []

        def fake_download(url, dest, progress, sha256=None):
            downloads.append(url)
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(buf.getvalue() if url.endswith(".zip") else b"weights")
        with patch("subprocess.run", side_effect=fake_run), patch("ruvoice.engines.install._download", side_effect=fake_download):
            install.install("rvc", lambda m: None)
        uv = calls[0]
        self.assertIn("--no-deps", uv)
        self.assertIn("--target", uv)
        self.assertIn(str(self.engines_dir / "rvc"), uv)
        self.assertIn(str(self.root / "requirements-rvc.txt"), uv)
        self.assertTrue((self.engines_dir / "rvc" / "applio" / "core.py").is_file())
        self.assertTrue((self.engines_dir / "rvc" / "applio" / "rvc" / "__init__.py").is_file())
        self.assertEqual(downloads[0], install.ENVS["rvc"]["applio"]["url"])
        for url, rel in install.ENVS["rvc"]["pretrained"]:
            self.assertTrue((self.engines_dir / "rvc" / rel).is_file(), rel)
            self.assertIn(url, downloads)

    def test_omnivoice_install_queues_model_download(self):
        with patch("subprocess.run", return_value=Mock(returncode=0, stdout="", stderr="")), \
             patch("ruvoice.models.download_async") as dl:
            install.install("omnivoice", lambda m: None)
        dl.assert_called_once_with(["omnivoice"])

    def test_pip_failure_raises_with_stderr(self):
        with patch("subprocess.run", return_value=Mock(returncode=1, stdout="", stderr="boom: no such package")):
            with self.assertRaises(RuntimeError) as cm:
                install.install("voxcpm", lambda m: None)
        self.assertIn("no such package", str(cm.exception))

    def test_status_reports_running_and_error(self):
        gate = threading.Event()

        def slow_install(engine, progress):
            progress("идёт")
            gate.wait(5)
            raise RuntimeError("boom")
        with patch("ruvoice.engines.install.install", side_effect=slow_install):
            st = install.install_async("rvc")
            self.assertTrue(st["rvc"]["running"])
            self.assertFalse(st["omnivoice"]["running"])
            gate.set()
            install._state["thread"].join(5)
        st = install.status()
        self.assertFalse(st["rvc"]["running"])
        self.assertEqual(st["rvc"]["error"], "boom")
        self.assertIn("installed", st["omnivoice"])
        with self.assertRaises(ValueError):
            install.install_async("nope")

    def test_api_install_route_starts_install(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ruvoice.api_dub import create_router
        jobs, lib = Mock(), Mock()
        jobs.current = None
        with patch("ruvoice.accel.info", return_value={}), patch("ruvoice.engines.install.install_async") as start:
            start.return_value = {"rvc": {"installed": False, "running": True, "message": "…", "error": None}}
            app = FastAPI()
            app.include_router(create_router(lib, jobs))
            with TestClient(app) as client:
                res = client.post("/api/system/engines/install", json={"engine": "rvc"})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertTrue(res.json()["rvc"]["running"])
                start.assert_called_once_with("rvc")
                self.assertEqual(client.post("/api/system/engines/install", json={"engine": "nope"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
