import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruvoice.api_dub import create_router
from ruvoice.project import Project


class VoicesApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [patch("ruvoice.project.PROJECTS_DIR", root / "projects"), patch("ruvoice.accel.info", return_value={})]
        for p in self.patches:
            p.start()
        self.gid = "test:voices"
        self.p = Project(self.gid)
        self.p.set_characters([{"id": "hero", "name": "Герой"}, {"id": "npc", "name": "Прохожий"}])
        rows = [{"id": f"h{i:02d}", "char": "hero", "orig": f"orig/h{i:02d}.wav", "duration": 10.0 + i, "en": "x", "src": {}} for i in range(20)]
        rows += [{"id": "n00", "char": "npc", "orig": "orig/n00.wav", "duration": 5.0, "en": "x", "src": {}}]
        self.p.add_lines(rows)
        self.p.set("voice_f0", {"hero": 118.0})
        self.p.set("donors", {"hero": {"id": "m110", "pitch": 1, "manual": False, "picked_at": 1}})
        self.p.set("voices", {"hero": {"method": "rvc", "pth": "voices/hero/rvc/model.pth", "val": {"accept": False}, "weak": True,
                                       "reason": "похожесть не выросла"}})
        self.bank = [{"id": "m110", "gender": "male", "age": "adult", "f0": 110.0, "source": "t", "license": "t", "dir": str(root),
                      "clips": {"calm": {"file": "m110_calm.flac", "text": "т"}}}]
        (root / "m110_calm.flac").write_bytes(b"fLaC")
        self.jobs, lib = Mock(), Mock()
        lib.get.return_value = {"name": "Игра"}
        self.jobs.project.return_value = self.p
        self.jobs.current = None
        self.jobs.settings.return_value = {"timbre_level": "normal", "timbre_min_seconds": 120, "speech_source": "auto",
                                           "clone_engine": "auto", "episodic_mode": "clone"}
        self.jobs.profile_f0.side_effect = lambda p, char: 118.0
        self.more = [patch("ruvoice.donors.load", return_value=self.bank)]
        for p in self.more:
            p.start()
        app = FastAPI()
        app.include_router(create_router(lib, self.jobs))
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.p.close()
        for p in self.more + self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_voices_api_lists_tier_donor_and_timbre(self):
        res = self.client.get(f"/api/projects/{self.gid}/voices")
        self.assertEqual(res.status_code, 200, res.text)
        rows = {r["id"]: r for r in res.json()}
        hero, npc = rows["hero"], rows["npc"]
        self.assertEqual((hero["tier"], hero["lines"], hero["f0"], hero["voice_type"]), ("main", 20, 118.0, "мужской"))
        self.assertEqual(hero["donor"], {"id": "m110", "gender": "male", "age": "adult", "pitch": 1, "manual": False})
        self.assertEqual(hero["timbre"], {"state": "weak", "reason": "похожесть не выросла"})
        self.assertEqual(hero["samples"], ["h19", "h18", "h17"])  # самые длинные
        self.assertEqual((npc["tier"], npc["donor"], npc["timbre"]["state"]), ("episodic", None, "none"))

    def test_voices_reason_explains_fast_path(self):
        with patch("ruvoice.engines.rvc_available", return_value=False):
            rows = {r["id"]: r for r in self.client.get(f"/api/projects/{self.gid}/voices").json()}
        self.assertEqual(rows["hero"]["timbre"]["reason"], "перенос тембра не установлен")
        self.assertEqual(rows["npc"]["timbre"]["reason"], "эпизодический персонаж")
        with patch("ruvoice.engines.rvc_available", return_value=True):
            rows = {r["id"]: r for r in self.client.get(f"/api/projects/{self.gid}/voices").json()}
        self.assertEqual(rows["hero"]["timbre"]["reason"], "похожесть не выросла")

    def test_voices_tier_override_and_donor_choice(self):
        res = self.client.post(f"/api/projects/{self.gid}/voices/tier", json={"char": "npc", "tier": "train"})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(self.p.get("timbre_override"), {"npc": "train"})
        self.assertEqual({r["id"]: r["tier"] for r in self.client.get(f"/api/projects/{self.gid}/voices").json()}["npc"], "main")
        self.client.post(f"/api/projects/{self.gid}/voices/tier", json={"char": "npc", "tier": "auto"})
        self.assertEqual(self.p.get("timbre_override"), {})
        self.assertEqual(self.client.post(f"/api/projects/{self.gid}/voices/tier", json={"char": "npc", "tier": "nope"}).status_code, 400)
        with patch("ruvoice.donors.set_manual", return_value={"id": "m110", "pitch": 1, "manual": True}) as sm:
            res = self.client.post(f"/api/projects/{self.gid}/voices/donor", json={"char": "npc", "donor": "m110"})
            self.assertEqual(res.status_code, 200, res.text)
            sm.assert_called_once()
        with patch("ruvoice.donors.set_manual", side_effect=ValueError("Нет такого донора")):
            self.assertEqual(self.client.post(f"/api/projects/{self.gid}/voices/donor", json={"char": "npc", "donor": "zz"}).status_code, 400)

    def test_voices_api_busy_guard(self):
        self.jobs.current = self.gid
        self.assertEqual(self.client.post(f"/api/projects/{self.gid}/voices/donor", json={"char": "npc", "donor": "m110"}).status_code, 409)
        self.assertEqual(self.client.post(f"/api/projects/{self.gid}/voices/tier", json={"char": "npc", "tier": "train"}).status_code, 409)

    def test_donor_bank_listing_and_audio(self):
        res = self.client.get("/api/donors")
        self.assertEqual(res.status_code, 200)
        d = res.json()[0]
        self.assertEqual((d["id"], d["gender"], d["styles"]), ("m110", "male", ["calm"]))
        self.assertIn("calm", d["url"])
        res = self.client.get("/api/donors/audio", params={"id": "m110", "style": "calm"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.client.get("/api/donors/audio", params={"id": "zz", "style": "calm"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
