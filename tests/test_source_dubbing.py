import json
from contextlib import ExitStack
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ruvoice import audio, performance, subtitles, discovery
from ruvoice.project import Project
from ruvoice.adapters.loose import LooseAdapter
from ruvoice.adapters.zip_audio import ZipAudioAdapter


class DiscoveryTests(unittest.TestCase):
    def test_renamed_zip_and_unknown_container_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with zipfile.ZipFile(root / "resources.dat", "w") as z:
                z.writestr("voice/en/greet.wav", b"RIFF" + b"0" * 30)
                z.writestr("subtitles/ru/story.srt", "1\n00:00:01,000 --> 00:00:02,000\nПривет\n")
            (root / "unknown.pak").write_bytes(b"proprietary")
            result = discovery.scan(root)
            self.assertEqual(result["counts"]["archived_audio"], 1)
            self.assertEqual(result["counts"]["archived_subtitles"], 1)
            self.assertEqual(result["counts"]["container"], 2)
            self.assertTrue(any(x["format"] == "zip" for x in result["assets"]))

    def test_scan_limit(self):
        with tempfile.TemporaryDirectory() as td:
            for i in range(8):
                (Path(td) / f"{i}.wav").write_bytes(b"RIFF1234")
            result = discovery.scan(Path(td), max_files=3)
            self.assertTrue(result["truncated"])
            self.assertEqual(result["scanned"], 3)

    def test_cues_match_timestamps_not_shifted_line_numbers(self):
        en = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:04,000 --> 00:00:05,000\nGoodbye\n"
        ru = "1\n00:00:04,000 --> 00:00:05,000\nДо свидания\n"
        a, b = subtitles._kv(".srt", en), subtitles._kv(".srt", ru)
        self.assertEqual([a[k] for k in a if k in b], ["Goodbye"])
        self.assertEqual(next(iter(b.values())), "До свидания")

    def test_ambiguous_subtitle_rejected(self):
        pool = [{"key": "greet", "en": "Hello", "ru": "Привет"},
                {"key": "greet", "en": "Hello", "ru": "Здравствуйте"}]
        self.assertIsNone(subtitles.exact_asset_match(pool, "voices/greet.wav"))
        self.assertIsNone(subtitles.Matcher(pool).match("Hello")[0])

    def test_ass_and_vtt(self):
        self.assertEqual(len(subtitles._kv(".vtt", "WEBVTT\n\n00:01.000 --> 00:02.000\nHello\n")), 1)
        cue = "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}Привет\\Nмир"
        self.assertIn("Привет мир", subtitles._kv(".ass", cue).values())

    def test_zip_localisations_in_renamed_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with zipfile.ZipFile(root / "content.pak", "w") as z:
                z.writestr("locales/en.json", json.dumps({"greet": "Hello there"}))
                z.writestr("locales/ru.json", json.dumps({"greet": "Привет тебе"}, ensure_ascii=False))
            self.assertEqual(subtitles.harvest(root)[0]["ru"], "Привет тебе")


class SourceSoundTests(unittest.TestCase):
    def fixture(self):
        sr = 16000
        orig = np.zeros(sr * 2, dtype=np.float32)
        for a, b, f in [(.2, .6, 900), (.82, 1.1, 700), (1.4, 1.8, 1000)]:
            n = np.arange(int((b - a) * sr))
            orig[int(a * sr):int(a * sr) + len(n)] = .2 * np.sin(2 * np.pi * f * n / sr)
        gen = np.zeros(sr * 2, dtype=np.float32)
        for a, b in [(.1, .8), (1.1, 1.8)]:
            n = np.arange(int((b - a) * sr))
            gen[int(a * sr):int(a * sr) + len(n)] = .15 * np.sin(2 * np.pi * 300 * n / sr)
        return sr, orig, gen

    def test_isolated_internal_event_preserved(self):
        sr, orig, gen = self.fixture()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "original.wav"
            audio.write(path, orig, sr)
            words = [{"start": .2, "end": .6, "text": "first"}, {"start": 1.4, "end": 1.8, "text": "second"}]
            classify = lambda w, s: [{"label": "Cough", "score": .91}, {"label": "Speech", "score": .01}]
            plan = performance.analyse(path, words, classify, "dry")
            self.assertEqual(len(plan["events"]), 1)
            out, meta = performance.compose(gen, sr, orig, sr, plan)
            self.assertIn("Cough", meta["kept"])
            self.assertTrue(meta["requires_review"])
            self.assertTrue(np.all(np.isfinite(out)))

    def test_overlapping_source_speech_never_copied(self):
        sr, orig, gen = self.fixture()
        plan = {"words": [{"start": .2, "end": .6}],
                "events": [{"start": .3, "end": .5, "confidence": 1., "label": "Cough", "before_words": 0}]}
        out, meta = performance.compose(gen, sr, orig, sr, plan)
        expected, _ = audio.fit_to_original(gen, sr, orig, sr, max_speedup=1.1)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(meta["kept"], [])

    def test_no_word_alignment_means_no_copy(self):
        sr, orig, gen = self.fixture()
        plan = {"words": [], "events": [{"start": 0, "end": 1, "confidence": 1., "label": "Laughter"}]}
        _, meta = performance.compose(gen, sr, orig, sr, plan)
        self.assertEqual(meta["kept"], [])

    def test_uncertain_effects_require_review_without_events(self):
        sr, orig, gen = self.fixture()
        _, meta = performance.compose(gen, sr, orig, sr, {"warnings": ["Unknown room effects"]})
        self.assertTrue(meta["requires_review"])
        self.assertEqual(meta["kept"], [])

    def test_explicit_background_and_room_response(self):
        dry = np.array([.1, .2, .1], dtype=np.float32)
        out = performance.mix_stems(dry, 16000, np.array([.1, 0, 0, 0], dtype=np.float32),
                                    np.array([1., .5], dtype=np.float32), wet=.5)
        self.assertEqual(len(out), 4)
        self.assertGreater(out[-1], 0)
        self.assertLessEqual(float(np.max(np.abs(out))), .98)


class AdapterRoundTripTests(unittest.TestCase):
    def test_library_note_does_not_parse_archives(self):
        from ruvoice.library import _dub_note
        with patch("ruvoice.adapters.for_game", side_effect=AssertionError("Archive work in listing")):
            self.assertIsNone(_dub_note({"analysis": {}}))
            self.assertEqual(_dub_note({"analysis": {"dub_note": "cached"}}), "cached")

    def test_loose_and_zip_without_game_specific_rules(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game = root / "untitled"
            (game / "voices/en/NPC").mkdir(parents=True)
            (game / "voices/ru/NPC").mkdir(parents=True)
            (game / "locales").mkdir()
            for lang, text in [("en", "Hello there"), ("ru", "Привет тебе")]:
                (game / f"locales/{lang}.json").write_text(json.dumps({"greet": text}, ensure_ascii=False), "utf-8")
                audio.write(game / f"voices/{lang}/NPC/greet.wav", np.ones(16000, dtype=np.float32) * .1, 16000)
            with patch("ruvoice.project.PROJECTS_DIR", root / "projects"), ExitStack() as resources:
                p = Project("unknown-game")
                resources.callback(p.close)
                self.assertEqual(LooseAdapter().extract(p, {"path": str(game)}, lambda *a: None), 1)
                self.assertEqual(p.lines()[0]["ru_official"], "Привет тебе")
                p.close()
                archive = root / "zipgame"
                archive.mkdir()
                with zipfile.ZipFile(archive / "content.dat", "w") as z:
                    z.write(game / "voices/en/NPC/greet.wav", "voices/en/NPC/greet.wav")
                    z.writestr("locales/en.json", json.dumps({"greet": "Hello there"}))
                    z.writestr("locales/ru.json", json.dumps({"greet": "Привет тебе"}, ensure_ascii=False))
                    z.writestr("untouched.dat", b"preserve exactly")
                before = (archive / "content.dat").read_bytes()
                adapter = ZipAudioAdapter()
                self.assertTrue(adapter.detect({"path": str(archive)}))
                p = Project("unknown-archive")
                resources.callback(p.close)
                self.assertEqual(adapter.extract(p, {"path": str(archive)}, lambda *a: None), 1)
                line = p.lines()[0]
                p.update_line(line["id"], gen=line["orig"], status="approved")
                self.assertEqual(adapter.install(p, {"path": str(archive)}, lambda *a: None), 1)
                with zipfile.ZipFile(archive / "content.dat") as z:
                    self.assertEqual(z.read("untouched.dat"), b"preserve exactly")
                self.assertTrue(adapter.is_installed(p, {}))
                adapter.uninstall(p, {"path": str(archive)}, lambda *a: None)
                self.assertEqual(before, (archive / "content.dat").read_bytes())
                self.assertFalse(adapter.is_installed(p, {}))
                p.close()


if __name__ == "__main__":
    unittest.main()
