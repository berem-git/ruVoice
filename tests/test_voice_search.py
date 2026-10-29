import unittest
import types
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from ruvoice.voice_search import select


class CandidateSelectionTests(unittest.TestCase):
    def candidate(self, name, similarity=.8, **changes):
        row = dict(name=name, cer=0., duration_ratio=1., clipped_fraction=0.,
                   ve_source=similarity, ve_holdout=similarity, ecapa_source=similarity)
        row.update(changes)
        return row

    def test_wrong_words_cannot_win_with_high_identity(self):
        good = self.candidate('good')
        self.assertIs(select([good, self.candidate('wrong', .99, cer=.4)]), good)

    def test_clipping_and_runaway_audio_cannot_win(self):
        good = self.candidate('good')
        bad = [self.candidate('clipped', .99, clipped_fraction=.1),
               self.candidate('runaway', .99, duration_ratio=3.)]
        self.assertIs(select(bad + [good]), good)

    def test_invalid_measurement_is_rejected(self):
        self.assertIsNone(select([self.candidate('nan', ve_source=float('nan'))]))

    def test_previous_version_is_kept_when_it_scores_better(self):
        baseline = self.candidate('previous', .9)
        self.assertIs(select([self.candidate('new', .85), baseline]), baseline)

    def test_no_valid_candidate_is_explicit(self):
        self.assertIsNone(select([self.candidate('wrong', cer=.5)]))


class ContinuationProtocolTests(unittest.TestCase):
    def engine(self):
        from ruvoice.engines.worker import VoxCPMEngine
        engine = VoxCPMEngine()
        engine.model = Mock()
        engine.model.generate.return_value = [.1, -.1]
        engine.model.tts_model.sample_rate = 48000
        return engine

    def test_continuation_does_not_speak_style_directions(self):
        engine = self.engine()
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace()}):
            engine.generate(dict(text='Привет.', ref_audio='voice.wav', prompt_audio='source.wav',
                                 prompt_text='Hello.', params={'instruct': 'sad, very quiet'}))
        kw = engine.model.generate.call_args.kwargs
        self.assertEqual(kw['text'], 'Привет.')
        self.assertEqual(kw['prompt_wav_path'], 'source.wav')
        self.assertEqual(kw['prompt_text'], 'Hello.')

    def test_reference_only_keeps_style_control(self):
        engine = self.engine()
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace()}):
            engine.generate(dict(text='Привет.', ref_audio='voice.wav', params={'instruct': 'quiet'}))
        self.assertEqual(engine.model.generate.call_args.kwargs['text'], '(quiet)Привет.')

    def test_unpaired_transcript_is_rejected(self):
        engine = self.engine()
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace()}), self.assertRaises(ValueError):
            engine.generate(dict(text='Привет.', ref_audio='voice.wav', prompt_audio='source.wav'))
        engine.model.generate.assert_not_called()


class SavedRecipeTests(unittest.TestCase):
    def test_recipe_is_invalidated_by_text_or_source_change(self):
        from ruvoice.dub import DubJobs
        from ruvoice.performance import fingerprint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source.wav').write_bytes(b'original')
            (root / 'reference.wav').write_bytes(b'reference')
            chosen = dict(audio='reference.wav', ru='Привет.', source=fingerprint(root / 'source.wav'),
                          prompt_audio='source.wav', prompt_text='Hello.', seed=42)
            line = dict(orig='source.wav', ru='Привет.', info={'voice_recipe': chosen})
            project = types.SimpleNamespace(dir=root)
            jobs = object.__new__(DubJobs)
            fallback = {'audio': 'default.wav'}
            result = jobs._ref_for_line(project, line, 'voxcpm', fallback, {})
            self.assertEqual(result['prompt_text'], 'Hello.')
            self.assertEqual(result['seed'], 42)
            line['ru'] = 'Другой текст.'
            self.assertIs(jobs._ref_for_line(project, line, 'voxcpm', fallback, {}), fallback)
            line['ru'] = 'Привет.'
            (root / 'source.wav').write_bytes(b'changed recording')
            self.assertIs(jobs._ref_for_line(project, line, 'voxcpm', fallback, {}), fallback)

    def test_recipe_cannot_reference_outside_project(self):
        from ruvoice.dub import DubJobs
        from ruvoice.performance import fingerprint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source.wav').write_bytes(b'original')
            line = dict(orig='source.wav', ru='Привет.', info={'voice_recipe': dict(
                audio='../outside.wav', ru='Привет.', source=fingerprint(root / 'source.wav'))})
            fallback = {'audio': 'default.wav'}
            self.assertIs(object.__new__(DubJobs)._ref_for_line(types.SimpleNamespace(dir=root),
                          line, 'voxcpm', fallback, {}), fallback)


if __name__ == '__main__':
    unittest.main()
