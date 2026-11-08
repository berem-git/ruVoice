import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np

from ruvoice import audio, identity, tts, voice_trials, judge
from ruvoice.project import Project
from ruvoice.dub import DubJobs


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch('ruvoice.project.PROJECTS_DIR', Path(self.tmp.name))
        self.patch.start()
        self.p = Project('test:unrelated')
        self.p.add_lines([{'id': 'one', 'char': 'voice1', 'orig': 'orig/one.wav', 'duration': 3,
                           'en': 'Hello there', 'src': {}},
                          {'id': 'two', 'char': 'voice2', 'orig': 'orig/two.wav', 'duration': 3,
                           'en': 'Other speaker', 'src': {}}])
        self.p.gen_dir.mkdir(exist_ok=True)
        self.source = self.p.gen_dir / 'one.wav'
        audio.write(self.source, np.sin(np.arange(48000) * 0.1).astype('float32') * 0.2, 16000)

    def tearDown(self):
        self.p.close()
        self.patch.stop()
        self.tmp.cleanup()

    def propose(self):
        return identity.propose(self.p, 'voice1', 'gen/one.wav', 'Привет, как дела?', 'voxcpm', 'one')

    def test_proposal_is_not_activated_and_snapshot_survives_regeneration(self):
        row = self.propose()
        self.assertIsNone(identity.active(self.p, 'voice1'))
        original = (self.p.dir / row['audio']).read_bytes()
        audio.write(self.source, np.ones(48000, dtype='float32') * .1, 16000)
        self.assertEqual((self.p.dir / row['audio']).read_bytes(), original)
        identity.activate(self.p, row['id'])
        self.assertEqual(identity.active(self.p, 'voice1')['id'], row['id'])
        self.assertEqual(identity.reference(identity.active(self.p, 'voice1'))['language'], 'ru')
        reopened = Project('test:unrelated')
        try:
            self.assertEqual(identity.active(reopened, 'voice1')['id'], row['id'])
        finally:
            reopened.close()

    def test_reject_tampered_or_reassigned_anchor(self):
        row = self.propose()
        identity.activate(self.p, row['id'])
        with self.p.db() as c:
            c.execute("update lines set char='voice2' where id='one'")
        with self.assertRaises(ValueError):
            identity.active(self.p, 'voice1')
        with self.p.db() as c:
            c.execute("update lines set char='voice1' where id='one'")
        (self.p.dir / row['audio']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            identity.active(self.p, 'voice1')

    def test_wrong_engine_and_source_are_rejected(self):
        row = self.propose()
        identity.activate(self.p, row['id'])
        with self.assertRaises(ValueError):
            identity.active(self.p, 'voice1', 'ru')
        with self.assertRaises(ValueError):
            identity.propose(self.p, 'voice2', 'gen/one.wav', 'Привет', 'voxcpm', 'one')
        with self.assertRaises(ValueError):
            identity.propose(self.p, 'voice1', '../one.wav', 'Привет', 'voxcpm', 'one')
        with self.assertRaises(ValueError):
            identity.propose(self.p, 'voice1', 'gen/one.wav', 'English', 'voxcpm', 'one')

    def test_approved_anchor_cannot_be_moved_by_new_bad_takes(self):
        row = self.propose()
        identity.activate(self.p, row['id'])
        embed = Mock(return_value=np.array([[1, 0]], dtype='float32'))
        a, floor = identity.centroid(self.p, identity.active(self.p, 'voice1'), embed)
        jobs = DubJobs.__new__(DubJobs)
        jobs._ru_voice_add(self.p, 'voice1', np.array([0., 1.]), .9)
        b, _ = identity.centroid(self.p, identity.active(self.p, 'voice1'), embed)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(embed.call_count, 1)
        self.assertGreater(floor, 0)
        with patch('ruvoice.judge.ecapa_ready', return_value=True), patch('ruvoice.judge.ecapa_embed', embed):
            e, _ = jobs._ru_voice(self.p, 'voice1')
            np.testing.assert_array_equal(e, a)
        self.assertIsNone(self.p.get('ru_voice'))

    def test_voxcpm_uses_pin_and_emotion_does_not_change_reference(self):
        row = self.propose()
        identity.activate(self.p, row['id'])
        jobs = DubJobs.__new__(DubJobs)
        with patch.object(jobs, 'character_profile', side_effect=AssertionError('English profile was used')):
            ref = jobs._refs_for(self.p, 'voxcpm', {'voice1'})['voice1']
        self.p.set('settings', {'emotion_refs': True})
        self.assertIs(jobs._ref_for_line(self.p, self.p.line('one'), 'ru', ref, {'emotion': 'angry'}), ref)
        self.assertEqual(ref['identity'], row['id'])

    def test_merge_invalidates_active_anchors(self):
        row = self.propose()
        identity.activate(self.p, row['id'])
        self.p.merge_characters('voice2', ['voice1'])
        self.assertIsNone(identity.active(self.p, 'voice1'))

    def test_rejected_accent_sample_cannot_be_activated(self):
        row = self.propose()
        identity.activate(self.p, row['id'])
        identity.reject(self.p, row['id'], 'Сильный акцент, проверено пользователем')
        self.assertIsNone(identity.active(self.p, 'voice1'))
        with self.assertRaises(ValueError):
            identity.activate(self.p, row['id'])

    def test_prompt_cache_keeps_only_active_voice(self):
        model = Mock()
        model.create_voice_clone_prompt.return_value = [object()]
        with patch.object(tts, '_model', model), patch.object(tts, '_prompts', {}):
            first = tts._ru_prompt('voice1.wav', None)
            self.assertIs(tts._ru_prompt('voice1.wav', None), first)
            tts._ru_prompt('voice2.wav', 'Русский эталон')
            self.assertEqual(list(tts._prompts), [('voice2.wav', 'Русский эталон')])
            self.assertEqual(model.create_voice_clone_prompt.call_count, 2)

    def test_wrong_voice_cannot_pass_on_good_text_and_duration(self):
        res = judge.judge_line(str(self.source), 'Привет', str(self.source), None,
                               heard='Привет', full=False, consistency=.3, cons_floor=.55)
        self.assertEqual(res['score'], 75)
        self.assertEqual(res['verdict'], 'rework')
        self.assertIn('drift', res['codes'])

    def test_voxcpm_continuation_is_russian_only_and_opt_in(self):
        from ruvoice.engines.worker import VoxCPMEngine
        engine = VoxCPMEngine()
        engine.model = Mock()
        engine.model.generate.return_value = np.zeros(100)
        engine.model.tts_model.sample_rate = 24000
        with patch.dict('sys.modules', {'torch': Mock()}):
            engine.generate({'text': 'Привет', 'ref_audio': 'english.wav'})
            self.assertNotIn('prompt_text', engine.model.generate.call_args.kwargs)
            engine.generate({'text': 'Привет', 'ref_audio': 'russian.wav', 'russian_anchor_text': 'Добрый день',
                             'params': {'instruct': 'Чёткое произношение'}})
            self.assertEqual(engine.model.generate.call_args.kwargs['prompt_text'], 'Добрый день')
            self.assertEqual(engine.model.generate.call_args.kwargs['text'], 'Привет')
            with self.assertRaises(ValueError):
                engine.generate({'text': 'Привет', 'ref_audio': 'english.wav', 'russian_anchor_text': 'Hello'})

    def test_api_explicit_selection_busy_guard_and_auto_approval(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ruvoice.api_dub import create_router
        jobs, lib = Mock(), Mock()
        lib.get.return_value = {'name': 'Unrelated game'}
        jobs.project.return_value = self.p
        jobs.current = None
        self.p.update_line('one', ru='Привет', gen='gen/one.wav', status='generated', info={'engine': 'voxcpm', 'judge': {'score': 75, 'verdict': 'rework'}})
        with patch('ruvoice.accel.info', return_value={}):
            app = FastAPI()
            app.include_router(create_router(lib, jobs))
            with TestClient(app) as client:
                url = '/api/projects/test:unrelated/voice-identities'
                jobs.current = 'other-project'
                self.assertEqual(client.post(url, json={'line': 'one'}).status_code, 409)
                jobs.current = None
                res = client.post(url, json={'line': 'one'})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertIn('voice1', res.json()['anchors'])
                res = client.post('/api/projects/test:unrelated/mark', json={'status': 'approved', 'min_score': 70})
                self.assertEqual(res.json()['changed'], 0)
                res = client.post(url, json={'clear': True, 'char': 'voice1'})
                self.assertEqual(res.json()['anchors'], {})

    def test_russian_anchor_enables_icl_but_english_reference_does_not(self):
        model = Mock()
        model.generate_voice_clone.return_value = ([np.zeros(100)], 24000)
        with patch.object(tts, '_model', model), patch.object(tts, 'load_ru'), \
             patch.object(tts, '_ru_prompt', return_value=object()) as prompt:
            tts._ru_generate(['Привет'], [{'audio': 'en.wav', 'text': 'Hello'}], [3], {})
            prompt.assert_called_with('en.wav', None)
            tts._ru_generate(['Привет'], [{'audio': 'ru.wav', 'text': 'Добрый день', 'identity': 'x', 'language': 'ru'}], [3], {})
            prompt.assert_called_with('ru.wav', 'Добрый день')

    def test_trials_require_acoustic_matches_instead_of_using_all_group_lines(self):
        seed = self.p.line('one')
        self.p.update_line('one', ru='Привет')
        self.p.update_line('two', ru='Другой голос')
        with patch('ruvoice.judge.ecapa_embed', return_value=np.array([[1., 0.]])):
            with self.assertRaises(ValueError):
                voice_trials.select_lines(self.p, seed)

    def test_source_selection_excludes_other_voice_in_same_character_group(self):
        self.p.orig_dir.mkdir(exist_ok=True)
        ids = ['one', 'a', 'b', 'c', 'impostor']
        for i in ids:
            audio.write(self.p.orig_dir / f'{i}.wav', np.ones(48000, dtype='float32') * .1, 16000)
            self.p.add_lines([{'id': i, 'char': 'voice1', 'orig': f'orig/{i}.wav', 'duration': 3, 'src': {}, 'en': 'Original'}])
            self.p.update_line(i, ru=f'Русская реплика {i}')
        E = np.array([[1, 0], [.99, .01], [.98, .02], [.97, .03], [0, 1]], dtype='float32')
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.savez(self.p.dir / 'speakers.npz', ids=np.array(ids), E=E)
        chosen, audit = voice_trials.select_lines(self.p, self.p.line('one'))
        self.assertEqual({r['id'] for r in chosen}, {'a', 'b', 'c'})
        self.assertEqual(audit['below_threshold'], 1)

    def test_inactive_lora_adapters_are_offloaded(self):
        from ruvoice import ttsft
        class Layer:
            def __init__(self):
                self.lora_A = {'a': Mock(), 'b': Mock()}
                self.lora_B = {'a': Mock(), 'b': Mock()}
                self.enable_adapters = Mock()
                self.set_adapter = Mock()
        layer = Layer()
        talker = Mock()
        talker.parameters.return_value = iter([Mock(device='cuda:0')])
        talker.modules.return_value = [layer]
        with patch.dict('sys.modules', {'peft.tuners.tuners_utils': Mock(BaseTunerLayer=Layer)}):
            ttsft.activate(talker, 'b')
        layer.lora_A['a'].to.assert_called_once_with('cpu')
        layer.lora_B['a'].to.assert_called_once_with('cpu')
        layer.lora_A['b'].to.assert_called_once_with('cuda:0')
        layer.set_adapter.assert_called_once_with('b')


if __name__ == '__main__':
    unittest.main()
