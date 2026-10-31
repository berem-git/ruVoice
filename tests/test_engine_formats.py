import hashlib
import io
import json
import pickle
import struct
import tempfile
import unittest
import zipfile
import zlib
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ruvoice import adapters, audio, detect, subtitles, discovery
from ruvoice.adapters.open_packages import PCK, RPA
from ruvoice.adapters.packed_audio import GodotPCKAdapter, RenpyRPAAdapter
from ruvoice.engine_catalog import candidates, capability
from ruvoice.project import Project


def pck(path, files, version=2, flags=0):
    # Independent fixture matching the published Godot 3.x/4.x directory layouts.
    header = b'GDPC' + struct.pack('<4I', version, 3 if version == 1 else 4, 0, 0)
    if version == 2:
        header += struct.pack('<IQ', flags, 0)
    header += bytes(64) + struct.pack('<I', len(files))
    names = [(name, name.encode() + b'\0' * (-len(name.encode()) % 4), data) for name, data in files.items()]
    base = len(header) + sum(4 + len(encoded) + 32 + (4 if version == 2 else 0) for _, encoded, _ in names)
    payload, index = b'', b''
    for name, encoded, data in names:
        index += struct.pack('<I', len(encoded)) + encoded
        index += struct.pack('<QQ', base + len(payload), len(data)) + hashlib.md5(data).digest()
        if version == 2:
            index += struct.pack('<I', 0)
        payload += data
    path.write_bytes(header + index + payload)


def rpa(path, files, version=3, split_prefix=False):
    payload, index, key = b'', {}, 0x1234 if version == 3 else 0
    for name, data in files.items():
        prefix = data[:3] if split_prefix and version == 3 else b''
        content = data[len(prefix):]
        index[name] = [(40 + len(payload) ^ key, len(content) ^ key, prefix)] if version == 3 else [(40 + len(payload), len(content))]
        payload += content
    offset = 40 + len(payload)
    header = (f'RPA-3.0 {offset:016x} {key:08x}\n' if version == 3 else f'RPA-2.0 {offset:016x}\n').encode().ljust(40, b'\0')
    path.write_bytes(header + payload + zlib.compress(pickle.dumps(index, protocol=2)))


class EngineDetectionTests(unittest.TestCase):
    def test_wwise_pck_is_not_godot_even_if_named_after_exe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'data.pck').write_bytes(b'AKPK' + bytes(64))
            (root / 'data.exe').write_bytes(b'MZ' + bytes(64))
            result = detect.analyze(str(root))
            self.assertNotEqual(result['engine'], 'Godot')
            self.assertIn('Wwise', result['middleware'])

    def test_pack_header_recognises_godot_without_title_rules(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pck(root / 'unrelated.pck', {'keep.txt': b'untouched'})
            result = detect.analyze(str(root))
            self.assertEqual(result['engine'], 'Godot')
            self.assertEqual(result['engine_candidates'][0]['confidence'], 'signature')

    def test_embedded_godot_is_recognised_but_not_marked_writable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            packed = b'GDPC' + bytes(48)
            (root / 'unknown.exe').write_bytes(b'MZ' + bytes(30) + packed + struct.pack('<Q', len(packed)) + b'GDPC')
            result = detect.analyze(str(root))
            self.assertEqual(result['engine'], 'Godot')
            self.assertIn('godot-embedded-pck', result['formats'])
            self.assertIsNone(adapters.for_game({'path': td}))
            self.assertEqual(capability(result, None)['status'], 'adapter_required')

    def test_engine_rules_and_partial_capability(self):
        for marker, expected in [('unityplayer.dll', 'unity'), ('gameinfo.gi', 'source2'),
                                 ('rmmz_core.js', 'rpg-maker-mz'), ('crysystem.dll', 'cryengine')]:
            self.assertEqual(candidates({marker}, Counter(), {})[0]['id'], expected)
        info = {'engine': 'Unity', 'formats': {'unity-bundle': ['content.bundle']}}
        self.assertEqual(capability(info, 'loose')['status'], 'partial')
        self.assertEqual(capability(info, None)['status'], 'adapter_required')

    def test_zip_based_engine_packages_are_usable(self):
        for extension in ('.love', '.pk3', '.pk4', '.iwd', '.nw'):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as td:
                path = Path(td) / ('content' + extension)
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('voices/en/greet.wav', b'RIFF' + bytes(50))
                self.assertEqual(adapters.for_game({'path': td}) .id, 'zip-audio')

    def test_support_api_does_not_require_gpu_or_start_generation(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ruvoice.api_dub import create_router
        with patch('ruvoice.api_dub.accel.info', return_value={}):
            app = FastAPI()
            app.include_router(create_router(None, None))
            with TestClient(app) as client:
                response = client.get('/api/support')
                self.assertEqual(response.status_code, 200)
                self.assertIn('godot-pck', [a['id'] for a in response.json()['adapters']])
                self.assertIn('renpy-rpa', [a['id'] for a in response.json()['adapters']])


class ArchiveFormatTests(unittest.TestCase):
    def test_read_rebuild_and_preserve_all_other_members(self):
        for reader, writer, versions in [(PCK, pck, (1, 2)), (RPA, rpa, (2, 3))]:
            for version in versions:
                with self.subTest(reader=reader.__name__, version=version), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    original = root / ('game.pck' if reader is PCK else 'game.rpa')
                    writer(original, {'voice/greet.wav': b'old words', 'keep.bin': b'unchanged\x00data'}, version)
                    archive = reader(original)
                    self.assertEqual(archive.read(archive.entries[0]), b'old words')
                    replacement = root / 'replacement'
                    replacement.write_bytes(b'longer new words in Russian')
                    rebuilt = root / 'rebuilt'
                    archive.rebuild(rebuilt, {'voice/greet.wav': replacement})
                    result = {e.name: reader(rebuilt).read(e) for e in reader(rebuilt).entries}
                    self.assertEqual(result, {'voice/greet.wav': replacement.read_bytes(), 'keep.bin': b'unchanged\x00data'})

    def test_split_rpa_prefix_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'archive.rpa'
            rpa(path, {'voice.wav': b'prefix and content'}, split_prefix=True)
            archive = RPA(path)
            self.assertEqual(archive.read(archive.entries[0]), b'prefix and content')

    def test_inventory_counts_packed_audio_and_renpy_localisations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = 'translate russian hello:\n    # e "Hello there"\n    e "Привет тебе"\n'
            rpa(root / 'content.rpa', {'voices/en/greet.wav': b'audio', 'tl/russian/dialogue.rpy': script.encode()})
            report = discovery.scan(root)
            self.assertEqual(report['counts']['archived_audio'], 1)
            self.assertEqual(report['counts']['archived_subtitles'], 1)
            rows = subtitles.harvest(root)
            self.assertTrue(any(r['en'] == 'Hello there' and r['ru'] == 'Привет тебе' for r in rows))
            limited = discovery.scan(root, max_files=1)
            self.assertTrue(limited['truncated'])

    def test_renpy_old_new_and_voice_directives(self):
        rows = subtitles._from_rpy('old "Continue"\nnew "Продолжить"\n\n# voice "audio.ogg"\ne "Речь"', 'test')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['en'], 'Continue')
        self.assertTrue(subtitles._usable('Continue', 'Продолжить'))
        self.assertFalse(subtitles._usable('STRING_IDENTIFIER', 'Продолжить'))

    def test_malformed_and_encrypted_archives_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pck(root / 'encrypted.pck', {'voice.wav': b'data'}, flags=1)
            with self.assertRaises(ValueError):
                PCK(root / 'encrypted.pck')
            pck(root / 'bad.pck', {'../outside.wav': b'data'})
            with self.assertRaises(ValueError):
                PCK(root / 'bad.pck')
            rpa(root / 'bad.rpa', {'../outside.wav': b'data'})
            with self.assertRaises(ValueError):
                RPA(root / 'bad.rpa')
            (root / 'short.pck').write_bytes(b'GDPC')
            with self.assertRaises(ValueError):
                PCK(root / 'short.pck')

    def test_rpa_cannot_execute_pickled_objects(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'bad.rpa'
            path.write_bytes(b'RPA-3.0 0000000000000028 00000000\n'.ljust(40, b'\0') + zlib.compress(pickle.dumps(eval)))
            with self.assertRaises(ValueError):
                RPA(path)


class AdapterEndToEndTests(unittest.TestCase):
    def test_extract_official_text_install_and_exact_rollback(self):
        for cls, writer, suffix in [(GodotPCKAdapter, pck, '.pck'), (RenpyRPAAdapter, rpa, '.rpa')]:
            with self.subTest(adapter=cls.id), tempfile.TemporaryDirectory() as td, ExitStack() as stack:
                root = Path(td)
                game = root / 'game'
                game.mkdir()
                memory = io.BytesIO()
                audio.sf.write(memory, np.sin(np.arange(16000) * .1).astype(np.float32) * .1, 16000, format='WAV')
                files = {'voices/en/NPC/greet.wav': memory.getvalue(), 'keep.bin': b'unchanged',
                         'locales/en.json': json.dumps({'greet': 'Hello there'}).encode(),
                         'locales/ru.json': json.dumps({'greet': 'Привет тебе'}, ensure_ascii=False).encode()}
                archive = game / ('content' + suffix)
                writer(archive, files)
                original = archive.read_bytes()
                with patch('ruvoice.project.PROJECTS_DIR', root / 'projects'):
                    project = Project('unknown-title')
                    stack.callback(project.close)
                    adapter = cls()
                    self.assertTrue(adapter.detect({'path': str(game)}))
                    self.assertEqual(adapter.extract(project, {'path': str(game)}, lambda *a: None), 1)
                    line = project.lines()[0]
                    self.assertEqual(line['ru_official'], 'Привет тебе')
                    project.update_line(line['id'], gen=line['orig'], status='approved')
                    self.assertEqual(adapter.install(project, {'path': str(game)}, lambda *a: None), 1)
                    self.assertTrue(adapter.is_installed(project, {}))
                    rebuilt = (PCK if suffix == '.pck' else RPA)(archive)
                    keep = next(e for e in rebuilt.entries if e.name == 'keep.bin')
                    self.assertEqual(rebuilt.read(keep), b'unchanged')
                    adapter.uninstall(project, {'path': str(game)}, lambda *a: None)
                    self.assertEqual(archive.read_bytes(), original)
                    self.assertFalse(adapter.is_installed(project, {}))


if __name__ == '__main__':
    unittest.main()
