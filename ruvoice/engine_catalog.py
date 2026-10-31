"""Engine evidence and format capabilities; recognition is separate from a working dubbing adapter."""
from __future__ import annotations

SIGNATURES = [
    (b'GDPC', 'godot-pck'), (b'RPA-', 'renpy-rpa'),
    (b'UnityFS', 'unity-bundle'), (b'UnityWeb', 'unity-bundle'), (b'UnityRaw', 'unity-bundle'),
    (b'AKPK', 'wwise-package'), (b'BKHD', 'wwise-bank'), (b'FSB5', 'fmod-fsb5'),
    (b'BHD5', 'bhd5'), (b'BND4', 'bnd4'), (b'BTDX', 'bethesda-ba2'), (b'BSA\0', 'bethesda-bsa'),
    (b'RDAR', 'redengine-archive'), (b'RPF7', 'rage-rpf'), (b'RPF8', 'rage-rpf'),
    (b'MPQ\x1a', 'blizzard-mpq'), (b'PACK', 'quake-pak'), (b'PK\x03\x04', 'zip'),
    (b'OggS', 'ogg'), (b'fLaC', 'flac'),
]
PROBE_EXT = {'.pck', '.rpa', '.assets', '.bundle', '.unity3d', '.resource', '.dat', '.bin', '.pak',
             '.ba2', '.bsa', '.archive', '.rpf', '.mpq', '.bnk', '.fsb', '.bank', '.bhd', '.bnd',
             '.zip', '.pk3', '.pk4', '.iwd', '.love', '.nw', '.wav', '.wem', '.ogg', '.flac'}

# Rules require actual file/layout evidence, never a game title or executable title.
RULES = [
    ('unity', 'Unity', {'unityplayer.dll', 'globalgamemanagers', 'data.unity3d'}, set(), {'unity-bundle'}),
    ('unreal', 'Unreal Engine 4.26+ / 5', set(), {'.utoc', '.ucas'}, set()),
    ('source2', 'Source 2', {'gameinfo.gi'}, set(), set()),
    ('godot', 'Godot', {'project.godot'}, set(), {'godot-pck', 'godot-embedded-pck'}),
    ('renpy', "Ren'Py", {'renpy'}, {'.rpyc'}, {'renpy-rpa'}),
    ('fromsoft', 'FromSoftware', set(), set(), {'bhd5', 'bnd4'}),
    ('reengine', 'RE Engine', set(), set(), set()),
    ('creation', 'Creation Engine (Bethesda)', set(), {'.ba2', '.bsa'}, {'bethesda-ba2', 'bethesda-bsa'}),
    ('redengine', 'REDengine', set(), set(), {'redengine-archive'}),
    ('rage', 'RAGE', set(), {'.rpf'}, {'rage-rpf'}),
    ('cryengine', 'CryEngine', {'crysystem.dll', 'cryengine.dll'}, set(), set()),
    ('rpg-maker-mz', 'RPG Maker MZ', {'rmmz_core.js'}, set(), set()),
    ('rpg-maker-mv', 'RPG Maker MV', {'rpg_core.js'}, set(), set()),
    ('rpg-maker-vx', 'RPG Maker VX Ace', set(), {'.rgss3a'}, set()),
    ('rpg-maker-xp', 'RPG Maker XP/VX', set(), {'.rgss2a', '.rgssad'}, set()),
    ('gamemaker', 'GameMaker', {'data.win', 'game.unx'}, set(), set()),
    ('monogame', 'XNA / FNA / MonoGame', {'monogame.framework.dll', 'fna.dll'}, {'.xnb'}, set()),
    ('love', 'LÖVE', {'love.dll'}, {'.love'}, set()),
    ('anvil', 'Anvil', set(), {'.forge'}, set()),
    ('glacier', 'Glacier', set(), {'.rpkg'}, set()),
    ('blizzard', 'Blizzard MPQ', set(), {'.mpq'}, {'blizzard-mpq'}),
    ('quake', 'Quake / PACK', set(), set(), {'quake-pak'}),
    ('idtech3', 'id Tech 3 / PK3', set(), {'.pk3'}, set()),
    ('idtech4', 'id Tech 4 / PK4', set(), {'.pk4'}, set()),
    ('nwjs', 'NW.js', {'nw.dll', 'package.nw'}, set(), set()),
]

FORMAT_NOTES = {
    'godot-embedded-pck': 'Godot PCK встроен в exe; для его безопасного изменения нужен отдельный адаптер.',
    'unity-bundle': 'Unity bundle распознан; внутренние AudioClip пока требуют отдельного адаптера.',
    'bethesda-ba2': 'BA2 распознан; извлечение и сборка этого контейнера пока не реализованы.',
    'bethesda-bsa': 'BSA распознан; извлечение и сборка этого контейнера пока не реализованы.',
    'redengine-archive': 'REDengine archive распознан; нужен адаптер контейнера.',
    'rage-rpf': 'RAGE RPF распознан; нужен адаптер контейнера.',
    'blizzard-mpq': 'MPQ распознан; нужен адаптер контейнера.',
    'quake-pak': 'Quake PACK распознан; этот формат отличается от ZIP/PK3.',
}


def signature(header):
    if header[:4] in (b'RIFF', b'RIFX'):
        return 'wav' if header[8:12] == b'WAVE' else 'fmod-bank' if header[8:12] == b'FEV ' else 'riff'
    return next((kind for prefix, kind in SIGNATURES if header.startswith(prefix)), None)


def candidates(names, exts, evidence):
    found = []
    for ident, name, markers, suffixes, formats in RULES:
        matches = sorted(markers & names) + sorted(e for e in suffixes if exts[e])
        magic = sorted(formats & evidence.keys())
        if matches or magic:
            found.append(dict(id=ident, name=name, confidence='signature' if magic else 'layout',
                              evidence=matches + [f'{kind}: {evidence[kind][0]}' for kind in magic]))
    def add(ident, name, signs):
        if not any(x['id'] == ident for x in found):
            found.append(dict(id=ident, name=name, confidence='layout', evidence=signs))
    if exts['.pak'] and 'paks' in names and not any(n.startswith('re_chunk_') for n in names):
        add('unreal', 'Unreal Engine', ['Paks/', '.pak'])
    if 'gameinfo.txt' in names and exts['.vpk']:
        add('source', 'Source', ['gameinfo.txt', '.vpk'])
    if any(n.startswith('re_chunk_') and n.endswith('.pak') for n in names):
        add('reengine', 'RE Engine', ['re_chunk_*.pak'])
    if exts['.cas'] and exts['.cat']:
        add('frostbite', 'Frostbite', ['.cas', '.cat'])
    if exts['.fat'] and exts['.dat']:
        add('dunia', 'Dunia', ['.fat', '.dat'])
    if exts['.bhd'] and exts['.bdt']:
        add('fromsoft', 'FromSoftware', ['.bhd', '.bdt'])
    # Specific engines outrank generic web/ZIP technology hints.
    return sorted(found, key=lambda x: (x['id'] in {'nwjs', 'idtech3', 'idtech4'}, x['confidence'] != 'signature'))


def capability(analysis, adapter_id):
    notes = list(analysis.get('format_warnings') or []) + [FORMAT_NOTES[k] for k in (analysis.get('formats') or {}) if k in FORMAT_NOTES]
    if adapter_id:
        partial = bool(notes and adapter_id in {'loose', 'zip-audio', 'wwise', 'fmod'})
        return dict(status='partial' if partial else 'available', adapter=adapter_id,
                    note='Доступен найденный звук; часть ресурсов остаётся в неподдерживаемых контейнерах.' if partial else
                         'Есть адаптер извлечения и установки. Состав реплик проверяется при подготовке.', blockers=notes)
    return dict(status='adapter_required' if analysis.get('engine') or analysis.get('formats') else 'unknown',
                adapter=None, note=notes[0] if notes else
                'Поддерживаемый звук не найден. Для упакованных или зашифрованных ресурсов нужен подходящий адаптер.', blockers=notes)


def catalog():
    return {'version': 1, 'engines': [{'id': ident, 'name': name} for ident, name, *_ in RULES] +
            [{'id': key, 'name': name} for key, name in [('unreal-legacy', 'Unreal Engine'), ('source', 'Source'),
                                                       ('frostbite', 'Frostbite'), ('dunia', 'Dunia')]],
            'new_archive_adapters': ['godot-pck', 'renpy-rpa'],
            'note': 'Распознавание движка не означает полную поддержку всех версий, шифрования и форматов озвучки.'}
