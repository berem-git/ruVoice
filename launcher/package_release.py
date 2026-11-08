"""Build a clean Windows portable ZIP from an explicit allowlist, never the working directory wholesale."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'__pycache__', '.git', '.cache', '.pytest_cache'}
SOURCE_ROOTS = {'ruvoice', 'launcher', 'tests'}
SOURCE_FILES = {'README.md', 'pyproject.toml', 'uv.lock', 'requirements-voxcpm.txt', 'requirements-rvc.txt',
                'requirements-omnivoice.txt', 'KNOWN_ISSUES.md', 'RELEASE_NOTES.md', 'THIRD_PARTY_NOTICES.md'}
SOURCE_TOOLS = {'tools/env.ps1', 'tools/make_donors.py'}
RUNTIME_ROOTS = ('runtime/python', 'runtime/engines/voxcpm')
TOOL_ROOTS = ('tools/vgmstream', 'tools/llama.cpp', 'tools/repak', 'tools/retoc', 'tools/uv', 'tools/fromsoft')
# Откуда брать runtime и tools: по умолчанию из папки программы; --base — распакованный прошлый релиз
# (там лежат проверенные среда и инструменты, которых нет в рабочей копии разработчика)
BASES: dict[str, Path] = {}


def clean_file(path: Path) -> bool:
    return not any(s in SKIP for s in path.parts) and path.suffix.lower() not in {'.pyc', '.pyo', '.log', '.tmp'}


def source_files() -> list[Path]:
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True, capture_output=True)
    paths = [Path(s.decode('utf-8')) for s in result.stdout.split(b'\0') if s]
    return [ROOT / p for p in paths if clean_file(p) and
            (p.parts[0] in SOURCE_ROOTS or p.as_posix() in SOURCE_FILES or p.as_posix() in SOURCE_TOOLS)]


def inputs() -> list[tuple[Path, Path]]:
    """(исходный файл, путь внутри пакета)."""
    files = [(f, f.relative_to(ROOT)) for f in source_files()]
    for relative in RUNTIME_ROOTS + TOOL_ROOTS:
        base = BASES.get(relative.split('/')[0], ROOT)
        root = base / relative
        if not root.is_dir():
            raise RuntimeError(f'Missing release component: {relative} (в {base})')
        files.extend((f, f.relative_to(base)) for f in root.rglob('*') if f.is_file() and clean_file(f))
    tools_base = BASES.get('tools', ROOT)
    files.extend((tools_base / rel, Path(rel)) for rel in ('tools/ffmpeg/bin/ffmpeg.exe', 'tools/ffmpeg/LICENSE.txt'))
    for path, _ in files:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'Invalid package input: {path}')
    return sorted(set(files), key=lambda x: x[1].as_posix())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def stage(version: str, base: Path) -> Path:
    dest = base / f'RuVoice-{version}'
    if dest.exists():
        raise RuntimeError(f'Staging already exists; choose a new build directory: {dest}')
    dest.mkdir(parents=True)
    files = inputs()
    print(f'Staging {len(files)} files (no models or user data)', flush=True)
    for i, (path, rel) in enumerate(files):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if i % 5000 == 0:
            print(f'Copied {i}/{len(files)}', flush=True)
    exe = dest / 'RuVoice.exe'
    subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(ROOT / 'launcher/build.ps1'), '-Output', str(exe)], check=True, cwd=ROOT)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    distributions = {}
    for relative in ('runtime/python/Lib/site-packages', 'runtime/engines/voxcpm'):
        distributions[relative] = sorted([{'name': d.metadata['Name'], 'version': d.version}
                                         for d in importlib.metadata.distributions(path=[str(dest / relative)])],
                                        key=lambda d: d['name'].lower())
    manifest = {'version': version, 'platform': 'windows-x64', 'source_commit': commit,
                'built_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'models_included': False, 'user_data_included': False, 'cuda_delivery': 'official PyTorch download during setup',
                'speech_pipeline': 'v2: donor speech + per-character RVC timbre; engine envs installed from the app',
                'runtime_distributions': distributions, 'checks': {},
                'launcher_sha256': sha256(exe), 'file_count': len(files) + 3}
    (dest / 'release-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    (dest / 'START_HERE.txt').write_text((ROOT / 'RELEASE_NOTES.md').read_text('utf-8'), 'utf-8-sig')
    shutil.copy2(ROOT / 'launcher/install-cuda.cmd', dest / 'Установить CUDA.cmd')
    return dest


def pack(dest: Path, version: str) -> Path:
    target = dest.parent / f'RuVoice-{version}-Windows-x64.zip'
    if target.exists():
        raise RuntimeError(f'Archive already exists: {target}')
    files = sorted(f for f in dest.rglob('*') if f.is_file() and clean_file(f)
                   and f.relative_to(dest).parts[:3] != ('runtime', 'accel', 'cuda'))
    print(f'Compressing {len(files)} files', flush=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i, path in enumerate(files):
            rel = path.relative_to(dest)
            if rel.parts[:3] == ('runtime', 'accel', 'cuda'):
                continue
            if rel.parts[0] in {'data', 'models', 'cache', '.git'}:
                raise RuntimeError(f'User data in release: {rel}')
            archive.write(path, (Path(dest.name) / rel).as_posix())
            if i % 5000 == 0:
                print(f'Compressed {i}/{len(files)}', flush=True)
    digest = sha256(target)
    (target.parent / 'SHA256SUMS.txt').write_text(f'{digest}  {target.name}\n', 'ascii')
    print(json.dumps({'archive': str(target), 'bytes': target.stat().st_size, 'sha256': digest}), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='1.1.0')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--base', type=Path, default=None,
                        help='распакованный прошлый релиз: оттуда берутся runtime/ и tools/ вместо рабочей копии')
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--pack-only', action='store_true')
    args = parser.parse_args()
    if args.base:
        BASES.update(runtime=args.base.resolve(), tools=args.base.resolve())
    base = (args.output or ROOT / 'dist' / args.version).resolve()
    dest = base / f'RuVoice-{args.version}' if args.pack_only else stage(args.version, base)
    if not args.stage_only:
        pack(dest, args.version)
    else:
        print(f'Staged: {dest}', flush=True)


if __name__ == '__main__':
    main()
