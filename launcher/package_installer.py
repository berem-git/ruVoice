"""Split finalized portable ZIP below GitHub's asset limit and compile a checksum-pinned downloader."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = sys.argv[1] if len(sys.argv) > 1 else '1.1.1'
BASE = ROOT / 'dist' / VERSION
ZIP = BASE / f'RuVoice-{VERSION}-Windows-x64.zip'
CHUNK = 1536 * 1024**2


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    hashes = []
    if ZIP.stat().st_size <= CHUNK:
        i = 1
        hashes.append((ZIP.name, digest(ZIP)))
    else:
        i = split(hashes)
    build(i, hashes)


def split(hashes):
    with ZIP.open('rb') as src:
        i = 0
        while True:
            first = src.read(4 * 1024**2)
            if not first:
                break
            i += 1
            target = ZIP.with_name(ZIP.name + f'.{i:03d}')
            with target.open('xb') as out:
                out.write(first)
                remaining = CHUNK - len(first)
                while remaining:
                    b = src.read(min(4 * 1024**2, remaining))
                    if not b:
                        break
                    out.write(b)
                    remaining -= len(b)
            hashes.append((target.name, digest(target)))
            print('Split', target.name, target.stat().st_size, flush=True)
    return i


def build(i, hashes):
    source = (ROOT / 'launcher/Installer.cs').read_text('utf-8')
    source = source.replace('__ARCHIVE_SHA256__', digest(ZIP)).replace('__PARTS__', str(i))
    cs = BASE / 'Installer.generated.cs'
    cs.write_text(source, 'utf-8-sig')
    exe = BASE / f'RuVoice-{VERSION}-Setup.exe'
    csc = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(csc), '/nologo', '/target:winexe', '/optimize+', '/codepage:65001',
                    '/win32icon:' + str(ROOT / 'launcher/ruvoice.ico'), '/reference:System.Windows.Forms.dll',
                    '/reference:System.IO.Compression.dll', '/reference:System.IO.Compression.FileSystem.dll',
                    '/out:' + str(exe), str(cs)], check=True)
    hashes.append((exe.name, digest(exe)))
    if not any(name == ZIP.name for name, _ in hashes):
        hashes.append((ZIP.name, digest(ZIP)))
    (BASE / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {name}\n' for name, h in hashes), 'ascii')
    (BASE / 'upload-assets.json').write_text(json.dumps([name for name, _ in hashes if name != ZIP.name or i == 1] + ['SHA256SUMS.txt']), 'utf-8')
    print('Installer compiled', exe, flush=True)


if __name__ == '__main__':
    main()
