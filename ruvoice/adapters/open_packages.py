"""Bounded readers and streaming rebuilds for unencrypted standalone Godot PCK and Ren'Py RPA.

Format references: Godot core/io/file_access_pack.cpp (3.x/4.x), Ren'Py renpy/loader.py.
No archive member name is ever extracted as a filesystem path.
"""
from __future__ import annotations

import hashlib
import io
import pickle
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_INDEX = 32 << 20
MAX_ENTRIES = 200_000


def member_name(value) -> str:
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if not isinstance(value, str):
        raise ValueError('Invalid member name')
    name = value.removeprefix('res://').replace('\\', '/')
    if not name or len(name) > 8192 or '\0' in name or ':' in name or name.startswith('/') or '..' in PurePosixPath(name).parts:
        raise ValueError('Unsafe package member name')
    return name


def read_exact(stream, length):
    data = stream.read(length)
    if len(data) != length:
        raise ValueError('Truncated package')
    return data


def copy_range(src, dst, start, length):
    src.seek(start)
    while length:
        chunk = read_exact(src, min(length, 1 << 20))
        dst.write(chunk)
        length -= len(chunk)


@dataclass
class Entry:
    name: str
    size: int
    pieces: tuple = ()
    key: object = None
    flags: int = 0
    slot: int = 0


class PCK:
    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        self.entries = []
        with self.path.open('rb') as stream:
            if read_exact(stream, 4) != b'GDPC':
                raise ValueError('Not a standalone Godot PCK')
            self.version, *engine = struct.unpack('<4I', read_exact(stream, 16))
            if self.version not in (1, 2):
                raise ValueError(f'Godot PCK version {self.version} needs another reader')
            base = 0
            if self.version == 2:
                flags, base = struct.unpack('<IQ', read_exact(stream, 12))
                if flags & ~2:
                    raise ValueError('Encrypted or unsupported Godot directory')
            read_exact(stream, 64)
            count, = struct.unpack('<I', read_exact(stream, 4))
            if count > MAX_ENTRIES:
                raise ValueError('Package directory too large')
            names = set()
            for _ in range(count):
                length, = struct.unpack('<I', read_exact(stream, 4))
                if not 0 < length <= 8192 or stream.tell() + length + 36 > MAX_INDEX:
                    raise ValueError('Invalid Godot directory')
                name = member_name(read_exact(stream, length).rstrip(b'\0'))
                if name.casefold() in names:
                    raise ValueError('Duplicate Godot member')
                names.add(name.casefold())
                slot = stream.tell()
                offset, length = struct.unpack('<QQ', read_exact(stream, 16))
                read_exact(stream, 16)  # MD5 is regenerated for replacements.
                flags = struct.unpack('<I', read_exact(stream, 4))[0] if self.version == 2 else 0
                if flags & ~3 or offset + base + length > size:
                    raise ValueError('Invalid Godot member extent')
                self.entries.append(Entry(name, length, ((offset + base, length),), flags=flags, slot=slot))
            end = stream.tell()
            if any(e.size and not e.flags & 2 and e.pieces[0][0] < end for e in self.entries):
                raise ValueError('Godot member overlaps directory')
            stream.seek(0)
            self.header = stream.read(end)
        self.writable = not any(e.flags & 1 for e in self.entries)

    def read(self, entry, limit=256 << 20):
        if entry.flags:
            raise ValueError('Encrypted or removed Godot member')
        if entry.size > limit:
            raise ValueError('Package member exceeds read limit')
        with self.path.open('rb') as stream:
            stream.seek(entry.pieces[0][0])
            return read_exact(stream, entry.size)

    def rebuild(self, destination, replacements):
        if not self.writable:
            raise ValueError('Encrypted Godot members cannot be rebuilt')
        if set(replacements) - {e.name for e in self.entries}:
            raise ValueError('Unknown replacement member')
        with self.path.open('rb') as src, Path(destination).open('w+b') as out:
            out.write(self.header)
            out.write(b'\0' * (-out.tell() % 32))
            base = out.tell() if self.version == 2 else 0
            if self.version == 2:
                out.seek(24)
                out.write(struct.pack('<Q', base))
                out.seek(base)
            for entry in self.entries:
                if entry.flags & 2:
                    continue
                position = out.tell()
                replacement = replacements.get(entry.name)
                digest = None
                if replacement:
                    hasher = hashlib.md5()
                    with Path(replacement).open('rb') as data:
                        while chunk := data.read(1 << 20):
                            out.write(chunk)
                            hasher.update(chunk)
                    length = out.tell() - position
                    digest = hasher.digest()
                else:
                    length = entry.size
                    copy_range(src, out, *entry.pieces[0])
                out.write(b'\0' * (-out.tell() % 32))
                next_position = out.tell()
                out.seek(entry.slot)
                out.write(struct.pack('<QQ', position - base, length))
                if digest is not None:
                    out.write(digest)
                out.seek(next_position)


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {('__builtin__', 'bytes'), ('builtins', 'bytes')}:
            def empty_bytes():
                return b''
            return empty_bytes
        if (module, name) == ('_codecs', 'encode'):
            def encode(text, encoding='latin1'):
                if not isinstance(text, str) or encoding.lower() not in ('latin1', 'latin-1', 'utf-8'):
                    raise ValueError('Unsupported legacy string')
                return text.encode(encoding)
            return encode
        raise ValueError('Executable objects in RPA index are forbidden')

    def persistent_load(self, pid):
        raise ValueError('Persistent pickle references are forbidden')


class RPA:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        size = self.path.stat().st_size
        with self.path.open('rb') as stream:
            head = stream.read(40)
            if head[:8] not in (b'RPA-2.0 ', b'RPA-3.0 '):
                raise ValueError('Unsupported Ren\'Py RPA version')
            self.version = int(chr(head[4]))
            offset = int(head[8:24], 16)
            self.xor_key = int(head[25:33], 16) if self.version == 3 else 0
            if not 24 <= offset < size or size - offset > MAX_INDEX:
                raise ValueError('Invalid RPA index offset')
            stream.seek(offset)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(stream.read(), MAX_INDEX + 1)
            if len(raw) > MAX_INDEX or decoder.unconsumed_tail or not decoder.eof:
                raise ValueError('Invalid or oversized RPA index')
        index = _DataOnlyUnpickler(io.BytesIO(raw), encoding='bytes').load()
        if not isinstance(index, dict) or len(index) > MAX_ENTRIES:
            raise ValueError('Invalid RPA index')
        names, extent_count = set(), 0
        for key, chunks in index.items():
            name = member_name(key)
            if name.casefold() in names or not isinstance(chunks, (list, tuple)) or not 1 <= len(chunks) <= 10000:
                raise ValueError('Duplicate or invalid RPA entry')
            names.add(name.casefold())
            extent_count += len(chunks)
            if extent_count > 1_000_000:
                raise ValueError('RPA has too many extents')
            pieces, length = [], 0
            for chunk in chunks:
                if not isinstance(chunk, tuple) or len(chunk) not in (2, 3):
                    raise ValueError('Invalid RPA extent')
                start, count = chunk[:2]
                if not isinstance(start, int) or not isinstance(count, int):
                    raise ValueError('Invalid RPA extent')
                start, count = start ^ self.xor_key, count ^ self.xor_key
                if start < (34 if self.version == 3 else 24) or count < 0 or start + count > offset:
                    raise ValueError('RPA extent outside data region')
                prefix = chunk[2] if len(chunk) == 3 else b''
                if isinstance(prefix, str):
                    prefix = prefix.encode('latin1')
                if not isinstance(prefix, bytes):
                    raise ValueError('Invalid RPA prefix')
                if prefix:
                    pieces.append(prefix)
                pieces.append((start, count))
                length += len(prefix) + count
            self.entries.append(Entry(name, length, tuple(pieces), key=key))
        self.writable = True

    def read(self, entry, limit=256 << 20):
        if entry.size > limit:
            raise ValueError('Package member exceeds read limit')
        out = io.BytesIO()
        with self.path.open('rb') as src:
            for piece in entry.pieces:
                out.write(piece) if isinstance(piece, bytes) else copy_range(src, out, *piece)
        return out.getvalue()

    def rebuild(self, destination, replacements):
        if set(replacements) - {e.name for e in self.entries}:
            raise ValueError('Unknown replacement member')
        index = {}
        with self.path.open('rb') as src, Path(destination).open('wb') as out:
            out.write(b'\0' * 40)
            for entry in self.entries:
                position = out.tell()
                replacement = replacements.get(entry.name)
                if replacement:
                    with Path(replacement).open('rb') as data:
                        while chunk := data.read(1 << 20):
                            out.write(chunk)
                else:
                    for piece in entry.pieces:
                        out.write(piece) if isinstance(piece, bytes) else copy_range(src, out, *piece)
                length = out.tell() - position
                index[entry.key] = [(position ^ self.xor_key, length ^ self.xor_key, b'')] if self.version == 3 else [(position, length)]
            offset = out.tell()
            out.write(zlib.compress(pickle.dumps(index, protocol=2)))
            out.seek(0)
            header = (f'RPA-3.0 {offset:016x} {self.xor_key:08x}\n' if self.version == 3 else f'RPA-2.0 {offset:016x}\n').encode()
            out.write(header.ljust(40, b'\0'))
