"""Чтение контейнеров Unreal Engine IoStore (.utoc + .ucas, UE 4.26+ / UE5) на чистом Python.

Поддержаны версии TOC 2–8 (UE 4.26 … UE 5.6), шифрование AES (ключ даёт пользователь), сжатие Oodle/Zlib/LZ4.
Файлы ищутся по индексу каталогов (directory index) — у отгруженных игр он есть всегда.
"""
import struct
from pathlib import Path

from .uepak import NeedsKey, Reader, aes_decrypt, decompress

TOC_MAGIC = b"-==--==--==--==-"
FLAG_COMPRESSED, FLAG_ENCRYPTED, FLAG_SIGNED, FLAG_INDEXED = 1, 2, 4, 8
V_PERFECT_HASH, V_PERFECT_HASH_OVERFLOW, V_IOHASH = 4, 5, 8
# Тип чанка — последний байт FIoChunkId
CHUNK_TYPES_UE5 = {1: "ExportBundleData", 2: "BulkData", 3: "OptionalBulkData", 4: "MemoryMappedBulkData",
                   5: "ScriptObjects", 6: "ContainerHeader", 7: "ExternalFile", 8: "ShaderCodeLibrary",
                   9: "ShaderCode", 10: "PackageStoreEntry", 11: "DerivedData", 12: "EditorDerivedData",
                   13: "PackageResource"}


class IoStore:
    """Один контейнер (.utoc и его .ucas-разделы). files — путь → индекс чанка."""

    def __init__(self, utoc: Path, key: bytes | None = None, extra_dll_dirs: list[Path] = ()):
        self.utoc = Path(utoc)
        self.key = key
        self.extra_dll_dirs = list(extra_dll_dirs)
        self.files: dict[str, int] = {}
        self.mount_point = ""
        self._handles: dict[int, object] = {}
        self._parse()

    # --- TOC ---

    def _parse(self):
        data = self.utoc.read_bytes()
        if data[:16] != TOC_MAGIC:
            raise ValueError("Не похоже на .utoc Unreal Engine")
        self.version = data[16]
        (hdr_size, self.entry_count, n_blocks, block_entry_size, n_methods, method_len, self.block_size,
         dir_index_size, self.partition_count) = struct.unpack_from("<9I", data, 20)
        self.container_id = struct.unpack_from("<Q", data, 56)[0]
        self.flags = data[80]
        n_seeds = struct.unpack_from("<I", data, 84)[0]
        self.partition_size = struct.unpack_from("<Q", data, 88)[0] or (1 << 62)
        n_overflow = struct.unpack_from("<I", data, 96)[0]
        pos = hdr_size
        self.chunk_ids = [data[pos + i * 12: pos + i * 12 + 12] for i in range(self.entry_count)]
        pos += 12 * self.entry_count
        self.offsets: list[tuple[int, int]] = []
        for i in range(self.entry_count):
            b = data[pos + i * 10: pos + i * 10 + 10]
            self.offsets.append((int.from_bytes(b[:5], "big"), int.from_bytes(b[5:], "big")))
        pos += 10 * self.entry_count
        if self.version >= V_PERFECT_HASH:
            pos += 4 * n_seeds
        if self.version >= V_PERFECT_HASH_OVERFLOW:
            pos += 4 * n_overflow
        self.blocks: list[tuple[int, int, int, int]] = []  # (offset, compressed, uncompressed, method)
        for i in range(n_blocks):
            b = data[pos + i * block_entry_size: pos + (i + 1) * block_entry_size]
            self.blocks.append((int.from_bytes(b[:5], "little"), int.from_bytes(b[5:8], "little"),
                                int.from_bytes(b[8:11], "little"), b[11]))
        pos += n_blocks * block_entry_size
        self.methods = []
        for i in range(n_methods):
            self.methods.append(data[pos:pos + method_len].split(b"\0")[0].decode("ascii", "replace"))
            pos += method_len
        if self.flags & FLAG_SIGNED:
            hash_size = struct.unpack_from("<i", data, pos)[0]
            pos += 4 + 2 * hash_size + 20 * n_blocks
        if self.flags & FLAG_INDEXED and dir_index_size > 0:
            index = data[pos:pos + dir_index_size]
            if self.flags & FLAG_ENCRYPTED:
                if not self.key:
                    raise NeedsKey("Контейнер .utoc зашифрован — нужен AES-ключ игры")
                index = aes_decrypt(index, self.key)
            self._parse_directory(index)
        pos += dir_index_size
        # далее — метаданные чанков (хеши), они не нужны

    def _parse_directory(self, index: bytes):
        r = Reader(index)
        self.mount_point = r.fstring()
        n_dirs = r.u32()
        dirs = [struct.unpack_from("<4I", index, r.pos + i * 16) for i in range(n_dirs)]
        r.pos += 16 * n_dirs
        n_files = r.u32()
        files = [struct.unpack_from("<3I", index, r.pos + i * 12) for i in range(n_files)]
        r.pos += 12 * n_files
        n_strings = r.u32()
        strings = [r.fstring() for _ in range(n_strings)]
        if not dirs:
            return
        NONE = 0xFFFFFFFF
        mount = self.mount_point.replace("\\", "/")
        while mount.startswith("../"):
            mount = mount[3:]
        mount = mount.strip("/")
        stack = [(0, mount)]
        while stack:
            di, path = stack.pop()
            name_i, first_child, next_sibling, first_file = dirs[di]
            if di != 0 and name_i != NONE:
                path = f"{path}/{strings[name_i]}" if path else strings[name_i]
            fi = first_file
            while fi != NONE:
                fname_i, next_file, user_data = files[fi]
                self.files[f"{path}/{strings[fname_i]}" if path else strings[fname_i]] = user_data
                fi = next_file
            ci = first_child
            while ci != NONE:
                stack.append((ci, path))
                ci = dirs[ci][2]

    # --- данные ---

    def chunk_type(self, idx: int) -> int:
        return self.chunk_ids[idx][11]

    def _partition(self, idx: int):
        fh = self._handles.get(idx)
        if fh is None:
            name = self.utoc.stem + (f"_s{idx}" if idx else "") + ".ucas"
            fh = open(self.utoc.with_name(name), "rb")
            self._handles[idx] = fh
        return fh

    def close(self):
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()

    def read_index(self, idx: int) -> bytes:
        offset, length = self.offsets[idx]
        if length == 0:
            return b""
        first, last = offset // self.block_size, (offset + length - 1) // self.block_size
        out = bytearray()
        for bi in range(first, last + 1):
            b_off, csize, usize, method = self.blocks[bi]
            fh = self._partition(b_off // self.partition_size)
            fh.seek(b_off % self.partition_size)
            n = (csize + 15) & ~15 if self.flags & FLAG_ENCRYPTED else csize
            raw = fh.read(n)
            if self.flags & FLAG_ENCRYPTED:
                if not self.key:
                    raise NeedsKey("Данные .ucas зашифрованы — нужен AES-ключ игры")
                raw = aes_decrypt(raw, self.key)[:csize]
            if method == 0:
                out += raw[:usize]
            else:
                name = self.methods[method - 1] if method - 1 < len(self.methods) else f"#{method}"
                out += decompress(name, raw, usize, self.extra_dll_dirs)
        start = offset - first * self.block_size
        return bytes(out[start:start + length])

    def read(self, name: str) -> bytes:
        return self.read_index(self.files[name])

    def __contains__(self, name: str) -> bool:
        return name in self.files

    def __len__(self):
        return len(self.files)
