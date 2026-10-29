"""Чтение архивов VPK движка Source (версии 1 и 2)."""
import struct
from pathlib import Path


class VPK:
    def __init__(self, dir_path: Path):
        self.dir_path = Path(dir_path)
        self.prefix = str(self.dir_path)[: -len("_dir.vpk")]
        self.entries: dict[str, tuple] = {}
        self._read()

    def _read(self):
        raw = self.dir_path.read_bytes()
        sig, version, tree_size = struct.unpack_from("<III", raw, 0)
        if sig != 0x55AA1234:
            raise ValueError(f"Не VPK: {self.dir_path}")
        header = 12 if version == 1 else 28
        self.data_offset = header + tree_size
        pos = header

        def cstr():
            nonlocal pos
            end = raw.index(b"\0", pos)
            s = raw[pos:end].decode("utf-8", "replace")
            pos = end + 1
            return s

        while True:
            ext = cstr()
            if not ext:
                break
            while True:
                folder = cstr()
                if not folder:
                    break
                while True:
                    name = cstr()
                    if not name:
                        break
                    crc, preload_len, archive, offset, length, _term = struct.unpack_from("<IHHIIH", raw, pos)
                    pos += 18
                    preload = raw[pos:pos + preload_len]
                    pos += preload_len
                    path = (f"{folder}/" if folder.strip() else "") + f"{name}.{ext}"
                    self.entries[path.lower()] = (archive, offset, length, preload)

    def __contains__(self, path: str) -> bool:
        return path.lower().replace("\\", "/") in self.entries

    def files(self, prefix: str = "", ext: str | None = None):
        prefix = prefix.lower()
        for p in self.entries:
            if p.startswith(prefix) and (ext is None or p.endswith(ext)):
                yield p

    def read(self, path: str) -> bytes:
        archive, offset, length, preload = self.entries[path.lower().replace("\\", "/")]
        if length == 0:
            return preload
        if archive == 0x7FFF:
            with open(self.dir_path, "rb") as f:
                f.seek(self.data_offset + offset)
                return preload + f.read(length)
        with open(f"{self.prefix}_{archive:03d}.vpk", "rb") as f:
            f.seek(offset)
            return preload + f.read(length)
