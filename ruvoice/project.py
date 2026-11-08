"""Проект озвучки одной игры: data/projects/<id>/project.db + orig/ (оригинал) + gen/ (русский) + backup/."""
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from .paths import DATA_DIR

PROJECTS_DIR = DATA_DIR / "projects"

# Статусы реплики: new → translated → generated → approved / rejected; skipped — не озвучивать
STATUSES = ("new", "translated", "generated", "approved", "rejected", "skipped", "error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY, name TEXT, color TEXT, ref_line TEXT, include INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS lines (
    id TEXT PRIMARY KEY, char TEXT, src TEXT, orig TEXT, duration REAL,
    en TEXT, ru TEXT, ru_official TEXT, ru_source TEXT, status TEXT DEFAULT 'new',
    gen TEXT, gen_duration REAL, info TEXT, error TEXT, updated REAL);
CREATE INDEX IF NOT EXISTS lines_char ON lines(char);
CREATE INDEX IF NOT EXISTS lines_status ON lines(status);
"""


def project_id(game_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", game_id)


class Project:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.id = project_id(game_id)
        self._remember_game_id = True  # после открытия базы записать game_id в meta (для списка «мусора»)
        self.dir = PROJECTS_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.orig_dir, self.gen_dir, self.backup_dir = self.dir / "orig", self.dir / "gen", self.dir / "backup"
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []  # все подключения из всех потоков — чтобы закрыть при удалении
        self.lock = threading.RLock()
        with self.db() as c:
            c.executescript(SCHEMA)

    @staticmethod
    def exists(game_id: str) -> bool:
        return (PROJECTS_DIR / project_id(game_id) / "project.db").exists()

    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.dir / "project.db", timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
            self._conns.append(conn)
        return conn

    def close(self):
        """Закрыть базу во всех потоках (иначе Windows не даст удалить папку проекта)."""
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        self._conns.clear()
        self._local = threading.local()

    # --- meta ---

    def get(self, key: str, default=None):
        if getattr(self, "_remember_game_id", False):
            self._remember_game_id = False
            try:
                with self.db() as c:
                    c.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", ("game_id", json.dumps(self.game_id)))
            except Exception:
                pass
        row = self.db().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value):
        with self.db() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))

    # --- реплики ---

    def add_lines(self, lines: list[dict]):
        now = time.time()
        with self.db() as c:
            c.executemany(
                "INSERT OR REPLACE INTO lines (id, char, src, orig, duration, en, ru_official, status, updated) "
                "VALUES (:id, :char, :src, :orig, :duration, :en, :ru_official, 'new', :updated)",
                [{"ru_official": None, "duration": None, **l, "src": json.dumps(l.get("src", {}), ensure_ascii=False),
                  "updated": now} for l in lines])

    def update_line(self, line_id: str, **fields):
        fields["updated"] = time.time()
        if "info" in fields and not isinstance(fields["info"], (str, type(None))):
            fields["info"] = json.dumps(fields["info"], ensure_ascii=False)
        cols = ", ".join(f"{k}=:{k}" for k in fields)
        with self.db() as c:
            c.execute(f"UPDATE lines SET {cols} WHERE id=:id", {**fields, "id": line_id})

    def line(self, line_id: str) -> dict | None:
        row = self.db().execute("SELECT * FROM lines WHERE id=?", (line_id,)).fetchone()
        return _line(row) if row else None

    def lines(self, status: str | None = None, char: str | None = None, query: str | None = None,
              limit: int | None = None, offset: int = 0, ids: list[str] | None = None,
              rework: bool = False) -> list[dict]:
        sql, args = "SELECT * FROM lines WHERE 1=1", []
        if status:
            statuses = status.split(",")
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            args += statuses
        if rework:
            sql += " AND gen IS NOT NULL AND json_extract(info, '$.judge.verdict') = 'rework'"
        if char:
            sql += " AND char=?"
            args.append(char)
        if query:
            sql += " AND (en LIKE ? OR ru LIKE ? OR id LIKE ?)"
            args += [f"%{query}%"] * 3
        if ids is not None:
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args += ids
        sql += " ORDER BY char, id"
        if limit:
            sql += " LIMIT ? OFFSET ?"
            args += [limit, offset]
        return [_line(r) for r in self.db().execute(sql, args)]

    def count(self, **filters) -> int:
        return len(self.lines(**filters))

    def stats(self) -> dict:
        rows = self.db().execute("SELECT status, COUNT(*), SUM(duration) FROM lines GROUP BY status").fetchall()
        by = {r[0]: r[1] for r in rows}
        rework = self.db().execute(
            "SELECT COUNT(*) FROM lines WHERE gen IS NOT NULL AND json_extract(info, '$.judge.verdict') = 'rework'").fetchone()[0]
        return {"total": sum(by.values()), "by_status": by, "rework": rework,
                "seconds": round(sum((r[2] or 0) for r in rows)),
                "with_official": self.db().execute(
                    "SELECT COUNT(*) FROM lines WHERE ru_official IS NOT NULL AND ru_official != ''").fetchone()[0]}

    # --- персонажи ---

    def set_characters(self, chars: list[dict]):
        with self.db() as c:
            c.executemany("INSERT OR REPLACE INTO characters (id, name, color, ref_line, include) "
                          "VALUES (:id, :name, :color, :ref_line, :include)",
                          [{"color": None, "ref_line": None, "include": 1, **ch} for ch in chars])

    def characters(self) -> list[dict]:
        rows = self.db().execute(
            "SELECT c.*, COUNT(l.id) AS lines, SUM(l.duration) AS seconds, "
            "SUM(CASE WHEN l.status!='skipped' THEN l.duration ELSE 0 END) AS speech_seconds, "  # без криков и стонов
            "SUM(l.status!='skipped') AS speech_lines, "
            "SUM(l.status='approved') AS approved, SUM(l.status IN ('generated','approved','rejected')) AS generated "
            "FROM characters c LEFT JOIN lines l ON l.char=c.id GROUP BY c.id ORDER BY lines DESC").fetchall()
        return [dict(r) for r in rows]

    def update_character(self, char_id: str, **fields):
        cols = ", ".join(f"{k}=:{k}" for k in fields)
        with self.db() as c:
            c.execute(f"UPDATE characters SET {cols} WHERE id=:id", {**fields, "id": char_id})

    def merge_characters(self, into: str, others: list[str]):
        with self.db() as c:
            for o in others:
                if o == into:
                    continue
                c.execute("UPDATE lines SET char=? WHERE char=?", (into, o))
                c.execute("DELETE FROM characters WHERE id=?", (o,))
        # A merge changes the meaning of a speaker; references must be chosen again.
        for key in ("profiles", "ru_voice", "voice_anchors", "voice_f0"):
            self.set(key, None)

    def reset(self):
        for key in ("voice_anchors", "ru_voice", "profiles", "voice_f0"):
            self.set(key, None)
        with self.db() as c:
            c.execute("DELETE FROM lines")
            c.execute("DELETE FROM characters")
            # раскладка по голосам и её следы относились к прежним репликам
            c.execute("DELETE FROM meta WHERE key IN ('speakers_applied', 'speakers_reliable', 'speaker_proposal', "
                      "'subtitles_scanned')")
        (self.dir / "speakers.npz").unlink(missing_ok=True)
        (self.dir / "subtitles.jsonl").unlink(missing_ok=True)  # пул субтитров относился к прежним репликам


def _line(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["src"] = json.loads(d["src"]) if d.get("src") else {}
    d["info"] = json.loads(d["info"]) if d.get("info") else {}
    return d
