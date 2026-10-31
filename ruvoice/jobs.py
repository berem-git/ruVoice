"""Фоновая очередь: поиск игр, анализ папок, запросы в Steam. Один поток, чтобы не грузить диск."""
import itertools
import queue
import threading
import time

from . import detect, scanners
from .library import Library


class Jobs:
    def __init__(self, lib: Library):
        self.lib = lib
        # Приоритеты: анализ (его ждёт пользователь) раньше поиска, запросы в Steam — в конце
        self.q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self.message = ""
        self.busy = False
        self.errors: dict[str, str] = {}
        self.version = 0  # растёт при каждом изменении библиотеки — интерфейс по нему обновляется
        threading.Thread(target=self._run, daemon=True).start()

    def status(self) -> dict:
        return {"busy": self.busy or not self.q.empty(), "message": self.message,
                "pending": self.q.qsize(), "errors": self.errors, "version": self.version}

    def _put(self, priority: int, task: str, arg):
        self.q.put((priority, next(self._seq), task, arg))

    def scan(self, force: bool = False):
        self._put(1, "scan", force)

    def analyze(self, gid: str):
        self._put(0, "analyze", gid)

    def scan_dir(self, folder: str):
        self._put(1, "scan_dir", folder)

    def _run(self):
        while True:
            _, _, task, arg = self.q.get()
            self.busy = True
            try:
                getattr(self, f"_do_{task}")(arg)
            except Exception as e:
                self.errors[task] = str(e)
            finally:
                self.version += 1
                self.busy = False
                if self.q.empty():
                    self.message = ""

    def _do_scan(self, force: bool):
        self.message = "Поиск игр в лаунчерах…"
        found, self.errors = scanners.scan_launchers()
        self.lib.merge_found(found)
        for folder in list(self.lib.library_dirs):
            self._do_scan_dir(folder, queue_analysis=False)
        self.version += 1
        for gid, g in list(self.lib.games.items()):
            if force or not g.get("analysis") or (g.get('analysis') or {}).get('analysis_version', 0) < 2:
                self.analyze(gid)
            if g.get("appid") and (force or not self.lib.steam_store(g["appid"])):
                self._put(2, "steam", gid)

    def _do_scan_dir(self, folder: str, queue_analysis: bool = True):
        self.message = f"Поиск игр в папке {folder}…"
        for path in scanners.scan_library_dir(folder):
            if self.lib.find_by_path(path):
                continue
            g = self.lib.add_manual(path, source="folder")
            if queue_analysis:
                self.analyze(g["id"])

    def _do_analyze(self, gid: str):
        g = self.lib.get(gid)
        if not g:
            return
        self.message = f"Анализ: {g['name']}"
        self.lib.update(gid, analysis=detect.analyze(g["path"]))

    def _do_steam(self, gid: str):
        g = self.lib.get(gid)
        if not g or not g.get("appid"):
            return
        self.message = f"Языки в Steam: {g['name']}"
        info = scanners.fetch_steam_store(g["appid"])
        if info is not None:
            self.lib.set_steam_store(g["appid"], info)
        time.sleep(0.4)  # Steam ограничивает частоту запросов
