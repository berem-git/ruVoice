"""Запуск движка отдельным процессом и обмен JSON-строками через stdin/stdout."""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from ..paths import DATA_DIR, PROJECT_DIR

_procs: dict[str, "EngineProcess"] = {}
_lock = threading.Lock()


class EngineError(RuntimeError):
    pass


class EngineProcess:
    def __init__(self, engine: str):
        self.engine = engine
        python = Path(sys.executable).with_name("python.exe")
        log_dir = DATA_DIR / "engines"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir / f"{engine}.log", "a", encoding="utf-8", buffering=1)
        from .. import power
        env = power.env_for_children({**os.environ, "PYTHONPATH": str(PROJECT_DIR), "PYTHONUTF8": "1",
                                      "PYTHONNOUSERSITE": "1", "HF_HUB_DISABLE_XET": "1"})
        self.proc = subprocess.Popen([str(python), "-u", "-m", "ruvoice.engines.worker", engine],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, encoding="utf-8", cwd=str(PROJECT_DIR), env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        power.apply_priority(self.proc.pid)
        self.io_lock = threading.Lock()
        first = self.request({"op": "ping"}, timeout=120)
        if not first.get("ok"):
            raise EngineError(first.get("error", "движок не ответил"))

    def request(self, msg: dict, timeout: float = 900, on_progress=None) -> dict:
        """Один запрос — один ответ; строки {"progress": …} по пути передаются в on_progress."""
        with self.io_lock:
            if self.proc.poll() is not None:
                raise EngineError(f"Процесс движка {self.engine} завершился (см. data\\engines\\{self.engine}.log)")
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            while True:
                line = _readline(self.proc.stdout, timeout)
                if line is None:
                    self.close()
                    raise EngineError(f"Движок {self.engine} не ответил за {timeout:.0f} с")
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    raise EngineError(f"Движок {self.engine}: неверный ответ {line[:200]!r}")
                if "progress" in reply and "ok" not in reply:
                    if on_progress:
                        on_progress(reply["progress"])
                    continue
                if not reply.get("ok"):
                    raise EngineError(reply.get("error", "ошибка движка"))
                return reply

    def close(self, force: bool = False):
        """Остановить движок. force — убить сразу вместе с дочерними процессами (обучение RVC и т.п.)."""
        try:
            if self.proc.poll() is None:
                if force:
                    _kill_tree(self.proc.pid)
                else:
                    try:
                        self.proc.stdin.write('{"op": "exit"}\n')
                        self.proc.stdin.flush()
                        self.proc.wait(10)
                    except Exception:
                        _kill_tree(self.proc.pid)
        finally:
            try:
                self.log.close()
            except Exception:
                pass


def _kill_tree(pid: int):
    try:
        import psutil
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        parent.kill()
    except Exception:
        pass


def _readline(stream, timeout: float):
    result = []

    def read():
        result.append(stream.readline())
    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or not result or not result[0]:
        return None
    return result[0]


def get(engine: str) -> EngineProcess:
    with _lock:
        p = _procs.get(engine)
        if p is None or p.proc.poll() is not None:
            p = EngineProcess(engine)
            _procs[engine] = p
        return p


def stop(engine: str | None = None, force: bool = False):
    """Выгрузить движок (освободить видеопамять); без аргумента — все. force — прервать немедленно."""
    with _lock:
        for k in list(_procs):
            if engine is None or k == engine:
                _procs.pop(k).close(force=force)
