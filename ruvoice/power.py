"""Нагрузка на систему: «низкая / средняя / полная».

Процесс небыстрый, и пользователю может быть нужен компьютер для другой работы. Уровень влияет на:
  • приоритет процессов программы и движков (Windows отдаёт остальным программам процессор первыми);
  • число потоков CPU (torch, llama.cpp, подготовка данных);
  • размер пачек для видеокарты и паузы между пачками (видеокарта работает не всё время, а «импульсами»).
Обучение голоса (RVC) всё равно загружает видеокарту на время эпохи — его смягчает только размер пачки
и приоритет. Уровень читается заново перед каждой пачкой, поэтому ползунок действует сразу."""
import json
import os
import threading
import time

from .paths import DATA_DIR

LEVELS = {
    "low": {"name": "Низкая", "desc": "компьютер остаётся отзывчивым, озвучка в 2–3 раза дольше",
            "threads": 0.25, "batch": 0.34, "pause": 1.5, "priority": "idle"},
    "medium": {"name": "Средняя", "desc": "можно работать в браузере и офисных программах",
               "threads": 0.5, "batch": 0.5, "pause": 0.5, "priority": "below"},
    "full": {"name": "Полная", "desc": "максимальная скорость, компьютер лучше не трогать",
             "threads": 1.0, "batch": 1.0, "pause": 0.0, "priority": "normal"},
}
PREFS = DATA_DIR / "prefs.json"
_lock = threading.Lock()
_cache = {"t": 0.0, "level": "full"}


def level() -> str:
    now = time.time()
    if now - _cache["t"] > 2:
        try:
            _cache["level"] = json.loads(PREFS.read_text("utf-8")).get("power", "full")
        except Exception:
            _cache["level"] = "full"
        _cache["t"] = now
    return _cache["level"] if _cache["level"] in LEVELS else "full"


def set_level(value: str):
    if value not in LEVELS:
        raise ValueError(value)
    with _lock:
        PREFS.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(PREFS.read_text("utf-8"))
        except Exception:
            data = {}
        data["power"] = value
        PREFS.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    _cache.update(t=0.0)
    apply_priority()


def threads() -> int:
    cores = os.cpu_count() or 4
    return max(1, int(cores * LEVELS[level()]["threads"]))


def batch(n: int) -> int:
    return max(1, round(n * LEVELS[level()]["batch"]))


def pause(work_seconds: float):
    """Пауза после пачки работы: даёт видеокарте и процессору «отдохнуть» для других программ."""
    p = LEVELS[level()]["pause"]
    if p > 0 and work_seconds > 0:
        time.sleep(min(work_seconds * p, 20))


def apply_priority(pid: int | None = None):
    """Приоритет процесса (себя или движка) по текущему уровню."""
    try:
        import psutil
        proc = psutil.Process(pid) if pid else psutil.Process()
        pri = LEVELS[level()]["priority"]
        proc.nice({"idle": psutil.IDLE_PRIORITY_CLASS, "below": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                   "normal": psutil.NORMAL_PRIORITY_CLASS}[pri])
        for child in proc.children(recursive=True):
            try:
                child.nice(proc.nice())
            except Exception:
                pass
    except Exception:
        pass


def env_for_children(env: dict) -> dict:
    """Переменные окружения для движков и их дочерних процессов."""
    n = str(threads())
    env.update(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, NUMEXPR_NUM_THREADS=n, RUVOICE_POWER=level())
    return env


def describe() -> dict:
    return {"level": level(), "levels": {k: {"name": v["name"], "desc": v["desc"]} for k, v in LEVELS.items()}}
