"""Запуск: окно приложения (pywebview) или `--browser` для открытия в обычном браузере."""
import ctypes
import os
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from .api import create_app
from .jobs import Jobs
from .library import Library
from .paths import DATA_DIR, PROJECT_DIR

ICON = PROJECT_DIR / "launcher" / "ruvoice.ico"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _redirect_output_if_no_console():
    """Под pythonw.exe нет консоли (sys.stdout = None), и uvicorn падает при старте — пишем в журнал."""
    if sys.stdout is None or sys.stderr is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log = open(DATA_DIR / "ruvoice.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


INSTANCE_FILE = DATA_DIR / "instance.json"
_mutex = None  # держим до выхода: пока жив процесс, вторая копия видит, что RuVoice уже запущен


def _first_instance() -> bool:
    """Одна копия программы. Первая создаёт именованный мьютекс и записывает свой порт (INSTANCE_FILE);
    вторая находит мьютекс, просит первую показать окно (POST /api/system/show) и выходит. Мьютекс
    освобождается системой, даже если первая копия упала, — ложных «уже запущен» не бывает."""
    global _mutex
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    _mutex = k32.CreateMutexW(None, False, "Local\\RuVoice.SingleInstance")
    if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
        return True
    import json
    import urllib.request
    try:
        info = json.loads(INSTANCE_FILE.read_text("utf-8"))
        # право вывести окно на передний план — у только что запущенной копии; отдаём его первой
        ctypes.windll.user32.AllowSetForegroundWindow(int(info.get("pid") or -1))
        req = urllib.request.Request(f"http://127.0.0.1:{info['port']}/api/system/show", data=b"{}",
                                     headers={"Content-Type": "application/json"}, method="POST")
        shown = json.loads(urllib.request.urlopen(req, timeout=5).read() or b"{}").get("ok")
        if not shown and info.get("url"):  # первая копия без окна (--browser/--server) — открыть её в браузере
            webbrowser.open(info["url"])
        print(f"RuVoice уже запущен (pid {info.get('pid')}) — показал его окно", flush=True)
    except Exception as e:
        print(f"RuVoice уже запущен, но окно показать не удалось: {e}", flush=True)
        ctypes.windll.user32.MessageBoxW(None, "RuVoice уже запущен — его значок в области уведомлений, у часов.",
                                         "RuVoice", 0x40)
    return False


def _write_instance(port: int, url: str):
    import json
    try:
        INSTANCE_FILE.write_text(json.dumps({"pid": os.getpid(), "port": port, "url": url}), "utf-8")
    except OSError as e:
        print(f"instance.json: {e}")


def release_instance():
    """Перед перезапуском из программы: отпустить мьютекс единственной копии и запись о порте,
    чтобы новая копия не приняла старую, ещё закрывающуюся, за работающую."""
    global _mutex
    if _mutex:
        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            k32.CloseHandle(_mutex)
        except Exception as e:
            print(f"мьютекс: {e}")
        _mutex = None
    _drop_instance()


def _drop_instance():
    import json
    try:
        if json.loads(INSTANCE_FILE.read_text("utf-8")).get("pid") == os.getpid():
            INSTANCE_FILE.unlink()
    except Exception:
        pass


def _set_window_icon(window):
    """pywebview на Windows не умеет задавать иконку — ставим её у формы WinForms напрямую."""
    try:
        from System import Action
        from System.Drawing import Icon
        form = window.native
        form.Invoke(Action(lambda: setattr(form, "Icon", Icon(str(ICON)))))
    except Exception as e:
        print(f"Не удалось установить иконку окна: {e}")


def main():
    _redirect_output_if_no_console()
    # падение внутри библиотек (CUDA, llama.cpp, WebView2) — стек всех потоков в журнал, а не тишина
    import faulthandler
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass
    # сообщения pywebview/WinForms (ошибки создания окна WebView2) — тоже в журнал, иначе окно может
    # молча не открыться, и программа закроется без единой строки о причине
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    print(f"--- запуск RuVoice {time.strftime('%Y-%m-%d %H:%M:%S')} (pid {os.getpid()})", flush=True)
    if not _first_instance():
        return
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(PROJECT_DIR / "models" / "huggingface"))
    from . import accel
    accel.activate()  # подключить сборку PyTorch под видеокарту до любых импортов torch
    from .dub import DubJobs

    lib = Library()
    jobs = Jobs(lib)
    jobs.scan()
    dub = DubJobs(lib)
    dub.resume_interrupted()  # прошлый запуск умер посреди озвучки — продолжить с того места

    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(lib, jobs, dub), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}/?t={int(time.time())}"  # новый адрес при каждом запуске — без старого кэша
    _write_instance(port, url)
    import atexit
    atexit.register(_drop_instance)

    if "--browser" in sys.argv or "--server" in sys.argv:
        print(f"RuVoice: {url}", flush=True)
        if "--browser" in sys.argv:
            webbrowser.open(url)
        thread.join()
        return

    if not _has_webview2():
        # Без WebView2 pywebview откатывается на старый Internet Explorer, где интерфейс не работает,
        # поэтому открываем программу в обычном браузере
        print("WebView2 не найден — открываю в браузере", flush=True)
        webbrowser.open(url)
        thread.join()
        return

    try:
        import webview
        # Своя группа на панели задач, иначе Windows покажет значок Python
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RuVoice.App")
        window = webview.create_window("RuVoice — русская озвучка игр", url, width=1320, height=860,
                                       min_size=(960, 620), background_color="#0f1115")
        window.events.shown += lambda: _set_window_icon(window)
        # Пока идёт озвучка, крестик прячет окно в трей, а не закрывает программу
        from .tray import Tray
        tray = Tray(window, dub, ICON)
        window.events.shown += tray.install
        window.events.closing += tray.on_closing
        shown_at = []
        window.events.shown += lambda: shown_at.append(time.time())
        started = time.time()
        webview.start(gui="edgechromium")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Окно не открылось ({e}) — открываю в браузере", flush=True)
        webbrowser.open(url)
        thread.join()
        return
    if not shown_at:
        # окно так и не показалось (WebView2 не создался) — не выходим молча, а открываем в браузере
        print(f"Окно не показалось за {time.time() - started:.0f} с — открываю в браузере: {url}", flush=True)
        webbrowser.open(url)
        thread.join()
        return
    print(f"окно закрыто — выход ({time.strftime('%H:%M:%S')})", flush=True)
    server.should_exit = True


def _has_webview2() -> bool:
    """WebView2 Runtime (встроен в Windows 11, в Windows 10 ставится с Edge) — ищем в реестре, как pywebview."""
    import winreg
    guid = r"\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    for hive, key in [(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node" + guid),
                      (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE" + guid),
                      (winreg.HKEY_CURRENT_USER, r"SOFTWARE" + guid)]:
        try:
            with winreg.OpenKey(hive, key) as k:
                if winreg.QueryValueEx(k, "pv")[0] not in ("", "0.0.0.0"):
                    return True
        except OSError:
            continue
    return False


if __name__ == "__main__":
    # Перезапуск импортирует ruvoice.main и должен отпустить мьютекс этого процесса.
    if __spec__ is not None:
        sys.modules[__spec__.name] = sys.modules[__name__]
    main()
