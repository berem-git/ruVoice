"""Значок в области уведомлений (трее): пока идёт работа (озвучка, обучение, установка), крестик окна
не закрывает программу, а прячет её сюда — работа продолжается. Двойной щелчок по значку — открыть окно,
меню — «Открыть» / «Выйти». Когда работа закончится, всплывёт уведомление.

Окно — pywebview на WinForms, поэтому значок — обычный NotifyIcon из .NET (через pythonnet), все обращения
к нему — в потоке интерфейса (BeginInvoke)."""
import threading
import time

_current = None  # значок/окно этой копии программы — для show_main()


def show_main() -> bool:
    """Показать окно и вывести на передний план (вторая копия программы просит об этом первую)."""
    t = _current
    if t is None:
        return False
    t.restore()
    return True


class Tray:
    def __init__(self, window, dub, icon_path):
        global _current
        _current = self
        self.window = window
        self.dub = dub
        self.icon_path = str(icon_path)
        self.ni = None
        self.hidden = False
        self.quitting = False
        self._watch = None

    # --- служебное ---

    def _ui(self, fn):
        """Выполнить fn в потоке интерфейса окна."""
        try:
            from System import Action
            self.window.native.BeginInvoke(Action(fn))
        except Exception as e:
            print(f"трей: {e}")

    def _busy(self) -> bool:
        return self.dub.current is not None or not self.dub.q.empty()

    def _status(self) -> str:
        gid = self.dub.current
        if not gid:
            return "Готово"
        st = self.dub.state.get(gid) or {}
        return st.get("message") or "Работаю…"

    # --- создание значка ---

    def install(self):
        def make():
            from System.Drawing import Icon
            from System.Windows.Forms import ContextMenuStrip, NotifyIcon, ToolStripMenuItem
            ni = NotifyIcon()
            ni.Icon = Icon(self.icon_path)
            ni.Text = "RuVoice"
            ni.Visible = False
            menu = ContextMenuStrip()
            open_item = ToolStripMenuItem("Открыть RuVoice")
            open_item.Click += lambda s, e: self.restore()
            exit_item = ToolStripMenuItem("Выйти (текущая работа остановится)")
            exit_item.Click += lambda s, e: self.quit()
            menu.Items.Add(open_item)
            menu.Items.Add(exit_item)
            ni.ContextMenuStrip = menu
            ni.DoubleClick += lambda s, e: self.restore()
            ni.BalloonTipClicked += lambda s, e: self.restore()
            self.ni = ni
        self._ui(make)

    # --- события окна ---

    def on_closing(self):
        """Крестик: если идёт работа — спрятать окно в трей и не закрывать. Возврат False отменяет закрытие."""
        if self.quitting or not self._busy() or self.ni is None:
            self._dispose()
            return True
        threading.Thread(target=self._hide, daemon=True).start()
        return False

    def _hide(self):
        try:
            self.window.hide()
        except Exception as e:
            print(f"трей: не спрятать окно: {e}")
            return
        self.hidden = True

        def show_icon():
            self.ni.Visible = True
            self.ni.Text = self._tip()
            self.ni.ShowBalloonTip(5000, "RuVoice работает в фоне",
                                   f"{self._status()[:120]}\nОткрыть — двойной щелчок по значку.", self._tip_icon("Info"))
        self._ui(show_icon)
        if not self._watch or not self._watch.is_alive():
            self._watch = threading.Thread(target=self._watch_loop, daemon=True)
            self._watch.start()

    def _tip(self) -> str:
        return ("RuVoice: " + self._status())[:63]  # подсказка значка в Windows — не длиннее 63 символов

    @staticmethod
    def _tip_icon(name):
        from System.Windows.Forms import ToolTipIcon
        return getattr(ToolTipIcon, name)

    def _watch_loop(self):
        """Пока окно спрятано: обновлять подсказку значка, а по окончании работы — уведомить."""
        was_busy = True
        while self.hidden and not self.quitting:
            busy = self._busy()
            if was_busy and not busy:
                msg = ""
                for g, st in self.dub.state.items():
                    if st.get("error"):
                        msg = f"Ошибка: {st['error'][:100]}"
                        break
                    msg = msg or (st.get("message") or "")
                text = msg or "Работа завершена"
                self._ui(lambda: (setattr(self.ni, "Text", "RuVoice: готово"),
                                  self.ni.ShowBalloonTip(8000, "RuVoice: работа завершена", text[:200],
                                                         self._tip_icon("Info"))))
            elif busy:
                tip = self._tip()
                self._ui(lambda: setattr(self.ni, "Text", tip))
            was_busy = busy
            time.sleep(3)

    def restore(self):
        self.hidden = False
        ni = self.ni
        if ni is not None:
            self._ui(lambda: setattr(ni, "Visible", False))
        threading.Thread(target=self._show, daemon=True).start()

    def _show(self):
        try:
            self.window.show()
            self.window.restore()
        except Exception as e:
            print(f"трей: не показать окно: {e}")

        def front():  # поверх остальных окон: свёрнутое — развернуть, спрятанное за другими — поднять
            form = self.window.native
            form.TopMost = True
            form.TopMost = False
            form.Activate()
            form.BringToFront()
        self._ui(front)

    def quit(self):
        """«Выйти» из меню значка: пользователь сам останавливает работу — её не продолжать при следующем запуске."""
        self.quitting = True
        self.hidden = False
        try:
            if self.dub.current:
                self.dub.cancel(self.dub.current)
            self.dub.clear_job()
        except Exception as e:
            print(f"трей: остановка задачи: {e}")
        self._dispose()
        threading.Thread(target=self.window.destroy, daemon=True).start()

    def _dispose(self):
        ni = self.ni
        if ni is None:
            return
        self.ni = None

        def drop():
            ni.Visible = False
            ni.Dispose()
        self._ui(drop)
