"""Игры на RE Engine (Capcom) со звуком Wwise: Resident Evil 2/3/7/Village, Devil May Cry 5, Monster Hunter Rise…

Откуда что берётся (без знания имён файлов — в архивах только хеши путей):
  • озвучка — английские файлы Wwise (пакеты .pck и банки .bnk с языком english(us)) — wwisevoice.py;
  • текст — файлы субтитров .msg (GMSG): в каждом все языки, английский и русский рядом. Связи «звук ↔ строка»
    в файлах нет, поэтому реплики сопоставляются с субтитрами по распознанной речи (общий шаг подготовки,
    subtitles.py) — пул строк адаптер пишет при извлечении.
Установка — свой патч-архив re_chunk_000.pak.patch_NNN.pak (первый свободный номер; пустую заглушку менеджера
модов он заменяет): в нём пакеты и банки с русскими репликами под теми же хешами путей.
Архивы игры не меняются; «Удалить озвучку» убирает патч (пустую заглушку возвращает)."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from .. import subtitles
from . import rechunk, remsg, wwisevoice


def _root(game: dict) -> Path:
    return Path(game["path"])


class REEngineAdapter:
    id = "reengine"
    name = "RE Engine (архивы re_chunk, звук Wwise)"

    @staticmethod
    def detect(game: dict) -> bool:
        root = _root(game)
        return root.is_dir() and rechunk.detect(root) is None

    def describe(self, game: dict) -> dict:
        why = rechunk.detect(_root(game))
        notes = ["Озвучка — английские файлы Wwise из архивов re_chunk_000.pak и патчей; субтитры (все языки сразу) — "
                 "из файлов .msg. Реплики сопоставляются с русскими субтитрами по распознанной английской речи.",
                 "Установка — отдельный патч-архив re_chunk_000.pak.patch_NNN.pak рядом с игрой; архивы игры не "
                 "меняются, «Удалить озвучку» убирает патч. В игре оставьте английскую озвучку и русские субтитры.",
                 "Если обновление игры добавит патч с бо́льшим номером, установите озвучку заново."]
        if why:
            notes.insert(0, f"Не поддерживается: {why}")
        return {"adapter": self.id, "name": self.name, "supports_official_ru": False, "notes": notes}

    def _own_patch(self, project) -> str | None:
        return (project.get("re_installed") or {}).get("patch")

    def _archive(self, project, game) -> rechunk.Archive:
        own = self._own_patch(project)  # свой патч не читаем: нужны оригиналы
        return rechunk.Archive(_root(game), exclude={own.lower()} if own else None)

    # --- извлечение ---

    def extract(self, project, game: dict, progress):
        why = rechunk.detect(_root(game))
        if why:
            raise RuntimeError(f"Эта игра на RE Engine пока не поддерживается: {why}")
        t0 = time.time()
        progress("Чтение оглавления архивов…")
        arc = self._archive(project, game)
        msg_rows: list[dict] = []

        def sound():
            """Файлы Wwise по одному (в памяти остаётся только английская озвучка); субтитры — по пути."""
            for e, data in arc.find(lambda h: wwisevoice.is_sound(h) or remsg.is_msg(h)):
                if remsg.is_msg(data[:8]):
                    try:
                        for r in remsg.parse(data):
                            t = r["texts"]
                            if t.get(remsg.EN) and t.get(remsg.RU):
                                msg_rows.append({"key": r["key"], "en": t[remsg.EN], "ru": t[remsg.RU],
                                                 "group": f"{e.hash_lo:08x}"})
                    except Exception:
                        continue
                else:
                    yield f"{e.hash_lo:08x}", data

        try:
            streams, embedded, has_ru = wwisevoice.scan(sound(), progress)
        finally:
            arc.close()
        if not streams and not embedded:
            raise RuntimeError("В архивах не найдено английской озвучки Wwise")
        project.reset()
        lines, chars = wwisevoice.to_lines(project, streams, embedded, progress)
        project.add_lines(lines)
        project.set_characters([{"id": c["id"], "name": c["name"], "ref_line": c["best"]} for c in chars.values()])
        kept = subtitles.save_pool(project, msg_rows)
        project.set("re_engine", {"voices": len(lines), "streams": sum(1 for l in lines if l["src"]["kind"] == "stream"),
                                  "subtitles": kept, "has_ru_voice": has_ru, "seconds": round(time.time() - t0)})
        return len(lines)

    # --- установка ---

    def install(self, project, game: dict, progress) -> int:
        root = _root(game)
        approved = [l for l in project.lines(status="approved") if l["src"].get("sid") is not None]
        if not approved:
            return 0
        work = project.dir / "stage_re"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        new = wwisevoice.encode(project, approved, work, progress)
        kinds = {int(l["src"]["sid"]): l["src"]["kind"] for l in approved}
        progress("Сборка пакетов и банков с русскими репликами…")
        arc = self._archive(project, game)
        try:
            files = list(wwisevoice.patch(arc.find(wwisevoice.is_sound), new, kinds))
        finally:
            arc.close()
        if not files:
            raise RuntimeError("Не нашлось файлов игры с этими репликами — извлеките реплики заново")
        own = self._own_patch(project)
        slot = root / own if own else rechunk.patch_slot(root)
        st = project.get("re_installed") or {}
        placeholder = st.get("placeholder") if own else (slot.exists() and slot.stat().st_size == 0)
        progress(f"Запись патча {slot.name} ({len(files)} файлов)…")
        rechunk.write_patch(slot, files)
        shutil.rmtree(work, ignore_errors=True)
        project.set("re_installed", {"patch": slot.name, "placeholder": bool(placeholder), "files": len(files),
                                     "lines": len(approved), "when": time.time()})
        project.set("installed_count", len(approved))
        return len(approved)

    def uninstall(self, project, game: dict, progress):
        st = project.get("re_installed") or {}
        if st.get("patch"):
            path = _root(game) / st["patch"]
            progress(f"Удаление патча {path.name}…")
            if st.get("placeholder"):
                path.write_bytes(b"")  # пустая заглушка была до нас — возвращаем как было
            else:
                path.unlink(missing_ok=True)
        project.set("re_installed", None)
        project.set("installed_count", 0)

    def is_installed(self, project, game: dict) -> bool:
        st = project.get("re_installed") or {}
        path = _root(game) / st["patch"] if st.get("patch") else None
        return bool(path and path.exists() and path.stat().st_size > 16)
