import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .jobs import Jobs
from .library import Library, game_view
from .paths import WEB_DIR


class AddGame(BaseModel):
    path: str
    name: str | None = None


class FolderPath(BaseModel):
    path: str


class Rename(BaseModel):
    name: str | None = None
    aes_key: str | None = None


def create_app(lib: Library, jobs: Jobs, dub=None) -> FastAPI:
    from . import __version__
    app = FastAPI(title="RuVoice", version=__version__)

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Окно программы (WebView2) кэширует интерфейс — без этого обновления не подхватываются
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    if dub is not None:
        from .api_dub import create_router
        app.include_router(create_router(lib, dub))

    def _game(gid: str) -> dict:
        g = lib.get(gid)
        if not g:
            raise HTTPException(404, "Игра не найдена")
        return g

    @app.get("/api/games")
    def games():
        return [game_view(lib, g) for g in list(lib.games.values())]

    @app.get("/api/status")
    def status():
        return jobs.status()

    @app.post("/api/system/show")
    def show_window():
        """Вторая копия программы просит первую показать окно (одна копия RuVoice на компьютер)."""
        from . import tray
        return {"ok": tray.show_main()}

    @app.post("/api/scan")
    def scan(force: bool = False):
        jobs.scan(force)
        return jobs.status()

    @app.post("/api/games")
    def add_game(body: AddGame):
        try:
            g = lib.add_manual(body.path, (body.name or "").strip() or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not g.get("analysis"):
            jobs.analyze(g["id"])
        return game_view(lib, g)

    @app.patch("/api/games/{gid}")
    def rename(gid: str, body: Rename):
        _game(gid)
        fields = {}
        if body.name is not None:
            name = body.name.strip()
            if not name:
                raise HTTPException(400, "Имя не может быть пустым")
            fields.update(name=name, custom_name=True)
        if body.aes_key is not None:
            # Ключ архивов Unreal Engine: проверяем формат, храним в библиотеке игр
            from .adapters.uepak import parse_aes_key
            key = body.aes_key.strip()
            try:
                parse_aes_key(key)
            except ValueError as e:
                raise HTTPException(400, str(e))
            fields["aes_key"] = key
        if not fields:
            raise HTTPException(400, "Нечего менять")
        return game_view(lib, lib.update(gid, **fields))

    @app.delete("/api/games/{gid}")
    def remove(gid: str):
        _game(gid)
        lib.remove(gid)
        return {"ok": True}

    @app.post("/api/games/{gid}/analyze")
    def analyze(gid: str):
        _game(gid)
        jobs.analyze(gid)
        return {"ok": True}

    @app.post("/api/games/{gid}/open")
    def open_folder(gid: str):
        path = _game(gid)["path"]
        if not os.path.isdir(path):
            raise HTTPException(400, "Папка не найдена")
        os.startfile(path)
        return {"ok": True}

    @app.post("/api/games/unhide")
    def unhide():
        lib.unhide_all()
        return {"ok": True}

    @app.get("/api/library-dirs")
    def library_dirs():
        return lib.library_dirs

    @app.post("/api/library-dirs")
    def add_library_dir(body: FolderPath):
        try:
            path = lib.add_library_dir(body.path)
        except ValueError as e:
            raise HTTPException(400, str(e))
        jobs.scan_dir(path)
        return lib.library_dirs

    @app.delete("/api/library-dirs")
    def remove_library_dir(body: FolderPath):
        lib.remove_library_dir(body.path)
        return lib.library_dirs

    @app.post("/api/pick-folder")
    def pick_folder():
        """Системный диалог выбора папки (только в окне приложения, не в браузере)."""
        try:
            import webview
        except ImportError:
            raise HTTPException(501, "Диалог недоступен")
        if not webview.windows:
            raise HTTPException(501, "Диалог доступен только в окне приложения")
        kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or webview.FOLDER_DIALOG
        result = webview.windows[0].create_file_dialog(kind)
        return {"path": result[0] if result else None}

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
