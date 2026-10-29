"""Адаптеры движков: как достать реплики из игры и как установить русскую озвучку обратно.

Каждый адаптер реализует:
  detect(game) -> bool              подходит ли игра
  extract(project, game, progress)  реплики → project (оригинальный звук в orig/, текст, персонажи)
  install(project, game, progress)  одобренные реплики → в игру (с возможностью отката)
  uninstall(project, game, progress)
  is_installed(project, game) -> bool
"""


_registered = []


def register(adapter):
    """Register a format reader without changing the dubbing pipeline."""
    required = ("detect", "extract", "install", "uninstall", "is_installed")
    if not getattr(adapter, "id", None) or not all(callable(getattr(adapter, name, None)) for name in required):
        raise ValueError("Адаптер должен иметь id и реализовать интерфейс чтения/установки")
    if any(cls.id == adapter.id for cls in _all()):
        raise ValueError("Адаптер с таким id уже зарегистрирован")
    _registered.append(adapter)


def _all():
    from .fmod_loose import FmodLooseAdapter
    from .fromsoft import FromsoftAdapter
    from .loose import LooseAdapter
    from .reengine import REEngineAdapter
    from .source import SourceAdapter
    from .unreal import UnrealAdapter
    from .wwise_loose import WwiseLooseAdapter
    from .zip_audio import ZipAudioAdapter
    # Сначала читатели структурированных форматов (знают связи текста и звука), затем общие по звуковой
    # системе (Wwise, FMOD в папке игры — любой движок), последним — отдельные звуковые файлы
    return list(_registered) + [SourceAdapter, FromsoftAdapter, REEngineAdapter, UnrealAdapter,
            WwiseLooseAdapter, FmodLooseAdapter, ZipAudioAdapter, LooseAdapter]


def for_game(game: dict):
    for cls in _all():
        try:
            if cls.detect(game):
                return cls()
        except (OSError, ValueError, RuntimeError) as error:
            # A broken/unknown container must not hide other usable resources.
            import logging
            logging.getLogger(__name__).warning("Проверка адаптера %s: %s", cls.id, error)
    return None


def by_id(adapter_id: str):
    return next((cls() for cls in _all() if cls.id == adapter_id), None)
