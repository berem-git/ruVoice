"""Тембр персонажа: кто получает обученную RVC-модель (ступень 2 конвейера v2) и на каком уровне.

Главный персонаж — тот, у кого речи не меньше `timbre_min_seconds` и реплик не меньше 12: ему обучается
своя модель тембра. Остальные — эпизодические, идут быстрым путём (клонирование или донорский голос).
Пользователь может переопределить уровень вручную («обучать» / «эпизодический»)."""

TIMBRE_LEVELS = {
    "off": None,
    "fast": {"name": "Быстро", "epochs": 150, "minutes": 8},
    "normal": {"name": "Обычно", "epochs": 300, "minutes": 15},
    "deep": {"name": "Долго", "epochs": 500, "minutes": 25},
}
MIN_LINES = 12


def tier(char: dict, settings: dict, override: str | None) -> str:
    """«main» — обучать тембр, «episodic» — быстрый путь. char — строка Project.characters()."""
    if TIMBRE_LEVELS.get(settings.get("timbre_level", "normal")) is None:
        return "episodic"
    if override == "train":
        return "main"
    if override == "episodic":
        return "episodic"
    enough = (char.get("seconds") or 0) >= float(settings.get("timbre_min_seconds", 120))
    return "main" if enough and (char.get("lines") or 0) >= MIN_LINES else "episodic"
