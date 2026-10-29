"""Справочник языков: русские названия, варианты написания в файлах игр и в Steam."""
import re

# код -> (название по-русски, варианты имён папок/файлов)
LANGS: dict[str, tuple[str, list[str]]] = {
    "ru": ("Русский", ["russian", "ru", "ru-ru", "ru_ru", "rus", "russia"]),
    "en": ("Английский", ["english", "en", "en-us", "en_us", "en-gb", "en_gb", "eng", "enus", "engb"]),
    "de": ("Немецкий", ["german", "deutsch", "de", "de-de", "de_de", "ger", "deu"]),
    "fr": ("Французский", ["french", "francais", "fr", "fr-fr", "fr_fr", "fre", "fra"]),
    "es": ("Испанский", ["spanish", "espanol", "es", "es-es", "es_es", "es-mx", "es_mx", "es-419", "spa", "latam"]),
    "it": ("Итальянский", ["italian", "it", "it-it", "it_it", "ita"]),
    "ja": ("Японский", ["japanese", "ja", "ja-jp", "ja_jp", "jp", "jpn"]),
    "zh": ("Китайский", ["chinese", "schinese", "tchinese", "zh", "zh-cn", "zh_cn", "zh-tw", "zh_tw",
                          "zh-hans", "zh-hant", "cn", "chs", "cht", "mandarin"]),
    "ko": ("Корейский", ["korean", "ko", "ko-kr", "ko_kr", "kr", "kor"]),
    "pl": ("Польский", ["polish", "pl", "pl-pl", "pl_pl", "pol"]),
    "pt": ("Португальский", ["portuguese", "brazilian", "pt", "pt-br", "pt_br", "pt-pt", "pt_pt", "ptbr", "por"]),
    "tr": ("Турецкий", ["turkish", "tr", "tr-tr", "tr_tr", "tur"]),
    "uk": ("Украинский", ["ukrainian", "uk-ua", "uk_ua", "ukr", "ua"]),
    "cs": ("Чешский", ["czech", "cs", "cs-cz", "cs_cz", "cze", "ces"]),
    "hu": ("Венгерский", ["hungarian", "hu", "hu-hu", "hun"]),
    "nl": ("Нидерландский", ["dutch", "nl", "nl-nl", "nld"]),
    "sv": ("Шведский", ["swedish", "sv", "sv-se", "swe"]),
    "ar": ("Арабский", ["arabic", "ar", "ar-sa", "ara"]),
    "th": ("Тайский", ["thai", "th", "th-th", "tha"]),
}

ALIAS_TO_CODE = {alias: code for code, (_, aliases) in LANGS.items() for alias in aliases}

# Полные названия языков внутри имени файла: voice_english.pck, Russian_vo.bnk
_FULL_NAMES = sorted({a for _, al in LANGS.values() for a in al if len(a) > 4 and a.isalpha()}, key=len, reverse=True)
FULL_NAME_RE = re.compile(r"(?:^|[^a-z])(" + "|".join(_FULL_NAMES) + r")(?:[^a-z]|$)")
# Короткий код в конце имени файла: pakchunk3_ru, dialog-en-us
CODE_SUFFIX_RE = re.compile(r"[_\-.](ru|en|de|fr|es|it|ja|jp|zh|ko|pl|pt|tr|cs)(?:[_\-][a-z]{2,4})?$")

STEAM_LANGS = {
    "Russian": "ru", "English": "en", "German": "de", "French": "fr",
    "Spanish - Spain": "es", "Spanish - Latin America": "es", "Italian": "it",
    "Japanese": "ja", "Simplified Chinese": "zh", "Traditional Chinese": "zh",
    "Korean": "ko", "Polish": "pl", "Portuguese - Brazil": "pt", "Portuguese - Portugal": "pt",
    "Turkish": "tr", "Ukrainian": "uk", "Czech": "cs", "Hungarian": "hu", "Dutch": "nl",
    "Swedish": "sv", "Arabic": "ar", "Thai": "th",
}


def code_from_component(name: str) -> str | None:
    """Язык по имени папки: 'Russian', 'English(US)', 'ru-RU'."""
    low = name.lower()
    code = ALIAS_TO_CODE.get(low)
    if code is None and "(" in low:
        code = ALIAS_TO_CODE.get(re.sub(r"\(.*?\)", "", low).strip())
    return code


def code_from_dirname(name: str) -> str | None:
    """Полное название языка внутри имени папки: hl2_russian, Russian_VO."""
    m = FULL_NAME_RE.search(name.lower())
    return ALIAS_TO_CODE[m.group(1)] if m else None


def code_from_filename(stem: str) -> str | None:
    m = FULL_NAME_RE.search(stem)
    if m:
        return ALIAS_TO_CODE[m.group(1)]
    m = CODE_SUFFIX_RE.search(stem)
    if m:
        return ALIAS_TO_CODE.get(m.group(1))
    return None


def display_name(code: str) -> str:
    return LANGS[code][0] if code in LANGS else code
