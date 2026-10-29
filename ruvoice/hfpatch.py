"""Все загрузки с Hugging Face — в models\\<имя репозитория> обычными файлами.

Стандартный кэш HF строится на символических ссылках, которые в Windows без режима разработчика
не создаются, и такую папку неудобно переносить. Патч подменяет hf_hub_download / snapshot_download,
поэтому любая библиотека (transformers, Seed-VC, BigVGAN…) кладёт файлы в models\\. Вызывать до импорта
transformers и других библиотек, которые импортируют эти функции по имени."""
import os

from .paths import PROJECT_DIR

MODELS_DIR = PROJECT_DIR / "models"
_installed = False


def local_dir_for(repo_id: str):
    return MODELS_DIR / repo_id.split("/")[-1]


def install():
    global _installed
    if _installed:
        return
    _installed = True
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import huggingface_hub
    from huggingface_hub import file_download, _snapshot_download

    orig_file = file_download.hf_hub_download
    orig_snapshot = _snapshot_download.snapshot_download

    def hf_hub_download(repo_id, filename, *args, **kwargs):
        if kwargs.get("repo_type") in (None, "model"):
            kwargs.pop("cache_dir", None)
            kwargs.setdefault("local_dir", str(local_dir_for(repo_id)))
        return orig_file(repo_id, filename, *args, **kwargs)

    def snapshot_download(repo_id, *args, **kwargs):
        if kwargs.get("repo_type") in (None, "model"):
            kwargs.pop("cache_dir", None)
            kwargs.setdefault("local_dir", str(local_dir_for(repo_id)))
        return orig_snapshot(repo_id, *args, **kwargs)

    orig_try = file_download.try_to_load_from_cache

    def try_to_load_from_cache(repo_id, filename, cache_dir=None, revision=None, repo_type=None):
        # transformers после скачивания ищет файл в стандартном кэше — подсказываем ему нашу папку
        if repo_type in (None, "model"):
            p = local_dir_for(repo_id) / filename
            if p.is_file():
                return str(p)
        return orig_try(repo_id, filename, cache_dir=cache_dir, revision=revision, repo_type=repo_type)

    for mod in (huggingface_hub, file_download):
        mod.hf_hub_download = hf_hub_download
        mod.try_to_load_from_cache = try_to_load_from_cache
    for mod in (huggingface_hub, _snapshot_download):
        mod.snapshot_download = snapshot_download
    try:  # transformers держит собственные ссылки на эти функции
        import huggingface_hub.utils as _u  # noqa: F401
        from huggingface_hub import hub_mixin
        hub_mixin.hf_hub_download = hf_hub_download
        hub_mixin.snapshot_download = snapshot_download
    except Exception:
        pass
