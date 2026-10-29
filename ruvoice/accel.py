"""Выбор ускорителя для нейросетей: NVIDIA (CUDA) или AMD (ROCm). Процессорного режима нет — озвучка
требует видеокарту; без неё программа открывается, но озвучка недоступна.

Сборки PyTorch лежат отдельно в runtime\\accel\\<вариант>; нужная подключается в sys.path до первого
`import torch`. Поэтому одна и та же папка программы работает и на ПК с NVIDIA, и на ноутбуке с AMD.
"""
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from .paths import DATA_DIR, PROJECT_DIR

ACCEL_DIR = PROJECT_DIR / "runtime" / "accel"
SETTINGS = DATA_DIR / "accel.json"
VARIANTS = {"cuda": "NVIDIA CUDA", "rocm": "AMD ROCm"}  # озвучка только на видеокарте

# Официальные сборки AMD для Windows (ROCm 7.2.1, Python 3.12). Поддерживаются Radeon RX 9000,
# часть RX 7000 и Ryzen AI 300/400/Max — на остальных AMD программа работает на процессоре.
ROCM_BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
ROCM_WHEELS = [
    "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "rocm-7.2.1.tar.gz",
    "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
]

_active: str | None = None
_gpus: list[str] | None = None
_lock = threading.Lock()


def gpus() -> list[str]:
    """Названия видеокарт из Windows (без загрузки драйверов и PyTorch)."""
    global _gpus
    if _gpus is None:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
            _gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        except Exception:
            _gpus = []
    return _gpus


def ram_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30)
    except Exception:
        return None


def vendor() -> str | None:
    names = " ".join(gpus()).lower()
    if "nvidia" in names:
        return "nvidia"
    if "amd" in names or "radeon" in names:
        return "amd"
    return None


def installed() -> list[str]:
    return [v for v in VARIANTS if (ACCEL_DIR / v / "torch").is_dir()]


def preferred() -> str:
    try:
        return json.loads(SETTINGS.read_text("utf-8")).get("variant", "auto")
    except Exception:
        return "auto"


def set_preferred(variant: str):
    if variant != "auto" and variant not in VARIANTS:
        raise ValueError(variant)
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps({"variant": variant}), "utf-8")


def choose() -> str:
    import os
    have = installed()
    want = os.environ.get("RUVOICE_ACCEL") or preferred()  # переменная — для тестов и дочерних процессов
    if want != "auto" and want in have:
        return want
    v = vendor()
    if v == "nvidia" and "cuda" in have:
        return "cuda"
    if v == "amd" and "rocm" in have:
        return "rocm"
    # Процессорного пути нет: без поддерживаемой видеокарты сборка CUDA всё равно подключается (для окна,
    # библиотеки и проверок), а озвучка отказывается работать с понятным сообщением (gpu_ready / require_gpu)
    return "cuda" if "cuda" in have else (have[0] if have else "cuda")


def activate() -> str:
    """Подключить сборку PyTorch. Вызывать до `import torch`; повторные вызовы ничего не делают."""
    global _active
    with _lock:
        if _active is None:
            _active = choose()
            path = str(ACCEL_DIR / _active)
            if path not in sys.path:
                sys.path.insert(0, path)
        return _active


def device() -> str:
    activate()
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype():
    import torch
    if device() == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def info() -> dict:
    """Сводка для интерфейса. Импортирует torch, поэтому вызывать в фоне."""
    variant = activate()
    result = {"variant": variant, "variant_name": VARIANTS.get(variant, variant), "preferred": preferred(),
              "installed": installed(), "gpus": gpus(), "vendor": vendor(), "device": "cpu",
              "device_name": None, "vram_gb": None, "ram_gb": ram_gb(), "error": None, "amd": amd_family()}
    try:
        import torch
        result["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result.update(device="cuda", device_name=props.name, vram_gb=round(props.total_memory / 2**30, 1))
    except Exception as e:
        result["error"] = str(e)
    return result


# Семейства AMD: официальные сборки AMD (RDNA3/RDNA4, Ryzen AI) и ночные сборки TheRock для RDNA2 (RX 6000) —
# последние неофициальные: AMD в Windows-сборки gfx103x не включает, но проект TheRock их собирает, и на RX 6800
# они работают (с оговорками). Индекс: https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/
ROCM_NIGHTLY_INDEX = {"gfx103X": "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/"}


def amd_family() -> dict | None:
    """Поколение видеокарты AMD по названию: official (RX 7000/9000, Ryzen AI) или rdna2 (RX 6000)."""
    import re
    names = " ".join(gpus())
    if not names or vendor() != "amd":
        return None
    if re.search(r"RX\s*6\d{3}", names, re.I):
        return {"family": "rdna2", "gfx": "gfx103X", "name": names, "experimental": True,
                "note": "RX 6000 (RDNA2): официальной сборки для Windows нет, ставится ночная сборка проекта ROCm/TheRock — "
                        "работает на RX 6800/6900, на RX 6600/6700 без гарантий. Нужен драйвер AMD 26.1.1 и новее."}
    if re.search(r"RX\s*[79]\d{3}|Radeon AI|Ryzen AI|8[0-9]{2}M|890M|880M", names, re.I):
        return {"family": "official", "gfx": "gfx11xx/12xx", "name": names, "experimental": False, "note": ""}
    return {"family": "unknown", "gfx": None, "name": names, "experimental": True,
            "note": "Эта AMD не входит в списки поддержки ROCm для Windows; можно попробовать ночную сборку, но скорее всего не заработает."}


def _prune_duplicates(target: Path):
    """Из целевой папки убрать пакеты, которые уже есть в основной среде (numpy, sympy…): папка ускорителя стоит
    первой в sys.path, и другая версия numpy сломала бы всё остальное."""
    import shutil
    main = Path(sys.executable).parent / "Lib" / "site-packages"
    keep = {"torch", "torchaudio", "torchvision", "functorch", "torchgen", "triton", "_rocm_sdk_core", "rocm", "rocm_sdk"}
    for d in list(target.iterdir()):
        base = d.name.split("-")[0].lower()
        if base in keep or base.startswith(("rocm", "_rocm", "torch", "hip", "amd")):
            continue
        if (main / d.name).exists() or any(main.glob(base + "-*.dist-info")) or (main / base).exists():
            shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)


def _verify_rocm(progress) -> bool:
    """Проверить сборку в отдельном процессе: видит ли PyTorch видеокарту."""
    code = ("import sys; sys.path.insert(0, r'%s'); import torch; "
            "print('OK' if torch.cuda.is_available() else 'NOGPU', torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
            % str(ACCEL_DIR / "rocm"))
    r = subprocess.run([str(Path(sys.executable).with_name("python.exe")), "-c", code], capture_output=True, text=True,
                       timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    out = (r.stdout or "").strip() + " " + (r.stderr or "")[-300:]
    progress(("Проверка: " + out.strip())[:300])
    return r.stdout.startswith("OK")


def install_rocm_nightly(gfx: str, progress=lambda msg: None):
    """Ночная сборка PyTorch ROCm (TheRock) под семейство gfx — для RX 6000. Около 3 ГБ."""
    target = ACCEL_DIR / "rocm"
    index = ROCM_NIGHTLY_INDEX[gfx]
    uv = PROJECT_DIR / "tools" / "uv" / "uv.exe"
    progress(f"Скачивание ночной сборки ROCm для {gfx} (это неофициальная сборка, ~3 ГБ)…")
    r = subprocess.run([str(uv), "pip", "install", "--target", str(target), "--link-mode", "copy",
                        "--python", str(Path(sys.executable).with_name("python.exe")),
                        "--index-url", index, "--extra-index-url", "https://pypi.org/simple",
                        "--index-strategy", "unsafe-best-match", "torch", "torchaudio"],
                       capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:
        raise RuntimeError("установка ночной сборки не удалась: " + (r.stderr or r.stdout)[-600:])
    _prune_duplicates(target)
    ok = _verify_rocm(progress)
    progress("Готово: перезапустите программу — озвучка пойдёт на видеокарте AMD" if ok
             else "Сборка поставлена, но PyTorch видеокарту не увидел: обновите драйвер AMD (26.1.1+) и перезапустите; "
                  "если не поможет — эта карта ночной сборкой не поддерживается")


def install_rocm(progress=lambda msg: None):
    """Сборка PyTorch для AMD в runtime\\accel\\rocm: официальная (RX 7000/9000) или ночная TheRock (RX 6000)."""
    fam = amd_family() or {}
    if fam.get("family") in ("rdna2", "unknown"):
        return install_rocm_nightly("gfx103X", progress)
    target = ACCEL_DIR / "rocm"
    tmp = PROJECT_DIR / "cache" / "rocm-wheels"
    tmp.mkdir(parents=True, exist_ok=True)
    files = []
    for i, name in enumerate(ROCM_WHEELS, 1):
        dest = tmp / name.replace("%2B", "+")
        if not dest.exists():
            progress(f"Скачивание {i}/{len(ROCM_WHEELS)}: {dest.name}")
            part = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(ROCM_BASE + name, part)
            part.replace(dest)
        files.append(str(dest))
    progress("Установка сборки для AMD…")
    uv = PROJECT_DIR / "tools" / "uv" / "uv.exe"
    subprocess.run([str(uv), "pip", "install", "--target", str(target), "--no-deps", "--link-mode", "copy",
                    "--python", str(Path(sys.executable).with_name("python.exe")), *files],
                   check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    ok = _verify_rocm(progress)
    progress("Готово: перезапустите программу, чтобы включить ускорение AMD" if ok
             else "Сборка поставлена, но PyTorch видеокарту не увидел — обновите драйвер AMD (26.2.2+) и перезапустите")


def restart_app(delay: float = 0.7):
    """Перезапустить программу (после смены ускорителя): новый процесс через RuVoice.exe, этот — закрыть."""
    import os
    import time

    def go():
        time.sleep(delay)
        exe = PROJECT_DIR / "RuVoice.exe"
        if exe.exists():
            subprocess.Popen([str(exe)], cwd=str(PROJECT_DIR), creationflags=subprocess.DETACHED_PROCESS)
        else:
            subprocess.Popen([sys.executable, "-m", "ruvoice.main"], cwd=str(PROJECT_DIR),
                             creationflags=subprocess.DETACHED_PROCESS)
        os._exit(0)
    threading.Thread(target=go, daemon=True).start()


def gpu_ready() -> bool:
    """Есть ли видеокарта, на которой могут работать нейросети (NVIDIA с CUDA или AMD с ROCm)."""
    try:
        return device() == "cuda"
    except Exception:
        return False


def require_gpu():
    if not gpu_ready():
        v = vendor()
        hint = ("установите ускорение AMD (ROCm) в «Система и нейросети» — поддерживаются Radeon RX 9000, часть RX 7000 "
                "и Ryzen AI 300/400/Max" if v == "amd" else "нужна NVIDIA с 6 ГБ видеопамяти и больше или поддерживаемая AMD")
        raise RuntimeError(f"Для озвучки нужна видеокарта: {hint}. На процессоре программа не озвучивает.")
