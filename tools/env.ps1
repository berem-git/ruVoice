# Окружение RuVoice для разработки. Пути считаются от папки программы, поэтому её можно переносить.
# Подключение в PowerShell:  . <папка RuVoice>\tools\env.ps1
$root = Split-Path -Parent $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = "$root\tools\python"
$env:UV_CACHE_DIR          = "$root\cache\uv"
$env:UV_TOOL_DIR           = "$root\tools\uv-tools"
$env:HF_HOME               = "$root\models\huggingface"
$env:TORCH_HOME            = "$root\models\torch"
$env:PIP_CACHE_DIR         = "$root\cache\pip"
$env:npm_config_cache      = "$root\cache\npm"
$env:PYTHONPATH            = $root
$env:PYTHONNOUSERSITE      = "1"
$env:HF_HUB_DISABLE_XET    = "1"
$env:RUVOICE_PYTHON        = "$root\runtime\python\python.exe"
$wwise = Get-ChildItem "$root\tools\Wwise" -Directory -Filter "Wwise_*" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if ($wwise) {
    $env:WWISEROOT     = $wwise.FullName
    $env:WWISE_CONSOLE = "$($wwise.FullName)\Authoring\x64\Release\bin\WwiseConsole.exe"
}
$env:PATH = @(
    "$root\runtime\python",
    "$root\tools\uv",
    "$root\tools\ffmpeg\bin",
    "$root\tools\vgmstream",
    "$root\tools\quickbms",
    $env:PATH
) -join ";"
# Установка новых пакетов во встроенный Python:
#   uv pip install --python $env:RUVOICE_PYTHON <пакет>
