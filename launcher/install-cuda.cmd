@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "UV_CACHE_DIR=%~dp0cache\uv"
set "PYTHONUTF8=1"
echo Установка CUDA из официального репозитория PyTorch. Потребуется несколько ГБ.
"%~dp0tools\uv\uv.exe" pip install --no-deps --link-mode copy --target "%~dp0runtime\accel\cuda" --python "%~dp0runtime\python\python.exe" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128
if errorlevel 1 (
  echo Установка не завершена. Проверьте интернет и повторите.
  pause
  exit /b 1
)
echo Готово. Запустите RuVoice.exe. Если программа уже открыта, выйдите через трей и запустите снова.
pause
