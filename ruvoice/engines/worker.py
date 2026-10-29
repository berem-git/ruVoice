"""Процесс-работник голосового движка: `python -m ruvoice.engines.worker <engine>`.

Протокол: по строке JSON на запрос в stdin, по строке JSON с ответом в stdout. Всё лишнее уходит в stderr
(журнал data\\engines\\<engine>.log). Папка движка runtime\\engines\\<env> подключается первой в sys.path,
поэтому его версии библиотек (например, другой transformers) не мешают основной программе."""
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS = PROJECT_DIR / "models"


def _setup(env: str | None):
    sys.path.insert(0, str(PROJECT_DIR))
    from ruvoice import accel
    accel.activate()
    if env:
        # Папка движка — раньше основных пакетов: у него могут быть свои transformers и huggingface_hub
        sys.path.insert(0, str(PROJECT_DIR / "runtime" / "engines" / env))
    from ruvoice import hfpatch
    hfpatch.install()
    import torch, torchaudio, soundfile as sf  # noqa: F401

    def _load(path, **kw):  # torchaudio 2.11 читает файлы только через torchcodec — обходим через soundfile
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr
    torchaudio.load = _load
    return torch


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


class RVC:
    """RVC (реализация Applio): обучение модели голоса персонажа и перенос тембра на русскую речь.

    В отличие от Seed-VC, RVC берёт содержание и произношение только из исходника (русской речи),
    а тембр — из обученной на персонаже модели, поэтому не тянет за собой английский выговор.
    Шаги обучения — скрипты Applio, запускаемые отдельными процессами (их вывод уходит в журнал)."""
    env = "rvc"
    APPLIO = PROJECT_DIR / "runtime" / "engines" / "rvc" / "applio"
    SR = 40000

    def load(self):
        import subprocess
        self.torch = _setup(self.env)
        sys.path.insert(0, str(self.APPLIO))
        os.chdir(self.APPLIO)  # Applio ищет свои модели по относительным путям rvc/models/...
        self.subprocess = subprocess
        self.vc = None

    def _env(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = ";".join([str(PROJECT_DIR / "runtime" / "engines" / "rvc"),
                                      str(PROJECT_DIR / "runtime" / "accel" / "cuda"), str(PROJECT_DIR),
                                      str(self.APPLIO)])
        env["PYTHONUTF8"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        return env

    def _run(self, args, progress, phase, total=None):
        import re
        python = Path(sys.executable).with_name("python.exe")
        proc = self.subprocess.Popen([str(python), *map(str, args)], cwd=str(self.APPLIO), env=self._env(),
                                     stdout=self.subprocess.PIPE, stderr=self.subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace",
                                     creationflags=self.subprocess.CREATE_NO_WINDOW)
        try:
            from ruvoice import power
            power.apply_priority(proc.pid)
        except Exception:
            pass
        last_epoch = 0
        for line in proc.stdout:
            print(f"[{phase}] {line}", end="", file=sys.stderr, flush=True)
            m = re.search(r"[Ee]poch\D{0,3}(\d+)", line)
            if m and total:
                ep = int(m.group(1))
                if ep != last_epoch:
                    last_epoch = ep
                    progress({"phase": phase, "epoch": ep, "total": total})
        if proc.wait() != 0:
            raise RuntimeError(f"RVC: шаг «{phase}» завершился с ошибкой (см. data\\engines\\rvc.log)")

    def train(self, req, progress):
        """req = {model_name, dataset_dir, out_dir, epochs, batch_size}; возвращает {pth, index}."""
        import glob
        import shutil
        name = req["model_name"]
        epochs = int(req.get("epochs", 200))
        batch = int(req.get("batch_size", 8))
        cores = max(1, (os.cpu_count() or 4) // 2)
        try:  # уровень нагрузки на систему: меньше потоков и пачка поменьше
            from ruvoice import power
            cores = max(1, min(cores, power.threads()))
            batch = max(2, power.batch(batch))
        except Exception:
            pass
        logs = self.APPLIO / "logs" / name
        shutil.rmtree(logs, ignore_errors=True)
        self.vc = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        progress({"phase": "preprocess", "message": "Подготовка записей (нарезка)…"})
        self._run(["rvc/train/preprocess/preprocess.py", logs, req["dataset_dir"], self.SR, cores, "Automatic",
                   "False", "3.0", "0.3", "none"], progress, "preprocess")
        progress({"phase": "extract", "message": "Извлечение признаков речи и высоты тона…"})
        self._run(["rvc/train/extract/extract.py", logs, "rmvpe", cores, "0", self.SR, "contentvec", "None", "2"],
                  progress, "extract")
        pg = self.APPLIO / "rvc" / "models" / "pretraineds" / "hifi-gan" / f"f0G{self.SR // 1000}k.pth"
        pd = self.APPLIO / "rvc" / "models" / "pretraineds" / "hifi-gan" / f"f0D{self.SR // 1000}k.pth"
        progress({"phase": "train", "epoch": 0, "total": epochs})
        self._run(["rvc/train/train.py", name, epochs, epochs, pg, pd, "0", batch, self.SR, "True", "True",
                   "False", "False", "HiFi-GAN", "False"], progress, "train", epochs)
        progress({"phase": "index", "message": "Построение индекса голоса…"})
        self._run(["rvc/train/process/extract_index.py", logs, "Auto"], progress, "index")
        pths = [p for p in glob.glob(str(logs / "*.pth")) if not os.path.basename(p).startswith(("G_", "D_"))]
        if not pths:
            raise RuntimeError("RVC: обучение не создало модель (см. data\\engines\\rvc.log)")
        pth = max(pths, key=os.path.getmtime)
        idx = sorted(glob.glob(str(logs / "*.index")), key=lambda p: ("added" not in p, p))
        out = Path(req["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pth, out / "model.pth")
        if idx:
            shutil.copy2(idx[0], out / "model.index")
        shutil.rmtree(logs, ignore_errors=True)
        return {"pth": str(out / "model.pth"), "index": str(out / "model.index") if idx else None}

    def convert(self, req):
        import soundfile as sf
        if self.vc is None:
            from rvc.infer.infer import VoiceConverter
            self.vc = VoiceConverter()
        p = req.get("params") or {}
        # index_rate 0 — не подмешивать фрагменты английских записей (они дают акцент);
        # protect 0.5 — беречь глухие согласные и дыхание от искажений
        self.vc.convert_audio(audio_input_path=req["source"], audio_output_path=req["out"], model_path=req["pth"],
                              index_path=req.get("index") or "", pitch=int(p.get("pitch", 0)), f0_method="rmvpe",
                              index_rate=float(p.get("index_rate", 0.0)), volume_envelope=1.0,
                              protect=float(p.get("protect", 0.5)), embedder_model="contentvec",
                              export_format="WAV", proposed_pitch=bool(p.get("auto_pitch", True)))
        wav, sr = sf.read(req["out"], dtype="float32", always_2d=True)
        return wav.mean(axis=1), sr


class VoxCPMEngine:
    """Isolated dependencies: never replace Qwen's transformers or the shared PyTorch runtime."""
    def load(self):
        torch = _setup("voxcpm")
        from voxcpm import VoxCPM
        self.model = VoxCPM.from_pretrained(str(MODELS / "VoxCPM2"), load_denoiser=False,
                                           optimize=False, device=_device(torch), local_files_only=True)

    def generate(self, req):
        import re
        import numpy as np
        import torch
        if req.get("seed") is not None:
            torch.manual_seed(int(req["seed"]))
            np.random.seed(int(req["seed"]) % 2**32)
        p = req.get("params") or {}
        instruction = re.sub(r"[()\r\n]", " ", str(p.get("instruct") or ""))[:400]
        text = req["text"]
        if instruction:
            text = f"({instruction}){text}"
        wav = self.model.generate(text=text, reference_wav_path=req["ref_audio"],
                                  cfg_value=float(p.get("cfg_value", 2.0)), inference_timesteps=int(p.get("steps", 10)),
                                  max_len=max(20, int(float(req.get("max_seconds", 15)) * 6.25) + 20),
                                  normalize=False, denoise=False)
        return np.asarray(wav, dtype=np.float32).reshape(-1), self.model.tts_model.sample_rate


ENGINES = {"rvc": RVC, "voxcpm": VoxCPMEngine}


def main():
    name = sys.argv[1]
    out = sys.stdout
    sys.stdout = sys.stderr  # болтовня библиотек не должна попадать в канал ответов

    def reply(obj):
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    engine = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "exit":
                reply({"ok": True})
                break
            if engine is None:
                engine = ENGINES[name]()
                engine.load()
            if op == "ping":
                reply({"ok": True, "engine": name})
            elif op == "train":
                result = engine.train(req, lambda info: reply({"progress": info}))
                reply({"ok": True, **result})
            elif op in ("generate", "convert"):
                import soundfile as sf
                import numpy as np
                wav, sr = getattr(engine, op)(req)
                wav = np.clip(np.nan_to_num(wav), -1, 1)
                Path(req["out"]).parent.mkdir(parents=True, exist_ok=True)
                sf.write(req["out"], wav, sr, subtype="FLOAT")
                reply({"ok": True, "sr": int(sr), "seconds": round(len(wav) / sr, 3)})
            else:
                reply({"ok": False, "error": f"неизвестная операция {op}"})
        except Exception as e:
            traceback.print_exc()
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
