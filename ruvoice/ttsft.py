"""Дообучение модели речи под голос персонажа (LoRA поверх русского Qwen3-TTS).

Полное дообучение (official sft_12hz.py) даёт голос, который на слух ближе всего к персонажу, но весит
4 ГБ на персонажа. Здесь обучаются только LoRA-адаптеры (r=16) в talker и code_predictor — ~60 МБ на
персонажа; базовая модель одна, адаптер подключается на лету.

Запуск отдельным процессом: `python -m ruvoice.ttsft <job.json>` (job: dataset_jsonl, out_dir, epochs, lr).
Прогресс — строки JSON в stdout. Данные готовит dub.DubJobs._do_train_tts (нарезка, расшифровка, коды)."""
import json
import os
import sys
import time
from pathlib import Path

from .paths import PROJECT_DIR

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FINETUNE_DIR = PROJECT_DIR / "tools" / "qwen3-tts-finetune"
TOKENIZER_KEY = "tts-tok"
VOICE_FILE = "voice.safetensors"


def lora_config(r: int = 16, alpha: int = 32):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")


def extract_codes(items: list[dict], tokenizer_dir: str, device: str = "cuda:0", batch: int = 8) -> list[dict]:
    """items: [{"audio", "text", "ref_audio"}] → те же записи с audio_codes (12 Гц токены речи)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(tokenizer_dir, device_map=device)
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        enc = tok.encode([it["audio"] for it in chunk])
        for it, code in zip(chunk, enc.audio_codes):
            out.append({**it, "audio_codes": code.cpu().tolist()})
    del tok
    return out


def train(job: dict, report=print):
    """Обучить LoRA-адаптер голоса. job: dataset_jsonl (с audio_codes), out_dir, epochs, lr, base (ключ модели)."""
    from . import accel, models
    accel.activate()
    import torch
    from peft import get_peft_model_state_dict, inject_adapter_in_model
    from torch.utils.data import DataLoader
    from transformers import AutoConfig
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    sys.path.insert(0, str(FINETUNE_DIR))
    from dataset import TTSDataset  # официальный загрузчик данных Qwen3-TTS

    base = models.path(job.get("base", "tts-ru"))
    if not base:
        raise RuntimeError("Базовая модель речи не скачана")
    dev = "cuda:0" if accel.device() == "cuda" else "cpu"
    qwen = Qwen3TTSModel.from_pretrained(str(base), torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=dev)
    model = qwen.model
    config = AutoConfig.from_pretrained(str(base))
    data = [json.loads(l) for l in open(job["dataset_jsonl"], encoding="utf-8") if l.strip()]
    dataset = TTSDataset(data, qwen.processor, config)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

    for prm in model.parameters():
        prm.requires_grad_(False)
    inject_adapter_in_model(lora_config(int(job.get("rank", 16)), int(job.get("alpha", 32))), model.talker, adapter_name="voice")
    trainable = [prm for prm in model.talker.parameters() if prm.requires_grad]
    for prm in trainable:
        prm.data = prm.data.float()  # LoRA-веса в fp32 — устойчивее при bf16-активациях
    report({"message": f"LoRA: {sum(x.numel() for x in trainable) / 1e6:.1f} млн обучаемых параметров"})
    try:  # без reentrant: иначе при замороженных эмбеддингах градиент до LoRA-слоёв не доходит
        model.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    opt = torch.optim.AdamW(trainable, lr=float(job.get("lr", 2e-4)), weight_decay=0.01)
    epochs, accum = int(job.get("epochs", 4)), int(job.get("accum", 8))
    total_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / 20) * max(0.05, 1 - st / max(1, total_steps / accum)))
    target_spk = None
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for bi, batch in enumerate(loader):
            input_ids, codec_ids = batch["input_ids"].to(dev), batch["codec_ids"].to(dev)
            ref_mels = batch["ref_mels"].to(dev).to(torch.bfloat16)
            text_mask, codec_mask_e = batch["text_embedding_mask"].to(dev), batch["codec_embedding_mask"].to(dev)
            attn, labels, codec_mask = batch["attention_mask"].to(dev), batch["codec_0_labels"].to(dev), batch["codec_mask"].to(dev)
            with torch.no_grad():
                spk = model.speaker_encoder(ref_mels)
            if target_spk is None:
                target_spk = spk[0].detach().float().cpu()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                emb = model.talker.model.text_embedding(input_ids[:, :, 0]) * text_mask \
                    + model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_mask_e
                emb[:, 6, :] = spk.to(emb.dtype)
                emb = emb.detach().requires_grad_(True)
                for i in range(1, 16):
                    emb = emb + model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i]) * codec_mask.unsqueeze(-1)
                out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=attn[:, :-1], labels=labels[:, 1:],
                                   output_hidden_states=True)
                hidden = out.hidden_states[0][-1][codec_mask[:, :-1]]
                _, sub_loss = model.talker.forward_sub_talker_finetune(codec_ids[codec_mask], hidden)
                loss = (out.loss + 0.3 * sub_loss) / accum
            loss.backward()
            if (bi + 1) % accum == 0 or bi + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                report({"epoch": epoch + 1, "total": epochs, "step": step, "steps": total_steps,
                        "loss": round(float(loss.item()) * accum, 3), "elapsed": round(time.time() - t0)})
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in get_peft_model_state_dict(model.talker, adapter_name="voice").items()}
    save_file(sd, str(out_dir / VOICE_FILE))
    torch.save(target_spk, out_dir / "speaker.pt")
    (out_dir / "meta.json").write_text(json.dumps({"base": job.get("base", "tts-ru"), "rank": int(job.get("rank", 16)),
                                                   "alpha": int(job.get("alpha", 32)), "epochs": epochs,
                                                   "samples": len(data), "ref_audio": data[0]["ref_audio"],
                                                   "train_seconds": round(time.time() - t0)}, ensure_ascii=False), "utf-8")
    if job.get("probe_text"):  # проба сразу после обучения, тем же процессом — отделяет ошибки обучения от ошибок загрузки
        import soundfile as sf
        model.eval()
        for prm in trainable:
            prm.data = prm.data.to(torch.bfloat16)
        with torch.no_grad():
            prompt = qwen.create_voice_clone_prompt(ref_audio=data[0]["ref_audio"], ref_text=None, x_vector_only_mode=True)[0]
            for lang in ("auto", "Russian"):
                wavs, sr = qwen.generate_voice_clone(text=[job["probe_text"]], language=[lang], voice_clone_prompt=[prompt],
                                                     max_new_tokens=400, temperature=0.8, repetition_penalty=1.05)
                sf.write(str(out_dir / f"probe_{lang}.wav"), wavs[0], sr)
        report({"message": "проба записана"})
    report({"message": "готово", "done": True})
    return str(out_dir)


def share_tokenizer(ckpt: Path, base: Path) -> int:
    """Токенайзер речи (speech_tokenizer, ~650 МБ) в каждом дообученном голосе — точная копия базового.
    Заменяем копии жёсткими ссылками на файлы базовой модели (тот же диск): загрузка не меняется, место
    не тратится. Возвращает, сколько байт освобождено. Если файлы различаются или диск другой — не трогаем."""
    import filecmp
    src_dir, dst_dir = Path(base) / "speech_tokenizer", Path(ckpt) / "speech_tokenizer"
    freed = 0
    if not (src_dir.is_dir() and dst_dir.is_dir()):
        return 0
    for dst in dst_dir.iterdir():
        src = src_dir / dst.name
        if not (dst.is_file() and src.is_file()) or dst.stat().st_size != src.stat().st_size:
            continue
        try:
            if os.path.samefile(src, dst):
                continue  # уже ссылка
            if not filecmp.cmp(src, dst, shallow=False):
                continue
            tmp = dst.with_name(dst.name + ".lnk_tmp")
            os.link(src, tmp)
            size = dst.stat().st_size
            os.replace(tmp, dst)
            freed += size
        except OSError:
            continue
    return freed


def train_full(job: dict, report=print) -> str:
    """Полное дообучение (официальный sft_12hz.py, адаптированный под Windows: sdpa, 8-битный AdamW,
    gradient checkpointing). Даёт лучший голос на слух, но чекпойнт — ~4 ГБ на персонажа.
    Возвращает папку чекпойнта (модель типа custom_voice с диктором job["speaker"])."""
    import re
    import subprocess
    from . import models
    base = models.path(job.get("base", "tts-ru"))
    if not base:
        raise RuntimeError("Базовая модель речи не скачана")
    out_dir = Path(job["out_dir"])
    epochs = int(job.get("epochs", 4))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_DIR / "runtime" / "accel" / "cuda") + os.pathsep + str(PROJECT_DIR)
    env["PYTHONUTF8"] = "1"
    env["SFT_ACCUM"] = str(job.get("accum", 8))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    exe = str(Path(sys.executable).with_name("python.exe"))
    args = [exe, str(FINETUNE_DIR / "sft_12hz_win.py"), "--init_model_path", str(base), "--output_model_path", str(out_dir),
            "--train_jsonl", job["dataset_jsonl"], "--batch_size", str(int(job.get("batch", 1))), "--lr", str(job.get("lr", 1e-5)),
            "--num_epochs", str(epochs), "--speaker_name", job["speaker"]]
    proc = subprocess.Popen(args, cwd=str(FINETUNE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.time()
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-30:]
        m = re.match(r"Epoch (\d+) \| Step (\d+) \| Loss: ([\d.]+)", line)
        if m:
            report({"epoch": int(m.group(1)) + 1, "total": epochs, "step": int(m.group(2)), "loss": float(m.group(3)),
                    "elapsed": round(time.time() - t0)})
    proc.wait()
    if proc.returncode:
        raise RuntimeError("дообучение не удалось: " + " | ".join(tail[-5:]))
    ckpt = out_dir / f"checkpoint-epoch-{epochs - 1}"
    if not (ckpt / "model.safetensors").exists():
        raise RuntimeError("дообучение не записало чекпойнт")
    (ckpt / "meta.json").write_text(json.dumps({"kind": "full", "speaker": job["speaker"], "epochs": epochs,
                                                "train_seconds": round(time.time() - t0)}, ensure_ascii=False), "utf-8")
    return str(ckpt)


# ---------- подключение адаптеров при озвучке ----------

def attach(talker, voice_dir: str, name: str):
    """Загрузить адаптер голоса в talker под именем name (один раз); активировать — activate()."""
    import torch
    from peft import inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    meta = json.loads((Path(voice_dir) / "meta.json").read_text("utf-8"))
    inject_adapter_in_model(lora_config(meta.get("rank", 16), meta.get("alpha", 32)), talker, adapter_name=name)
    sd = load_file(str(Path(voice_dir) / VOICE_FILE))
    res = set_peft_model_state_dict(talker, sd, adapter_name=name)
    missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
    unexpected = list(getattr(res, "unexpected_keys", []))
    if missing or unexpected:
        print(f"адаптер {name}: не совпали ключи — не хватает {len(missing)}, лишних {len(unexpected)}: {(missing + unexpected)[:3]}")
    dev = next(talker.parameters()).device
    for m in talker.modules():
        if hasattr(m, "lora_A") and name in m.lora_A:
            m.lora_A[name].to(dev).to(torch.bfloat16)
            m.lora_B[name].to(dev).to(torch.bfloat16)
    return meta


def activate(talker, name: str | None):
    """Включить адаптер name (None — чистая базовая модель)."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    dev = next(talker.parameters()).device
    for m in talker.modules():
        if isinstance(m, BaseTunerLayer):
            # All trained voices may live on disk/CPU; only the selected adapter uses VRAM.
            for attr in ("lora_A", "lora_B"):
                for key, adapter in getattr(m, attr, {}).items():
                    adapter.to(dev if key == name else "cpu")
            if name is None:
                m.enable_adapters(False)
            else:
                m.enable_adapters(True)
                m.set_adapter(name)


def main():
    job = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    out = sys.stdout
    sys.stdout = sys.stderr

    def report(info):
        out.write(json.dumps(info, ensure_ascii=False) + "\n")
        out.flush()
    try:
        if job.get("op") == "codes":
            from . import accel
            accel.activate()
            items = [json.loads(l) for l in open(job["input_jsonl"], encoding="utf-8") if l.strip()]
            rows = extract_codes(items, job["tokenizer_dir"])
            Path(job["output_jsonl"]).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
            report({"message": "коды готовы", "done": True, "count": len(rows)})
        elif job.get("op") == "train_full":
            ckpt = train_full(job, report)
            report({"message": "готово", "done": True, "ckpt": ckpt})
        else:
            train(job, report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        report({"error": f"{type(e).__name__}: {e}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
