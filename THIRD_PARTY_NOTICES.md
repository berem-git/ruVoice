# Third-party components

RuVoice's portable package includes third-party runtime components. PyTorch/CUDA are downloaded separately by the installer. Their
copyright and license files are retained in the runtime and tool directories.
The release manifest lists the installed Python distributions and versions.
Model weights are downloaded separately and have their own model licenses.

| Component | Source / license information |
| --- | --- |
| Python 3.12 | https://www.python.org/ — see runtime/python/LICENSE.txt |
| PyTorch / TorchAudio | https://github.com/pytorch/pytorch and https://github.com/pytorch/audio — licenses in their dist-info directories |
| NVIDIA CUDA / cuDNN libraries shipped with PyTorch | NVIDIA redistributable components; see licenses/notices included with PyTorch |
| VoxCPM | https://github.com/OpenBMB/VoxCPM — Apache-2.0; runtime/engines/voxcpm |
| Qwen3-TTS | https://github.com/QwenLM/Qwen3-TTS — Apache-2.0; runtime/python/Lib/site-packages |
| FFmpeg | https://ffmpeg.org/ — GPL-3.0-or-later build; tools/ffmpeg/LICENSE.txt |
| FFmpeg build scripts | https://github.com/BtbN/FFmpeg-Builds — MIT; build N-128013-g6b05263b44-20261024, corresponding FFmpeg source https://github.com/FFmpeg/FFmpeg/tree/6b05263b44 |
| vgmstream | https://github.com/vgmstream/vgmstream — tools/vgmstream/COPYING and bundled component notices |
| llama.cpp | https://github.com/ggml-org/llama.cpp — MIT; LLVM OpenMP license retained |
| repak | https://github.com/trumank/repak — Apache-2.0 |
| retoc | https://github.com/trumank/retoc — MIT |
| uv | https://github.com/astral-sh/uv — Apache-2.0 / MIT |
| Applio (RVC) | https://github.com/IAHispano/Applio — MIT; ставится из программы в runtime/engines/rvc (релиз 3.7.1), претрейны HiFi-GAN / RMVPE / ContentVec с https://huggingface.co/IAHispano/Applio |
| OmniVoice | https://github.com/k2-fsa/OmniVoice — Apache-2.0; ставится из программы в runtime/engines/omnivoice, веса k2-fsa/OmniVoice |
| Silero TTS v5.6 | https://github.com/snakers4/silero-models — CC BY-NC 4.0 (некоммерческая); скачивается только при явном выборе бэкенда |
| Донорский банк голосов (ruvoice/donors) | 16 синтетических русских голосов, созданных Qwen3-TTS VoiceDesign (Apache-2.0) по текстовым описаниям; записей реальных людей не содержит |

FFmpeg's complete configuration is available using `tools/ffmpeg/bin/ffmpeg.exe
-version`. Upstream build recipes, dependency versions and source download
instructions are maintained by BtbN/FFmpeg-Builds. No changes to that binary
are made by RuVoice. This notice does not replace the individual licenses.

Wwise Authoring, Oodle, game assets, generated game voice recordings and
third-party model weights are not included in the portable release.
