# SimVoiceVLA

Simulation-based bilingual Vision-Language-Action training on LIBERO with Chinese voice instruction support.

SimVoiceVLA trains a compact Florence-2 VLA policy with LoRA on a single NVIDIA V100S 32GB GPU. It supports English LIBERO instructions, reviewed Chinese translations, paraphrase augmentation, and an end-to-end `.wav -> Whisper ASR -> VLA -> headless LIBERO rollout -> MP4` demo.

## Key Constraints

- Single GPU target: NVIDIA V100S 32GB.
- Precision: fp16 only. bf16 is explicitly disabled in configs.
- Rendering: headless MuJoCo EGL via `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.
- No GUI windows, no `human` render mode, no FlashAttention-2.
- Default backbone: `microsoft/Florence-2-base`; `Florence-2-large` is configurable.

## Quick Start

```bash
cd simvoicevla
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
pytest
```

Remote setup:

```bash
bash env_setup/setup_remote_server.sh
source .venv/bin/activate
source .env.headless
python env_setup/verify_libero.py
```

If you are already inside a Python 3.10/3.11 conda environment, install into that
active environment instead of creating `.venv`:

```bash
bash env_setup/setup_active_env.sh
source .env.headless
python env_setup/verify_gpu.py
python env_setup/verify_headless_rendering.py
python env_setup/verify_libero.py
```

## Data

Place official LIBERO HDF5 demonstrations under `data/libero/`.

```bash
bash scripts/download_libero_demos.sh
python scripts/prepare_bilingual_instructions.py
python scripts/prepare_paraphrases.py
```

The translation and paraphrase scripts write reviewed-offline JSON placeholders first. Replace placeholder text with reviewed Chinese translations and final paraphrases before reporting results.

## Training

Stage 1, English behavioral cloning:

```bash
bash scripts/run_bc_train.sh
```

Stage 2, bilingual training:

```bash
bash scripts/run_bc_train_bilingual.sh
```

Stage 3, bilingual paraphrase augmentation:

```bash
bash scripts/run_bc_train_paraphrase.sh
```

Checkpoints are written under `checkpoints/`. The trainer saves LoRA adapters and the action head.

## Evaluation

```bash
bash scripts/run_eval_all.sh
```

Outputs:

- JSON success-rate reports in `data/eval_videos/`.
- Per-episode MP4 videos rendered offscreen with H.264.
- Bilingual EN/ZH success-rate gap.
- Held-out paraphrase robustness results.

## Voice Demo

```bash
bash scripts/run_voice_demo.sh \
  --audio command.wav \
  --checkpoint-path checkpoints/bc_v100s/best \
  --task-id 0
```

The demo transcribes Chinese speech with Whisper, sends the transcript directly to the bilingual VLA model, executes in LIBERO, and saves an MP4 plus JSON log.

## 中文说明

SimVoiceVLA 是一个基于 LIBERO 仿真的双语视觉-语言-动作项目。目标是在单张 NVIDIA V100S 32GB GPU 上，用 Florence-2 小模型和 LoRA 完成 VLA 行为克隆训练，并支持中文语音指令演示。

核心流程：

- 英文 LIBERO 指令训练基线。
- 英文 + 中文指令混合训练。
- 英文/中文 paraphrase 指令增强，验证模型是否真正利用语言条件。
- 中文 `.wav` 语音通过 Whisper 转写后直接送入 VLA，在无图形界面的远程服务器上运行仿真并保存视频。

注意事项：

- 只使用 fp16，不使用 bf16。
- MuJoCo/LIBERO 必须使用 EGL headless 渲染。
- 训练和评估脚本不会打开窗口，也不依赖 X server。
- 结果表格暂留 `TBD`，不要在没有真实实验前填写 benchmark 数字。

See `docs/` for detailed setup, architecture, training, evaluation, hardware, and data notes.
