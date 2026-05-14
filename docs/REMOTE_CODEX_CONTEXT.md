# Remote Codex Context

Project path on remote:

```text
/cloud/cloud-ssd1/Listen2Act
```

Main goal:

Train and evaluate a simulation-based bilingual VLA model on LIBERO Object, with Chinese instruction and voice demo support. Hardware target is single V100S 32GB. Use fp16 only, no bf16, no FlashAttention-2. Headless rendering uses MuJoCo `osmesa` on this container because EGL failed.

## Current Environment

Use conda env:

```bash
cd /cloud/cloud-ssd1/Listen2Act
conda activate py310
source .env.headless
```

Important: do not use `py312`. Project requires Python `>=3.10,<3.12`.

GPU verified:

- Tesla V100-SXM2-32GB
- fp16 matmul OK

Rendering:

- EGL fails because container lacks NVIDIA graphics capability:
  `Cannot initialize a EGL device display`
- OSMesa works:
  `SUCCESS: rendered frame shape=(64,64,3), MUJOCO_GL=osmesa`
- `.env.headless` should set:
  - `MUJOCO_GL=osmesa`
  - `PYOPENGL_PLATFORM=osmesa`

LIBERO:

- Installed under `external/LIBERO`
- `~/.libero/config.yaml` points to:
  - `benchmark_root: /cloud/cloud-ssd1/Listen2Act/external/LIBERO/libero/libero`
  - `datasets: /cloud/cloud-ssd1/Listen2Act/data/libero`
- `verify_libero.py` passed:
  - `LIBERO imports OK. robosuite=1.4.1`
  - `SUCCESS: LIBERO reset/step/render OK`

Tests:

```bash
python -m pytest
```

Remote passed earlier with `6 passed`; local project later had `11 passed` after more fixes.

Dataset:

- `libero_object` demos downloaded into `data/libero`
- `find data/libero -name "*.hdf5" -o -name "*.h5" | head` should show files
- Dataset indexed `74507` LIBERO timesteps.

Instruction prep:

```bash
python scripts/prepare_bilingual_instructions.py
python scripts/prepare_paraphrases.py
```

Early versions only produced 1 instruction due to HDF5 metadata fallback. This was fixed in `src/data/libero_dataset.py` by recovering instruction from LIBERO benchmark metadata, BDDL, and filenames. Make sure remote has latest `git pull`.

## Important Fixes Already Made Locally

The remote Codex should check these files exist after `git pull`:

### Python/version/setup

- `env_setup/assert_python_version.py`
- `env_setup/setup_active_env.sh`
- `env_setup/init_libero_config.py`
- `env_setup/configure_headless_backend.py`
- `env_setup/verify_libero_deps.py`
- `env_setup/verify_model_deps.py`

### LIBERO deps

Pinned:

- `robosuite==1.4.1`
- `bddl==1.0.1`
- `future==0.18.2`
- `gym==0.25.2`
- `cloudpickle==2.1.0`
- `easydict==1.9`
- `thop==0.1.1.post2209072238`

Reason:

LIBERO expects old robosuite module:

```text
robosuite.environments.manipulation.single_arm_env
```

This module does not exist in robosuite 1.5.

### Florence-2 dependencies and model loading

Added:

- `timm==0.9.16`
- local `flash_attn/` shim package

Reason:

Florence-2 remote modeling imports `flash_attn` and `timm`. V100 cannot use FlashAttention-2, so the local shim satisfies import checks while keeping attention on SDPA/eager path.

Configs:

- `configs/model/florence2_base.yaml` includes pinned `model_revision`
- `configs/model/florence2_large.yaml` includes pinned `model_revision`

Model loading fallback:

`src/models/florence2_vla.py` has `_load_model_with_safetensors_fallback(...)` because Florence-2 safetensors metadata can be `None`; fallback loads `pytorch_model.bin` with `use_safetensors=False`.

### Florence-2 forward path

Important:

Florence-2 is encoder-decoder. Full `backbone(**inputs)` caused:

```text
ValueError: Please pass either input_ids or decoder_input_ids...
```

Fix:

`Florence2VLA.forward()` should call encoder-only helper:

```text
_encode_florence2_multimodal_features(...)
```

It does:

```text
_encode_image -> merge image/text embeddings -> get_encoder()
```

and does not invoke decoder.

### AMP / GradScaler fix

Error fixed:

```text
ValueError: Attempting to unscale FP16 gradients.
```

Cause:

LoRA trainable params became fp16. GradScaler requires trainable params/grads to be fp32.

Fix:

- In `src/models/florence2_vla.py`: `_cast_trainable_parameters(...)`
- In `src/training/bc_trainer.py`: `_cast_trainable_parameters_to_fp32(...)`

Frozen backbone remains fp16; trainable LoRA + action head are fp32.

### Data throughput fix

GPU utilization was about 30%.

Fixes made:

- `LiberoDataset` caches HDF5 handles per worker instead of opening the file per sample.
- DataLoader defaults:
  - `num_workers=8`
  - `persistent_workers=True`
  - `prefetch_factor=4`
- Added fast training config:
  - `configs/training/bc_v100s_fast.yaml`
  - `batch_size: 4`
  - `gradient_accumulation_steps: 2`
  - effective batch still 8

Run fast config:

```bash
TRAINING_CONFIG=configs/training/bc_v100s_fast.yaml bash scripts/run_bc_train.sh
```

If OOM, fall back:

```bash
bash scripts/run_bc_train.sh
```

## Known Git Issue

Remote had git pull failure:

```text
HTTP/2 stream 1 was not closed cleanly
```

Fix:

```bash
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
git pull --ff-only
```

If still failing, use SSH remote:

```bash
git remote set-url origin git@github.com:WWWzzzjy/Listen2Act.git
git pull --ff-only
```

## Before Training

Always:

```bash
cd /cloud/cloud-ssd1/Listen2Act
conda activate py310
source .env.headless
python --version
python env_setup/verify_gpu.py
python env_setup/verify_headless_rendering.py
python env_setup/verify_libero.py
python env_setup/verify_model_deps.py
python -m pytest
```

Check latest fixes are present:

```bash
grep -n "_encode_florence2_multimodal_features" src/models/florence2_vla.py
grep -n "_load_model_with_safetensors_fallback" src/models/florence2_vla.py
grep -n "_cast_trainable_parameters_to_fp32" src/training/bc_trainer.py
grep -n "model_revision" configs/model/florence2_base.yaml
```

## Training

Default Stage 1 English BC:

```bash
bash scripts/run_bc_train.sh
```

Fast config:

```bash
TRAINING_CONFIG=configs/training/bc_v100s_fast.yaml bash scripts/run_bc_train.sh
```

Logs:

```bash
tail -f checkpoints/bc_v100s/train_metrics.jsonl
tail -f checkpoints/bc_v100s_fast/train_metrics.jsonl
```

Checkpoint dirs:

- default: `checkpoints/bc_v100s`
- fast: `checkpoints/bc_v100s_fast`

Expected train time:

- default: roughly 12-48h depending throughput
- dataset: 74507 timesteps
- default config: 50000 optimizer steps, effective batch 8, about 5.4 epochs

## After Stage 1 Training

Check checkpoints:

```bash
ls -lh checkpoints/bc_v100s*/best
tail -n 20 checkpoints/bc_v100s*/train_metrics.jsonl
```

Small eval first. Edit `configs/eval/libero_eval.yaml`:

```yaml
checkpoint_path: checkpoints/bc_v100s_fast/best  # if using fast
episodes_per_task: 3
task_ids: [0, 1]
save_videos: true
```

Run:

```bash
python -m src.eval.eval_libero \
  --eval-config configs/eval/libero_eval.yaml \
  --model-config configs/model/florence2_base.yaml
```

Inspect:

```bash
cat data/eval_videos/libero_eval_results.json
ls data/eval_videos/*.mp4 | head
```

Then full eval:

```yaml
episodes_per_task: 20
task_ids: []
```

Stage 2 bilingual:

Review and replace placeholders in:

```text
data/instructions_zh/libero_object_zh.json
```

Then:

```bash
bash scripts/run_bc_train_bilingual.sh
python -m src.eval.eval_bilingual
```

Stage 3 paraphrase:

Review:

```text
data/paraphrases/libero_object_paraphrases.json
```

Then:

```bash
bash scripts/run_bc_train_paraphrase.sh
python -m src.eval.eval_paraphrase
```

## Warnings That Are OK

These appeared and are not fatal:

- robosuite private macro warning
- gym unmaintained warning
- EGL failure, because OSMesa works

## Things Not To Do

- Do not use Python 3.12 env.
- Do not install FlashAttention-2 on V100.
- Do not upgrade robosuite to 1.5.
- Do not remove local `flash_attn/` shim unless replacing Florence-2 loading strategy.
- Do not train if generated instruction JSON has only 1 key; pull latest dataset parsing fix first.

