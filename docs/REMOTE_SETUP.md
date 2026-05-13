# Remote Setup

Target machine: Ubuntu Linux over SSH with one NVIDIA V100S 32GB GPU.

## One-Shot Setup

```bash
git clone <your-fork-url> simvoicevla
cd simvoicevla
bash env_setup/setup_remote_server.sh
source .venv/bin/activate
source .env.headless
```

Then verify:

```bash
python env_setup/verify_gpu.py
python env_setup/verify_headless_rendering.py
python env_setup/verify_libero.py
```

## Active Conda Environment

If the server already provides a conda environment, activate it first and install
into that environment:

```bash
conda activate py310
bash env_setup/setup_active_env.sh
source .env.headless
python env_setup/verify_gpu.py
python env_setup/verify_headless_rendering.py
python env_setup/verify_libero.py
```

`setup_active_env.sh` also initializes LIBERO's `~/.libero/config.yaml`
non-interactively and writes `.env.headless` with the first working MuJoCo
backend. On containers without NVIDIA graphics capability this may be `osmesa`
instead of `egl`.

LIBERO currently expects robosuite 1.4.x APIs such as
`robosuite.environments.manipulation.single_arm_env`, so the setup scripts pin
`robosuite==1.4.1`. They also install `bddl==1.0.1`, matching LIBERO's
official requirements. The curated runtime requirements file avoids downgrading
SimVoiceVLA's Florence-2 stack while still installing LIBERO simulator imports.

## Conda Alternative

```bash
conda env create -f env_setup/conda_env.yaml
conda activate simvoicevla
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## No-Sudo Notes

If `libegl1`, NVIDIA driver libraries, or FFmpeg codecs are missing and you cannot use sudo, ask the cluster admin to install them globally. You can still install Python packages in `.venv` or Conda, but EGL driver libraries must match the host driver.

## VSCode Remote-SSH

Open the repository folder through Remote-SSH, then use the integrated terminal:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Do not rely on VSCode GUI forwarding for MuJoCo. All rendering should be EGL offscreen.

## WandB and tmux

```bash
wandb login
tmux new -s simvoicevla
bash scripts/run_bc_train.sh
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t simvoicevla`.
