# Evaluation

The primary evaluation suite is LIBERO-Object.

Run all evaluation modes:

```bash
bash scripts/run_eval_all.sh
```

`eval_libero.py` runs K episodes per task and writes:

```json
{
  "0": {
    "task_id": 0,
    "success_rate": 0.0,
    "n_episodes": 20,
    "episodes": []
  }
}
```

`eval_bilingual.py` compares English and Chinese instructions separately and reports the absolute success-rate gap.

`eval_paraphrase.py` evaluates held-out instruction variants. Keep train-time and held-out paraphrase splits separate when reporting robustness.

Videos are saved as MP4 files in `data/eval_videos/` using H.264 at low quality for size. All rendering is offscreen.

