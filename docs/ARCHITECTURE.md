# Architecture

SimVoiceVLA has four runtime layers:

1. Data: `LiberoDataset` reads official LIBERO HDF5 demos and returns `rgb`, `state`, `action_chunk`, and `instruction`.
2. Model: `Florence2VLA` loads Florence-2, applies LoRA to attention projections, pools hidden states, and predicts `H x 7` actions through `ActionHead`.
3. Training: `BCTrainer` runs fp16 behavioral cloning with gradient accumulation, warmup cosine LR, checkpointing, JSONL logs, and optional WandB.
4. Evaluation/demo: `LiberoEnv` enforces EGL headless rendering, `eval_*` scripts compute success rates, and the voice demo connects Whisper ASR to policy rollout.

Data flow:

```text
LIBERO HDF5 -> LiberoDataset -> collator -> Florence2VLA -> MSE(action_chunk)
                                                    |
Chinese wav -> Whisper ASR -> transcript -----------+-> LiberoEnv rollout -> MP4 + JSON
```

The project depends on official LIBERO and HuggingFace packages at runtime. It does not vendor LIBERO, StarVLA, OpenVLA, or model weights.

