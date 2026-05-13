# Training

Training uses behavioral cloning with MSE over `H x 7` action chunks.

Default V100S config:

- batch size: 2 per device
- gradient accumulation: 4
- learning rate: `5e-5`
- warmup: 500 steps
- total steps: 50,000
- LoRA rank/alpha: 16/32
- action chunk size: 8
- precision: fp16

## Stage 1: English BC

```bash
bash scripts/run_bc_train.sh
```

Expected wall-clock on V100S: roughly 12-24h depending on storage, CPU workers, and LIBERO demo size.

## Stage 2: Bilingual Training

```bash
python scripts/prepare_bilingual_instructions.py
# Review and replace placeholder Chinese translations.
bash scripts/run_bc_train_bilingual.sh
```

Expected extra wall-clock: roughly 6h for a continuation-style run, or longer if training from scratch.

## Stage 3: Paraphrase Augmentation

```bash
python scripts/prepare_paraphrases.py
# Review held-in/held-out paraphrase splits before experiments.
bash scripts/run_bc_train_paraphrase.sh
```

Expected extra wall-clock: roughly 12h.

## Expected Result Shape

Do not invent benchmark numbers. A healthy trajectory usually shows training loss decreasing first, English success improving earlier, Chinese success approaching English after bilingual training, and paraphrase gap shrinking after augmentation. Actual success rates are `TBD` until measured.

## OOM Tips

Use Florence-2 base first. If memory is tight, reduce `num_workers`, use batch size 1 with higher gradient accumulation, shorten evaluation video retention, and keep validation batch count small.

