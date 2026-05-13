# Data

LIBERO demonstrations are stored as HDF5 files under `data/libero/`.

The dataset wrapper searches for common fields:

- actions: `actions` or `action`
- RGB observations: `obs/agentview_rgb`, `obs/rgb`, `obs/image`, `obs/robot0_eye_in_hand_image`, or any RGB-shaped dataset
- state: `states`, `state`, `obs/state`, or `robot_states`
- instruction: HDF5 attributes such as `language_instruction`, `instruction`, or `task_description`

Each training sample returns:

```python
{
    "rgb": Tensor[3, 224, 224],
    "state": Tensor[*],
    "action_chunk": Tensor[8, 7],
    "instruction": str,
}
```

Action convention:

- xyz: first 3 continuous dimensions
- rpy: next 3 continuous dimensions
- gripper: final scalar

Chinese instructions live in `data/instructions_zh/*.json` as:

```json
{
  "pick up the red bowl": "拿起红色的碗"
}
```

Paraphrases live in `data/paraphrases/*.json` as:

```json
{
  "pick up the red bowl": {
    "en": ["grab the red bowl"],
    "zh": ["把红色的碗拿起来"]
  }
}
```

During bilingual training, language is sampled 50/50. During paraphrase training, variants are sampled uniformly from the configured mapping.

