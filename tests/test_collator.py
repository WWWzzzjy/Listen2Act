from __future__ import annotations

import torch

from src.data.collator import collate_vla_batch


def test_collator_batches_vla_samples() -> None:
    samples = [
        {
            "rgb": torch.zeros(3, 224, 224),
            "state": torch.zeros(2),
            "action_chunk": torch.zeros(8, 7),
            "instruction": "pick up the bowl",
        },
        {
            "rgb": torch.ones(3, 224, 224),
            "state": torch.ones(2),
            "action_chunk": torch.ones(8, 7),
            "instruction": "move the plate",
        },
    ]
    batch = collate_vla_batch(samples)
    assert batch["images"].shape == (2, 3, 224, 224)
    assert batch["states"].shape == (2, 2)
    assert batch["action_chunks"].shape == (2, 8, 7)
    assert batch["instructions"] == ["pick up the bowl", "move the plate"]

