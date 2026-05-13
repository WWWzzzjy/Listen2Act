from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.data.libero_dataset import LiberoDataset


def test_libero_dataset_reads_hdf5_and_pads_actions(tmp_path: Path) -> None:
    data_root = tmp_path / "libero"
    data_root.mkdir()
    hdf5_path = data_root / "demo.hdf5"
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    with h5py.File(hdf5_path, "w") as handle:
        demo = handle.create_group("demo_0")
        demo.attrs["language_instruction"] = "pick up the red bowl"
        demo.create_dataset("actions", data=actions)
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=images)
        obs.create_dataset("state", data=np.ones((3, 4), dtype=np.float32))

    translations_path = tmp_path / "zh.json"
    translations_path.write_text(json.dumps({"pick up the red bowl": "拿起红色的碗"}), encoding="utf-8")
    dataset = LiberoDataset(
        data_root=data_root,
        action_chunk_size=5,
        translations_path=translations_path,
        language_sampling="zh",
    )
    assert len(dataset) == 3
    sample = dataset[1]
    assert sample["rgb"].shape == (3, 224, 224)
    assert sample["state"].shape == (4,)
    assert sample["instruction"] == "拿起红色的碗"
    assert sample["action_chunk"].shape == (5, 7)
    np.testing.assert_allclose(sample["action_chunk"][-1].numpy(), actions[-1])

