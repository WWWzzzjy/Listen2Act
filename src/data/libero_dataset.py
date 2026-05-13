"""Dataset wrapper for official LIBERO HDF5 demonstration files."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augment import sample_instruction

LOGGER = logging.getLogger(__name__)
DEFAULT_ACTION_CHUNK_SIZE = 8
DEFAULT_ACTION_DIM = 7
DEFAULT_IMAGE_SIZE = 224
FLORENCE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
FLORENCE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
ACTION_DATASET_NAMES = ("actions", "action")
STATE_DATASET_NAMES = ("states", "state", "obs/state", "robot_states")
RGB_DATASET_NAMES = (
    "obs/agentview_rgb",
    "obs/rgb",
    "obs/image",
    "obs/robot0_eye_in_hand_image",
    "agentview_rgb",
    "rgb",
    "images",
)
INSTRUCTION_ATTR_NAMES = (
    "language_instruction",
    "instruction",
    "task_description",
    "lang",
    "natural_language_instruction",
)


@dataclass(frozen=True)
class DemoSampleIndex:
    """Index entry for a single timestep in an HDF5 demonstration."""

    hdf5_path: Path
    group_path: str
    step: int
    instruction: str


class LiberoDataset(Dataset[dict[str, Any]]):
    """Load LIBERO HDF5 demos and produce VLA training samples."""

    def __init__(
        self,
        data_root: str | Path,
        action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        translations_path: str | Path | None = None,
        paraphrases_path: str | Path | None = None,
        language_sampling: str = "en",
        use_paraphrases: bool = False,
        seed: int = 0,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_root: Directory containing official LIBERO ``.hdf5`` or ``.h5`` files.
            action_chunk_size: Number of future actions to predict per sample.
            image_size: Square image resize target, kept at or above 224.
            translations_path: Optional English-to-Chinese JSON mapping.
            paraphrases_path: Optional paraphrase JSON mapping.
            language_sampling: Language sampling strategy.
            use_paraphrases: Whether to sample paraphrases during training.
            seed: Local RNG seed used for instruction sampling.
        """
        self.data_root = Path(data_root)
        self.action_chunk_size = action_chunk_size
        self.image_size = max(image_size, DEFAULT_IMAGE_SIZE)
        self.translations = _load_json_mapping(translations_path)
        self.paraphrases = _load_json_mapping(paraphrases_path)
        self.language_sampling = language_sampling
        self.use_paraphrases = use_paraphrases
        self.rng = random.Random(seed)
        self.samples = self._build_index()

    def __len__(self) -> int:
        """Return the number of indexed timesteps."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample from an HDF5 file.

        Args:
            index: Dataset index.

        Returns:
            Dictionary with image tensor, state tensor, action chunk, and instruction.
        """
        sample = self.samples[index]
        with h5py.File(sample.hdf5_path, "r") as handle:
            group = handle[sample.group_path] if sample.group_path else handle
            actions = np.asarray(_require_dataset(group, ACTION_DATASET_NAMES))
            rgb_dataset = _find_rgb_dataset(group)
            rgb = np.asarray(rgb_dataset[sample.step])
            state = _load_state(group, sample.step)

        action_chunk = _make_action_chunk(actions, sample.step, self.action_chunk_size)
        instruction = sample_instruction(
            sample.instruction,
            translations=self.translations,
            paraphrases=self.paraphrases,
            language_sampling=self.language_sampling,
            use_paraphrases=self.use_paraphrases,
            rng=self.rng,
        )
        return {
            "rgb": _preprocess_rgb(rgb, self.image_size),
            "state": torch.as_tensor(state, dtype=torch.float32),
            "action_chunk": torch.as_tensor(action_chunk, dtype=torch.float32),
            "instruction": instruction,
        }

    def _build_index(self) -> list[DemoSampleIndex]:
        """Scan all HDF5 files and index action timesteps."""
        if not self.data_root.exists():
            LOGGER.warning("LIBERO data root does not exist: %s", self.data_root)
            return []

        paths = sorted(list(self.data_root.rglob("*.hdf5")) + list(self.data_root.rglob("*.h5")))
        index: list[DemoSampleIndex] = []
        for path in paths:
            try:
                with h5py.File(path, "r") as handle:
                    for group_path, group in _iter_demo_groups(handle):
                        actions = np.asarray(_require_dataset(group, ACTION_DATASET_NAMES))
                        instruction = _read_instruction(group, handle)
                        for step in range(actions.shape[0]):
                            index.append(
                                DemoSampleIndex(
                                    hdf5_path=path,
                                    group_path=group_path,
                                    step=step,
                                    instruction=instruction,
                                )
                            )
            except Exception as exc:
                LOGGER.warning("Skipping invalid HDF5 file %s: %s", path, exc)
        LOGGER.info("Indexed %d LIBERO timesteps from %s", len(index), self.data_root)
        return index


def _load_json_mapping(path: str | Path | None) -> dict[str, Any]:
    """Load an optional JSON mapping."""
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {json_path}")
    return data


def _iter_demo_groups(handle: h5py.File) -> list[tuple[str, h5py.Group]]:
    """Return groups containing an action dataset."""
    groups: list[tuple[str, h5py.Group]] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group) and _has_any_dataset(obj, ACTION_DATASET_NAMES):
            groups.append((name, obj))

    if _has_any_dataset(handle, ACTION_DATASET_NAMES):
        groups.append(("", handle))
    handle.visititems(visitor)
    return groups


def _has_any_dataset(group: h5py.Group | h5py.File, names: tuple[str, ...]) -> bool:
    """Return whether any candidate dataset path exists."""
    return any(_get_nested(group, name) is not None for name in names)


def _require_dataset(group: h5py.Group | h5py.File, names: tuple[str, ...]) -> h5py.Dataset:
    """Resolve the first existing dataset from candidates."""
    for name in names:
        dataset = _get_nested(group, name)
        if isinstance(dataset, h5py.Dataset):
            return dataset
    raise KeyError(f"None of the datasets exist: {names}")


def _get_nested(group: h5py.Group | h5py.File, path: str) -> h5py.Group | h5py.Dataset | None:
    """Resolve a slash-delimited HDF5 path under a group."""
    current: h5py.Group | h5py.Dataset = group
    for part in path.split("/"):
        if not isinstance(current, h5py.Group) or part not in current:
            return None
        current = current[part]
    return current


def _read_instruction(group: h5py.Group, handle: h5py.File) -> str:
    """Read a language instruction from group or file attributes."""
    for source in (group, handle):
        for key in INSTRUCTION_ATTR_NAMES:
            if key in source.attrs:
                return _decode_text(source.attrs[key])
    if "instruction" in group and isinstance(group["instruction"], h5py.Dataset):
        value = group["instruction"][()]
        return _decode_text(value)
    return "unknown task"


def _decode_text(value: Any) -> str:
    """Decode bytes, scalar arrays, or Python objects to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_text(value.item())
    return str(value)


def _find_rgb_dataset(group: h5py.Group) -> h5py.Dataset:
    """Find an RGB observation dataset."""
    for name in RGB_DATASET_NAMES:
        dataset = _get_nested(group, name)
        if isinstance(dataset, h5py.Dataset):
            return dataset
    found = _find_first_dataset(group, _looks_like_rgb)
    if found is None:
        raise KeyError("Could not find an RGB observation dataset.")
    return found


def _find_first_dataset(
    group: h5py.Group,
    predicate: Callable[[h5py.Dataset], bool],
) -> h5py.Dataset | None:
    """Find the first dataset under a group that matches a predicate."""
    result: h5py.Dataset | None = None

    def visitor(_name: str, obj: h5py.Group | h5py.Dataset) -> None:
        nonlocal result
        if result is None and isinstance(obj, h5py.Dataset) and predicate(obj):
            result = obj

    group.visititems(visitor)
    return result


def _looks_like_rgb(dataset: h5py.Dataset) -> bool:
    """Return whether a dataset shape plausibly stores RGB frames."""
    shape = dataset.shape
    return len(shape) >= 4 and (shape[-1] == 3 or shape[-3] == 3)


def _load_state(group: h5py.Group, step: int) -> np.ndarray:
    """Load a state vector for one timestep or return an empty vector."""
    for name in STATE_DATASET_NAMES:
        dataset = _get_nested(group, name)
        if isinstance(dataset, h5py.Dataset):
            values = np.asarray(dataset)
            if values.ndim == 1:
                return values.astype(np.float32)
            return values[min(step, values.shape[0] - 1)].astype(np.float32)
    return np.zeros((0,), dtype=np.float32)


def _make_action_chunk(actions: np.ndarray, start: int, chunk_size: int) -> np.ndarray:
    """Slice and pad a fixed-length action chunk."""
    chunk = actions[start : start + chunk_size]
    if chunk.shape[0] < chunk_size:
        pad_count = chunk_size - chunk.shape[0]
        pad_value = chunk[-1:] if chunk.shape[0] else np.zeros((1, DEFAULT_ACTION_DIM), dtype=np.float32)
        chunk = np.concatenate([chunk, np.repeat(pad_value, pad_count, axis=0)], axis=0)
    if chunk.shape[-1] < DEFAULT_ACTION_DIM:
        padded = np.zeros((chunk.shape[0], DEFAULT_ACTION_DIM), dtype=np.float32)
        padded[:, : chunk.shape[-1]] = chunk
        chunk = padded
    return chunk[:, :DEFAULT_ACTION_DIM].astype(np.float32)


def _preprocess_rgb(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    """Resize and normalize RGB image for Florence-style VLM input."""
    array = np.asarray(rgb)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array[..., :3])
    image = image.resize((image_size, image_size), Image.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
    return (tensor - FLORENCE_MEAN) / FLORENCE_STD
