"""Dataset wrapper for official LIBERO HDF5 demonstration files."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
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
ENV_ARGS_ATTR_NAMES = ("env_args", "env_kwargs", "environment_args")
UNKNOWN_INSTRUCTION = "unknown task"
FILENAME_SUFFIXES = ("_demo", "_demos", "_trajectory", "_trajectories")


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
                        instruction = _read_instruction(group, handle, path)
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


def _read_instruction(group: h5py.Group, handle: h5py.File, hdf5_path: Path) -> str:
    """Read a language instruction from group or file attributes."""
    sources = [group]
    if isinstance(group.parent, h5py.Group):
        sources.append(group.parent)
    sources.append(handle)
    for source in sources:
        for key in INSTRUCTION_ATTR_NAMES:
            if key in source.attrs:
                instruction = _decode_text(source.attrs[key]).strip()
                if instruction and instruction != UNKNOWN_INSTRUCTION:
                    return instruction
    if "instruction" in group and isinstance(group["instruction"], h5py.Dataset):
        value = group["instruction"][()]
        instruction = _decode_text(value).strip()
        if instruction and instruction != UNKNOWN_INSTRUCTION:
            return instruction

    instruction = _instruction_from_env_args(group, handle)
    if instruction:
        return instruction

    instruction = _instruction_from_benchmark_lookup(hdf5_path)
    if instruction:
        return instruction

    return _instruction_from_filename(hdf5_path)


def _decode_text(value: Any) -> str:
    """Decode bytes, scalar arrays, or Python objects to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_text(value.item())
    return str(value)


def _instruction_from_env_args(group: h5py.Group, handle: h5py.File) -> str | None:
    """Resolve an instruction through robomimic/LIBERO env args metadata."""
    for source in (group, group.parent, handle):
        if source is None:
            continue
        for key in ENV_ARGS_ATTR_NAMES:
            if key not in source.attrs:
                continue
            payload = _parse_jsonish_attr(source.attrs[key])
            instruction = _instruction_from_metadata(payload)
            if instruction:
                return instruction
    return None


def _parse_jsonish_attr(value: Any) -> Any:
    """Parse HDF5 attrs that may store JSON as bytes or strings."""
    text = _decode_text(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _instruction_from_metadata(payload: Any) -> str | None:
    """Extract an instruction from known metadata layouts."""
    if isinstance(payload, str):
        return _instruction_from_bddl_reference(payload)
    if not isinstance(payload, dict):
        return None
    for key in INSTRUCTION_ATTR_NAMES:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("bddl_file_name", "bddl_file", "problem_name"):
        value = payload.get(key)
        if isinstance(value, str):
            instruction = _instruction_from_bddl_reference(value)
            if instruction:
                return instruction
    for nested_key in ("env_kwargs", "env_args", "task"):
        instruction = _instruction_from_metadata(payload.get(nested_key))
        if instruction:
            return instruction
    return None


def _instruction_from_bddl_reference(reference: str) -> str | None:
    """Resolve an instruction from a BDDL path or filename."""
    lookup = _load_benchmark_instruction_lookup()
    keys = _candidate_lookup_keys(Path(reference))
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def _instruction_from_benchmark_lookup(path: Path) -> str | None:
    """Resolve an instruction by matching the HDF5 filename to LIBERO tasks."""
    lookup = _load_benchmark_instruction_lookup()
    for key in _candidate_lookup_keys(path):
        if key in lookup:
            return lookup[key]
    return None


@lru_cache(maxsize=1)
def _load_benchmark_instruction_lookup() -> dict[str, str]:
    """Build a best-effort mapping from LIBERO BDDL names to task language."""
    try:
        from libero.libero import benchmark
    except Exception:
        return {}
    lookup: dict[str, str] = {}
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
    except Exception:
        return lookup
    for suite_factory in benchmark_dict.values():
        try:
            suite = suite_factory()
            task_count = _infer_task_count(suite)
            for task_id in range(task_count):
                task = suite.get_task(task_id)
                language = _task_language(task)
                if not language:
                    continue
                for reference in _task_bddl_references(task):
                    for key in _candidate_lookup_keys(Path(reference)):
                        lookup[key] = language
        except Exception:
            continue
    return lookup


def _infer_task_count(suite: Any) -> int:
    """Infer the number of tasks in a LIBERO benchmark suite."""
    for attr in ("n_tasks", "num_tasks"):
        value = getattr(suite, attr, None)
        if isinstance(value, int):
            return value
    tasks = getattr(suite, "tasks", None)
    if isinstance(tasks, list):
        return len(tasks)
    if hasattr(suite, "get_num_tasks"):
        try:
            return int(suite.get_num_tasks())
        except Exception:
            pass
    return 0


def _task_language(task: Any) -> str | None:
    """Read task language across common LIBERO task object layouts."""
    for attr in ("language", "language_instruction", "task_description", "description"):
        value = getattr(task, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _task_bddl_references(task: Any) -> list[str]:
    """Collect BDDL-like references from a LIBERO task object."""
    references: list[str] = []
    for attr in ("bddl_file", "bddl_file_name", "problem_name"):
        value = getattr(task, attr, None)
        if value:
            references.append(str(value))
    problem_folder = getattr(task, "problem_folder", None)
    bddl_file = getattr(task, "bddl_file", None) or getattr(task, "bddl_file_name", None)
    if problem_folder and bddl_file:
        references.append(str(Path(str(problem_folder)) / str(bddl_file)))
    return references


def _candidate_lookup_keys(path: Path) -> list[str]:
    """Produce normalized lookup keys for BDDL/HDF5 paths."""
    keys = []
    for value in (path.name, path.stem, str(path), path.as_posix()):
        normalized = _normalize_lookup_key(value)
        if normalized and normalized not in keys:
            keys.append(normalized)
    instruction_key = _normalize_lookup_key(_filename_to_instruction_text(path.stem))
    if instruction_key and instruction_key not in keys:
        keys.append(instruction_key)
    return keys


def _normalize_lookup_key(value: str) -> str:
    """Normalize filenames and task names for fuzzy lookup."""
    text = value.lower().strip()
    for suffix in (".hdf5", ".h5", ".bddl"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    for suffix in FILENAME_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.replace("\\", "/").replace("_", " ").strip()


def _instruction_from_filename(path: Path) -> str:
    """Recover a readable instruction from an official LIBERO demo filename."""
    instruction = _filename_to_instruction_text(path.stem)
    return instruction if instruction else UNKNOWN_INSTRUCTION


def _filename_to_instruction_text(stem: str) -> str:
    """Convert a LIBERO file stem to a natural-language instruction."""
    text = stem.lower()
    for suffix in FILENAME_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    parts = [part for part in text.split("_") if part]
    while parts and (parts[0] in {"libero", "object", "spatial", "goal"} or parts[0].isdigit()):
        parts.pop(0)
    if len(parts) >= 2 and parts[0] in {"kitchen", "living", "study"} and parts[1].startswith("scene"):
        parts = parts[2:]
    if parts and parts[0].startswith("scene"):
        parts = parts[1:]
    return " ".join(parts).strip()


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
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    return (tensor - FLORENCE_MEAN) / FLORENCE_STD
