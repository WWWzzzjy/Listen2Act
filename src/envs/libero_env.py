"""Headless LIBERO environment wrapper."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.headless import enforce_headless

LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE_SIZE = 256
RGB_KEYS = (
    "agentview_rgb",
    "robot0_eye_in_hand_image",
    "rgb",
    "image",
    "front_rgb",
    "wrist_rgb",
)


class LiberoEnv:
    """Gym-style wrapper around an official LIBERO offscreen environment."""

    def __init__(
        self,
        suite_name: str = "libero_object",
        task_id: int = 0,
        seed: int = 0,
        image_size: int = DEFAULT_IMAGE_SIZE,
        record_video: bool = False,
        env_factory: Callable[..., Any] | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a headless LIBERO environment.

        Args:
            suite_name: LIBERO benchmark suite name.
            task_id: Task index inside the suite.
            seed: Environment seed.
            image_size: Offscreen camera size.
            record_video: Whether to keep RGB frames in memory.
            env_factory: Optional factory for tests or custom env construction.
            env_kwargs: Optional keyword arguments passed to the env factory.
        """
        enforce_headless()
        self.suite_name = suite_name
        self.task_id = task_id
        self.seed = seed
        self.image_size = image_size
        self.record_video = record_video
        self.frames: list[np.ndarray] = []
        self.env = (
            env_factory(**(env_kwargs or {}))
            if env_factory is not None
            else _make_offscreen_libero_env(suite_name, task_id, image_size)
        )
        self.instruction = _get_task_instruction(suite_name, task_id)
        self._seed_env(seed)

    def reset(self, seed: int | None = None, initial_state: Any | None = None) -> dict[str, Any]:
        """Reset the environment.

        Args:
            seed: Optional reset seed.
            initial_state: Optional LIBERO initial state.

        Returns:
            Observation dictionary with an ``rgb`` image.
        """
        if seed is not None:
            self._seed_env(seed)
        if initial_state is not None and hasattr(self.env, "set_init_state"):
            self.env.set_init_state(initial_state)
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        formatted = self._format_obs(obs)
        self.frames = []
        self._maybe_record(formatted["rgb"])
        return formatted

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Step the environment with a 7-DoF action.

        Args:
            action: Action array.

        Returns:
            Tuple of observation, reward, done, and info.
        """
        try:
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = result
            formatted = self._format_obs(obs)
            self._maybe_record(formatted["rgb"])
            return formatted, float(reward), bool(done), dict(info or {})
        except Exception as exc:
            LOGGER.exception("LIBERO step failed: %s", exc)
            obs = {"rgb": np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)}
            return obs, 0.0, True, {"error": str(exc), "success": False}

    def render_rgb(self) -> np.ndarray:
        """Render one offscreen RGB frame without opening a GUI."""
        try:
            frame = self.env.render(mode="rgb_array")
        except TypeError:
            frame = self.env.render()
        return _to_uint8_rgb(frame)

    def close(self) -> None:
        """Close the underlying environment."""
        if hasattr(self.env, "close"):
            self.env.close()

    def _format_obs(self, obs: Any) -> dict[str, Any]:
        """Normalize environment observations."""
        if isinstance(obs, dict):
            rgb = _extract_rgb_from_dict(obs)
            if rgb is None:
                rgb = self.render_rgb()
            output = dict(obs)
            output["rgb"] = _to_uint8_rgb(rgb)
            return output
        return {"rgb": _to_uint8_rgb(obs)}

    def _maybe_record(self, frame: np.ndarray) -> None:
        """Append a frame when video recording is enabled."""
        if self.record_video:
            self.frames.append(_to_uint8_rgb(frame))

    def _seed_env(self, seed: int) -> None:
        """Seed a LIBERO or gym-style environment when supported."""
        if hasattr(self.env, "seed"):
            self.env.seed(seed)


def _make_offscreen_libero_env(suite_name: str, task_id: int, image_size: int) -> Any:
    """Construct the official LIBERO OffScreenRenderEnv."""
    _ensure_project_libero_config()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "Could not import LIBERO runtime modules. Run "
            "`bash env_setup/setup_active_env.sh` in the active Python environment. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_factory = benchmark_dict[suite_name]
    suite = suite_factory()
    task = suite.get_task(task_id)
    bddl_file = _resolve_task_bddl(task, bddl_root=Path(get_libero_path("bddl_files")))
    env_args = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": image_size,
        "camera_widths": image_size,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    }
    try:
        return OffScreenRenderEnv(**env_args)
    except TypeError as exc:
        LOGGER.warning("Retrying OffScreenRenderEnv with minimal LIBERO args after: %s", exc)
        return OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=image_size,
            camera_widths=image_size,
        )


def _ensure_project_libero_config() -> None:
    """Create LIBERO's config file from the project checkout when absent."""
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    project_root = Path.cwd()
    libero_root = Path(os.environ.get("LIBERO_ROOT", project_root / "external" / "LIBERO")).expanduser()
    if not libero_root.is_absolute():
        libero_root = project_root / libero_root
    benchmark_root = libero_root.resolve() / "libero" / "libero"
    if not benchmark_root.exists():
        return
    if str(libero_root.resolve()) not in sys.path:
        sys.path.insert(0, str(libero_root.resolve()))
    datasets = Path(os.environ.get("LIBERO_DATASETS", project_root / "data" / "libero")).expanduser()
    if not datasets.is_absolute():
        datasets = project_root / datasets
    datasets.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": datasets.resolve(),
        "assets": benchmark_root / "assets",
    }
    config_file.write_text(
        "\n".join(f"{key}: {value}" for key, value in config.items()) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Initialized LIBERO config at %s", config_file)


def _resolve_task_bddl(task: Any, bddl_root: Path | None = None) -> Path | str:
    """Resolve a LIBERO task BDDL path across common task object layouts."""
    problem_folder = getattr(task, "problem_folder", None)
    bddl_file = getattr(task, "bddl_file", None) or getattr(task, "bddl_file_name", None)
    if problem_folder and bddl_file:
        relative = Path(problem_folder) / str(bddl_file)
        return bddl_root / relative if bddl_root is not None else relative
    if bddl_file:
        path = Path(str(bddl_file))
        if path.is_absolute() or bddl_root is None:
            return path
        return bddl_root / path
    raise AttributeError("Could not resolve BDDL file from LIBERO task object.")


def _get_task_instruction(suite_name: str, task_id: int) -> str:
    """Read the official task language when LIBERO is installed."""
    try:
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(task_id)
        return str(getattr(task, "language", getattr(task, "language_instruction", "unknown task")))
    except Exception as exc:
        LOGGER.warning("Could not read LIBERO task instruction: %s", exc)
        return "unknown task"


def _extract_rgb_from_dict(obs: dict[str, Any]) -> np.ndarray | None:
    """Extract the first RGB-like array from an observation dict."""
    for key in RGB_KEYS:
        if key in obs:
            return np.asarray(obs[key])
    for value in obs.values():
        if isinstance(value, dict):
            nested = _extract_rgb_from_dict(value)
            if nested is not None:
                return nested
        array = np.asarray(value)
        if array.ndim == 3 and (array.shape[-1] == 3 or array.shape[0] == 3):
            return array
    return None


def _to_uint8_rgb(frame: Any) -> np.ndarray:
    """Convert an arbitrary image-like array to HWC uint8 RGB."""
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array[..., :3]
