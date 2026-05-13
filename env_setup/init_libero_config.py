"""Create LIBERO's non-interactive path config file."""

from __future__ import annotations

import os
from pathlib import Path
import argparse


DEFAULT_LIBERO_ROOT = "external/LIBERO"
DEFAULT_DATASETS = "data/libero"
DEFAULT_CONFIG_DIR = "~/.libero"


def write_libero_config(
    libero_root: str | Path = DEFAULT_LIBERO_ROOT,
    datasets: str | Path = DEFAULT_DATASETS,
    config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Write LIBERO path config without importing LIBERO.

    Args:
        libero_root: Path to cloned LIBERO repository.
        datasets: Path to LIBERO HDF5 datasets.
        config_dir: Directory containing ``config.yaml``.
        project_root: Repository root used to resolve relative paths.

    Returns:
        Path to the written config file.
    """
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    config_root = config_dir or os.environ.get("LIBERO_CONFIG_PATH", DEFAULT_CONFIG_DIR)
    libero_path = _resolve_path(root, libero_root)
    benchmark_root = libero_path / "libero" / "libero"
    datasets_path = _resolve_path(root, datasets)
    config_path_dir = Path(config_root).expanduser().resolve()
    config_path_dir.mkdir(parents=True, exist_ok=True)
    datasets_path.mkdir(parents=True, exist_ok=True)

    config = _build_config(benchmark_root=benchmark_root, datasets=datasets_path)
    config_path = config_path_dir / "config.yaml"
    config_path.write_text(_format_config(config), encoding="utf-8")
    return config_path


def main() -> None:
    """Write ``~/.libero/config.yaml`` without importing LIBERO."""
    parser = argparse.ArgumentParser(description="Initialize LIBERO path config non-interactively.")
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT, help="Path to cloned LIBERO repository.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS, help="Path to LIBERO HDF5 datasets.")
    parser.add_argument(
        "--config-dir",
        default=os.environ.get("LIBERO_CONFIG_PATH", DEFAULT_CONFIG_DIR),
        help="Directory containing LIBERO config.yaml.",
    )
    args = parser.parse_args()
    project_root = Path.cwd()
    config_path = write_libero_config(
        libero_root=args.libero_root,
        datasets=args.datasets,
        config_dir=args.config_dir,
        project_root=project_root,
    )
    print(f"Wrote LIBERO config to {config_path}")
    benchmark_root = _resolve_path(project_root, args.libero_root) / "libero" / "libero"
    config = _build_config(benchmark_root=benchmark_root, datasets=_resolve_path(project_root, args.datasets))
    for key, value in config.items():
        print(f"{key}: {value}")


def _resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve a possibly relative path under a repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _build_config(benchmark_root: Path, datasets: Path) -> dict[str, Path]:
    """Build LIBERO config values from resolved paths."""
    return {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": datasets,
        "assets": benchmark_root / "assets",
    }


def _format_config(config: dict[str, Path]) -> str:
    """Format LIBERO's simple YAML config without requiring PyYAML."""
    return "\n".join(f"{key}: {value}" for key, value in config.items()) + "\n"


if __name__ == "__main__":
    main()
