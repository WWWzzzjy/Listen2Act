"""Verify LIBERO runtime Python dependencies before simulator startup."""

from __future__ import annotations

import importlib
import sys

REQUIRED_IMPORTS = (
    "bddl",
    "cloudpickle",
    "easydict",
    "future",
    "gym",
    "robosuite",
    "thop",
)


def main() -> int:
    """Import required LIBERO runtime modules and report all missing packages."""
    missing: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"{module_name}: OK ({version})")
    if missing:
        print("ERROR: Missing or broken LIBERO runtime dependencies:")
        for item in missing:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

