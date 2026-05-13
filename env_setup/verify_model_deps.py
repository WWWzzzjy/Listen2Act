"""Verify Florence-2 runtime imports without downloading model weights."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_IMPORTS = ("timm", "flash_attn", "flash_attn.bert_padding")


def main() -> int:
    """Check Florence-2 local runtime dependencies."""
    missing: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        print(f"{module_name}: OK ({getattr(module, '__file__', 'built-in')})")
    if missing:
        print("ERROR: Missing Florence-2 runtime imports:")
        for item in missing:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

