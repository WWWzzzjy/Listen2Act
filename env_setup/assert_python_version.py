"""Fail fast when the active Python version is unsupported."""

from __future__ import annotations

import sys

MIN_VERSION = (3, 10)
MAX_EXCLUSIVE_VERSION = (3, 12)


def main() -> int:
    """Validate the active Python interpreter version."""
    version = sys.version_info[:2]
    if MIN_VERSION <= version < MAX_EXCLUSIVE_VERSION:
        return 0
    print(
        "ERROR: SimVoiceVLA requires Python >=3.10,<3.12, "
        f"but active Python is {version[0]}.{version[1]} at {sys.executable}.\n"
        "Activate the project environment first, for example:\n"
        "  conda activate py310\n"
        "  source .env.headless",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

