"""Verify CUDA availability and fp16 execution for V100S-style training."""

from __future__ import annotations

import sys

import torch


def main() -> int:
    """Run a small CUDA fp16 sanity check."""
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is false.")
        return 1
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"CUDA device: {props.name}")
    print(f"Total memory: {props.total_memory / (1024**3):.2f} GB")
    print("bf16: disabled by project policy")
    x = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    y = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    z = x @ y
    torch.cuda.synchronize()
    print(f"fp16 matmul OK, dtype={z.dtype}, mean={z.float().mean().item():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

