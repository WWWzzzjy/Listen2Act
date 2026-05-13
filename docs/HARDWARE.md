# Hardware

Target GPU: NVIDIA V100S 32GB.

Project constraints:

- fp16 mixed precision only.
- bf16 is disabled in configs because V100S does not support it efficiently.
- FlashAttention-2 is not used because Volta is incompatible.
- Attention implementation uses PyTorch SDPA where HuggingFace supports it.

Approximate memory budget:

- Florence-2 base/large model: 2-4GB depending on variant and optimizer state.
- LoRA optimizer states: about 1GB.
- Activations: about 10-15GB depending on sequence length and image batch.
- Evaluation/video buffers: up to 5GB if many frames are retained.
- Runtime overhead and fragmentation: leave several GB free.

If OOM occurs:

1. Use `microsoft/Florence-2-base`.
2. Reduce per-device batch size to 1.
3. Increase `gradient_accumulation_steps`.
4. Lower validation frequency or max validation batches.
5. Disable saving videos during evaluation sweeps.

