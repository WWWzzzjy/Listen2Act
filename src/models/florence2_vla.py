"""Florence-2 based VLA policy with LoRA adapters and an action head."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from src.models.action_head import ActionHead
from src.models.tokenization import ensure_tokenizer_padding, normalize_instruction

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "microsoft/Florence-2-base"
DEFAULT_ACTION_CHUNK_SIZE = 8
DEFAULT_ACTION_DIM = 7
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")


class Florence2VLA(nn.Module):
    """Florence-2 VLM plus a lightweight chunked action head."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = None,
        action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
        action_dim: int = DEFAULT_ACTION_DIM,
        action_head_hidden_dim: int = 1024,
        action_head_depth: int = 3,
        lora_rank: int = DEFAULT_LORA_RANK,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
        trust_remote_code: bool = True,
        load_backbone: bool = True,
        feature_dim: int | None = None,
    ) -> None:
        """Initialize the VLA model.

        Args:
            model_name: HuggingFace Florence-2 model identifier.
            model_revision: Optional pinned HuggingFace revision for remote code safety.
            action_chunk_size: Number of actions predicted per model call.
            action_dim: Action dimension.
            action_head_hidden_dim: Hidden width for the MLP action head.
            action_head_depth: Number of linear layers in the action head.
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout probability.
            lora_target_modules: Attention projection modules to adapt.
            trust_remote_code: Whether to allow Florence remote model code.
            load_backbone: Whether to load HuggingFace weights immediately.
            feature_dim: Optional feature dimension for tests or custom backbones.
        """
        super().__init__()
        self.model_name = model_name
        self.model_revision = model_revision
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.processor: Any | None = None
        self.backbone: nn.Module | None = None
        self._cached_chunk: torch.Tensor | None = None
        self._cached_index = 0

        if load_backbone:
            self.processor, self.backbone = self._load_backbone_with_lora(
                model_name=model_name,
                model_revision=model_revision,
                trust_remote_code=trust_remote_code,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
            )
            resolved_dim = _resolve_hidden_dim(self.backbone)
        else:
            resolved_dim = feature_dim
        if resolved_dim is None:
            raise ValueError("Could not resolve VLM hidden dimension.")

        self.action_head = ActionHead(
            input_dim=resolved_dim,
            action_chunk_size=action_chunk_size,
            action_dim=action_dim,
            hidden_dim=action_head_hidden_dim,
            depth=action_head_depth,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Florence2VLA":
        """Build a model from a dictionary-like config.

        Args:
            config: Model configuration.

        Returns:
            Initialized model.
        """
        return cls(
            model_name=config.get("model_name", DEFAULT_MODEL_NAME),
            model_revision=config.get("model_revision"),
            action_chunk_size=int(config.get("action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE)),
            action_dim=int(config.get("action_dim", DEFAULT_ACTION_DIM)),
            action_head_hidden_dim=int(config.get("action_head_hidden_dim", 1024)),
            action_head_depth=int(config.get("action_head_depth", 3)),
            lora_rank=int(config.get("lora_rank", DEFAULT_LORA_RANK)),
            lora_alpha=int(config.get("lora_alpha", DEFAULT_LORA_ALPHA)),
            lora_dropout=float(config.get("lora_dropout", DEFAULT_LORA_DROPOUT)),
            lora_target_modules=tuple(config.get("lora_target_modules", DEFAULT_TARGET_MODULES)),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
        )

    def forward(self, images: Any, instructions: Sequence[str]) -> torch.Tensor:
        """Predict action chunks from images and language instructions.

        Args:
            images: Batch image tensor or processor-compatible image list.
            instructions: Natural-language instructions.

        Returns:
            Tensor of shape ``(batch, H, 7)``.
        """
        if self.backbone is None:
            raise RuntimeError("Florence2VLA was initialized without a backbone.")
        model_inputs = self._prepare_inputs(images, instructions)
        outputs = self.backbone(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = _extract_hidden(outputs)
        attention_mask = model_inputs.get("attention_mask")
        pooled = _masked_mean_pool(hidden, attention_mask)
        return self.action_head(pooled)

    @torch.no_grad()
    def predict(self, image: Any, instruction: str) -> torch.Tensor:
        """Predict one rollout action from a chunk cache.

        Args:
            image: Single image observation.
            instruction: Language instruction.

        Returns:
            Tensor with shape ``(7,)``.
        """
        if self._cached_chunk is None or self._cached_index >= self.action_chunk_size:
            chunk = self.forward(_single_image_batch(image), [instruction])
            self._cached_chunk = chunk[0].detach().cpu()
            self._cached_index = 0
        action = self._cached_chunk[self._cached_index]
        self._cached_index += 1
        return action

    def reset_rollout(self) -> None:
        """Clear cached chunk state before a new environment rollout."""
        self._cached_chunk = None
        self._cached_index = 0

    def save_pretrained(self, path: str) -> None:
        """Save LoRA-enabled backbone adapters when supported.

        Args:
            path: Output directory.
        """
        if self.backbone is not None and hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(path)

    def _load_backbone_with_lora(
        self,
        model_name: str,
        model_revision: str | None,
        trust_remote_code: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
    ) -> tuple[Any, nn.Module]:
        """Load Florence-2 and apply LoRA adapters."""
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError("Install transformers and peft to load Florence2VLA.") from exc

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=model_revision,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            ensure_tokenizer_padding(tokenizer)
        backbone = _load_model_with_safetensors_fallback(
            AutoModelForCausalLM,
            model_name=model_name,
            revision=model_revision,
            trust_remote_code=trust_remote_code,
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(lora_target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
        LOGGER.info("Loaded %s with LoRA rank=%d alpha=%d", model_name, lora_rank, lora_alpha)
        return processor, backbone

    def _prepare_inputs(self, images: Any, instructions: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize text and prepare image tensors for the backbone."""
        if self.processor is None:
            raise RuntimeError("Processor is unavailable.")
        normalized = [normalize_instruction(text) for text in instructions]
        device = next(self.parameters()).device
        if isinstance(images, torch.Tensor):
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                encoded = self.processor(text=normalized, return_tensors="pt", padding=True)
            else:
                encoded = tokenizer(normalized, return_tensors="pt", padding=True)
            encoded["pixel_values"] = images
        else:
            encoded = self.processor(text=normalized, images=images, return_tensors="pt", padding=True)
        return {key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def _resolve_hidden_dim(model: nn.Module | None) -> int | None:
    """Resolve a likely hidden dimension from a HuggingFace model config."""
    if model is None:
        return None
    config = getattr(model, "config", None)
    candidates = (
        "hidden_size",
        "d_model",
        "text_config.hidden_size",
        "vision_config.hidden_size",
        "base_model.config.hidden_size",
        "base_model.config.d_model",
        "base_model.model.config.hidden_size",
        "base_model.model.config.d_model",
        "projection_dim",
    )
    for candidate in candidates:
        value = _get_nested_attr(config, candidate)
        if isinstance(value, int):
            return value
    return None


def _get_nested_attr(obj: Any, path: str) -> Any:
    """Resolve dotted attributes from an object."""
    current = obj
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _extract_hidden(outputs: Any) -> torch.Tensor:
    """Extract the last hidden state from common HuggingFace output layouts."""
    for attr in ("encoder_last_hidden_state", "last_hidden_state"):
        hidden = getattr(outputs, attr, None)
        if hidden is not None:
            return hidden
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    if isinstance(outputs, dict):
        for key in ("encoder_last_hidden_state", "last_hidden_state", "hidden_states"):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)) and value:
                return value[-1]
    raise RuntimeError("Could not extract hidden states from Florence-2 outputs.")


def _load_model_with_safetensors_fallback(
    auto_model_cls: Any,
    model_name: str,
    revision: str | None,
    trust_remote_code: bool,
) -> nn.Module:
    """Load Florence-2 and fall back when safetensors metadata is absent.

    Args:
        auto_model_cls: HuggingFace auto model class.
        model_name: Model identifier.
        revision: Optional pinned revision.
        trust_remote_code: Whether remote code is allowed.

    Returns:
        Loaded model.
    """
    backbone_kwargs = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",
    }
    try:
        return auto_model_cls.from_pretrained(model_name, **backbone_kwargs)
    except AttributeError as exc:
        if "'NoneType' object has no attribute 'get'" not in str(exc):
            raise
        LOGGER.warning(
            "Safetensors metadata was missing for %s; retrying with pytorch_model.bin.",
            model_name,
        )
        return auto_model_cls.from_pretrained(
            model_name,
            **backbone_kwargs,
            use_safetensors=False,
        )


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool hidden states using an attention mask when lengths match."""
    if attention_mask is None or attention_mask.shape[-1] != hidden.shape[1]:
        return hidden.mean(dim=1)
    mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def _single_image_batch(image: Any) -> Any:
    """Wrap a single image into a batch shape accepted by ``forward``."""
    if isinstance(image, torch.Tensor):
        return image.unsqueeze(0) if image.ndim == 3 else image
    return [image]
