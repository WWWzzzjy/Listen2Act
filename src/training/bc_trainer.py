"""Behavioral cloning trainer for SimVoiceVLA."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split

from src.data.collator import collate_vla_batch
from src.data.libero_dataset import LiberoDataset
from src.models.florence2_vla import Florence2VLA
from src.training.losses import action_mse_loss
from src.training.schedulers import build_warmup_cosine_scheduler
from src.utils.checkpointing import save_checkpoint
from src.utils.logging_utils import JsonlLogger, setup_logging
from src.utils.seeding import seed_everything

LOGGER = logging.getLogger(__name__)
DEFAULT_VALIDATION_FRACTION = 0.05
DEFAULT_MAX_VALIDATION_BATCHES = 50


@dataclass
class BCTrainerConfig:
    """Configuration for single-GPU behavioral cloning."""

    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_train_steps: int = 50000
    eval_every_steps: int = 2000
    save_every_steps: int = 5000
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    precision: str = "fp16"
    lora_rank: int = 16
    lora_alpha: int = 32
    action_chunk_size: int = 8
    image_size: int = 224
    num_workers: int = 4
    log_every_steps: int = 20
    checkpoint_dir: str = "checkpoints/bc_v100s"
    seed: int = 42

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "BCTrainerConfig":
        """Create a config from a mapping.

        Args:
            values: Mapping parsed from YAML.

        Returns:
            Trainer configuration.
        """
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


class BCTrainer:
    """Single-GPU behavioral cloning trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: BCTrainerConfig,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: VLA model.
            train_loader: Training data loader.
            val_loader: Optional validation loader.
            config: Trainer config.
            device: Optional device override.
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        _cast_trainable_parameters_to_fp32(self.model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.num_train_steps,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metric_logger = JsonlLogger(Path(config.checkpoint_dir) / "train_metrics.jsonl")

    def train(self) -> None:
        """Run the behavioral cloning training loop."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        last_log_time = time.time()
        samples_since_log = 0

        while self.global_step < self.config.num_train_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.num_train_steps:
                    break
                try:
                    loss = self._forward_loss(batch) / self.config.gradient_accumulation_steps
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite training loss at step {self.global_step}")
                    self.scaler.scale(loss).backward()
                    micro_step += 1
                    samples_since_log += int(batch["images"].shape[0])
                    if micro_step % self.config.gradient_accumulation_steps == 0:
                        metrics = self._optimizer_step(loss.detach(), samples_since_log, last_log_time)
                        samples_since_log = 0
                        last_log_time = time.time()
                        self._maybe_log_and_checkpoint(metrics)
                except Exception as exc:
                    LOGGER.exception("Skipping failed training batch at step %d: %s", self.global_step, exc)
                    self.optimizer.zero_grad(set_to_none=True)
                    micro_step = 0

    def validate(self, max_batches: int = DEFAULT_MAX_VALIDATION_BATCHES) -> float:
        """Evaluate validation loss on a bounded number of batches.

        Args:
            max_batches: Maximum validation batches.

        Returns:
            Mean validation loss, or infinity if validation fails.
        """
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                try:
                    loss = self._forward_loss(batch)
                    if torch.isfinite(loss):
                        losses.append(float(loss.detach().cpu()))
                except Exception as exc:
                    LOGGER.warning("Validation batch %d failed: %s", batch_idx, exc)
        self.model.train()
        if not losses:
            return float("inf")
        return sum(losses) / len(losses)

    def _forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Move a batch to device, forward the model, and compute loss."""
        images = batch["images"].to(self.device, non_blocking=True)
        targets = batch["action_chunks"].to(self.device, non_blocking=True)
        instructions = batch["instructions"]
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda" and self.config.precision == "fp16",
        ):
            predicted = self.model(images, instructions)
            return action_mse_loss(predicted, targets)

    def _optimizer_step(
        self,
        scaled_loss: torch.Tensor,
        samples_since_log: int,
        last_log_time: float,
    ) -> dict[str, float]:
        """Apply optimizer, scheduler, gradient clipping, and collect metrics."""
        self.scaler.unscale_(self.optimizer)
        grad_norm = float(clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1
        elapsed = max(time.time() - last_log_time, 1e-6)
        lr = float(self.optimizer.param_groups[0]["lr"])
        memory_gb = (
            float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
            if self.device.type == "cuda"
            else 0.0
        )
        return {
            "step": float(self.global_step),
            "train_loss": float(scaled_loss.detach().cpu()) * self.config.gradient_accumulation_steps,
            "grad_norm": grad_norm,
            "lr": lr,
            "samples_per_sec": float(samples_since_log / elapsed),
            "gpu_memory_gb": memory_gb,
        }

    def _maybe_log_and_checkpoint(self, metrics: dict[str, float]) -> None:
        """Log metrics and save periodic or best checkpoints."""
        step = self.global_step
        if step % self.config.log_every_steps == 0:
            LOGGER.info(
                "step=%d loss=%.5f lr=%.6g grad_norm=%.3f samples/s=%.2f mem=%.2fGB",
                step,
                metrics["train_loss"],
                metrics["lr"],
                metrics["grad_norm"],
                metrics["samples_per_sec"],
                metrics["gpu_memory_gb"],
            )
            self.metric_logger.log(metrics)
        if step % self.config.eval_every_steps == 0:
            val_loss = self.validate()
            self.metric_logger.log({"step": step, "val_loss": val_loss})
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(self.model, Path(self.config.checkpoint_dir) / "best", step, {"val_loss": val_loss})
        if step % self.config.save_every_steps == 0:
            save_checkpoint(
                self.model,
                Path(self.config.checkpoint_dir) / f"step_{step:06d}",
                step,
                {"train_loss": metrics["train_loss"]},
            )


def build_dataloaders(
    data_config: dict[str, Any],
    train_config: BCTrainerConfig,
) -> tuple[DataLoader, DataLoader | None]:
    """Build train and validation dataloaders from config.

    Args:
        data_config: Data YAML mapping.
        train_config: Trainer configuration.

    Returns:
        Train loader and optional validation loader.
    """
    dataset = LiberoDataset(
        data_root=data_config.get("data_root", "data/libero"),
        action_chunk_size=int(data_config.get("action_chunk_size", train_config.action_chunk_size)),
        image_size=int(data_config.get("image_size", train_config.image_size)),
        translations_path=data_config.get("translations_path"),
        paraphrases_path=data_config.get("paraphrases_path"),
        language_sampling=data_config.get("language_sampling", "en"),
        use_paraphrases=bool(data_config.get("use_paraphrases", False)),
        seed=train_config.seed,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No LIBERO samples found under {data_config.get('data_root', 'data/libero')}. "
            "Run scripts/download_libero_demos.sh first."
        )
    val_size = max(1, int(len(dataset) * DEFAULT_VALIDATION_FRACTION))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_vla_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_vla_batch,
        drop_last=False,
    )
    return train_loader, val_loader


def _cast_trainable_parameters_to_fp32(model: torch.nn.Module) -> None:
    """Keep optimizer-owned parameters in fp32 for AMP GradScaler compatibility.

    Args:
        model: Training model.
    """
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()


def main() -> None:
    """CLI entrypoint for behavioral cloning training."""
    parser = argparse.ArgumentParser(description="Train SimVoiceVLA with behavioral cloning.")
    parser.add_argument("--training-config", default="configs/training/bc_v100s.yaml")
    parser.add_argument("--model-config", default="configs/model/florence2_base.yaml")
    parser.add_argument("--data-config", default="configs/data/libero_object.yaml")
    args = parser.parse_args()
    setup_logging()

    train_values = OmegaConf.to_container(OmegaConf.load(args.training_config), resolve=True)
    model_values = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
    data_values = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    if not isinstance(train_values, dict) or not isinstance(model_values, dict) or not isinstance(data_values, dict):
        raise ValueError("Config files must contain mappings.")
    config = BCTrainerConfig.from_mapping(train_values)
    seed_everything(config.seed)
    train_loader, val_loader = build_dataloaders(data_values, config)
    model_values = {**model_values, "lora_rank": config.lora_rank, "lora_alpha": config.lora_alpha}
    model = Florence2VLA.from_config(model_values)
    trainer = BCTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
