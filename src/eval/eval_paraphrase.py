"""Evaluate held-out instruction paraphrases."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.data.instruction_paraphraser import load_paraphrases
from src.envs.libero_env import LiberoEnv
from src.eval.eval_libero import evaluate_task, load_policy_from_checkpoint
from src.utils.headless import enforce_headless
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def evaluate_paraphrases(
    policy: Any,
    suite_name: str,
    task_ids: list[int],
    paraphrases_path: str | Path,
    output_dir: str | Path,
    episodes_per_variant: int,
    max_episode_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate paraphrase variants for each task.

    Args:
        policy: Loaded policy.
        suite_name: LIBERO suite name.
        task_ids: Task IDs.
        paraphrases_path: Paraphrase JSON mapping.
        output_dir: Output directory.
        episodes_per_variant: Episodes per paraphrase.
        max_episode_steps: Maximum rollout length.
        seed: Base seed.

    Returns:
        JSON-serializable paraphrase evaluation results.
    """
    paraphrases = load_paraphrases(paraphrases_path) if Path(paraphrases_path).exists() else {}
    results: dict[str, Any] = {}
    for task_id in task_ids:
        env = LiberoEnv(suite_name=suite_name, task_id=task_id, seed=seed)
        base_instruction = env.instruction
        env.close()
        variants = paraphrases.get(base_instruction, {}).get("en", []) + paraphrases.get(
            base_instruction, {}
        ).get("zh", [])
        if not variants:
            variants = [base_instruction]
        variant_results = []
        for variant_idx, variant in enumerate(variants):
            task_result = evaluate_task(
                policy,
                suite_name,
                task_id,
                episodes_per_task=episodes_per_variant,
                max_episode_steps=max_episode_steps,
                output_dir=Path(output_dir) / f"task_{task_id:03d}" / f"variant_{variant_idx:03d}",
                seed=seed + variant_idx,
                instruction_override=str(variant),
            )
            task_result["instruction"] = str(variant)
            variant_results.append(task_result)
        results[str(task_id)] = {
            "base_instruction": base_instruction,
            "variants": variant_results,
            "mean_success_rate": sum(float(item["success_rate"]) for item in variant_results)
            / max(1, len(variant_results)),
        }
    return results


def main() -> None:
    """CLI entrypoint for paraphrase robustness evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate held-out paraphrase robustness.")
    parser.add_argument("--eval-config", default="configs/eval/libero_eval.yaml")
    parser.add_argument("--model-config", default="configs/model/florence2_base.yaml")
    parser.add_argument("--paraphrases-path", default="data/paraphrases/libero_object_paraphrases.json")
    parser.add_argument("--results-path", default="data/eval_videos/paraphrase_eval_results.json")
    parser.add_argument("--episodes-per-variant", type=int, default=5)
    args = parser.parse_args()
    setup_logging()
    enforce_headless()

    eval_values = OmegaConf.to_container(OmegaConf.load(args.eval_config), resolve=True)
    model_values = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
    if not isinstance(eval_values, dict) or not isinstance(model_values, dict):
        raise ValueError("Config files must contain mappings.")
    policy = load_policy_from_checkpoint(model_values, eval_values.get("checkpoint_path"))
    task_ids = [int(task_id) for task_id in (eval_values.get("task_ids") or list(range(10)))]
    results = evaluate_paraphrases(
        policy=policy,
        suite_name=str(eval_values.get("suite_name", "libero_object")),
        task_ids=task_ids,
        paraphrases_path=args.paraphrases_path,
        output_dir=eval_values.get("output_dir", "data/eval_videos"),
        episodes_per_variant=args.episodes_per_variant,
        max_episode_steps=int(eval_values.get("max_episode_steps", 600)),
        seed=int(eval_values.get("seed", 123)),
    )
    path = Path(args.results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote paraphrase evaluation results to %s", path)


if __name__ == "__main__":
    main()
