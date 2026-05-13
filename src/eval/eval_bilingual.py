"""Evaluate English and Chinese instructions separately."""

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

from src.data.instruction_translator import load_instruction_mapping
from src.envs.libero_env import LiberoEnv
from src.eval.eval_libero import evaluate_task, load_policy_from_checkpoint
from src.utils.headless import enforce_headless
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def evaluate_bilingual(
    policy: Any,
    suite_name: str,
    task_ids: list[int],
    translations_path: str | Path,
    output_dir: str | Path,
    episodes_per_task: int,
    max_episode_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run separate English and Chinese evaluation sweeps.

    Args:
        policy: Loaded policy.
        suite_name: LIBERO suite name.
        task_ids: Task IDs.
        translations_path: English-to-Chinese mapping path.
        output_dir: Output directory.
        episodes_per_task: Episodes per task and language.
        max_episode_steps: Maximum rollout length.
        seed: Base seed.

    Returns:
        Bilingual evaluation results with absolute language gap.
    """
    translations = load_instruction_mapping(translations_path) if Path(translations_path).exists() else {}
    language_results: dict[str, dict[str, Any]] = {"en": {}, "zh": {}}
    for task_id in task_ids:
        env = LiberoEnv(suite_name=suite_name, task_id=task_id, seed=seed)
        english = env.instruction
        env.close()
        language_results["en"][str(task_id)] = evaluate_task(
            policy,
            suite_name,
            task_id,
            episodes_per_task=episodes_per_task,
            max_episode_steps=max_episode_steps,
            output_dir=Path(output_dir) / "en",
            seed=seed,
            instruction_override=english,
        )
        language_results["zh"][str(task_id)] = evaluate_task(
            policy,
            suite_name,
            task_id,
            episodes_per_task=episodes_per_task,
            max_episode_steps=max_episode_steps,
            output_dir=Path(output_dir) / "zh",
            seed=seed,
            instruction_override=translations.get(english, english),
        )
    en_rate = _aggregate_rate(language_results["en"])
    zh_rate = _aggregate_rate(language_results["zh"])
    return {
        "en": language_results["en"],
        "zh": language_results["zh"],
        "aggregate": {
            "en_success_rate": en_rate,
            "zh_success_rate": zh_rate,
            "absolute_gap": abs(en_rate - zh_rate),
        },
    }


def _aggregate_rate(task_results: dict[str, Any]) -> float:
    """Compute mean task success rate."""
    if not task_results:
        return 0.0
    return sum(float(result["success_rate"]) for result in task_results.values()) / len(task_results)


def main() -> None:
    """CLI entrypoint for bilingual evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate EN vs ZH instruction success rates.")
    parser.add_argument("--eval-config", default="configs/eval/libero_eval.yaml")
    parser.add_argument("--model-config", default="configs/model/florence2_base.yaml")
    parser.add_argument("--translations-path", default="data/instructions_zh/libero_object_zh.json")
    parser.add_argument("--results-path", default="data/eval_videos/bilingual_eval_results.json")
    args = parser.parse_args()
    setup_logging()
    enforce_headless()

    eval_values = OmegaConf.to_container(OmegaConf.load(args.eval_config), resolve=True)
    model_values = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
    if not isinstance(eval_values, dict) or not isinstance(model_values, dict):
        raise ValueError("Config files must contain mappings.")
    policy = load_policy_from_checkpoint(model_values, eval_values.get("checkpoint_path"))
    task_ids = [int(task_id) for task_id in (eval_values.get("task_ids") or list(range(10)))]
    results = evaluate_bilingual(
        policy=policy,
        suite_name=str(eval_values.get("suite_name", "libero_object")),
        task_ids=task_ids,
        translations_path=args.translations_path,
        output_dir=eval_values.get("output_dir", "data/eval_videos"),
        episodes_per_task=int(eval_values.get("episodes_per_task", 20)),
        max_episode_steps=int(eval_values.get("max_episode_steps", 600)),
        seed=int(eval_values.get("seed", 123)),
    )
    path = Path(args.results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote bilingual evaluation results to %s", path)


if __name__ == "__main__":
    main()
