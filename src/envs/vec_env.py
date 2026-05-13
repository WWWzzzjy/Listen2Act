"""Multiprocessing vector wrapper for LIBERO evaluation."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from src.envs.libero_env import LiberoEnv

LOGGER = logging.getLogger(__name__)
DEFAULT_NUM_ENVS = 2


class VecLiberoEnv:
    """A small multiprocessing vector environment for evaluation."""

    def __init__(self, num_envs: int = DEFAULT_NUM_ENVS, env_kwargs: dict[str, Any] | None = None) -> None:
        """Launch worker environments.

        Args:
            num_envs: Number of parallel environments.
            env_kwargs: Keyword arguments passed to each ``LiberoEnv``.
        """
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs or {}
        self.ctx = mp.get_context("spawn")
        self.parents: list[Connection] = []
        self.processes: list[mp.Process] = []
        for idx in range(num_envs):
            parent, child = self.ctx.Pipe()
            kwargs = {**self.env_kwargs, "seed": int(self.env_kwargs.get("seed", 0)) + idx}
            process = self.ctx.Process(target=_worker, args=(child, kwargs), daemon=True)
            process.start()
            child.close()
            self.parents.append(parent)
            self.processes.append(process)

    def reset(self) -> list[dict[str, Any]]:
        """Reset all environments."""
        for parent in self.parents:
            parent.send(("reset", None))
        return [parent.recv() for parent in self.parents]

    def step(self, actions: list[np.ndarray]) -> list[tuple[dict[str, Any], float, bool, dict[str, Any]]]:
        """Step all environments.

        Args:
            actions: List of actions, one per environment.

        Returns:
            List of gym-style step tuples.
        """
        for parent, action in zip(self.parents, actions, strict=True):
            parent.send(("step", action))
        return [parent.recv() for parent in self.parents]

    def close(self) -> None:
        """Close all worker environments."""
        for parent in self.parents:
            try:
                parent.send(("close", None))
            except Exception as exc:
                LOGGER.warning("Failed to send close to worker: %s", exc)
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()


def _worker(connection: Connection, env_kwargs: dict[str, Any]) -> None:
    """Worker loop for one LIBERO environment."""
    env = LiberoEnv(**env_kwargs)
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(env.reset())
            elif command == "step":
                connection.send(env.step(payload))
            elif command == "close":
                env.close()
                connection.close()
                break
            else:
                connection.send({"error": f"Unknown command: {command}"})
    except EOFError:
        env.close()
