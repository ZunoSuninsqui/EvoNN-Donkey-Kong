"""Replay a recorded action sequence (best_inputs.txt) in the Donkey Kong game."""
from __future__ import annotations

import json
from pathlib import Path

import pygame

from Game.ExecuteGame import GameEnv


def load_actions(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("best_inputs.txt must contain a JSON list")
    return data


def main() -> None:
    path = Path("best_inputs.txt")
    if not path.exists():
        raise FileNotFoundError("best_inputs.txt not found. Run training first.")

    actions = load_actions(path)
    env = GameEnv(mode="ai", render=True, fast_mode=False, record=False)
    env.reset()

    running = True
    step_index = 0
    clock = pygame.time.Clock()

    while running and step_index < len(actions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        entry = actions[step_index]
        if isinstance(entry, dict):
            action = entry.get("action", 0)
        else:
            action = int(entry)

        _, _, done, info = env.step(action)
        step_index += 1

        if done:
            running = False

        clock.tick(60)

    # Keep the window open for a moment if the episode ended naturally.
    if env.render:
        pygame.time.delay(2000)
    pygame.quit()


if __name__ == "__main__":
    main()

