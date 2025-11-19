"""Minimal deterministic Donkey Kong style environment.

The module exposes:
* ACTIONS: list of valid action strings.
* Game: core environment class.
* run_episode(actions): helper to execute an entire episode from a list of actions.

The game world is a discrete grid of 15 columns by 10 rows. Y=0 is the top row and
Y=9 the ground. Platforms and ladders are static and defined below. Barrels follow
predefined deterministic paths, so an identical list of inputs always leads to the
same outcome. This makes the environment suitable for supervised or reinforcement
learning setups that need determinism.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Tile encodings used by ``Game.get_state``.
TILE_EMPTY = 0
TILE_PLATFORM = 1
TILE_LADDER = 2
TILE_PLAYER = 3
TILE_BARREL = 4
TILE_GOAL = 5

# Public list of valid actions. ``WAIT`` is equivalent to doing nothing for one tick.
ACTIONS: List[str] = ["LEFT", "RIGHT", "UP", "DOWN", "JUMP", "WAIT"]

# Deterministic barrel schedule: spawn ticks and the shared path that each barrel follows.
# Using sparse spawn times makes it possible to clear the level with a carefully timed
# sequence of inputs while remaining fully deterministic.
BARREL_SPAWN_TICKS = [4, 24, 44]
BARREL_PATH = [
    (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
    (6, 2), (6, 3),
    (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3),
    (12, 4), (12, 5),
    (11, 5), (10, 5), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5), (4, 5), (3, 5),
    (3, 6), (3, 7),
    (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7),
    (11, 8), (11, 9),
    (10, 9), (9, 9), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (3, 9),
    (2, 9), (1, 9), (0, 9),
]


@dataclass
class Player:
    x: int
    y: int


@dataclass
class Barrel:
    """Barrel with a predefined deterministic path."""

    path_index: int = -1
    active: bool = True

    def position(self) -> Optional[tuple[int, int]]:
        if 0 <= self.path_index < len(BARREL_PATH):
            return BARREL_PATH[self.path_index]
        return None

    def advance(self) -> Optional[tuple[int, int]]:
        if not self.active:
            return None
        next_index = self.path_index + 1
        if next_index >= len(BARREL_PATH):
            self.active = False
            return None
        self.path_index = next_index
        return BARREL_PATH[self.path_index]


class Game:
    """Deterministic Donkey Kong inspired environment."""

    width = 15
    height = 10

    def __init__(self) -> None:
        # Sets of coordinates that contain platforms and ladders.
        self.platforms = self._build_platforms()
        self.ladders = self._build_ladders()
        self.goal = (13, 0)
        self.player_start = (1, 9)
        self.barrels: List[Barrel] = []
        self.reset()

    def _build_platforms(self) -> set[tuple[int, int]]:
        platforms: set[tuple[int, int]] = set()
        # Ground.
        for x in range(self.width):
            platforms.add((x, 9))
        # Additional rows (left-to-right ranges inclusive).
        for x in range(0, 12):
            platforms.add((x, 7))
        for x in range(3, self.width):
            platforms.add((x, 5))
        for x in range(0, 13):
            platforms.add((x, 3))
        for x in range(2, self.width):
            platforms.add((x, 1))
        return platforms

    def _build_ladders(self) -> set[tuple[int, int]]:
        ladders: set[tuple[int, int]] = set()
        ladder_specs = [
            (11, 7, 9),  # connect ground to row 7
            (3, 5, 7),
            (12, 3, 5),
            (6, 1, 3),
            (13, 0, 1),  # final climb to the goal
        ]
        for x, y_start, y_end in ladder_specs:
            for y in range(min(y_start, y_end), max(y_start, y_end) + 1):
                ladders.add((x, y))
        return ladders

    def reset(self) -> None:
        """Reset to the initial state."""
        self.player = Player(*self.player_start)
        self.barrels = []
        self.tick = 0
        self.done = False
        self.outcome: Optional[str] = None

    # ------------------------------------------------------------------
    # Helper predicates.
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_platform(self, x: int, y: int) -> bool:
        return (x, y) in self.platforms

    def _is_ladder(self, x: int, y: int) -> bool:
        return (x, y) in self.ladders

    def _is_walkable(self, x: int, y: int) -> bool:
        return self._is_platform(x, y) or self._is_ladder(x, y)

    # ------------------------------------------------------------------
    def step(self, action: str) -> Dict[str, object]:
        """Advance the game by one tick applying ``action``.

        Returns a dict containing ``state`` (see :meth:`get_state`), a boolean ``done`` and the
        ``reason`` when the episode has ended.
        """
        if self.done:
            raise RuntimeError("Game already finished. Call reset() before stepping again.")

        if action not in ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Valid actions: {ACTIONS}")

        self._apply_action(action)
        self._spawn_barrels()
        self._advance_barrels()
        self._check_collisions()

        self.tick += 1
        state = self.get_state()
        info = {"state": state, "done": self.done, "reason": self.outcome}
        return info

    def _apply_action(self, action: str) -> None:
        dx = dy = 0
        if action == "LEFT":
            dx = -1
        elif action == "RIGHT":
            dx = 1
        elif action == "UP":
            dy = -1
        elif action == "DOWN":
            dy = 1
        elif action == "JUMP":
            # Simple jump: move up one tile if the destination is walkable.
            dy = -1
        elif action == "WAIT":
            pass

        target_x = self.player.x + dx
        target_y = self.player.y + dy
        if not self._in_bounds(target_x, target_y):
            return

        if action in {"LEFT", "RIGHT"}:
            if self._is_walkable(target_x, target_y):
                self.player.x = target_x
                self.player.y = target_y
        elif action in {"UP", "DOWN", "JUMP"}:
            if self._is_ladder(self.player.x, self.player.y) or self._is_ladder(target_x, target_y):
                if self._is_walkable(target_x, target_y):
                    self.player.x = target_x
                    self.player.y = target_y
        else:  # WAIT
            return

    def _spawn_barrels(self) -> None:
        if self.tick in BARREL_SPAWN_TICKS:
            self.barrels.append(Barrel())

    def _advance_barrels(self) -> None:
        active: List[Barrel] = []
        for barrel in self.barrels:
            if not barrel.active:
                continue
            pos = barrel.advance()
            if pos is not None and barrel.active:
                active.append(barrel)
        self.barrels = active

    def _check_collisions(self) -> None:
        for barrel in self.barrels:
            pos = barrel.position()
            if pos is None:
                continue
            if pos == (self.player.x, self.player.y):
                self.done = True
                self.outcome = "dead"
                return
        if (self.player.x, self.player.y) == self.goal:
            self.done = True
            self.outcome = "completed"

    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, object]:
        """Return the current state as a dictionary.

        The dictionary contains:
        * ``grid``: matrix of ints using ``TILE_*`` constants.
        * ``player``: dict with ``x`` and ``y``.
        * ``barrels``: list of dicts (``x``, ``y``) for active barrels.
        * ``goal``: goal coordinates.
        * ``tick``: current tick number.
        """
        grid = [[TILE_EMPTY for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.platforms:
            grid[y][x] = TILE_PLATFORM
        for x, y in self.ladders:
            grid[y][x] = TILE_LADDER
        gx, gy = self.goal
        grid[gy][gx] = TILE_GOAL

        # Overlay barrels and player.
        for barrel in self.barrels:
            pos = barrel.position()
            if pos is None:
                continue
            x, y = pos
            if self._in_bounds(x, y):
                grid[y][x] = TILE_BARREL
        grid[self.player.y][self.player.x] = TILE_PLAYER

        barrel_state = []
        for barrel in self.barrels:
            pos = barrel.position()
            if pos is not None:
                barrel_state.append({"x": pos[0], "y": pos[1]})

        state = {
            "grid": grid,
            "player": {"x": self.player.x, "y": self.player.y},
            "barrels": barrel_state,
            "goal": {"x": gx, "y": gy},
            "tick": self.tick,
        }
        return state

    def render(self) -> None:
        symbol_map = {
            TILE_EMPTY: " ",
            TILE_PLATFORM: "=",
            TILE_LADDER: "H",
            TILE_PLAYER: "P",
            TILE_BARREL: "B",
            TILE_GOAL: "G",
        }
        state = self.get_state()["grid"]
        for y, row in enumerate(state):
            chars = "".join(symbol_map[val] for val in row)
            print(f"{y:02d}|{chars}")
        print()


def run_episode(actions: List[str]) -> Dict[str, object]:
    """Execute a full episode using ``actions``.

    Returns a dictionary describing the outcome with keys ``success`` (bool), ``steps`` (int),
    ``final_state`` (see :meth:`Game.get_state`) and ``reason`` (``completed`` | ``dead`` |
    ``out_of_actions``).
    """
    game = Game()
    steps = 0
    final_state = game.get_state()
    reason: Optional[str] = None
    for action in actions:
        info = game.step(action)
        final_state = info["state"]
        steps += 1
        if info["done"]:
            reason = info["reason"]
            break
    else:
        if not game.done:
            reason = "out_of_actions"
            game.outcome = reason

    success = reason == "completed"
    return {
        "success": success,
        "steps": steps,
        "final_state": final_state,
        "reason": reason,
    }


if __name__ == "__main__":
    # Example sequence that reaches the goal without touching any barrel.
    demo_actions = [
        "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT",
        "UP", "LEFT", "UP", "LEFT", "LEFT", "LEFT", "LEFT", "LEFT", "LEFT", "LEFT", "LEFT",
        "UP", "UP", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT",
        "UP", "UP", "LEFT", "LEFT", "LEFT", "LEFT", "LEFT", "LEFT", "UP", "UP",
        "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "UP",
    ]
    result = run_episode(demo_actions)
    print("Episode result:", result)
