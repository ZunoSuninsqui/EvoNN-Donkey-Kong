"""Pygame-based visualizer for the deterministic Donkey Kong environment.

This module focuses on a single static level and is meant to be deterministic so
that it can be used during neural-network training. The rendering derives
entirely from the environment state returned by :class:`dk_game.game.Game`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pygame

from dk_game.game import (
    ACTIONS,
    TILE_BARREL,
    TILE_GOAL,
    TILE_LADDER,
    TILE_PLATFORM,
    TILE_PLAYER,
    Game,
)

# Screen and tile configuration.
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 48  # Each grid cell maps to a 48x48 pixel tile.
MARGIN = 20  # Padding around the grid to center it nicely.
FPS = 60

# Basic palette.
BG_COLOR = (15, 15, 35)
PLATFORM_COLOR = (200, 90, 50)
LADDER_COLOR = (70, 160, 220)
PLAYER_COLOR = (80, 230, 120)
PLAYER_ALT_COLOR = (120, 255, 160)
BARREL_COLOR = (230, 200, 60)
GOAL_COLOR = (230, 120, 200)
TEXT_COLOR = (230, 230, 230)


def _cell_to_pixel(x: int, y: int) -> pygame.Rect:
    """Convert grid coordinates to a pygame.Rect in screen space."""
    return pygame.Rect(MARGIN + x * TILE_SIZE, MARGIN + y * TILE_SIZE, TILE_SIZE, TILE_SIZE)


@dataclass
class Renderer:
    """Minimal renderer for the fixed Donkey-Kong-style level."""

    surface: pygame.Surface
    font: pygame.font.Font
    tick_font: pygame.font.Font

    def render(self, state: Dict[str, object], outcome: Optional[str] = None) -> None:
        """Draw the full scene using the provided ``state`` dictionary."""
        self.surface.fill(BG_COLOR)
        grid: List[List[int]] = state["grid"]  # type: ignore[index]

        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                rect = _cell_to_pixel(x, y)
                if tile == TILE_PLATFORM:
                    pygame.draw.rect(self.surface, PLATFORM_COLOR, rect)
                elif tile == TILE_LADDER:
                    ladder_rect = rect.inflate(-TILE_SIZE * 0.6, 0)
                    pygame.draw.rect(self.surface, LADDER_COLOR, ladder_rect)
                elif tile == TILE_BARREL:
                    center = rect.center
                    pygame.draw.circle(self.surface, BARREL_COLOR, center, TILE_SIZE // 3)
                elif tile == TILE_PLAYER:
                    # Slight color shift based on tick for a subtle animation.
                    color = PLAYER_COLOR if (state["tick"] % 20) < 10 else PLAYER_ALT_COLOR  # type: ignore[index]
                    player_rect = rect.inflate(-TILE_SIZE * 0.15, -TILE_SIZE * 0.15)
                    pygame.draw.rect(self.surface, color, player_rect, border_radius=6)
                elif tile == TILE_GOAL:
                    goal_rect = rect.inflate(-TILE_SIZE * 0.2, -TILE_SIZE * 0.2)
                    pygame.draw.rect(self.surface, GOAL_COLOR, goal_rect, border_radius=4)

        self._draw_grid_outline(grid)
        self._draw_labels(state, outcome)
        pygame.display.flip()

    def _draw_grid_outline(self, grid: List[List[int]]) -> None:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        total_width = cols * TILE_SIZE
        total_height = rows * TILE_SIZE
        outer_rect = pygame.Rect(MARGIN, MARGIN, total_width, total_height)
        pygame.draw.rect(self.surface, (60, 60, 90), outer_rect, width=2)

    def _draw_labels(self, state: Dict[str, object], outcome: Optional[str]) -> None:
        tick_surface = self.tick_font.render(f"Tick: {state['tick']}", True, TEXT_COLOR)
        self.surface.blit(tick_surface, (MARGIN, SCREEN_HEIGHT - tick_surface.get_height() - 10))

        if outcome:
            text = f"Outcome: {outcome}"
            outcome_surface = self.font.render(text, True, TEXT_COLOR)
            rect = outcome_surface.get_rect(center=(SCREEN_WIDTH // 2, 20))
            self.surface.blit(outcome_surface, rect)


def _init_pygame() -> pygame.Surface:
    pygame.init()
    pygame.display.set_caption("Deterministic Donkey Kong Visualizer")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    return screen


def _action_from_keys(pressed: pygame.key.ScancodeWrapper) -> str:
    """Translate currently pressed keys into a game action."""
    if pressed[pygame.K_LEFT]:
        return "LEFT"
    if pressed[pygame.K_RIGHT]:
        return "RIGHT"
    if pressed[pygame.K_UP]:
        return "UP"
    if pressed[pygame.K_DOWN]:
        return "DOWN"
    if pressed[pygame.K_SPACE]:
        return "JUMP"
    return "WAIT"


def run_human_loop() -> None:
    """Run an interactive loop with keyboard controls for debugging the environment."""
    screen = _init_pygame()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 22)
    tick_font = pygame.font.SysFont("consolas", 18)
    renderer = Renderer(screen, font, tick_font)

    game = Game()
    running = True
    info: Dict[str, object] = {"state": game.get_state(), "done": False, "reason": None}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()
                info = {"state": game.get_state(), "done": False, "reason": None}

        action = _action_from_keys(pygame.key.get_pressed())
        if not info["done"]:  # type: ignore[index]
            info = game.step(action)
        renderer.render(info["state"], outcome=info.get("reason"))  # type: ignore[arg-type]
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


def run_action_sequence(actions: Iterable[str], fps: int = FPS) -> Dict[str, object]:
    """Replay a predefined list of actions with rendering.

    This is useful to visualize the instructions produced by a neural network. If the
    environment finishes early the function returns immediately with the final ``info``.
    """
    screen = _init_pygame()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 22)
    tick_font = pygame.font.SysFont("consolas", 18)
    renderer = Renderer(screen, font, tick_font)

    game = Game()
    info: Dict[str, object] = {"state": game.get_state(), "done": False, "reason": None}
    for action in actions:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Valid actions: {ACTIONS}")
        if not info["done"]:  # type: ignore[index]
            info = game.step(action)
        renderer.render(info["state"], outcome=info.get("reason"))  # type: ignore[arg-type]
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if info["done"]:  # type: ignore[index]
            break
    return info


if __name__ == "__main__":
    # Example usage: run a human-controlled loop.
    # Press arrow keys to move, space to jump, and R to reset the level.
    run_human_loop()
