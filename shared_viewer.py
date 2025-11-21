"""Shared-map multi-agent viewer for NEAT training.

This module leaves the training loop and the core game engine untouched while
providing a lightweight visualization that places several genomes into a single
Donkey Kong map simultaneously. Each agent runs its own NEAT network and acts
independently on its own Player instance while sharing barrels, ladders, and the
same background.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import neat
import pygame

from Game.ExecuteGame import (
    Game,
    LADDER_ALIGN_TOLERANCE,
    PLAYER_COLOR,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STATE_GAME_OVER,
    STATE_PLAYING,
    STATE_WIN,
    Player,
    action_to_keys,
)


Color = Tuple[int, int, int]


def _argmax_action(outputs: Sequence[float]) -> int:
    return int(max(range(len(outputs)), key=lambda i: outputs[i]))


@dataclass
class VisualAgent:
    genome_id: int
    net: neat.nn.FeedForwardNetwork
    player: Player
    color: Color
    status: str = STATE_PLAYING

    @property
    def alive(self) -> bool:
        return self.status == STATE_PLAYING


class SharedGameViewer:
    """Render several genomes inside a single, shared Donkey Kong map."""

    def __init__(
        self,
        max_agents: int = 8,
        frames_per_generation: int = 1000,
        fps: int = 60,
        palette: Sequence[Color] | None = None,
    ) -> None:
        self.max_agents = max_agents
        self.frames_per_generation = frames_per_generation
        self.fps = fps
        self.palette = list(palette) if palette else self._default_palette()

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.game: Game | None = None

    def render_generation(self, genomes, config: neat.Config) -> None:
        if not genomes:
            return

        self._ensure_window()

        sorted_genomes = sorted(genomes, key=lambda g: g[1].fitness or 0.0, reverse=True)[
            : self.max_agents
        ]

        self.game = Game()
        agents = self._build_agents(sorted_genomes, config)

        for _ in range(self.frames_per_generation):
            self._process_events()
            self._advance_environment()

            active = False
            for agent in agents:
                if not agent.alive:
                    continue

                active = True
                if self.game.can_control:
                    self._apply_action(agent)

                agent.player.update(self.game.platforms, self.game.ladders, self.game.screen_rect)
                self._check_collisions(agent)

            self._draw_frame(agents)

            if not active:
                break

            self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _ensure_window(self) -> None:
        if self.screen is not None:
            return

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NEAT shared-map live viewer")
        self.clock = pygame.time.Clock()

    def _build_agents(self, genomes, config: neat.Config) -> List[VisualAgent]:
        assert self.game is not None
        start_pos = (self.game.player.rect.x, self.game.player.rect.y)
        agents: List[VisualAgent] = []

        for idx, (genome_id, genome) in enumerate(genomes):
            color = self.palette[idx % len(self.palette)]
            player = Player(start_pos)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            agents.append(VisualAgent(genome_id=genome_id, net=net, player=player, color=color))

        return agents

    def _default_palette(self) -> List[Color]:  # pragma: no cover - trivial
        r, g, b = PLAYER_COLOR
        return [
            (r, g, b),
            (230, 70, 70),
            (70, 200, 120),
            (70, 140, 220),
            (240, 190, 60),
            (200, 120, 200),
            (120, 200, 200),
            (200, 200, 120),
        ]

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def _process_events(self) -> None:
        if self.screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Keep the visualization alive even if the window close button is hit.
                pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    def _advance_environment(self) -> None:
        assert self.game is not None

        self.game.frame_count += 1
        if not self.game.can_control and self.game.frame_count >= self.game.start_delay_frames:
            self.game.can_control = True

        if self.game.frame_count % self.game.barrel_spawn_frames == 0:
            self.game.spawn_barrel()

        for barrel in list(self.game.barrels):
            barrel.update(self.game.platforms, self.game.screen_rect, self.game.turn_zones)
            if barrel.rect.top > SCREEN_HEIGHT:
                self.game.barrels.remove(barrel)

    def _apply_action(self, agent: VisualAgent) -> None:
        action = _argmax_action(agent.net.activate(self._agent_state(agent.player)))
        keys = action_to_keys(action)
        agent.player.handle_input(keys, self.game.ladders, self.game.platforms)

    def _check_collisions(self, agent: VisualAgent) -> None:
        if self.game is None:
            return

        for barrel in self.game.barrels:
            if agent.player.rect.colliderect(barrel.rect):
                agent.status = STATE_GAME_OVER
                return

        if agent.player.rect.colliderect(self.game.princess_rect):
            agent.status = STATE_WIN

    def _agent_state(self, player: Player) -> List[float]:
        assert self.game is not None

        px = player.rect.centerx / SCREEN_WIDTH
        py = player.rect.centery / SCREEN_HEIGHT
        is_grounded = 1.0 if player.on_ground else 0.0
        is_climbing = 1.0 if player.on_ladder else 0.0

        ladder, near_ladder = self._nearest_ladder(player)
        ladder_x_norm = ladder.centerx / SCREEN_WIDTH if ladder else 0.0
        is_near_ladder = 1.0 if near_ladder else 0.0

        barrel = self._nearest_barrel(player)
        if barrel:
            barrel_x = barrel.rect.centerx / SCREEN_WIDTH
            barrel_y = barrel.rect.centery / SCREEN_HEIGHT
            barrel_vx = barrel.vel_x / 10.0
            barrel_vy = barrel.vel_y / 10.0
        else:
            barrel_x = barrel_y = barrel_vx = barrel_vy = 0.0

        return [
            px,
            py,
            is_grounded,
            is_climbing,
            is_near_ladder,
            ladder_x_norm,
            barrel_x,
            barrel_y,
            barrel_vx,
            barrel_vy,
        ]

    def _nearest_ladder(self, player: Player):
        assert self.game is not None
        player_x = player.rect.centerx
        nearest = None
        best_dx = float("inf")
        for ladder in self.game.ladders:
            dx = abs(player_x - ladder.centerx)
            if dx < best_dx:
                best_dx = dx
                nearest = ladder
        return nearest, bool(best_dx <= LADDER_ALIGN_TOLERANCE * 1.5)

    def _nearest_barrel(self, player: Player):
        assert self.game is not None
        if not self.game.barrels:
            return None
        player_pos = player.rect.center
        return min(
            self.game.barrels,
            key=lambda b: (b.rect.centerx - player_pos[0]) ** 2 + (b.rect.centery - player_pos[1]) ** 2,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _draw_frame(self, agents: Sequence[VisualAgent]) -> None:
        assert self.screen is not None
        assert self.game is not None

        self.screen.fill((0, 0, 0))
        self.screen.blit(self.game.background, self.game.background_rect)

        for barrel in self.game.barrels:
            barrel.draw(self.screen)

        for agent in agents:
            outline = (0, 0, 0)
            pygame.draw.rect(self.screen, agent.color, agent.player.rect)
            pygame.draw.rect(self.screen, outline, agent.player.rect, width=2)

        pygame.display.flip()
