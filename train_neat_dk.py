"""NEAT training script for the Donkey Kong prototype.

Usage
-----
python train_neat_dk.py --generations 50
python train_neat_dk.py --visual       # slower, shows multiple agents at once
python train_neat_dk.py --live         # live grid with multiple agents per generation
python train_neat_dk.py --live-shared  # shared-map viewer with several agents together
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import List
import threading

import neat

from Game.ExecuteGame import GameEnv, SCREEN_HEIGHT, SCREEN_WIDTH
from shared_viewer import SharedGameViewer

BEST_RUN: dict = {"fitness": float("-inf"), "record": None, "genome": None}
RUN_VISUAL = False
RUN_LIVE = False
RUN_LIVE_SHARED = False
SHARED_VIEWER: SharedGameViewer | None = None


def argmax_action(outputs: List[float]) -> int:
    return int(max(range(len(outputs)), key=lambda i: outputs[i]))


def evaluate_single(genome: neat.DefaultGenome, config: neat.Config) -> GameEnv:
    env = GameEnv(mode="ai", render=False, fast_mode=True, record=True, show_debug=False)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = env.reset()
    done = False

    while not done:
        outputs = net.activate(state)
        action = argmax_action(outputs)
        state, _, done, _ = env.step(action)

    genome.fitness = env.total_reward

    if env.game and env.game.game_state == "WIN" and env.total_reward > BEST_RUN["fitness"]:
        BEST_RUN.update({"fitness": env.total_reward, "record": env.get_recorded_actions(), "genome": genome})

    return env


def render_top_agents(genomes, config: neat.Config, top_n: int = 4, frames: int = 600) -> None:
    """Visualize the best agents of the current generation side-by-side."""

    sorted_genomes = sorted(genomes, key=lambda g: g[1].fitness or 0.0, reverse=True)[:top_n]
    if not sorted_genomes:
        return

    cols = min(top_n, 2)
    rows = math.ceil(len(sorted_genomes) / cols)
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((cols * SCREEN_WIDTH, rows * SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT visual generation preview")
    clock = pygame.time.Clock()

    envs: List[GameEnv] = []
    nets = []
    positions = []

    for idx, (_, genome) in enumerate(sorted_genomes):
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        env = GameEnv(render=True, fast_mode=True, show_debug=False, surface=surf)
        env.reset()
        envs.append(env)
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        col = idx % cols
        row = idx // cols
        positions.append((col * SCREEN_WIDTH, row * SCREEN_HEIGHT))

    for _ in range(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((0, 0, 0))
        for env, net, pos in zip(envs, nets, positions):
            if not env.done:
                action = argmax_action(net.activate(env.get_state()))
                env.step(action)
            screen.blit(env.screen, pos)

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


def render_generation_live(
    genomes,
    config: neat.Config,
    stop_event: threading.Event,
    max_agents: int = 6,
    fps: int = 120,
) -> None:
    """Render several agents of the current generation simultaneously.

    This function runs in a background thread when ``--live`` is enabled so the
    training loop can continue without waiting for the visualization to finish.
    """

    import pygame

    if not genomes:
        return

    pygame.init()

    count = min(max_agents, len(genomes))
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    screen = pygame.display.set_mode((cols * SCREEN_WIDTH, rows * SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT live generation monitor")
    clock = pygame.time.Clock()

    envs: List[GameEnv] = []
    nets = []
    positions = []

    for idx, (_, genome) in enumerate(genomes[:count]):
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        env = GameEnv(mode="ai", render=True, fast_mode=True, record=False, show_debug=False, surface=surface)
        env.reset()
        envs.append(env)
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        col = idx % cols
        row = idx // cols
        positions.append((col * SCREEN_WIDTH, row * SCREEN_HEIGHT))

    running = True
    while running and not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                running = False
                break

        screen.fill((0, 0, 0))
        active = False
        for env, net, pos in zip(envs, nets, positions):
            if not env.done:
                active = True
                action = argmax_action(net.activate(env.get_state()))
                env.step(action)
            screen.blit(env.screen, pos)

        pygame.display.flip()
        if not active:
            break
        clock.tick(fps)

    # Do not pygame.quit() here to avoid interfering with headless environments
    # created elsewhere. Simply leave the display as-is; it will be reused or
    # closed when the process exits.


def eval_genomes(genomes, config):
    live_stop: threading.Event | None = None
    live_thread: threading.Thread | None = None

    if RUN_LIVE and not RUN_LIVE_SHARED:
        live_stop = threading.Event()
        live_thread = threading.Thread(
            target=render_generation_live, args=(genomes, config, live_stop), daemon=True
        )
        live_thread.start()

    try:
        for _, genome in genomes:
            genome.fitness = 0.0
            evaluate_single(genome, config)
    finally:
        if live_stop is not None:
            live_stop.set()
        if live_thread is not None:
            live_thread.join(timeout=1.0)

    if RUN_VISUAL:
        render_top_agents(genomes, config)

    if RUN_LIVE_SHARED:
        global SHARED_VIEWER
        if SHARED_VIEWER is None:
            SHARED_VIEWER = SharedGameViewer()
        SHARED_VIEWER.render_generation(genomes, config)


def save_artifacts(genome: neat.DefaultGenome, record: list | None) -> None:
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(genome, f)

    if record:
        with open("best_inputs.txt", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)


def main() -> None:
    global RUN_VISUAL, RUN_LIVE, RUN_LIVE_SHARED, LIVE_MAX_AGENTS

    parser = argparse.ArgumentParser(description="Train NEAT to play Donkey Kong")
    parser.add_argument("--config", default="config-neat-dk.ini", help="Path to NEAT config file")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to train")
    parser.add_argument("--visual", action="store_true", help="Render a small visual demo each generation")
    parser.add_argument("--live", action="store_true", help="Live grid of multiple agents per generation")
    parser.add_argument(
        "--live-shared",
        action="store_true",
        help="Show several agents together in a single shared Donkey Kong map",
    )
    args = parser.parse_args()

    RUN_VISUAL = args.visual
    global RUN_LIVE
    RUN_LIVE = args.live
    global RUN_LIVE_SHARED
    RUN_LIVE_SHARED = args.live_shared

    config_path = Path(args.config)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, args.generations)

    best_genome = BEST_RUN.get("genome") or winner

    if BEST_RUN.get("record") is None:
        # Evaluate the winning genome once more to capture its action trace.
        env = GameEnv(mode="ai", render=False, fast_mode=True, record=True, show_debug=False)
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        state = env.reset()
        done = False
        while not done:
            action = argmax_action(net.activate(state))
            state, _, done, _ = env.step(action)
        BEST_RUN.update({"record": env.get_recorded_actions(), "fitness": env.total_reward})

    save_artifacts(best_genome, BEST_RUN.get("record"))

    print("Training finished. Best fitness:", BEST_RUN.get("fitness"))
    print("Artifacts saved: best_genome.pkl, best_inputs.txt")


if __name__ == "__main__":
    main()

