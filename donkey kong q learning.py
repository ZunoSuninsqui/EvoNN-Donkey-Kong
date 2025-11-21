
from __future__ import annotations

import pygame
import numpy as np
import random
import json
from typing import List, Tuple
from dataclasses import dataclass
import os

GRAVITY = 0.7
MOVE_SPEED = 3
LADDER_SPEED = 3
JUMP_FORCE = 12
MAX_FALL_SPEED = 12
LADDER_ALIGN_TOLERANCE = 18
SMALL_STEP_HEIGHT = 12

BARREL_SPAWN_INTERVAL_SECONDS = 2.0
BARREL_SPEED_X = 3
BARREL_SIZE = 22
START_DELAY_SECONDS = 0.5

SCREEN_WIDTH = 812
SCREEN_HEIGHT = 782
PLAYER_WIDTH = 28
PLAYER_HEIGHT = 32
PLAYER_COLOR = (50, 100, 255)
FPS = 60

STATE_MENU = "MENU"
STATE_PLAYING = "PLAYING"
STATE_TRAINING = "TRAINING"
STATE_AI_DEMO = "AI_DEMO"
STATE_GAME_OVER = "GAME_OVER"
STATE_WIN = "WIN"

BACKGROUND_PATH = r"C:\Users\julia\Desktop\Imagen de WhatsApp 2025-11-20 a las 19.58.56_f7958190.jpg"
AI_WEIGHTS_PATH = "donkey_kong_ai_weights.json"


@dataclass
class TrainingStats:
    episodes: int = 0
    wins: int = 0
    best_score: int = 0
    avg_score: float = 0.0
    epsilon: float = 1.0


class NeuralNetwork:
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2
    
    def predict(self, state):
        state_array = np.array(state).reshape(1, -1)
        q_values = self.forward(state_array)
        return q_values[0]
    
    def save_weights(self, filepath: str):
        weights = {
            'w1': self.w1.tolist(),
            'b1': self.b1.tolist(),
            'w2': self.w2.tolist(),
            'b2': self.b2.tolist(),
        }
        with open(filepath, 'w') as f:
            json.dump(weights, f)
    
    def load_weights(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                weights = json.load(f)
            self.w1 = np.array(weights['w1'])
            self.b1 = np.array(weights['b1'])
            self.w2 = np.array(weights['w2'])
            self.b2 = np.array(weights['b2'])
            return True
        return False


class ReplayBuffer:
    
    def __init__(self, max_size: int = 2000):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def size(self):
        return len(self.buffer)


class AIAgent:
    
    ACTIONS = [
        [],
        [pygame.K_LEFT],
        [pygame.K_RIGHT],
        [pygame.K_SPACE],
        [pygame.K_UP],
        [pygame.K_DOWN],
        [pygame.K_LEFT, pygame.K_SPACE],
        [pygame.K_RIGHT, pygame.K_SPACE],
    ]
    
    def __init__(self):
        input_size = 11
        hidden_size = 64
        output_size = len(self.ACTIONS)
        
        self.network = NeuralNetwork(input_size, hidden_size, output_size)
        self.replay_buffer = ReplayBuffer(max_size=2000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        self.stats = TrainingStats()
        self.current_episode_score = 0
        self.last_distance_to_princess = float('inf')
        
    def get_state(self, game) -> List[float]:
        player = game.player
        
        closest_barrel_x, closest_barrel_y, closest_barrel_vel_x = 0, 0, 0
        if game.barrels:
            closest = min(game.barrels, 
                         key=lambda b: ((b.rect.x - player.rect.x)**2 + 
                                       (b.rect.y - player.rect.y)**2)**0.5)
            closest_barrel_x = (closest.rect.x - player.rect.x) / SCREEN_WIDTH
            closest_barrel_y = (closest.rect.y - player.rect.y) / SCREEN_HEIGHT
            closest_barrel_vel_x = closest.vel_x / BARREL_SPEED_X
        else:
            closest_barrel_x = 0
            closest_barrel_y = 1
        
        state = [
            player.rect.x / SCREEN_WIDTH,
            player.rect.y / SCREEN_HEIGHT,
            player.vel_y / MAX_FALL_SPEED,
            1.0 if player.on_ground else 0.0,
            1.0 if player.on_ladder else 0.0,
            closest_barrel_x,
            closest_barrel_y,
            closest_barrel_vel_x,
            game.princess_rect.x / SCREEN_WIDTH,
            game.princess_rect.y / SCREEN_HEIGHT,
            self._distance_to_princess(player, game.princess_rect) / (SCREEN_WIDTH + SCREEN_HEIGHT),
        ]
        return state
    
    def _distance_to_princess(self, player, princess_rect):
        return abs(player.rect.x - princess_rect.x) + abs(player.rect.y - princess_rect.y)
    
    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, len(self.ACTIONS) - 1)
        
        q_values = self.network.predict(state)
        return np.argmax(q_values)
    
    def calculate_reward(self, game, prev_state, action) -> float:
        reward = 0.01
        
        current_distance = self._distance_to_princess(game.player, game.princess_rect)
        if current_distance < self.last_distance_to_princess:
            reward += 0.5
        elif current_distance > self.last_distance_to_princess:
            reward -= 0.1
        self.last_distance_to_princess = current_distance
        
        if game.player.rect.y < 400:
            reward += 0.3
        
        if game.game_state == STATE_GAME_OVER:
            reward = -10.0
        
        if game.game_state == STATE_WIN:
            reward = 100.0
        
        return reward
    
    def train_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_array = np.array(state).reshape(1, -1)
            next_state_array = np.array(next_state).reshape(1, -1)
            
            current_q = self.network.forward(state_array)[0]
            target_q = current_q.copy()
            
            if done:
                target_q[action] = reward
            else:
                next_q = self.network.forward(next_state_array)[0]
                target_q[action] = reward + self.gamma * np.max(next_q)
            
            self._update_weights(state_array, target_q.reshape(1, -1))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_weights(self, state, target_q):
        z1 = np.dot(state, self.network.w1) + self.network.b1
        a1 = self.network.relu(z1)
        z2 = np.dot(a1, self.network.w2) + self.network.b2
        
        dz2 = z2 - target_q
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.network.w2.T)
        dz1 = da1 * (z1 > 0)
        dw1 = np.dot(state.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        self.network.w2 -= self.learning_rate * dw2
        self.network.b2 -= self.learning_rate * db2
        self.network.w1 -= self.learning_rate * dw1
        self.network.b1 -= self.learning_rate * db1
    
    def save_model(self):
        self.network.save_weights(AI_WEIGHTS_PATH)
        print(f"Modelo guardado en {AI_WEIGHTS_PATH}")
    
    def load_model(self):
        return self.network.load_weights(AI_WEIGHTS_PATH)


class Player:
    def __init__(self, start_pos: tuple[int, int]):
        self.rect = pygame.Rect(start_pos[0], start_pos[1], PLAYER_WIDTH, PLAYER_HEIGHT)
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.on_ladder = False

    def _current_ladder(self, ladders: List[pygame.Rect]) -> pygame.Rect | None:
        for ladder in ladders:
            if self.rect.colliderect(ladder):
                return ladder
        return None

    def handle_input(self, keys, ladders: List[pygame.Rect], platforms: List[pygame.Rect]) -> None:
        ladder = self._current_ladder(ladders)

        if self.on_ladder and ladder is None:
            self.on_ladder = False

        wants_climb = ladder and (keys[pygame.K_UP] or keys[pygame.K_DOWN])
        self.vel_x = 0

        if self.on_ladder:
            touching_platform = any(self.rect.colliderect(p) for p in platforms)
            if touching_platform:
                if ladder is not None:
                    self.rect.centerx = ladder.centerx
                if keys[pygame.K_UP]:
                    self.vel_y = -LADDER_SPEED
                elif keys[pygame.K_DOWN]:
                    self.vel_y = LADDER_SPEED
                else:
                    self.vel_y = 0
                return
            else:
                if keys[pygame.K_LEFT]:
                    self.on_ladder = False
                    self.vel_x = -MOVE_SPEED
                elif keys[pygame.K_RIGHT]:
                    self.on_ladder = False
                    self.vel_x = MOVE_SPEED
                else:
                    if ladder is not None:
                        self.rect.centerx = ladder.centerx
                    if keys[pygame.K_UP]:
                        self.vel_y = -LADDER_SPEED
                    elif keys[pygame.K_DOWN]:
                        self.vel_y = LADDER_SPEED
                    else:
                        self.vel_y = 0
                    return

        if keys[pygame.K_LEFT]:
            self.vel_x = -MOVE_SPEED
        if keys[pygame.K_RIGHT]:
            self.vel_x = MOVE_SPEED if self.vel_x == 0 else self.vel_x

        if wants_climb and ladder and abs(self.rect.centerx - ladder.centerx) <= LADDER_ALIGN_TOLERANCE:
            self.on_ladder = True
            self.on_ground = False
            self.rect.centerx = ladder.centerx
            if keys[pygame.K_UP]:
                self.vel_y = -LADDER_SPEED
            elif keys[pygame.K_DOWN]:
                self.vel_y = LADDER_SPEED
            else:
                self.vel_y = 0
            return

        if keys[pygame.K_SPACE] and self.on_ground:
            self.vel_y = -JUMP_FORCE

    def apply_gravity(self) -> None:
        if not self.on_ladder:
            self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)

    def move_and_collide(self, platforms: List[pygame.Rect], screen_rect: pygame.Rect) -> bool:
        old_rect = self.rect.copy()
        self.rect.x += self.vel_x
        
        if self.rect.left < 50:
            self.rect.left = 50
            self.vel_x = 0
        if self.rect.right > SCREEN_WIDTH - 50:
            self.rect.right = SCREEN_WIDTH - 50
            self.vel_x = 0

        hit = None
        for platform in platforms:
            if self.rect.colliderect(platform):
                hit = platform
                break

        if hit and self.vel_x != 0:
            climbed = False
            for h in range(1, SMALL_STEP_HEIGHT + 1):
                test_rect = old_rect.move(self.vel_x, -h)
                if not any(test_rect.colliderect(p) for p in platforms):
                    self.rect = test_rect
                    climbed = True
                    break
            if not climbed:
                if self.vel_x > 0:
                    self.rect.right = hit.left
                elif self.vel_x < 0:
                    self.rect.left = hit.right

        self.rect.y += self.vel_y
        self.on_ground = False

        for platform in platforms:
            if self.rect.colliderect(platform):
                if self.on_ladder:
                    continue
                if self.vel_y > 0:
                    self.rect.bottom = platform.top
                    self.vel_y = 0
                    self.on_ground = True
                elif self.vel_y < 0:
                    self.rect.top = platform.bottom
                    self.vel_y = 0

        if self.rect.top > SCREEN_HEIGHT + 50:
            return True
        
        return False

    def update(self, platforms: List[pygame.Rect], ladders: List[pygame.Rect], screen_rect: pygame.Rect) -> bool:
        self.apply_gravity()
        fell_off = self.move_and_collide(platforms, screen_rect)
        return fell_off

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PLAYER_COLOR, self.rect)


class Barrel:
    COLOR = (200, 120, 40)

    def __init__(self, start_pos: tuple[int, int], direction: int = 1):
        self.rect = pygame.Rect(start_pos[0], start_pos[1], BARREL_SIZE, BARREL_SIZE)
        self.vel_x = BARREL_SPEED_X * direction
        self.vel_y = 0

    def apply_gravity(self):
        self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)

    def update(self, platforms: List[pygame.Rect], screen_rect: pygame.Rect,
               turn_zones: List[pygame.Rect]):
        self.rect.x += self.vel_x
        for zone in turn_zones:
            if self.rect.colliderect(zone):
                if self.vel_x > 0:
                    self.rect.right = zone.left - 1
                else:
                    self.rect.left = zone.right + 1
                self.vel_x *= -1
                break
        self.apply_gravity()
        self.rect.y += self.vel_y
        for platform in platforms:
            if self.rect.colliderect(platform):
                if self.vel_y > 0:
                    self.rect.bottom = platform.top
                    self.vel_y = 0
                elif self.vel_y < 0:
                    self.rect.top = platform.bottom
                    self.vel_y = 0
        self.rect.clamp_ip(screen_rect)

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, self.COLOR, self.rect)


class Button:
    def __init__(self, x, y, width, height, text, color=(70, 130, 180)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = (100, 160, 210)
        self.is_hovered = False
    
    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 3, border_radius=10)
        
        font = pygame.font.SysFont("Arial", 24, bold=True)
        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False


class Game:
    def __init__(self, use_background=True) -> None:
        self.screen_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.use_background = use_background
        
        if use_background:
            try:
                self.background, self.background_rect = self._load_background()
            except:
                self.background = None
                print("No se pudo cargar la imagen de fondo. Usando color sólido.")
        else:
            self.background = None
            
        self.platforms = self._build_platforms()
        self.ladders = self._build_ladders()
        self.turn_zones = self._build_turn_zones()
        self.princess_rect = pygame.Rect(400, 60, 40, 40)

        start_x = 60
        start_y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40
        self.player = Player((start_x, start_y))

        self.barrels: List[Barrel] = []
        self.frame_count = 0
        self.barrel_spawn_frames = int(BARREL_SPAWN_INTERVAL_SECONDS * FPS)
        self.start_delay_frames = int(START_DELAY_SECONDS * FPS)
        self.can_control = False
        self.game_state = STATE_PLAYING

    def _load_background(self) -> tuple[pygame.Surface, pygame.Rect]:
        raw_bg = pygame.image.load(BACKGROUND_PATH).convert()
        scaled_bg = pygame.transform.smoothscale(raw_bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
        return scaled_bg, scaled_bg.get_rect(topleft=(0, 0))

    def _build_platforms(self) -> List[pygame.Rect]:
        PLATFORM_HEIGHT = 25
        STAIR_PLATFORM_WIDTH = 53
        platforms: List[pygame.Rect] = []

        platforms.append(pygame.Rect(60, 740, 365, PLATFORM_HEIGHT))
        start_x = 425
        start_y = 737
        for i in range(7):
            platforms.append(pygame.Rect(start_x + i * STAIR_PLATFORM_WIDTH, start_y - 3*i, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))
        start_x = 58
        start_y = 610
        for i in range(13):
            platforms.append(pygame.Rect(start_x + i * STAIR_PLATFORM_WIDTH, start_y + 3*i, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))
        start_x = 110
        start_y = 540
        for i in range(13):
            platforms.append(pygame.Rect(start_x + i * STAIR_PLATFORM_WIDTH, start_y - 3*i, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))
        start_x = 58
        start_y = 393
        for i in range(13):
            platforms.append(pygame.Rect(start_x + i * STAIR_PLATFORM_WIDTH, start_y + 3*i, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))
        start_x = 110
        start_y = 325
        for i in range(13):
            platforms.append(pygame.Rect(start_x + i * STAIR_PLATFORM_WIDTH, start_y - 3*i, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))
        start_x = 530
        start_y = 205
        for i in range(4):
            platforms.append(pygame.Rect(start_x + i * STAIR_PLATFORM_WIDTH, start_y + 3*i, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        platforms.extend([
            pygame.Rect(58, 202, 472, PLATFORM_HEIGHT),
            pygame.Rect(345, 110, 160, PLATFORM_HEIGHT),
        ])

        return platforms

    def _build_ladders(self) -> List[pygame.Rect]:
        ladder_width = 25
        raw_ladders = [
            pygame.Rect(270, 30, ladder_width, 172),
            pygame.Rect(320, 30, ladder_width, 172),
            pygame.Rect(478, 110, ladder_width, 92),
            pygame.Rect(348, 226, ladder_width, 88),
            pygame.Rect(662, 236, ladder_width, 58),
            pygame.Rect(295, 337, ladder_width, 72),
            pygame.Rect(165, 344, ladder_width, 59),
            pygame.Rect(427, 438, ladder_width, 86),
            pygame.Rect(662, 452, ladder_width, 59),
            pygame.Rect(374, 548, ladder_width, 84),
            pygame.Rect(164, 560, ladder_width, 58),
            pygame.Rect(662, 668, ladder_width, 59),
        ]
        TOP_EXTRA = 40
        ladders = [pygame.Rect(r.x, r.y - TOP_EXTRA, r.width, r.height + TOP_EXTRA) for r in raw_ladders]
        return ladders

    def _build_turn_zones(self) -> List[pygame.Rect]:
        return [pygame.Rect(55, 0, 8, SCREEN_HEIGHT), pygame.Rect(800, 0, 8, SCREEN_HEIGHT)]

    def spawn_barrel(self):
        self.barrels.append(Barrel((90, 150), direction=1))

    def update(self, keys):
        if self.game_state != STATE_PLAYING:
            return

        self.frame_count += 1
        if not self.can_control and self.frame_count >= self.start_delay_frames:
            self.can_control = True

        if self.frame_count % self.barrel_spawn_frames == 0:
            self.spawn_barrel()

        if self.can_control:
            self.player.handle_input(keys, self.ladders, self.platforms)

        fell_off = self.player.update(self.platforms, self.ladders, self.screen_rect)
        if fell_off:
            self.game_state = STATE_GAME_OVER
            return

        for barrel in list(self.barrels):
            barrel.update(self.platforms, self.screen_rect, self.turn_zones)
            if barrel.rect.top > SCREEN_HEIGHT:
                self.barrels.remove(barrel)

        for barrel in self.barrels:
            if self.player.rect.colliderect(barrel.rect):
                self.game_state = STATE_GAME_OVER

        if self.player.rect.colliderect(self.princess_rect):
            self.game_state = STATE_WIN

    def draw(self, surface: pygame.Surface, show_debug=False):
        if self.background:
            surface.blit(self.background, self.background_rect)
        else:
            surface.fill((20, 20, 40))
        
        border_color = (100, 100, 100)
        pygame.draw.rect(surface, border_color, pygame.Rect(45, 0, 5, SCREEN_HEIGHT))
        pygame.draw.rect(surface, border_color, pygame.Rect(SCREEN_WIDTH - 50, 0, 5, SCREEN_HEIGHT))
        
        if show_debug:
            self._debug_draw_platforms(surface)
        
        for barrel in self.barrels:
            barrel.draw(surface)
        self.player.draw(surface)

    def _debug_draw_platforms(self, surface):
        debug_color = (255, 255, 255, 100)
        for rect in self.platforms:
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(debug_color)
            surface.blit(s, (rect.x, rect.y))
        debug_color_ladders = (255, 255, 0, 100)
        for rect in self.ladders:
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(debug_color_ladders)
            surface.blit(s, (rect.x, rect.y))
        princess_color = (255, 0, 255, 160)
        s = pygame.Surface((self.princess_rect.width, self.princess_rect.height), pygame.SRCALPHA)
        s.fill(princess_color)
        surface.blit(s, (self.princess_rect.x, self.princess_rect.y))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Donkey Kong - IA con Aprendizaje")
    clock = pygame.time.Clock()
    
    game_mode = STATE_MENU
    game = None
    ai_agent = AIAgent()
    
    if ai_agent.load_model():
        print("Modelo cargado exitosamente!")
    
    training_speed = 1
    prev_state = None
    prev_action = None
    episodes_this_session = 0
    score_history = []
    
    button_play = Button(SCREEN_WIDTH//2 - 150, 250, 300, 60, "JUGAR (Humano)")
    button_train = Button(SCREEN_WIDTH//2 - 150, 340, 300, 60, "ENTRENAR IA", (180, 70, 70))
    button_ai_demo = Button(SCREEN_WIDTH//2 - 150, 430, 300, 60, "VER IA JUGANDO", (70, 180, 70))
    button_back = Button(20, 20, 120, 40, "MENU", (100, 100, 100))
    
    running = True
    while running:
        dt = clock.tick(FPS if game_mode != STATE_TRAINING else FPS * training_speed)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if game_mode == STATE_MENU:
                if button_play.handle_event(event):
                    game_mode = STATE_PLAYING
                    game = Game()
                elif button_train.handle_event(event):
                    game_mode = STATE_TRAINING
                    game = Game(use_background=False)
                    prev_state = None
                    prev_action = None
                    episodes_this_session = 0
                    ai_agent.current_episode_score = 0
                    ai_agent.last_distance_to_princess = float('inf')
                elif button_ai_demo.handle_event(event):
                    game_mode = STATE_AI_DEMO
                    game = Game()
                    prev_state = None
            
            elif game_mode in [STATE_PLAYING, STATE_AI_DEMO]:
                if button_back.handle_event(event):
                    game_mode = STATE_MENU
                    game = None
        
        if game_mode == STATE_MENU:
            screen.fill((20, 30, 50))
            
            font_title = pygame.font.SysFont("Arial", 48, bold=True)
            title = font_title.render("DONKEY KONG IA", True, (255, 200, 50))
            screen.blit(title, title.get_rect(center=(SCREEN_WIDTH//2, 150)))
            
            button_play.draw(screen)
            button_train.draw(screen)
            button_ai_demo.draw(screen)
            
            font_stats = pygame.font.SysFont("Arial", 18)
            stats_text = [
                f"Episodios entrenados: {ai_agent.stats.episodes}",
                f"Victorias: {ai_agent.stats.wins}",
                f"Mejor puntuación: {ai_agent.stats.best_score}",
                f"Epsilon (exploración): {ai_agent.epsilon:.3f}",
            ]
            y_offset = 540
            for text in stats_text:
                surf = font_stats.render(text, True, (200, 200, 200))
                screen.blit(surf, (SCREEN_WIDTH//2 - surf.get_width()//2, y_offset))
                y_offset += 30
        
        elif game_mode == STATE_PLAYING:
            keys = pygame.key.get_pressed()
            
            if game.game_state == STATE_PLAYING:
                game.update(keys)
            elif game.game_state in [STATE_GAME_OVER, STATE_WIN]:
                if keys[pygame.K_r]:
                    game = Game()
            
            game.draw(screen, show_debug=False)
            button_back.draw(screen)
            
            font = pygame.font.SysFont("Arial", 32, bold=True)
            if game.game_state == STATE_PLAYING and not game.can_control:
                text = font.render("PREPÁRATE...", True, (255, 255, 0))
                screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, 80)))
            elif game.game_state == STATE_GAME_OVER:
                text = font.render("GAME OVER - Pulsa R para reiniciar", True, (255, 50, 50))
                screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
            elif game.game_state == STATE_WIN:
                text = font.render("¡VICTORIA! - Pulsa R", True, (50, 255, 50))
                screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        
        elif game_mode == STATE_TRAINING:
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_ESCAPE]:
                game_mode = STATE_MENU
                ai_agent.save_model()
                game = None
                continue
            
            if keys[pygame.K_1]:
                training_speed = 1
            elif keys[pygame.K_2]:
                training_speed = 5
            elif keys[pygame.K_3]:
                training_speed = 10
            
            if game.game_state == STATE_PLAYING:
                current_state = ai_agent.get_state(game)
                
                action_idx = ai_agent.choose_action(current_state, training=True)
                action = ai_agent.ACTIONS[action_idx]
                
                simulated_keys = {k: False for k in [pygame.K_LEFT, pygame.K_RIGHT, 
                                                      pygame.K_UP, pygame.K_DOWN, 
                                                      pygame.K_SPACE]}
                for key in action:
                    simulated_keys[key] = True
                
                class KeyPress:
                    def __init__(self, key_dict):
                        self.keys = key_dict
                    def __getitem__(self, key):
                        return self.keys.get(key, False)
                
                keys_obj = KeyPress(simulated_keys)
                game.update(keys_obj)
                
                reward = ai_agent.calculate_reward(game, current_state, action_idx)
                ai_agent.current_episode_score += reward
                
                next_state = ai_agent.get_state(game)
                done = game.game_state in [STATE_GAME_OVER, STATE_WIN]
                
                if prev_state is not None:
                    ai_agent.replay_buffer.add((prev_state, prev_action, reward, 
                                                current_state, done))
                
                prev_state = current_state
                prev_action = action_idx
                
                if game.frame_count % 4 == 0:
                    ai_agent.train_step()
            
            elif game.game_state in [STATE_GAME_OVER, STATE_WIN]:
                ai_agent.stats.episodes += 1
                episodes_this_session += 1
                
                if game.game_state == STATE_WIN:
                    ai_agent.stats.wins += 1
                
                score_history.append(ai_agent.current_episode_score)
                if len(score_history) > 100:
                    score_history.pop(0)
                
                ai_agent.stats.avg_score = sum(score_history) / len(score_history)
                ai_agent.stats.best_score = max(ai_agent.stats.best_score, 
                                                ai_agent.current_episode_score)
                ai_agent.stats.epsilon = ai_agent.epsilon
                
                if ai_agent.stats.episodes % 50 == 0:
                    ai_agent.save_model()
                    print(f"Episodio {ai_agent.stats.episodes}: "
                          f"Score={ai_agent.current_episode_score:.1f}, "
                          f"Epsilon={ai_agent.epsilon:.3f}")
                
                game = Game(use_background=False)
                ai_agent.current_episode_score = 0
                ai_agent.last_distance_to_princess = float('inf')
                prev_state = None
                prev_action = None
            
            game.draw(screen, show_debug=True)
            
            panel_height = 180
            panel = pygame.Surface((SCREEN_WIDTH, panel_height), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 200))
            screen.blit(panel, (0, 0))
            
            font_title = pygame.font.SysFont("Arial", 28, bold=True)
            font_info = pygame.font.SysFont("Arial", 18)
            
            title_surf = font_title.render("MODO ENTRENAMIENTO", True, (255, 200, 50))
            screen.blit(title_surf, (20, 10))
            
            info_lines = [
                f"Episodio: {ai_agent.stats.episodes} (Sesión: {episodes_this_session})",
                f"Victorias totales: {ai_agent.stats.wins}",
                f"Score actual: {ai_agent.current_episode_score:.1f} | Mejor: {ai_agent.stats.best_score:.1f}",
                f"Score promedio (últimos 100): {ai_agent.stats.avg_score:.1f}",
                f"Epsilon: {ai_agent.epsilon:.4f} | Buffer: {ai_agent.replay_buffer.size()}",
                f"Velocidad: {training_speed}x (1/2/3 para cambiar) | ESC para salir y guardar",
            ]
            
            y_pos = 50
            for line in info_lines:
                surf = font_info.render(line, True, (200, 255, 200))
                screen.blit(surf, (20, y_pos))
                y_pos += 22
        
        elif game_mode == STATE_AI_DEMO:
            keys = pygame.key.get_pressed()
            
            if game.game_state == STATE_PLAYING:
                current_state = ai_agent.get_state(game)
                action_idx = ai_agent.choose_action(current_state, training=False)
                action = ai_agent.ACTIONS[action_idx]
                
                simulated_keys = {k: False for k in [pygame.K_LEFT, pygame.K_RIGHT, 
                                                      pygame.K_UP, pygame.K_DOWN, 
                                                      pygame.K_SPACE]}
                for key in action:
                    simulated_keys[key] = True
                
                class KeyPress:
                    def __init__(self, key_dict):
                        self.keys = key_dict
                    def __getitem__(self, key):
                        return self.keys.get(key, False)
                
                keys_obj = KeyPress(simulated_keys)
                game.update(keys_obj)
            
            elif game.game_state in [STATE_GAME_OVER, STATE_WIN]:
                if keys[pygame.K_r]:
                    game = Game()
            
            game.draw(screen, show_debug=False)
            button_back.draw(screen)
            
            font = pygame.font.SysFont("Arial", 24, bold=True)
            ai_text = font.render("IA JUGANDO", True, (100, 255, 100))
            screen.blit(ai_text, (SCREEN_WIDTH - 160, 20))
            
            font_msg = pygame.font.SysFont("Arial", 32, bold=True)
            if game.game_state == STATE_GAME_OVER:
                text = font_msg.render("GAME OVER - Pulsa R para reiniciar", True, (255, 50, 50))
                screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
            elif game.game_state == STATE_WIN:
                text = font_msg.render("¡LA IA GANÓ! - Pulsa R", True, (50, 255, 50))
                screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        
        pygame.display.flip()
    
    ai_agent.save_model()
    pygame.quit()


if __name__ == "__main__":
    main()
