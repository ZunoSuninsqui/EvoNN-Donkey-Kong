"""Playable Donkey Kong style prototype using Pygame.

The module renders the original background art and overlays invisible rectangles
for platforms and ladders so the player can climb and walk with simple physics.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pygame

# ==== PHYSICS TUNING SECTION =================================================
# Cambiar este valor para hacer que la gravedad sea más fuerte/suave.
GRAVITY = 0.7
# Cambiar este valor para que el jugador se mueva más rápido/lento en el eje X.
MOVE_SPEED = 3  # antes 4, ahora un poco más lento
# Cambiar este valor para ajustar la velocidad de subida/bajada en las escaleras.
LADDER_SPEED = 3
# Cambiar este valor para que el salto sea más potente/suave.
JUMP_FORCE = 12
# Límite de la velocidad de caída para evitar aceleraciones excesivas.
MAX_FALL_SPEED = 12
# Margen horizontal para considerar que el jugador está alineado con la escalera.
LADDER_ALIGN_TOLERANCE = 18
# Altura máxima de “escalón” que se puede subir sin saltar.
SMALL_STEP_HEIGHT = 12

# Barriles: ritmo y físicas
# Intervalo entre barriles (sin aleatoriedad, siempre igual entre partidas).
BARREL_SPAWN_INTERVAL_SECONDS = 2.0  # antes 1.0, ahora más separados
BARREL_SPEED_X = 3
BARREL_SIZE = 22  # antes 18, barril un poco más grande

# Tiempo de espera al inicio antes de poder mover a Mario.
START_DELAY_SECONDS = 2.0
# ============================================================================

# Tamaño de la ventana. Se usan las dimensiones originales de la imagen de fondo.
SCREEN_WIDTH = 812
SCREEN_HEIGHT = 782
PLAYER_WIDTH = 28
PLAYER_HEIGHT = 32
PLAYER_COLOR = (50, 100, 255)
BACKGROUND_PATH = Path(__file__).resolve().parent.parent / "dk_game" / "background donkey kong.png"
FPS = 60

# Estados del juego
STATE_PLAYING = "PLAYING"
STATE_GAME_OVER = "GAME_OVER"
STATE_WIN = "WIN"


class Player:
    """Rectangular placeholder for Mario with simple platforming physics."""

    def __init__(self, start_pos: tuple[int, int]):
        self.rect = pygame.Rect(start_pos[0], start_pos[1], PLAYER_WIDTH, PLAYER_HEIGHT)
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.on_ladder = False

    def _current_ladder(self, ladders: List[pygame.Rect]) -> pygame.Rect | None:
        """Return the ladder the player overlaps (if any)."""
        for ladder in ladders:
            if self.rect.colliderect(ladder):
                return ladder
        return None

    def handle_input(
        self,
        keys: pygame.key.ScancodeWrapper,
        ladders: List[pygame.Rect],
        platforms: List[pygame.Rect],
    ) -> None:
        """Procesa el input del jugador.

        Reglas importantes:
        - Cuando está en escalera (on_ladder=True) puede:
          - Moverse solo en Y si está solapando una plataforma (tramo dentro de viga).
          - Salir lateralmente con ←/→ si está en un tramo libre de plataforma.
        - Cuando NO está en escalera, movimiento normal + salto + entrada a escalera.
        """
        ladder = self._current_ladder(ladders)

        # Si estábamos en escalera pero ya no tocamos ninguna, salimos de modo escalera.
        if self.on_ladder and ladder is None:
            self.on_ladder = False

        wants_climb = ladder and (keys[pygame.K_UP] or keys[pygame.K_DOWN])

        # Por defecto, sin input horizontal.
        self.vel_x = 0

        # =========================
        #      MODO ESCALERA
        # =========================
        if self.on_ladder:
            # ¿Está el jugador solapando alguna plataforma?
            touching_platform = any(self.rect.colliderect(p) for p in platforms)

            if touching_platform:
                # Estamos en la zona donde la escalera cruza una viga:
                # no dejamos salir lateralmente, solo subir/bajar o quedarse quieto.
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
                # Tramo libre de escalera (entre vigas):
                # permitimos salir lateralmente.
                if keys[pygame.K_LEFT]:
                    self.on_ladder = False
                    self.vel_x = -MOVE_SPEED
                elif keys[pygame.K_RIGHT]:
                    self.on_ladder = False
                    self.vel_x = MOVE_SPEED
                else:
                    # Si no hay input lateral, seguimos usando la escalera normalmente.
                    if ladder is not None:
                        self.rect.centerx = ladder.centerx
                    if keys[pygame.K_UP]:
                        self.vel_y = -LADDER_SPEED
                    elif keys[pygame.K_DOWN]:
                        self.vel_y = LADDER_SPEED
                    else:
                        self.vel_y = 0
                    return

        # ==============================
        #   NO ESTAMOS EN ESCALERA
        # ==============================
        # Movimiento horizontal normal.
        if keys[pygame.K_LEFT]:
            self.vel_x = -MOVE_SPEED
        if keys[pygame.K_RIGHT]:
            self.vel_x = MOVE_SPEED if self.vel_x == 0 else self.vel_x

        # Intentar entrar a una escalera con ↑ o ↓ si estamos alineados.
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

        # Salto solo si no estamos en escalera.
        if keys[pygame.K_SPACE] and self.on_ground:
            self.vel_y = -JUMP_FORCE

    def apply_gravity(self) -> None:
        if not self.on_ladder:
            self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)

    def move_and_collide(self, platforms: List[pygame.Rect], screen_rect: pygame.Rect) -> None:
        # ---------------------
        # MOVIMIENTO HORIZONTAL
        # ---------------------
        old_rect = self.rect.copy()
        self.rect.x += self.vel_x
        self.rect.clamp_ip(screen_rect)

        hit = None
        for platform in platforms:
            if self.rect.colliderect(platform):
                hit = platform
                break

        if hit and self.vel_x != 0:
            # Intentar subir un pequeño escalón en lugar de chocar en seco.
            climbed = False
            for h in range(1, SMALL_STEP_HEIGHT + 1):
                test_rect = old_rect.move(self.vel_x, -h)
                if not any(test_rect.colliderect(p) for p in platforms):
                    self.rect = test_rect
                    climbed = True
                    break

            if not climbed:
                # Colisión horizontal normal.
                if self.vel_x > 0:
                    self.rect.right = hit.left
                elif self.vel_x < 0:
                    self.rect.left = hit.right

        # ---------------------
        # MOVIMIENTO VERTICAL
        # ---------------------
        self.rect.y += self.vel_y
        self.on_ground = False

        for platform in platforms:
            if self.rect.colliderect(platform):
                # IMPORTANTE: si estamos en escalera, ignoramos TODAS las plataformas.
                if self.on_ladder:
                    continue

                if self.vel_y > 0:  # cayendo
                    self.rect.bottom = platform.top
                    self.vel_y = 0
                    self.on_ground = True
                elif self.vel_y < 0:  # golpeando techo
                    self.rect.top = platform.bottom
                    self.vel_y = 0

        self.rect.clamp_ip(screen_rect)

    def update(self, platforms: List[pygame.Rect], ladders: List[pygame.Rect], screen_rect: pygame.Rect) -> None:
        self.apply_gravity()
        self.move_and_collide(platforms, screen_rect)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PLAYER_COLOR, self.rect)


class Barrel:
    """Barril que cae por gravedad y rebota en líneas verticales (turn zones)."""

    COLOR = (200, 120, 40)

    def __init__(self, start_pos: tuple[int, int], direction: int = 1):
        self.rect = pygame.Rect(start_pos[0], start_pos[1], BARREL_SIZE, BARREL_SIZE)
        self.vel_x = BARREL_SPEED_X * direction
        self.vel_y = 0

    def apply_gravity(self):
        self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)

    def update(
        self,
        platforms: List[pygame.Rect],
        screen_rect: pygame.Rect,
        turn_zones: List[pygame.Rect],
    ):
        # --- Movimiento horizontal constante ---
        self.rect.x += self.vel_x

        # Chequear colisión con líneas verticales de giro.
        for zone in turn_zones:
            if self.rect.colliderect(zone):
                if self.vel_x > 0:
                    self.rect.right = zone.left - 1
                elif self.vel_x < 0:
                    self.rect.left = zone.right + 1
                self.vel_x *= -1
                break

        # --- Gravedad + colisión vertical con plataformas ---
        self.apply_gravity()
        self.rect.y += self.vel_y

        for platform in platforms:
            if self.rect.colliderect(platform):
                if self.vel_y > 0:  # cae sobre la plataforma
                    self.rect.bottom = platform.top
                    self.vel_y = 0
                elif self.vel_y < 0:  # golpea techo (raro)
                    self.rect.top = platform.bottom
                    self.vel_y = 0

        self.rect.clamp_ip(screen_rect)

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, self.COLOR, self.rect)


class Game:
    """Agrupa el fondo, las plataformas invisibles, las escaleras, el jugador y los barriles."""

    def __init__(self) -> None:
        self.screen_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.background, self.background_rect = self._load_background()
        self.platforms = self._build_platforms()
        self.ladders = self._build_ladders()
        self.turn_zones = self._build_turn_zones()

        # Rectángulo de la "princesa" (meta)
        self.princess_rect = pygame.Rect(400, 60, 40, 40)

        start_x = 60
        start_y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40
        self.player = Player((start_x, start_y))

        # Barriles
        self.barrels: List[Barrel] = []
        self.frame_count = 0
        self.barrel_spawn_frames = int(BARREL_SPAWN_INTERVAL_SECONDS * FPS)

        # Retardo inicial de control
        self.start_delay_frames = int(START_DELAY_SECONDS * FPS)
        self.can_control = False

        # Estado del juego
        self.game_state = STATE_PLAYING

    def _load_background(self) -> tuple[pygame.Surface, pygame.Rect]:
        raw_bg = pygame.image.load(os.fspath(BACKGROUND_PATH)).convert()
        bg_rect = raw_bg.get_rect()
        scale = min(SCREEN_WIDTH / bg_rect.width, SCREEN_HEIGHT / bg_rect.height)
        scaled_size = (int(bg_rect.width * scale), int(bg_rect.height * scale))
        background = pygame.transform.smoothscale(raw_bg, scaled_size)
        background_rect = background.get_rect(center=self.screen_rect.center)
        return background, background_rect

    def _build_platforms(self) -> List[pygame.Rect]:
        PLATFORM_HEIGHT = 25
        STAIR_PLATFORM_WIDTH = 53

        platforms: List[pygame.Rect] = []

        # ----- VIGA INFERIOR IZQUIERDA (suelo grande) -----
        platforms.append(pygame.Rect(60, 740, 365, PLATFORM_HEIGHT))

        # ----- PRIMERA CAPA ESCALONADA (abajo a la derecha) -----
        start_x = 425
        start_y = 737
        dx = STAIR_PLATFORM_WIDTH  # +53 en X
        dy = -3  # -3 en Y

        for i in range(7):  # 7 plataformas
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        # ----- SEGUNDA CAPA ESCALONADA -----
        start_x = 58
        start_y = 610
        dx = STAIR_PLATFORM_WIDTH
        dy = 3

        for i in range(13):
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        # Tercera capa escalonada
        start_x = 110
        start_y = 540
        dx = STAIR_PLATFORM_WIDTH
        dy = -3

        for i in range(13):
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        # Cuarta capa escalonada
        start_x = 58
        start_y = 393
        dx = STAIR_PLATFORM_WIDTH
        dy = 3

        for i in range(13):
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        # Quinta capa escalonada
        start_x = 110
        start_y = 325
        dx = STAIR_PLATFORM_WIDTH
        dy = -3

        for i in range(13):
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        # Sexta (superior derecha)
        start_x = 530
        start_y = 205
        dx = STAIR_PLATFORM_WIDTH
        dy = 3

        for i in range(4):
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        platforms.extend([
            pygame.Rect(58, 202, 472, PLATFORM_HEIGHT),   # Plataforma superior.
            pygame.Rect(345, 110, 160, PLATFORM_HEIGHT),  # Plataforma princesa
        ])

        return platforms

    def _build_ladders(self) -> List[pygame.Rect]:
        """Escaleras invisibles alineadas con las blancas del fondo.

        La base de cada escalera se mantiene alineada al piso inferior,
        pero se extiende unos píxeles por encima de la plataforma superior
        para poder subir y bajar suavemente.
        """
        ladder_width = 25

        raw_ladders = [
            # 1–3: las tres que ya tenías bien colocadas
            pygame.Rect(270, 30, ladder_width, 172),  # Doble escalera izquierda (1)
            pygame.Rect(320, 30, ladder_width, 172),  # Doble escalera izquierda (2)
            pygame.Rect(478, 110, ladder_width, 92),  # Escalera corta superior derecha

            # 4: escalera sobre la segunda plataforma (parte superior izquierda)
            pygame.Rect(348, 226, ladder_width, 88),

            # 5: tramo de escalera hacia la derecha (sobre la misma plataforma alta)
            pygame.Rect(662, 236, ladder_width, 58),

            # 6–7: escaleras centrales en la parte media del nivel
            pygame.Rect(295, 337, ladder_width, 72),
            pygame.Rect(165, 344, ladder_width, 59),

            # 8–9: escaleras en la zona media–baja (casi centro y derecha)
            pygame.Rect(427, 438, ladder_width, 86),
            pygame.Rect(662, 452, ladder_width, 59),

            # 10–11: escaleras de la parte baja, izquierda y centro
            pygame.Rect(374, 548, ladder_width, 84),
            pygame.Rect(164, 560, ladder_width, 58),

            # 12: escalera inferior derecha que sube desde la penúltima plataforma
            pygame.Rect(662, 668, ladder_width, 59),
        ]

        # Extender solo por arriba: la base se mantiene, la parte superior sube.
        TOP_EXTRA = 40  # píxeles extra por arriba

        ladders: List[pygame.Rect] = []
        for r in raw_ladders:
            new_y = r.y - TOP_EXTRA
            new_h = r.height + TOP_EXTRA
            ladders.append(pygame.Rect(r.x, new_y, r.width, new_h))

        return ladders

    def _build_turn_zones(self) -> List[pygame.Rect]:
        """Líneas verticales para cambiar dirección de barriles."""
        zones: List[pygame.Rect] = []

        # Línea izquierda (ligeramente dentro del nivel).
        zones.append(pygame.Rect(55, 0, 8, SCREEN_HEIGHT))

        # Línea derecha (ligeramente dentro del nivel).
        zones.append(pygame.Rect(800, 0, 8, SCREEN_HEIGHT))

        return zones

    def debug_draw_platforms(self, surface):
        """Dibuja las plataformas invisibles y zonas de giro con transparencia para depuración."""
        debug_color = (255, 255, 255, 150)  # Blanco con opacidad 50

        for rect in self.platforms:
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(debug_color)
            surface.blit(s, (rect.x, rect.y))

        debug_color_ladders = (255, 255, 0, 150)
        for rect in self.ladders:
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(debug_color_ladders)
            surface.blit(s, (rect.x, rect.y))

        debug_color_turn = (0, 255, 255, 150)  # cian para ver las líneas de giro
        for rect in self.turn_zones:
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(debug_color_turn)
            surface.blit(s, (rect.x, rect.y))

        # Dibujar princesa (meta) en rosa.
        princess_color = (255, 0, 255, 160)
        s = pygame.Surface((self.princess_rect.width, self.princess_rect.height), pygame.SRCALPHA)
        s.fill(princess_color)
        surface.blit(s, (self.princess_rect.x, self.princess_rect.y))

    # ---------- Barriles ----------
    def spawn_barrel(self):
        """Crea un barril en la parte superior izquierda, cerca de los barriles del fondo."""
        spawn_x = 90
        spawn_y = 150  # un poco por encima para que caiga y se apoye
        barrel = Barrel((spawn_x, spawn_y), direction=1)  # primero hacia la derecha
        self.barrels.append(barrel)

    def update(self, keys: pygame.key.ScancodeWrapper) -> None:
        if self.game_state != STATE_PLAYING:
            return

        self.frame_count += 1

        # Activar control tras el retardo inicial.
        if not self.can_control and self.frame_count >= self.start_delay_frames:
            self.can_control = True

        # Spawnear barriles de forma determinista cada X frames.
        if self.frame_count % self.barrel_spawn_frames == 0:
            self.spawn_barrel()

        # Input solo si ya se cumplió el retardo inicial.
        if self.can_control:
            self.player.handle_input(keys, self.ladders, self.platforms)

        self.player.update(self.platforms, self.ladders, self.screen_rect)

        # Actualizar barriles y limpiar los que salen de pantalla
        for barrel in list(self.barrels):
            barrel.update(self.platforms, self.screen_rect, self.turn_zones)
            if barrel.rect.top > SCREEN_HEIGHT:
                self.barrels.remove(barrel)

        # ---- MECÁNICA DE MUERTE POR BARRIL ----
        for barrel in self.barrels:
            if self.player.rect.colliderect(barrel.rect):
                self.game_state = STATE_GAME_OVER
                break

        # ---- MECÁNICA DE VICTORIA (PRINCESA) ----
        if self.game_state == STATE_PLAYING and self.player.rect.colliderect(self.princess_rect):
            self.game_state = STATE_WIN

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((0, 0, 0))
        surface.blit(self.background, self.background_rect)
        self.debug_draw_platforms(surface)

        # Dibujar barriles
        for barrel in self.barrels:
            barrel.draw(surface)

        self.player.draw(surface)

        # Mensajes de estado
        font = pygame.font.SysFont("Arial", 32, bold=True)

        # Mensaje de inicio mientras aún no se puede mover.
        if self.game_state == STATE_PLAYING and not self.can_control:
            text = font.render("PREPÁRATE...", True, (255, 255, 0))
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 80))
            surface.blit(text, rect)

        if self.game_state == STATE_GAME_OVER:
            text = font.render("GAME OVER - Pulsa R para reiniciar", True, (255, 50, 50))
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            surface.blit(text, rect)
        elif self.game_state == STATE_WIN:
            text = font.render("HAS RESCATADO A LA PRINCESA! - Pulsa R", True, (50, 255, 50))
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            surface.blit(text, rect)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Donkey Kong Prototype")
    clock = pygame.time.Clock()
    game = Game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Reinicio desde GAME_OVER o WIN
        if game.game_state in (STATE_GAME_OVER, STATE_WIN) and keys[pygame.K_r]:
            game = Game()
        else:
            game.update(keys)

        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
