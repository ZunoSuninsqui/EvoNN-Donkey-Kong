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
MOVE_SPEED = 4
# Cambiar este valor para ajustar la velocidad de subida/bajada en las escaleras.
LADDER_SPEED = 3
# Cambiar este valor para que el salto sea más potente/suave.
JUMP_FORCE = 12
# Límite de la velocidad de caída para evitar aceleraciones excesivas.
MAX_FALL_SPEED = 12
# Margen horizontal para considerar que el jugador está alineado con la escalera.
LADDER_ALIGN_TOLERANCE = 18
# ============================================================================

# Tamaño de la ventana. Se usan las dimensiones originales de la imagen de fondo.
SCREEN_WIDTH = 812
SCREEN_HEIGHT = 782
PLAYER_WIDTH = 28
PLAYER_HEIGHT = 32
PLAYER_COLOR = (50, 100, 255)
BACKGROUND_PATH = Path(__file__).resolve().parent.parent / "dk_game" / "background donkey kong.png"
FPS = 60


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

    def handle_input(self, keys: pygame.key.ScancodeWrapper, ladders: List[pygame.Rect]) -> None:
        self.vel_x = 0

        if keys[pygame.K_LEFT]:
            self.vel_x = -MOVE_SPEED
        if keys[pygame.K_RIGHT]:
            self.vel_x = MOVE_SPEED if self.vel_x == 0 else self.vel_x

        ladder = self._current_ladder(ladders)
        wants_climb = ladder and (keys[pygame.K_UP] or keys[pygame.K_DOWN])

        if wants_climb and abs(self.rect.centerx - ladder.centerx) <= LADDER_ALIGN_TOLERANCE:
            # Bloquea el eje X a la escalera para evitar movimientos imposibles.
            self.rect.centerx = ladder.centerx
            self.on_ladder = True
            self.vel_y = 0
            if keys[pygame.K_UP]:
                self.vel_y = -LADDER_SPEED
            elif keys[pygame.K_DOWN]:
                self.vel_y = LADDER_SPEED
        else:
            # Salimos de la escalera si ya no la tocamos.
            if self.on_ladder and ladder is None:
                self.on_ladder = False
            if keys[pygame.K_SPACE] and self.on_ground:
                self.vel_y = -JUMP_FORCE

    def apply_gravity(self) -> None:
        if not self.on_ladder:
            self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)

    def move_and_collide(self, platforms: List[pygame.Rect], screen_rect: pygame.Rect) -> None:
        # Movimiento horizontal primero.
        self.rect.x += self.vel_x
        self.rect.clamp_ip(screen_rect)
        for platform in platforms:
            if self.rect.colliderect(platform):
                if self.vel_x > 0:
                    self.rect.right = platform.left
                elif self.vel_x < 0:
                    self.rect.left = platform.right

        # Movimiento vertical después para una resolución de colisión estable.
        self.rect.y += self.vel_y
        self.on_ground = False
        for platform in platforms:
            if self.rect.colliderect(platform):
                if self.vel_y > 0:
                    self.rect.bottom = platform.top
                    self.vel_y = 0
                    self.on_ground = True
                    self.on_ladder = False
                elif self.vel_y < 0:
                    self.rect.top = platform.bottom
                    self.vel_y = 0

        self.rect.clamp_ip(screen_rect)

    def update(self, platforms: List[pygame.Rect], ladders: List[pygame.Rect], screen_rect: pygame.Rect) -> None:
        self.apply_gravity()
        self.move_and_collide(platforms, screen_rect)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PLAYER_COLOR, self.rect)


class Game:
    """Agrupa el fondo, las plataformas invisibles y el jugador."""

    def __init__(self) -> None:
        self.screen_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.background, self.background_rect = self._load_background()
        self.platforms = self._build_platforms()
        self.ladders = self._build_ladders()
        start_x = 60
        start_y = SCREEN_HEIGHT - PLAYER_HEIGHT - 40
        self.player = Player((start_x, start_y))

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
        #  x: 425, 478, 531, 584, 637, 690, 743  -> suma 53 cada vez
        #  y: 737, 733, 729, 726, 723, 720, 717  -> resta 3 cada vez
        start_x = 425
        start_y = 737
        dx = STAIR_PLATFORM_WIDTH  # +53 en X
        dy = -3  # -3 en Y

        for i in range(7):  # 7 plataformas
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        # ----- SEGUNDA CAPA ESCALONADA -----
        #  x: 58, 111, 164, 217, 270, 323       -> +53 en X
        #  y: 610, 613, 616, 619, 622, 625      -> +3 en Y
        start_x = 58
        start_y = 610
        dx = STAIR_PLATFORM_WIDTH  # +53 en X
        dy = 3  # +3 en Y

        for i in range(13):  # 6 plataformas
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))


        #Tercera capa escalonada
        start_x = 110
        start_y = 540
        dx = STAIR_PLATFORM_WIDTH  # +53 en X
        dy = -3  # -4 en Y para hacerlas subir

        for i in range(13):  # 6 plataformas

            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))


        #Cuarta
        start_x = 58
        start_y = 393
        dx = STAIR_PLATFORM_WIDTH  # +53 en X
        dy = 3  # +3 en Y

        for i in range(13):  # 6 plataformas
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))



        # QUITA capa escalonada
        start_x = 110
        start_y = 325
        dx = STAIR_PLATFORM_WIDTH  # +53 en X
        dy = -3  # -4 en Y para hacerlas subir

        for i in range(13):  # 6 plataformas

            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))
        # ----- RESTO DE VIGAS (de momento rectas completas) -----

        # SEXTA
        start_x = 530
        start_y = 205
        dx = STAIR_PLATFORM_WIDTH  # +53 en X
        dy = 3  # +3 en Y

        for i in range(4):  # 6 plataformas
            x = start_x + i * dx
            y = start_y + i * dy
            platforms.append(pygame.Rect(x, y, STAIR_PLATFORM_WIDTH, PLATFORM_HEIGHT))

        platforms.extend([
            pygame.Rect(58, 202, 472, PLATFORM_HEIGHT),  # Plataforma superior.
            pygame.Rect(345, 110, 160, PLATFORM_HEIGHT), #PLATAFORMA PRINCESA

        ])

        return platforms

    def debug_draw_platforms(self, surface):
        """Dibuja las plataformas invisibles con transparencia para depuración."""
        debug_color = (255, 255, 255, 150)  # Blanco con opacidad 50

        for rect in self.platforms:
            # Crear surface con alpha
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(debug_color)
            surface.blit(s, (rect.x, rect.y))

    def _build_ladders(self) -> List[pygame.Rect]:
        """Escaleras invisibles alineadas con las blancas del fondo."""
        return [
            # Desde el suelo hasta la segunda viga (escalera derecha del fondo).
            pygame.Rect(720, 645, 22, 95),
            # Enlace segunda a tercera viga (escalera central izquierda).
            pygame.Rect(160, 565, 22, 90),
            # Enlace segunda a tercera viga (escalera central derecha).
            pygame.Rect(500, 565, 22, 90),
            # Enlace tercera a cuarta viga (escalera derecha).
            pygame.Rect(640, 485, 22, 90),
            # Enlace tercera a cuarta viga (escalera izquierda larga).
            pygame.Rect(90, 485, 22, 90),
            # Enlace cuarta a quinta viga.
            pygame.Rect(300, 405, 22, 90),
            # Enlace quinta a sexta viga.
            pygame.Rect(540, 325, 22, 90),
            # Escalera doble debajo de la parte superior (lado izquierdo).
            pygame.Rect(190, 245, 22, 90),
            pygame.Rect(230, 245, 22, 90),
            # Escalera final hacia la cima (lado derecho cerca del barril de fuego).
            pygame.Rect(660, 245, 22, 170),
        ]

    def update(self, keys: pygame.key.ScancodeWrapper) -> None:
        self.player.handle_input(keys, self.ladders)
        self.player.update(self.platforms, self.ladders, self.screen_rect)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((0, 0, 0))
        surface.blit(self.background, self.background_rect)
        self.debug_draw_platforms(surface)
        self.player.draw(surface)


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
        game.update(keys)

        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
