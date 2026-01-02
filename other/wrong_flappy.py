"""
Flappy Bird minimal en Python usando pygame
Guarda este archivo como flappy_bird.py y ejecútalo con:
  pip install pygame
  python flappy_bird.py

Controles:
 - Barra espaciadora o clic izquierdo: saltar
 - R o clic en pantalla cuando estés en Game Over: reiniciar

Código simple y comentado para facilitar modificaciones.
"""
import pygame
import random
import sys

# --- Configuración ---
WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.45
JUMP_VELOCITY = -9
PIPE_WIDTH = 70
PIPE_GAP = 300
PIPE_DISTANCE = 400  # distancia horizontal entre tuberías
GROUND_HEIGHT = 80
BG_COLOR = (135, 206, 235)  # cielo
BIRD_COLOR = (255, 255, 0)
PIPE_COLOR = (34, 139, 34)
GROUND_COLOR = (222, 184, 135)

# --- Inicialización ---
pygame.init()
surface = None
pygame.display.set_caption("Flappy Bird - Minimal")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

# --- Clases ---
class Bird:
    def __init__(self):
        self.x = WIDTH * 0.2
        self.y = HEIGHT / 2
        self.radius = 14
        self.vel = 0.0
        self.alive = True

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel

        # limitar para que no salga por arriba
        if self.y < 0:
            self.y = 0
            self.vel = 0

    def jump(self):
        self.vel = JUMP_VELOCITY

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def draw(self, surf):
        pygame.draw.circle(surf, BIRD_COLOR, (int(self.x), int(self.y)), self.radius)
        # ojo
        pygame.draw.circle(surf, (0,0,0), (int(self.x + 5), int(self.y - 3)), 3)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.passed = False
        # definir hueco vertical aleatorio
        margin = 50
        self.gap_y = random.randint(margin + 20, HEIGHT - GROUND_HEIGHT - margin - PIPE_GAP)

    def update(self, dx):
        self.x -= dx

    def off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, rect):
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP, self.width, HEIGHT - (self.gap_y + PIPE_GAP) - GROUND_HEIGHT)
        return rect.colliderect(top_rect) or rect.colliderect(bottom_rect)

    def draw(self, surf):
        top_rect = pygame.Rect(int(self.x), 0, self.width, int(self.gap_y))
        bottom_rect = pygame.Rect(int(self.x), int(self.gap_y + PIPE_GAP), self.width, int(HEIGHT - (self.gap_y + PIPE_GAP) - GROUND_HEIGHT))
        pygame.draw.rect(surf, PIPE_COLOR, top_rect)
        pygame.draw.rect(surf, PIPE_COLOR, bottom_rect)
        # "borde" sencillo
        pygame.draw.rect(surf, (0,0,0), top_rect, 2)
        pygame.draw.rect(surf, (0,0,0), bottom_rect, 2)

# --- Funciones de utilidad ---

def draw_ground(surf, offset):
    rect = pygame.Rect(0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT)
    pygame.draw.rect(surf, GROUND_COLOR, rect)
    # líneas para simular textura
    for i in range(-50, WIDTH + 50, 30):
        x = (i + offset) % (WIDTH + 50) - 25
        pygame.draw.rect(surf, (200,150,100), (x, HEIGHT - GROUND_HEIGHT + 10, 15, 10))


def draw_text_center(surf, text, y, size=36):
    txt = font.render(text, True, (0,0,0))
    r = txt.get_rect(center=(WIDTH//2, y))
    surf.blit(txt, r)

# --- Juego ---

def run_game():
    bird = Bird()
    pipes = []
    score = 0
    speed = 3.0
    frame = 0
    ground_offset = 0
    game_over = False

    # crear tuberías iniciales
    for i in range(3):
        pipes.append(Pipe(WIDTH + i * PIPE_DISTANCE))

    while True:
        dt = clock.tick(FPS)
        frame += 1

        # --- Eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_over:
                    bird.jump()
                if event.key == pygame.K_r and game_over:
                    return  # reiniciar (llamará a run_game de nuevo)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # clic izquierdo
                    if not game_over:
                        bird.jump()
                    else:
                        return

        # --- Lógica ---
        if not game_over:
            bird.update()
            # mover tuberías
            for p in pipes:
                p.update(speed)

            # añadir nuevas tuberías cuando sea necesario
            if pipes and pipes[-1].x < WIDTH - PIPE_DISTANCE:
                pipes.append(Pipe(WIDTH + 20))

            # eliminar tuberías fuera
            if pipes and pipes[0].off_screen():
                pipes.pop(0)

            # comprobar colisiones y pasar
            brect = bird.get_rect()
            for p in pipes:
                if not p.passed and p.x + p.width < bird.x:
                    p.passed = True
                    score += 1
                    # aumentar velocidad ligeramente por cada tubería pasada
                    speed += 0.05
                if p.collides_with(brect):
                    bird.alive = False
                    game_over = True

            # colisión con el suelo
            if bird.y + bird.radius > HEIGHT - GROUND_HEIGHT:
                bird.y = HEIGHT - GROUND_HEIGHT - bird.radius
                bird.alive = False
                game_over = True

            # colisión con techo (ya limitada en update)

            # animar suelo
            ground_offset += speed

        # --- Dibujado ---
        surface.fill(BG_COLOR)
        # nubes simples (decoración)
        for i in range(3):
            cx = (frame * 0.2 + i * 140) % (WIDTH + 100) - 50
            pygame.draw.ellipse(surface, (255,255,255), (cx, 40 + i*30, 80, 30))

        # pipes
        for p in pipes:
            p.draw(surface)

        # bird
        bird.draw(surface)

        # ground
        draw_ground(surface, ground_offset)

        # score
        score_surf = font.render(str(score), True, (0,0,0))
        surface.blit(score_surf, (WIDTH//2 - score_surf.get_width()//2, 20))

        if game_over:
            draw_text_center(surface, "GAME OVER", HEIGHT//2 - 20)
            draw_text_center(surface, f"Puntos: {score}", HEIGHT//2 + 20)
            hint = small_font.render("Pulsa R o haz clic para reiniciar", True, (0,0,0))
            surface.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2 + 60))

        pygame.display.flip()

def rl_init():
    global bird, pipes, score, speed, ground_offset, game_over
    bird = Bird()
    pipes = [Pipe(WIDTH + i * PIPE_DISTANCE) for i in range(3)]
    score = 0
    speed = 3.0
    ground_offset = 0
    game_over = False

def rl_step(action):
    global bird, pipes, score, speed, ground_offset, game_over

    reward = 0.01

    if action == 1:
        bird.jump()

    bird.update()

    for p in pipes:
        p.update(speed)

    if pipes[-1].x < WIDTH - PIPE_DISTANCE:
        pipes.append(Pipe(WIDTH + 20))

    if pipes[0].off_screen():
        pipes.pop(0)

    brect = bird.get_rect()
    for p in pipes:
        if not p.passed and p.x + PIPE_WIDTH < bird.x:
            p.passed = True
            score += 1
            reward += 1.0
            speed += 0.05

        if p.collides_with(brect):
            reward = -1.0
            game_over = True

    if bird.y + bird.radius > HEIGHT - GROUND_HEIGHT:
        reward = -1.0
        game_over = True

    ground_offset += speed

    return reward, game_over


def rl_obs():
    next_pipe = None
    for p in pipes:
        if p.x + PIPE_WIDTH > bird.x:
            next_pipe = p
            break

    if next_pipe is None:
        next_pipe_x = bird.x + PIPE_DISTANCE
        next_pipe_gap_y = HEIGHT / 2
    else:
        next_pipe_x = next_pipe.x
        next_pipe_gap_y = next_pipe.gap_y

    return (
        float(bird.y),
        float(bird.vel),
        float(next_pipe_x),
        float(next_pipe_gap_y)
    )

def rl_render():
    surface.fill(BG_COLOR)
    for p in pipes:
        p.draw(surface)
    bird.draw(surface)
    draw_ground(surface, ground_offset)
    return pygame.surfarray.array3d(surface)

if __name__ == '__main__':
    surface = pygame.display.set_mode((WIDTH, HEIGHT))  
    while True:
        run_game()
