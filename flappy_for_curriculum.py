# flappy.py
import pygame
import random
import sys

# --- Config ---
WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.45
JUMP_VELOCITY = -9
PIPE_WIDTH = 70
PIPE_GAP_MIN = 250
PIPE_GAP_MAX = 350
PIPE_DISTANCE = 400
GROUND_HEIGHT = 80
BG_COLOR = (135, 206, 235)  # ciel
BIRD_COLOR = (255, 255, 0)
PIPE_COLOR = (34, 139, 34)
GROUND_COLOR = (222, 184, 135)

# --- Initialisation ---
pygame.init()
surface = None
pygame.display.set_caption("Flappy Bird - Minimal")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

# --- Game state ---
game_state = {}

# --- Classes ---
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
        if self.y < 0:
            self.y = 0
            self.vel = 0

    def jump(self):
        self.vel = JUMP_VELOCITY

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius*2, self.radius*2)

    def draw(self, surf):
        pygame.draw.circle(surf, BIRD_COLOR, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surf, (0,0,0), (int(self.x+5), int(self.y-3)), 3)

class Pipe:
    def __init__(self, x, gap):
        self.x = x
        self.width = PIPE_WIDTH
        self.passed = False
        self.gap = gap
        margin = 50
        self.gap_y = random.randint(margin+20, HEIGHT - GROUND_HEIGHT - margin - self.gap)

    def update(self, dx):
        self.x -= dx

    def off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, rect):
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        bottom_rect = pygame.Rect(self.x, self.gap_y+self.gap, self.width, HEIGHT-(self.gap_y+self.gap)-GROUND_HEIGHT)
        return rect.colliderect(top_rect) or rect.colliderect(bottom_rect)

    def draw(self, surf):
        top_rect = pygame.Rect(int(self.x), 0, self.width, int(self.gap_y))
        bottom_rect = pygame.Rect(int(self.x), int(self.gap_y+self.gap), self.width, int(HEIGHT-(self.gap_y+self.gap)-GROUND_HEIGHT))
        pygame.draw.rect(surf, PIPE_COLOR, top_rect)
        pygame.draw.rect(surf, PIPE_COLOR, bottom_rect)
        pygame.draw.rect(surf, (0,0,0), top_rect, 2)
        pygame.draw.rect(surf, (0,0,0), bottom_rect, 2)

# --- Fonctions utilitaires ---
def draw_ground(surf, offset):
    rect = pygame.Rect(0, HEIGHT-GROUND_HEIGHT, WIDTH, GROUND_HEIGHT)
    pygame.draw.rect(surf, GROUND_COLOR, rect)
    for i in range(-50, WIDTH+50, 30):
        x = (i + offset) % (WIDTH+50) - 25
        pygame.draw.rect(surf, (200,150,100), (x, HEIGHT-GROUND_HEIGHT+10, 15, 10))

# --- RL API ---
# Dans la fonction rl_init
def rl_init(pipe_gap=200, pipe_speed=3.0, pipe_gap_max=PIPE_GAP_MAX, pipe_gap_min=PIPE_GAP_MIN):
    global game_state
    
    bird = Bird()
    pipes = [Pipe(WIDTH + i*PIPE_DISTANCE, random.randint(pipe_gap_min, pipe_gap_max)) for i in range(3)]
    
    game_state = {
        "bird": bird,
        "pipes": pipes,
        "score": 0,
        "base_speed": pipe_speed,
        "current_speed": pipe_speed,
        "ground_offset": 0,
        "game_over": False,
        "PIPE_GAP": pipe_gap,
        "PIPE_GAP_MAX": pipe_gap_max,
        "PIPE_GAP_MIN": pipe_gap_min
    }

# Dans la fonction rl_step
def rl_step(action):
    global game_state
    bird = game_state["bird"]
    pipes = game_state["pipes"]
    base_speed = game_state["base_speed"]  # Récupère la vitesse de base
    current_speed = game_state["current_speed"]  # Vitesse actuelle
    score = game_state["score"]
    ground_offset = game_state["ground_offset"]
    game_over = game_state["game_over"]
    PIPE_GAP = game_state["PIPE_GAP"]
    pipe_gap_min = game_state["PIPE_GAP_MIN"]
    pipe_gap_max = game_state["PIPE_GAP_MAX"]
    
    reward = 0.01
    if action == 1:
        bird.jump()
    
    bird.update()
    for p in pipes:
        p.update(current_speed)  # Utilise la vitesse actuelle
    
    if pipes[-1].x < WIDTH - PIPE_DISTANCE:
        rand = random.randint(pipe_gap_min, pipe_gap_max)
        pipes.append(Pipe(WIDTH+20, rand))
    
    if pipes[0].off_screen():
        pipes.pop(0)
    
    brect = bird.get_rect()
    next_pipe = None
    for p in pipes:
        if next_pipe is None and p.x + PIPE_WIDTH > bird.x:
            next_pipe = p
        if not p.passed and p.x + PIPE_WIDTH < bird.x:
            p.passed = True
            score += 1
            reward += 1.0
            # Augmente seulement la vitesse actuelle, pas la base
            current_speed += 0.05
        if p.collides_with(brect):
            reward = -1.0
            game_over = True
    
    if next_pipe is not None:
        center_y = next_pipe.gap_y + PIPE_GAP/2
        dy = bird.y - center_y
        reward += max(min(0.1 - 0.1*abs(dy)/100, 0.1), -0.1)
    
    if bird.y + bird.radius > HEIGHT - GROUND_HEIGHT:
        reward = -1.0
        game_over = True
    
    ground_offset += current_speed
    
    game_state.update({
        "score": score,
        "current_speed": current_speed,  # Met à jour la vitesse actuelle
        "ground_offset": ground_offset,
        "game_over": game_over
    })
    
    return reward, game_over


def rl_obs():
    bird = game_state["bird"]
    pipes = game_state["pipes"]
    PIPE_GAP = game_state["PIPE_GAP"]

    next_pipe = None
    for p in pipes:
        if p.x + PIPE_WIDTH > bird.x:
            next_pipe = p
            break

    if next_pipe is None:
        next_pipe_x = bird.x + PIPE_DISTANCE
        next_pipe_gap_y = HEIGHT/2
    else:
        next_pipe_x = next_pipe.x
        next_pipe_gap_y = next_pipe.gap_y

    return (float(bird.y), float(bird.vel), float(next_pipe_x), float(next_pipe_gap_y))

def rl_render():
    global surface
    bird = game_state["bird"]
    pipes = game_state["pipes"]
    ground_offset = game_state["ground_offset"]

    surface.fill(BG_COLOR)
    for i in range(3):
        cx = (pygame.time.get_ticks()*0.2 + i*140) % (WIDTH+100) - 50
        pygame.draw.ellipse(surface, (255,255,255), (cx, 40+i*30, 80, 30))
    for p in pipes:
        p.draw(surface)
    bird.draw(surface)
    draw_ground(surface, ground_offset)
    return pygame.surfarray.array3d(surface)

# --- Jeu classique ---
def run_game():
    rl_init()
    while True:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_state["game_over"]:
                    game_state["bird"].jump()
                if event.key == pygame.K_r and game_state["game_over"]:
                    rl_init()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not game_state["game_over"]:
                        game_state["bird"].jump()
                    else:
                        rl_init()

        rl_step(0)  # mise à jour sans action
        rl_render()
        pygame.display.flip()

if __name__ == "__main__":
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    run_game()
