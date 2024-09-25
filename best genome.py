import pygame
import neat
import pickle
import os
import time

# Initialize Pygame
pygame.init()

# Set up display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Space Invaders - Best Genome")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up game clock
clock = pygame.time.Clock()

# Load images
player_image = pygame.image.load(r"C:\\Users\\minnu\\OneDrive\\Documents\\Space_invaders_AI-using_NEAT\\assets\\images\\player.png").convert_alpha()
green_enemy_image = pygame.image.load(r"C:\\Users\\minnu\\OneDrive\\Documents\\Space_invaders_AI-using_NEAT\\assets\\images\\green.png").convert_alpha()
red_enemy_image = pygame.image.load(r"C:\\Users\\minnu\\OneDrive\\Documents\\Space_invaders_AI-using_NEAT\\assets\\images\\red.png").convert_alpha()
yellow_enemy_image = pygame.image.load(r"C:\\Users\\minnu\\OneDrive\\Documents\\Space_invaders_AI-using_NEAT\\assets\\images\\yellow.png").convert_alpha()

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = player_image
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 10
        self.speed = 5
        self.cooldown_time = 0  # Cooldown time for shooting bullets

    def handle_action(self, action):
        if action == 0 and self.rect.left > 0:  # Move Left
            self.rect.x -= self.speed
        elif action == 1 and self.rect.right < SCREEN_WIDTH:  # Move Right
            self.rect.x += self.speed

    def shoot(self):
        if self.cooldown_time == 0:
            bullet = Bullet(self.rect.centerx, self.rect.top)
            all_sprites.add(bullet)
            bullets.add(bullet)
            self.cooldown_time = 20  # Cooldown between shots

    def update(self):
        if self.cooldown_time > 0:
            self.cooldown_time -= 1

# Bullet class for player shooting
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.top = y
        self.speed = -10  # Bullet moves upward

    def update(self):
        self.rect.y += self.speed
        if self.rect.bottom < 0:  # Bullet goes off the top of the screen
            self.kill()

# Enemy class
class Enemy(pygame.sprite.Sprite):
    def __init__(self, image, x, y):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

# Function to create enemies
def create_enemy_grid(rows, cols, enemy_speed):
    enemies = pygame.sprite.Group()
    enemy_width = red_enemy_image.get_width()
    enemy_height = red_enemy_image.get_height()
    padding = 20
    start_x = (SCREEN_WIDTH - (enemy_width + padding) * cols) // 2

    for row in range(rows):
        for col in range(cols):
            x = start_x + col * (enemy_width + padding)
            y = 50 + row * (enemy_height + padding)

            if row < 2:
                enemy = Enemy(red_enemy_image, x, y)
            elif row < 4:
                enemy = Enemy(green_enemy_image, x, y)
            else:
                enemy = Enemy(yellow_enemy_image, x, y)

            enemies.add(enemy)

    return enemies, enemy_speed

# Function to run the best genome
def run_best_genome(genome, config):
    global all_sprites, bullets, enemies
    all_sprites = pygame.sprite.Group()
    bullets = pygame.sprite.Group()

    # Create player
    player = Player()
    all_sprites.add(player)

    # Create initial enemies
    enemies, enemy_speed = create_enemy_grid(5, 10, 2)
    all_sprites.add(enemies)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    enemy_direction = 1  # Track enemy movement direction
    running = True

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # AI player logic
        player_x = player.rect.centerx / SCREEN_WIDTH
        if enemies:
            closest_enemy = min(enemies, key=lambda e: abs(player.rect.centerx - e.rect.centerx))
            closest_enemy_distance = abs(player.rect.centerx - closest_enemy.rect.centerx) / SCREEN_WIDTH
            closest_enemy_y = closest_enemy.rect.y / SCREEN_HEIGHT
        else:
            closest_enemy_distance = 1.0
            closest_enemy_y = 1.0

        # Activate NEAT to decide the player's action
        inputs = (player_x, closest_enemy_distance, closest_enemy_y)
        output = net.activate(inputs)
        action = output.index(max(output))

        player.handle_action(action)

        if action == 2:  # If the action is shooting, shoot a bullet
            player.shoot()

        # Move enemies
        enemies_hit_edge = False
        for enemy in enemies:
            enemy.rect.x += enemy_speed * enemy_direction
            if enemy.rect.right >= SCREEN_WIDTH or enemy.rect.left <= 0:
                enemies_hit_edge = True

        if enemies_hit_edge:
            enemy_direction *= -1  # Reverse direction
            for enemy in enemies:
                enemy.rect.y += 10  # Move enemies down

        # Check for collisions between bullets and enemies
        for bullet in bullets:
            hit_enemies = pygame.sprite.spritecollide(bullet, enemies, True)
            if hit_enemies:
                bullet.kill()

        # Check for collisions between enemies and the player
        if pygame.sprite.spritecollide(player, enemies, False):
            print("Player hit by enemy! Game Over!")
            running = False

        # Check if enemies reach the bottom of the screen (lose condition)
        for enemy in enemies:
            if enemy.rect.bottom >= SCREEN_HEIGHT:
                print("Enemy reached the bottom! Game Over!")
                running = False

        # Update game objects
        all_sprites.update()
        bullets.update()

        # Draw everything
        screen.fill(BLACK)
        all_sprites.draw(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    # Load the best genome
    with open("best_genome.pkl", "rb") as f:
        best_genome = pickle.load(f)

    # Load the NEAT configuration
    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Run the best genome
    run_best_genome(best_genome, config)
