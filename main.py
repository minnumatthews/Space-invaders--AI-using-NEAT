import pygame
import neat
import os
import random
import visualize
import pickle  # Importing pickle for saving the best genome
import logging  # Importing logging to track performance
import time

# Initialize Pygame
pygame.init()

# Set up display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Space Invaders - AI")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up game clock
clock = pygame.time.Clock()

# Set up logging
logging.basicConfig(filename='game_performance.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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
        self.idle_time = 0  # Time the player has stayed in one place
        self.cooldown_time = 0  # Cooldown time for shooting bullets
        self.total_distance_moved = 0  # Track total player movement

    def update(self):
        pass

    def handle_action(self, action):
        previous_x = self.rect.x
        if action == 0 and self.rect.left > 0:  # Move Left
            self.rect.x -= self.speed
        elif action == 1 and self.rect.right < SCREEN_WIDTH:  # Move Right
            self.rect.x += self.speed

        # Track movement
        distance_moved = abs(self.rect.x - previous_x)
        self.total_distance_moved += distance_moved


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


# Main game function with level system up to level 10
def run_game(genomes, config):
    global all_sprites, enemies  # Make enemies global if needed elsewhere
    all_sprites = pygame.sprite.Group()
    bullets = pygame.sprite.Group()

    level = 1  # Starting level

    # Create player
    player = Player()
    all_sprites.add(player)

    # Create initial enemies
    enemies, enemy_speed = create_enemy_grid(5, 10, 2)
    all_sprites.add(enemies)

    score = 0
    total_bullets_fired = 0
    total_hits = 0
    total_misses = 0
    enemy_direction = 1
    cooldown_limit = 30
    start_time = time.time()  # Track the start time of the genome

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

    running = True
    max_level_time = 60  # Maximum allowed time to complete a level

    while running and level <= 10  # Limit levels to 10
        clock.tick(60)
        current_time = time.time()

        # Check if AI is stuck and penalize
        if current_time - start_time > max_level_time:
            genome.fitness -= 50  # Penalty for not progressing within time limit
            running = False
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for genome_id, genome in genomes:
            # AI should prioritize enemies across the whole screen, not just the center
            player_x = player.rect.centerx / SCREEN_WIDTH
            if enemies:
                closest_enemy = min(enemies, key=lambda e: abs(player.rect.centerx - e.rect.centerx))
                closest_enemy_distance = abs(player.rect.centerx - closest_enemy.rect.centerx) / SCREEN_WIDTH
                closest_enemy_y = closest_enemy.rect.y / SCREEN_HEIGHT  # Focus on enemies' Y-position too
            else:
                closest_enemy_distance = 1.0
                closest_enemy_y = 1.0

            inputs = (player_x, closest_enemy_distance, closest_enemy_y)

            output = net.activate(inputs)
            action = output.index(max(output))

            previous_position = player.rect.centerx
            player.handle_action(action)

            # Idle penalty and movement reward
            if player.rect.centerx != previous_position:
                genome.fitness += 0.1
                player.idle_time = 0
            else:
                player.idle_time += 1
                if player.idle_time > 60:
                    genome.fitness -= 1  # Increased penalty for idleness

            player.cooldown_time += 1
            if action == 2 and player.cooldown_time > cooldown_limit:
                bullet = Bullet(player.rect.centerx, player.rect.top)
                all_sprites.add(bullet)
                bullets.add(bullet)
                player.cooldown_time = 0
                total_bullets_fired += 1

            for bullet in bullets:
                hit_enemies = pygame.sprite.spritecollide(bullet, enemies, True)
                if hit_enemies:
                    score += 10
                    total_hits += 1
                    genome.fitness += 10
                    enemy_speed = min(enemy_speed + 0.1, 3)  # Limit enemy speed to prevent free-fall
                else:
                    total_misses += 1
                    genome.fitness -= 0.5  # Penalize for missed shots

        if total_bullets_fired < 5:
            genome.fitness -= 2  # Penalize for not shooting enough

        enemies_hit_edge = False
        for enemy in enemies:
            enemy.rect.x += enemy_speed * enemy_direction
            if enemy.rect.right >= SCREEN_WIDTH or enemy.rect.left <= 0:
                enemies_hit_edge = True

        if enemies_hit_edge:
            enemy_direction *= -1
            for enemy in enemies:
                enemy.rect.y += 10

            # Ensure enemy speed doesn't cause them to free fall
            enemy_speed = min(enemy_speed, 3)  # Cap the speed to avoid erratic behavior

        if pygame.sprite.spritecollide(player, enemies, False):
            genome.fitness -= 10  # Penalize for collision
            running = False

        # Progress to next level when enemies are cleared
        if len(enemies) == 0 and level < 10:
            genome.fitness += 50  # Increased reward for clearing a level
            if level == 5:
                genome.fitness += 100  # Bonus for reaching level 5
            elif level == 10:
                genome.fitness += 100  # Bonus for reaching level 10

            level += 1
            enemies.empty()  # Clear current enemies
            enemies, enemy_speed = create_enemy_grid(5, 10, 2 + level * 0.1)
            all_sprites.add(enemies)

        bullets.update()
        all_sprites.update()

        screen.fill(BLACK)
        all_sprites.draw(screen)
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        level_text = font.render(f"Level: {level}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(level_text, (SCREEN_WIDTH - 150, 10))

        pygame.display.flip()

        time_alive = current_time - start_time
        logging.info(f"Bullets Fired: {total_bullets_fired}, Hits: {total_hits}, Misses: {total_misses}, Total Distance Moved: {player.total_distance_moved}, Survival Time: {time_alive}, Highest Level: {level}")


# NEAT-related functions
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        run_game([(genome_id, genome)], config)

def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 50)

    print(f'Best genome:\n{winner}')

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Best genome saved as best_genome.pkl")
    visualize.plot_stats(stats, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == "__main__":
    config_path = r"C:\\Users\\minnu\\OneDrive\\Documents\\Space_invaders_AI-using_NEAT\\config-feedforward.txt"
    run_neat(config_path)
