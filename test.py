# pylint: disable=no-member
"""
I WILL BE TESTING GENERATOR AND ALSO POMCP
"""

import numpy as np
import pygame
from tree_builder import Cell, Action
from generator import Generator

# Setup
CELL_SIZE = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

input_maps = {
    0: [
        [0, 2, 0, 0, 2, 0],
        [2, 2, 0, 2, 2, 0],
        [0, 2, 0, 0, 2, 0],
        [2, 2, 2, 2, 2, 2],
        [0, 2, 0, 0, 2, 0],
        [2, 2, 0, 2, 2, 0],
        [0, 2, 0, 0, 2, 0]
    ]
}

map_id = 0
map_data = np.array(input_maps[map_id])

agent_r = 1
agent_start = (3, 0)
exit_pos = (0, 4)
reward = 100 # not used
penalty = -1
gen = Generator(map_data, agent_r, penalty)
agent_pos = agent_start
obs = gen.get_init_state(agent_pos)

# Pygame Init
pygame.init()
rows, cols = map_data.shape
print(rows)
print(cols)
screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
pygame.display.set_caption("Map Test UI")
running = True

font = pygame.font.SysFont(None, 24)

print(gen.rooms)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = Action.UP.value
            elif event.key == pygame.K_DOWN:
                action = Action.DOWN.value
            elif event.key == pygame.K_LEFT:
                action = Action.LEFT.value
            elif event.key == pygame.K_RIGHT:
                action = Action.RIGHT.value
            else:
                continue

            exit_found, agent_pos, obs,belief, reward = gen.generate(exit_pos, agent_pos, obs, action)
            if exit_found:
                print("Exit found!")

    screen.fill(WHITE)

    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = GRAY if (r, c) in obs else WHITE
            if map_data[r, c] == Cell.WALL.value:
                color = BLACK
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Draw agent
    agent_rect = pygame.Rect(agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, BLUE, agent_rect)

    # Draw exit
    exit_rect = pygame.Rect(exit_pos[1] * CELL_SIZE, exit_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, exit_rect)

    pygame.display.flip()

pygame.quit()
