import numpy as np
from value_iteration import ValueIteration
import pygame
from tree_builder import Cell

# Setup colors and cell size
CELL_SIZE = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

map = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1]
])

fragment = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
])

partition = [
    {"top left": (0,0), "reflect": False, "rotations": 0},
    {"top left": (3,0), "reflect": False, "rotations": 0}
]


pygame.init()
rows, cols = fragment.shape
screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))

for r in range(rows):
    for c in range(cols):
        rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = BLACK if fragment[r, c] == Cell.WALL.value else WHITE
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()

VI = ValueIteration()


entrances, entrance_to_part, part_visited = VI.get_fragment_entrances(map, fragment, partition)

policy_iter_1, num_iter = VI.generate_policy(map, entrances)
print(policy_iter_1)

pos = (5, 6)
VI.perform_search(map, policy_iter_1, pos, entrances, entrance_to_part, part_visited)

policy_iter_2, num_iter = VI.generate_policy(map, entrances)
print(policy_iter_2)

pos = (4, 5)
VI.perform_search(map, policy_iter_2, pos, entrances, entrance_to_part, part_visited)