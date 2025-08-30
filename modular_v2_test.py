from modular_v2 import *
import pygame
"""
global_map = np.array([
    [0, 0, 0, 2, 0, 0, 0],
    [2, 2, 0, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 2, 0],
    [2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0],
    [3, 2, 2, 0, 0, 0, 0],
])
fragment = np.array([
        [0, 0, 0],
        [2, 2, 0],
        [0, 2, 0],
])

agent_pos = (6, 0)


import collections
path = run_fragment_search(dict(), global_map, agent_pos)
print(path)
"""

# Constants
TILE_SIZE = 30
FPS = 40

# Colors
COLOR_BG = (30, 30, 30)
COLOR_FLOOR = (160, 160, 160)
COLOR_OBSERVED = (210, 210, 210)
COLOR_WALL = (50, 50, 50)
COLOR_AGENT = (255, 0, 0)
COLOR_GOAL = (0, 255, 0)
COLOR_TRAIL = (100, 100, 255)

# Arrow drawing helper – only draws the head when width=0


def draw_arrow(surface, start, end, color):
    """Draw just the arrowhead pointing from start → end."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    base_ang = math.atan2(dy, dx)
    head_len = 12
    spread   = math.radians(30)
    left_ang  = base_ang + math.pi - spread
    right_ang = base_ang + math.pi + spread

    left_pt  = (end[0] + head_len * math.cos(left_ang),
                end[1] + head_len * math.sin(left_ang))
    right_pt = (end[0] + head_len * math.cos(right_ang),
                end[1] + head_len * math.sin(right_ang))

    pygame.draw.polygon(surface, color, (left_pt, right_pt, end))

def visualize_after_checkpoint(map_array, pos_indices, agent_path):
    pygame.init()
    rows, cols = map_array.shape
    num_empty_spaces = 0
    for x in range(rows):
        for y in range(cols):
            if map_array[x][y] != Cell.WALL.value:
                num_empty_spaces += 1
                
    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE))
    pygame.display.set_caption("POMCP Moves Visualization")
    clock = pygame.time.Clock()

    generator = Generator(map_array)
    generator.get_init_state(agent_path[0])
    pygame.display.set_caption("Agent Full Path")

    # Font for annotations
    font = pygame.font.SysFont(None, 16)

    running = True
    checkpoint = 0
    past_pos = agent_path[0]
    path_index = 0
    action = 0
    past_path = [past_pos]
    observed = set()

    past_check_point = 0
    checkpoint_index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Press SPACE to get the next move from POMCP
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    
                    for _ in range(pos_indices[checkpoint_index] - past_check_point):

                        path_index += 1

                        if(past_pos[0] > agent_path[path_index][0]):
                            action = 0
                        elif(past_pos[0] < agent_path[path_index][0]):
                            action = 2
                        elif(past_pos[1] > agent_path[path_index][1]):
                            action = 1
                        elif(past_pos[1] < agent_path[path_index][1]):
                            action = 3
                        
                        past_path.append(past_pos)
                        generator.observed = generator.observed.union(generator.get_observation(past_pos))
                        
                        exit_found, new_agent_pos, new_obs, new_belief, reward = generator.generate((0,0), past_pos, set(), set(), action)
                        generator.observed = generator.observed.union(generator.get_observation(past_pos))
                        generator.observed = generator.observed.union(new_obs)
                        past_pos = agent_path[path_index]

                    generator.observed = generator.observed.union(generator.get_observation(past_pos))
                    past_check_point = pos_indices[checkpoint_index]
                    checkpoint_index += 1
                    #observed.update(new_obs)
                    #print(observed)
                    #print(generator.observed)
                            

        # Drawing routine
        map_surf = pygame.Surface(screen.get_size())
        for i in range(rows):
            for j in range(cols):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                tile = map_array[i][j]
                if tile == 0:
                    col = COLOR_WALL
                elif (i, j) in generator.observed:
                    col = COLOR_OBSERVED
                elif tile == 5 or tile == 3:
                    col = COLOR_GOAL
                else:
                    col = COLOR_FLOOR
                pygame.draw.rect(map_surf, col, rect)
                pygame.draw.rect(map_surf, (0,0,0), rect, 1)

        # 2) Trail surface: draw entire path at once
        trail_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        px_pts = [
            (p[1]*TILE_SIZE + TILE_SIZE//2, p[0]*TILE_SIZE + TILE_SIZE//2)
            for p in past_path
        ]
        if len(px_pts) > 1:
            pygame.draw.lines(trail_surf, (255, 0, 0), False, px_pts, 7)
            draw_arrow(trail_surf, px_pts[-2], px_pts[-1], (255, 0, 0))

        # 3) Annotations
        corner_used = {}  # (i,j) → 'top-right' or 'bottom-left'

        for idx, (i, j) in enumerate(past_path):
            pos = (i, j)
            text = font.render(str(idx), True, (0, 0, 0))  # black text
            text_rect = text.get_rect()

            top_right = (j * TILE_SIZE + TILE_SIZE - 2 - text_rect.width, i * TILE_SIZE + 2)
            bottom_left = (j * TILE_SIZE + 2, i * TILE_SIZE + TILE_SIZE - text_rect.height - 2)

            if corner_used.get(pos) != 'top-right':
                trail_surf.blit(text, top_right)
                corner_used[pos] = 'top-right'
            else:
                trail_surf.blit(text, bottom_left)
                corner_used[pos] = 'bottom-left'

        # 4) Final draw
        screen.blit(map_surf, (0,0))
        screen.blit(trail_surf, (0,0))


        end_i, end_j = past_path[-1]
        center = (end_j * TILE_SIZE + TILE_SIZE//2, end_i * TILE_SIZE + TILE_SIZE//2)
        pygame.draw.circle(screen, COLOR_AGENT, center, TILE_SIZE//3)

        start_i, start_j = past_path[0]
        start_center = (start_j * TILE_SIZE + TILE_SIZE//2, start_i * TILE_SIZE + TILE_SIZE//2)
        pygame.draw.circle(screen, (0, 255, 255), start_center, TILE_SIZE//3)

        explored_ratio = len(generator.observed) / num_empty_spaces
        explored_text = font.render(f"Explored: {explored_ratio:.2%}", True, (0, 0, 0))
        screen.blit(explored_text, (10, 10))

        pygame.display.flip()

        clock = pygame.time.Clock()
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# Test code
if __name__ == "__main__":
    # Test run_modular function
    print("Testing run_modular function...")
    
    map = np.array(
    [
        [0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    ])

    fragment = np.array([
        [2, 2, 2, 2, 2, 2, 2],
        [2, 0, 0, 2, 0, 2, 2],
        [2, 0, 0, 2, 0, 2, 2],
        [2, 2, 2, 2, 0, 2, 2],
    ])

    copies = [
        {"top left": (0,2), "reflect": False, "rotations": 0},
        {"top left": (0,15), "reflect": False, "rotations": 0},
        {"top left": (2,21), "reflect": True, "rotations": 0},
        {"top left": (6,1), "reflect": True, "rotations": 0},
        {"top left": (6,16), "reflect": False, "rotations": 0},
        {"top left": (11,1), "reflect": False, "rotations": 2},
        # {"top left": (15,1), "reflect": True, "rotations": 0},
        # {"top left": (15,7), "reflect": True, "rotations": 0},
        {"top left": (11,16), "reflect": True, "rotations": 2},
    ]
    

    map2 = np.array([
    [0, 0, 0, 2, 0, 0, 0],
    [2, 2, 0, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 2, 0],
    [2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0],
    [3, 2, 2, 0, 0, 0, 0],
    ])

    map_copy = map.copy()

    fragment2 = np.array([
        [0, 0, 0],
        [2, 2, 0],
        [0, 2, 0],
    ])

    copies2 = [
        {"top left": (0,0), "reflect": False, "rotations": 0},
        {"top left": (0,4), "reflect": False, "rotations": 0},

    ]

    
    print(f"Fragment shape: {fragment.shape}")
    print(f"Global map shape: {map.shape}")
    print(f"Number of copies: {len(copies)}")
    print(f"Initial agent position: (4, 4)")
    
    
    # Test the modular planning
    print("\nStarting modular planning...")
    agent_path, checkpoints = run_modular(map, fragment, copies)
    print("Modular planning completed successfully!")
    # agent_path = [(10, 1), (11, 1), (11, 1), (12, 1), (13, 1), (14, 1), (14, 2), (14, 3), (14, 4), (13, 4), (12, 4), (11, 4), (11, 5), (11, 6), (11, 7), (11, 7), (12, 7), (13, 7), (14, 7), (14, 6), (14, 6), (14, 5), (14, 4), (14, 3), (14, 2), (13, 2), (12, 2), (11, 2), (11, 1), (11, 1), (10, 1), (9, 1), (9, 1), (8, 1), (7, 1), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 7), (8, 7), (9, 7), (9, 7), (9, 6), (9, 5), (9, 4), (8, 4), (7, 4), (6, 4), (6, 3), (6, 2), (6, 1), (6, 1), (5, 1), (4, 1), (4, 2), (3, 2), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 7), (1, 7), (2, 7), (3, 7), (3, 7), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 16), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 20), (6, 19), (7, 19), (8, 19), (9, 19), (9, 19), (8, 19), (7, 19), (6, 19), (6, 20), (6, 21), (6, 21), (5, 21), (5, 21), (4, 21), (3, 21), (2, 21), (2, 22), (2, 23), (2, 24), (3, 24), (4, 24), (5, 24), (5, 25), (5, 26), (5, 27), (5, 27), (5, 26), (5, 25), (5, 24), (4, 24), (3, 24), (2, 24), (2, 23), (2, 22), (3, 22), (4, 22), (4, 21), (5, 21), (5, 21), (4, 21), (3, 21), (3, 21), (2, 21), (1, 21), (0, 21), (0, 20), (0, 19), (0, 18), (1, 18), (2, 18), (3, 18), (3, 17), (3, 16), (3, 15), (3, 15), (3, 16), (3, 17), (3, 18), (2, 18), (1, 18), (0, 18), (0, 19), (0, 20), (1, 20), (2, 20), (3, 20), (3, 21), (3, 21), (3, 20), (2, 20), (3, 20), (3, 21), (4, 21), (5, 21), (6, 21), (6, 20), (6, 19), (7, 19), (8, 19), (9, 19), (9, 18), (10, 18), (11, 18), (11, 18), (11, 19), (12, 19), (13, 19), (14, 19), (14, 20), (14, 21), (14, 20), (14, 19), (14, 18), (14, 17), (14, 16), (14, 16)]
    # checkpoints = [2, 15, 29, 32, 45, 55, 60, 69, 73, 86, 97, 103, 105, 118, 131, 134, 147, 160, 176, 188, 189]
    print("\nTest completed!")
    visualize_after_checkpoint(map_copy, checkpoints, agent_path)
    