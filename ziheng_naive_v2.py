
import numpy as np
from tree_builder import Tree
import math
import json
from maps.ziheng_maps import *
from modular_v2 import *

import builtins

def step_heuristic(tree, segmentation, node_id):
    steps = tree.nodes[node_id]["steps_from_parent"]

    if tree.nodes[node_id]['pos'] in segmentation.keys():
        return steps, [tree.nodes[node_id]['pos']]
    else:
        curr_min = float('inf')
        curr_min_child = None
        curr_min_path = None
        for child_id in tree.nodes[node_id]['children']:
            subsequent_steps, subsequent_path = step_heuristic(tree, segmentation, child_id)
            if subsequent_steps < curr_min:
   
                curr_min = subsequent_steps
                curr_min_child = tree.nodes[child_id]['pos']
                curr_min_path = subsequent_path

        return steps + curr_min, [tree.nodes[node_id]['pos']] + curr_min_path
       



def segment_map(fragment, copies):
    """
    Creates a dictionary mapping each index pair to its corresponding copy.
    Only indices inside a copy appear in the keys.
    """
    segmentation = {}
    height, width = len(fragment), len(fragment[0])
    for copy in copies:
        tl_i, tl_j = copy["top left"]
        if copy["rotations"] % 2 == 1:
            cp_height = width
            cp_width = height
        else:
            cp_height = height
            cp_width = width
        for del_i in range(cp_height):
            for del_j in range(cp_width):
                i, j = tl_i + del_i, tl_j + del_j
                # Would need overall size...
                # if i >= height or j >= width:
                #    continue

                # TODO: Optimize by creating and transforming
                # the full index matrix at once.
                
                # map del_i, del_j to offsets in original fragment

                mask = np.zeros((cp_height, cp_width))
                mask[del_i,del_j] = 1
                mask = np.rot90(mask, -copy["rotations"])
                if copy["reflect"]:
                    mask = np.flip(mask, 1)
                base_i,base_j = np.where(mask)
                base_i,base_j = int(base_i[0]),int(base_j[0])
                segmentation[(i,j)] = (copy, base_i, base_j)
    return segmentation
   

def construct_copy(fragment, reflect, rotations):
    if reflect:
        fragment = np.flip(fragment, 1)
    fragment = np.rot90(fragment, rotations)
    return fragment

def find_agent_pos(map: np.array):
    for i in range(map.shape[0]):
        for j in range (map.shape[1]):
            if map[i, j] == 3:
                return (i, j)
    
    return None

def test_single_iteration():
    map = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 2, 3]
    ])

    map2 = np.array([
        [0, 0, 0, 2, 0, 0, 0],
        [2, 2, 0, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 2, 0],
        [2, 2, 2, 2, 2, 2, 2],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 2, 2, 0],
        [3, 2, 2, 0, 0, 2, 0],
    ])


   
    map3 = np.array([
    [0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 2, 0],
    [3, 2, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 0, 2, 0, 2, 2, 2],
    [0, 2, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 2, 2, 0, 2, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
])

    fragment3 = np.array([
        [0, 0, 0],
        [2, 2, 0],
        [0, 2, 0],
    ])

    copies3 = [
        {"top left": (0,4), "reflect": False, "rotations": 0},
        {"top left": (6,3), "reflect": False, "rotations": 0},
        {"top left": (4,6), "reflect": False, "rotations": 0}
    ]

    fragment = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    copies = [
        {"top left": (0,0), "reflect": False, "rotations": 0},
        {"top left": (3,0), "reflect": False, "rotations": 0}
    ]

    segmentation = segment_map(fragment3, copies3)

    subtrees = dict()
    copies_unexplored = set()
    tree = Tree(map3, fragment3, segmentation, copies_unexplored, subtrees)
    
    print('nearest fragment path from ' + str((3,0)))
    result = step_heuristic(tree, segmentation, 0)

    print(result)

import pygame
import sys

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

def visualize(map_array, agent_path):
    pygame.init()
    rows, cols = map_array.shape
    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE))
    pygame.display.set_caption("Agent Full Path")

    # Font for annotations
    font = pygame.font.SysFont(None, 16)  # Small font for corner numbers

    # 1) Pre-render map once
    map_surf = pygame.Surface(screen.get_size())
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            tile = map_array[i][j]
            if tile == 0:
                col = COLOR_WALL
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
        for p in agent_path
    ]
    if len(px_pts) > 1:
        pygame.draw.lines(trail_surf, (255, 0, 0), False, px_pts, 7)
        draw_arrow(trail_surf, px_pts[-2], px_pts[-1], (255, 0, 0))

    # 3) Annotations
    corner_used = {}  # (i,j) → 'top-right' or 'bottom-left'

    for idx, (i, j) in enumerate(agent_path):
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

    end_i, end_j = agent_path[-1]
    center = (end_j * TILE_SIZE + TILE_SIZE//2, end_i * TILE_SIZE + TILE_SIZE//2)
    pygame.draw.circle(screen, COLOR_AGENT, center, TILE_SIZE//3)

    start_i, start_j = agent_path[0]
    start_center = (start_j * TILE_SIZE + TILE_SIZE//2, start_i * TILE_SIZE + TILE_SIZE//2)
    pygame.draw.circle(screen, (0, 255, 255), start_center, TILE_SIZE//3)

    pygame.display.flip()

    clock = pygame.time.Clock()
    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


def visualize_after_checkpoint(map_array, pos_indices, agent_path):

    explored_checkpoints = list()

    pygame.init()
    rows, cols = map_array.shape
    num_empty_spaces = 0
    for x in range(rows):
        for y in range(cols):
            if map_array[x][y] == 3 or map_array[x][y] == 2:
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

                        if(past_pos[0] > agent_path[path_index][0]):
                            action = 0
                        elif(past_pos[0] < agent_path[path_index][0]):
                            action = 3
                        elif(past_pos[1] > agent_path[path_index][1]):
                            action = 2
                        elif(past_pos[1] < agent_path[path_index][1]):
                            action = 4
                        
                        past_pos = agent_path[path_index]
                        past_path.append(past_pos)
                        exit_found, new_agent_pos, new_obs, new_belief, reward = generator.generate((0,0), past_pos, set(), set(), action)
                        generator.observed.add(new_agent_pos)

                        path_index += 1

                    past_check_point = pos_indices[checkpoint_index]
                    checkpoint_index += 1
                    #observed.update(new_obs)
                    #print(observed)
                    #print(generator.observed)

                    explored_ratio = len(generator.observed) / num_empty_spaces
                    explored_checkpoints.append(explored_ratio)

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
    return explored_checkpoints

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

def visualize(map_array, agent_path):
    pygame.init()
    rows, cols = map_array.shape
    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE))
    pygame.display.set_caption("Agent Full Path")

    # Font for annotations
    font = pygame.font.SysFont(None, 16)  # Small font for corner numbers

    # 1) Pre-render map once
    map_surf = pygame.Surface(screen.get_size())
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            tile = map_array[i][j]
            if tile == 0:
                col = COLOR_WALL
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
        for p in agent_path
    ]
    if len(px_pts) > 1:
        pygame.draw.lines(trail_surf, (255, 0, 0), False, px_pts, 7)
        draw_arrow(trail_surf, px_pts[-2], px_pts[-1], (255, 0, 0))

    # 3) Annotations
    corner_used = {}  # (i,j) → 'top-right' or 'bottom-left'

    for idx, (i, j) in enumerate(agent_path):
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

    end_i, end_j = agent_path[-1]
    center = (end_j * TILE_SIZE + TILE_SIZE//2, end_i * TILE_SIZE + TILE_SIZE//2)
    pygame.draw.circle(screen, COLOR_AGENT, center, TILE_SIZE//3)

    start_i, start_j = agent_path[0]
    start_center = (start_j * TILE_SIZE + TILE_SIZE//2, start_i * TILE_SIZE + TILE_SIZE//2)
    pygame.draw.circle(screen, (0, 255, 255), start_center, TILE_SIZE//3)

    pygame.display.flip()

    clock = pygame.time.Clock()
    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


def compute_explored_from_path(fragment: list[list[int, int]], agent_path: list[tuple[int, int]], clip_indices):
    
    generator = Generator(fragment)
    generator.get_init_state(agent_path[0])

    clip_ptr = 0
    clipped_percent_observed = list()
    
    for i in range(1, len(agent_path)):
        if (agent_path[i][0] > agent_path[i - 1][0]):
            action = 0
        elif (agent_path[i][0] < agent_path[i - 1][0]):
            action = 2
        elif (agent_path[i][1] > agent_path[i - 1][1]):
            action = 1
        elif (agent_path[i][1] < agent_path[i - 1][1]):
            action = 3
        else:
            action = 0
                        
        _, new_agent_pos, _, _, _ = generator.generate((0,0), agent_path[i - 1], set(), set(), action)
        generator.observed.add(new_agent_pos)

        if i == clip_indices[clip_ptr]:
            curr_percent_observed = float(len(generator.observed)) / float(len(generator.rooms))
            clipped_percent_observed.append(curr_percent_observed)
            clip_ptr += 1
            if clip_ptr >= len(clip_indices):
                break

    final_observations = generator.get_observation(agent_path[-1])
    generator.observed = generator.observed.union(final_observations)

    return clipped_percent_observed



def run_naive_pomcp_time(fragment: list[list[int, int]], checkpoints):

    print(checkpoints)
    # for experimentation
    # set max depth to number of empty rooms in fragment

    height, width = fragment.shape

    # find agent pos
    agent_pos = None
    for r in range(height):
        for c in range(width):
            if fragment[r, c] == Cell.AGENT.value:
                agent_pos = (r, c)
                break
        if agent_pos:
            break

    generator = Generator(fragment)
    pomcp_algorithm = FragmentPOMCP(generator, depth=len(generator.rooms))

    init_obs, init_belief = generator.get_init_state(agent_pos)
    root_node = Node(agent_pos, init_obs, init_belief, parent_id="", parent_a=0)

    globals.naive_start_time = time.time()

    path = list()
    moves = list()
    ctr = 0

    checkpoint_ptr = 0
    clip_indices = list()

    while ctr <= len(generator.rooms) * 10:

        path.append(root_node.agent_pos)

        if time.time() - globals.naive_start_time >= checkpoints[checkpoint_ptr]:
            clip_indices.append(ctr)
            checkpoint_ptr += 1
            if checkpoint_ptr >= len(checkpoints):
                break

        best_action = pomcp_algorithm.search(root_node, simul_limit=False, time_limit=True)
        moves.append(best_action)

        if len(root_node.children) == 0:
            break
        else:
            root_node = root_node.children[best_action]

        ctr += 1

    return path, moves, clip_indices


def run_naive_pomcp_simul(fragment: list[list[int, int]], checkpoints: list[float]):

    height, width = fragment.shape

    # find agent pos
    agent_pos = None
    for r in range(height):
        for c in range(width):
            if fragment[r, c] == Cell.AGENT.value:
                agent_pos = (r, c)
                break
        if agent_pos:
            break

    generator = Generator(fragment)
    pomcp_algorithm = FragmentPOMCP(generator, depth=len(generator.rooms))

    init_obs, init_belief = generator.get_init_state(agent_pos)
    root_node = Node(agent_pos, init_obs, init_belief, parent_id="", parent_a=0)

    path = list()
    moves = list()
    ctr = 0

    checkpoint_ptr = 0
    clip_indices = list()

    while ctr <= len(generator.rooms):

        path.append(root_node.agent_pos)

        if globals.total_simul_actions >= checkpoints[checkpoint_ptr]:
            clip_indices.append(ctr)
            checkpoint_ptr += 1
            if checkpoint_ptr >= len(checkpoints):
                break

        best_action = pomcp_algorithm.search(root_node, simul_limit = True, time_limit = False)
        moves.append(best_action)

        if len(root_node.children) == 0:
            break
        else:
            root_node = root_node.children[best_action]

        ctr += 1

    return path, moves, clip_indices


def run_naive_pomcp(fragment: list[list[int, int]]):

    height, width = fragment.shape

    # find agent pos
    agent_pos = None
    for r in range(height):
        for c in range(width):
            if fragment[r, c] == Cell.AGENT.value:
                agent_pos = (r, c)
                break
        if agent_pos:
            break

    generator = Generator(fragment)
    pomcp_algorithm = FragmentPOMCP(generator, depth=len(generator.rooms))

    init_obs, init_belief = generator.get_init_state(agent_pos)
    root_node = Node(agent_pos, init_obs, init_belief, parent_id="", parent_a=0)

    path = [agent_pos]
    ctr = 0
    while ctr <= len(generator.rooms):

        if globals.simul_rollout_count >= globals.simul_rollout_limit:
            break

        path.append(root_node.agent_pos)

        best_action = pomcp_algorithm.search(root_node, simul_limit = True, time_limit = False)

        if len(root_node.children) == 0:
            break
        else:
            root_node = root_node.children[best_action]

        ctr += 1

    return path, globals.simul_rollout_limit

# Example usage:
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python test_naive.py <map_number>")
        sys.exit(1)

    map_number = sys.argv[1]
    map_var = f"map_{map_number}"
    fragment_var = f"fragment_{map_number}"
    copies_var = f"copies_{map_number}"

    try:
        map_data = builtins.globals()[map_var]
        fragment_data = builtins.globals()[fragment_var]
        copies_data = builtins.globals()[copies_var]
    except KeyError as e:
        print(f"Error: Could not find variable {e} for map {map_number}")
        sys.exit(1)


    time_arr = [22.798896074295044, 32.55155968666077, 53.02738571166992, 65.22498369216919, 70.35267210006714, 94.09676575660706, 107.54444766044617, 513.8766357898712, 527.9445335865021, 542.7625856399536, 712.6229438781738]
    roll_arr = [778650, 1406996, 2247131, 2325814, 2545834, 3437207, 3636088, 5648220, 6611439, 6826377, 8934920]

    with open(f'ziheng_v2_results/map_{map_number}_naive.txt', 'a') as f:
        np.savetxt(f, map_data, fmt='%d')
        f.write('\n\n')

    globals.simul_rollout_count = 0
    globals.naive_start_time = 0


    agent_path_with_simul_limit, agent_moves_with_simul_limit, simul_clip_idx  = run_naive_pomcp_simul(map_data, roll_arr)
    agent_path_with_time_limit, agent_moves_with_time_elapsed, time_clip_idx = run_naive_pomcp_time(map_data, time_arr)

    percentage_explored_simul_limit = compute_explored_from_path(map_data, agent_path_with_simul_limit, simul_clip_idx)
    percentage_explored_time_limit = compute_explored_from_path(map_data, agent_path_with_time_limit, time_clip_idx)

    with open(f'ziheng_v2_results/map_{map_number}_naive.txt', 'a') as f:

        f.write("Naive percentage explored at time checkpoints\n")
        f.write(f'{percentage_explored_time_limit}')
        f.write('\n\n')

        f.write("Naive percentage explored at rollout checkpoints\n")
        f.write(f'{percentage_explored_simul_limit}')
        f.write('\n\n')

    visualize(map_data, agent_path_with_simul_limit)