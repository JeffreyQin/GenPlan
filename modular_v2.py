from collections import defaultdict
import numpy as np
import pygame
import time
import globals
from escape_search import EscapeMCTS
from tree_builder import Cell, Node, EscapeNode
from fragment_utils import segment_map, fragment_to_map_coords
from generator import Generator, BridgeGenerator
from pomcp import POMCP
from bridge_search import BridgePOMCP

def compute_exit_penalty(fragment: np.ndarray, copy: dict, map: np.ndarray) -> dict[tuple[int, int], float]:
    """
    compute penalty for each exit of fragment
    penalty is a maximally simplistic approximation of subsequent work required to reach other unexplored fragments
    """
    height, width = fragment.shape
    exits = set()
    exit_penalty = dict()

    # get all border cells
    for c in range(0, width - 1):
        if fragment[0, c] != Cell.WALL.value:
            exits.add((0, c))
        if fragment[height - 1, c] != Cell.WALL.value:
            exits.add((height - 1, c))
    for r in range(0, height - 1):
        if fragment[r, 0] != Cell.WALL.value:
            exits.add((r, 0))
        if fragment[r, width - 1] != Cell.WALL.value:
            exits.add((r, width - 1))

    # get global map coordinates for fragment
    global_coords = fragment_to_map_coords(fragment, copy)

    # get all unobserved coords in global map
    unobserved_cells = list()
    map_height, map_width = map.shape
    for i in range(map_height):
        for j in range(map_width):
            if map[i, j] == Cell.UNOBSERVED.value:
                unobserved_cells.append((i, j))

    # compute penalty for each exit
    for exit in exits:
        exit_global_coords = global_coords[exit]
            
        # calculate sum of manhattan distances to all unexplored cells
        total_distance = 0
        for (r, c) in unobserved_cells:
            distance = abs(exit_global_coords[0] - r) + abs(exit_global_coords[1] - c)
            total_distance += distance
        
        exit_penalty[exit] = total_distance
    
    return exit_penalty


def run_escape_search(fragment: np.ndarray, exit_penalty: dict, agent_pos: tuple[int, int]):
    """
    find optimal escape path from current explored fragment
    """
    
    # Create MCTS solver with computed penalties
    mcts = EscapeMCTS(fragment, exit_penalty)
    
    root_node = EscapeNode(agent_pos)
    escape_path = list()

    current_node = root_node
    
    # maximum steps as fragment size
    max_steps = fragment.shape[0] * fragment.shape[1]
    for _ in range(max_steps):
        escape_path.append(current_node.agent_pos)

        if mcts.is_boundary(current_node.agent_pos):
            break

        best_action = mcts.search(current_node)
        new_pos = mcts.get_new_position(current_node.agent_pos, best_action)

        # Update current node
        if current_node.children[best_action] is None:
            current_node.children[best_action] = EscapeNode(new_pos, current_node.id, best_action)
        
        current_node = current_node.children[best_action]
    
    return escape_path



def compute_fragment_utility(map: np.ndarray, fragment: np.ndarray, copy: dict):
    """
    compute the utility to explore current fragment
    """
    return np.sum(fragment == Cell.UNOBSERVED.value)

    
def run_bridge_search(map: np.ndarray, agent_pos: tuple[int, int], segmentation: dict):
    """
    return optimal path to next unexplored fragment
    """
    
    generator = BridgeGenerator(map, fragment)




def run_fragment_search(subtrees: dict, fragment: np.ndarray, agent_pos: tuple[int, int]):
    """
    return the path of exploration within the fragment by pomcp
    """

    generator = Generator(fragment)

    # initializing root node of subtree if not already exist
    if agent_pos not in subtrees.keys():
        init_obs, init_belief = generator.get_init_state(agent_pos)
        subtrees[agent_pos] = Node(agent_pos, init_obs, init_belief, parent_id="", parent_a=0)
    root_node = subtrees[agent_pos]

    pomcp_algorithm = POMCP(generator, discount = 0, num_simulations=globals.num_simulations)
    pomcp_algorithm.search(root_node)

    node_ptr = root_node
    path = list()
    
    ## recursively compute next cell in the fragment by optimal action
    while True:
        path.append(node_ptr.agent_pos)

        if len(node_ptr.children) == 0:
            break
        else:
            a_values: list[int] = [node_ptr.action_values[a] for a in range(4)]
            a_optimal: int = a_values.index(max(a_values))
            node_ptr = node_ptr.children[a_optimal]
    
    return path



def run_modular(map: np.ndarray, fragment: np.ndarray, copies: list[dict]):
    start_time = time.time()
    time_checkpoints = []
    bridge_time_checkpoints = []
    escape_time_checkpoints = []
    fragment_time_checkpoints = []
    escape_rollout_checkpoints = []
    pomcp_rollout_checkpoints = []
    bridge_rollout_checkpoints = []
    map_h, map_w = map.shape
    frag_h, frag_w = fragment.shape
    segmentation = segment_map(fragment, copies)

    # get initial agent pos
    for r in range(map_h):
        for c in range(map_w):
            if map[r, c] == Cell.AGENT.value:
                agent_pos = (r, c)
                map[r, c] = Cell.UNOBSERVED.value

    # compute fragment utilities
    fragment_utility = dict()
    for copy in copies:
        fragment_utility[copy['top left']] = compute_fragment_utility(fragment, map, copy)
    
    
    # pre-computed policy for in-fragment planning
    fragment_subtrees = dict()

    explored_copies = set()

    while len(copies) != len(explored_copies):
        #bridge section of the search
        bridge_start = time.time()
        bridge_path = run_bridge_search(map, agent_pos, fragment_utility, segmentation)
        bridge_end = time.time()
        bridge_time_checkpoints.append(bridge_end - bridge_start)
        agent_pos = bridge_path[-1] # fragment entrance
        copy, base_r, base_c = segmentation[agent_pos]

        # mapping from fragment coords to global coords
        coords_mapping = fragment_to_map_coords(fragment, copy)
        
        #fragment section of the search
        frag_start = time.time()
        fragment_path = run_fragment_search(fragment_subtrees, fragment, (base_r, base_c))
        frag_end = time.time()
        fragment_time_checkpoints.append(frag_end - frag_start)
        fragment_agent_pos = fragment_path[-1] # fragment-relative position to begin escape
        
        exit_penalty = compute_exit_penalty(fragment, copy, map)# compute exit penalties for current fragment

        esc_start = time.time()
        escape_path = run_escape_search(fragment, exit_penalty, fragment_agent_pos)
        esc_end = time.time()
        escape_time_checkpoints.append(esc_end - esc_start)
        
        escape_rollout_checkpoints.append(globals.escape_rollout_count)
        pomcp_rollout_checkpoints.append(globals.simul_rollout_count)
        bridge_rollout_checkpoints.append(globals.bridge_rollout_count)
        checkpoint_time = time.time()
        time_checkpoints.append(checkpoint_time - start_time)

        fragment_agent_pos = escape_path[-1]

        agent_pos = coords_mapping[fragment_agent_pos[0], fragment_agent_pos[1]]

        explored_copies.add(copy['top left'])
    

def visualize_after_checkpoint(map_array, pos_indices, agent_path):
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
    fragment = np.array([
        [0, 2, 0, 0, 2, 0],
        [0, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 2, 2],
        [0, 2, 0, 0, 0, 0],
        [0, 2, 2, 2, 3, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    
    # Create a sample copy dictionary for testing
    copy = {
        "top left": (0, 0),
        "rotations": 0,
        "reflect": False
    }
    
    # Create a sample global map for testing
    map = np.full((20, 20), Cell.UNOBSERVED.value)
    map[0:6, 0:6] = fragment  # Place fragment in the map
    
    agent_pos = (4, 4)
    
    # Test the escape search
    result = run_escape_search(fragment, agent_pos, copy, map)
    print("Escape path:", result)