from collections import defaultdict
import numpy as np
import pygame
import time
import globals
from escape_search import EscapeMCTS
from tree_builder import Cell, Node, EscapeNode, BridgeNode
from map_utils import segment_map, fragment_to_map_coords, update_map
from generator import Generator, BridgeGenerator
from fragment_search import FragmentPOMCP
from bridge_search import BridgePOMCP
import math

# Constants for visualization
TILE_SIZE = 30
COLOR_WALL = (100, 100, 100)
COLOR_OBSERVED = (200, 200, 200)
COLOR_GOAL = (0, 255, 0)
COLOR_FLOOR = (255, 255, 255)
COLOR_AGENT = (255, 0, 0)

def compute_exit_penalty(fragment: np.ndarray, copy: dict, map: np.ndarray, global_coords: dict, segmentation: dict) -> dict[tuple[int, int], float]:
    """
    compute penalty for each exit of fragment
    penalty is a maximally simplistic approximation of subsequent work required to reach other unexplored fragments
    """
    height, width = fragment.shape
    map_height, map_width = map.shape
    exit_penalty = dict()

    # get all border cells
    border_cells = set()
    for c in range(0, width):
        if fragment[0, c] != Cell.WALL.value:
            border_cells.add((0, c))
        if fragment[height - 1, c] != Cell.WALL.value:
            border_cells.add((height - 1, c))
    for r in range(0, height):
        if fragment[r, 0] != Cell.WALL.value:
            border_cells.add((r, 0))
        if fragment[r, width - 1] != Cell.WALL.value:
            border_cells.add((r, width - 1))

    # validate all border cells for exit
    exits = set()
    for border in border_cells:
        coords = global_coords[border]

        is_outside_frag = False

        neighbor = list()
        if coords[0] > 0 and map[coords[0] - 1, coords[1]] != Cell.WALL.value:
            neighbor.append((coords[0] - 1, coords[1]))
        if coords[0] < map_height - 1 and map[coords[0] + 1, coords[1]] != Cell.WALL.value:
            neighbor.append((coords[0] + 1, coords[1]))
        if coords[1] > 0 and map[coords[0], coords[1] - 1] != Cell.WALL.value:
            neighbor.append((coords[0], coords[1] - 1))
        if coords[1] < map_width - 1 and map[coords[0], coords[1] + 1] != Cell.WALL.value:
            neighbor.append((coords[0], coords[1] + 1))

        for nb in neighbor:
            if nb not in segmentation.keys():
                is_outside_frag = True
            else:
                neighbor_copy, base_r, base_c = segmentation[nb]
                if neighbor_copy['top left'] != copy['top left']:
                    is_outside_frag = True
            
        if is_outside_frag:
            exits.add(border)

    # get all unobserved coords in global map
    unobserved_cells = list()
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

        if current_node.agent_pos in mcts.exit_penalty.keys():
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



def run_bridge_search(map: np.ndarray, agent_pos: tuple[int, int], fragment: np.ndarray, copies: list[dict]):
    """
    return optimal path to next unexplored fragment
    """
    
    generator = BridgeGenerator(map, fragment, copies)
    pomcp_algorithm = BridgePOMCP(generator, depth=len(generator.rooms))

    init_obs, init_belief = generator.get_init_state(agent_pos)
    root_node = BridgeNode(agent_pos, init_obs, init_belief, parent_id="", parent_a=0)

    path = list()
    moves = list()
    ctr = 0
    ## recursively compute next cell in the fragment by optimal action
    while ctr <= len(generator.rooms) * 10:
        path.append(root_node.agent_pos)
        generator.update_observed(root_node.agent_pos) # keep track of new observations made on the way

        best_action = pomcp_algorithm.search(root_node)
        moves.append(best_action)

        if len(root_node.children) == 0:
            break
        elif best_action == 4: # begin fragment search
            break
        else:
            root_node = root_node.children[best_action]

        ctr += 1
    
    return path, moves, generator.observed


def run_fragment_search(subtrees, fragment: list[list[int, int]], agent_pos: tuple[int, int]):

    # for experimentation
    # set max depth to number of empty rooms in fragment

    num_fragment_rooms = 0
    height, width = fragment.shape
    for r in range(height):
        for c in range(width):
            if fragment[r, c] != Cell.WALL.value:
                num_fragment_rooms += 1

    generator = Generator(fragment)
    pomcp_algorithm = FragmentPOMCP(generator, depth=len(generator.rooms))


    if agent_pos not in subtrees.keys():
        init_obs, init_belief = generator.get_init_state(agent_pos)
        subtrees[agent_pos] = Node(agent_pos, init_obs, init_belief, parent_id="", parent_a=0)
    
    root_node = subtrees[agent_pos]

    path = list()
    moves = list()
    ctr = 0
    while ctr <= len(generator.rooms) * 10:
        path.append(root_node.agent_pos)
        generator.update_observed(root_node.agent_pos) # keep track of new observations made on the way

        best_action = pomcp_algorithm.search(root_node)
        moves.append(best_action)

        if len(root_node.children) == 0:
            break
        else:
            root_node = root_node.children[best_action]

        ctr += 1

    return path, moves, generator.observed

def reuse_computed_policy(subtrees, agent_pos: tuple[int, int]):
    
    node_ptr = subtrees[agent_pos]
    path = list()
    
    ## recursively compute next cell in the fragment by optimal action
    ctr = 0
    while ctr < globals.exploration_steps:
        path.append(node_ptr.agent_pos)

        if len(node_ptr.children) == 0:
            break
        else:
            a_values: list[int] = [node_ptr.action_values[a] for a in range(4)]
            a_optimal: int = a_values.index(max(a_values))
            node_ptr = node_ptr.children[a_optimal]

        ctr += 1

    return path


def run_modular(map: np.ndarray, fragment: np.ndarray, copies: list[dict]):

    start_time = time.time()
    agent_path = []
    checkpoints = []
    time_checkpoints = []
    bridge_time_checkpoints = []
    escape_time_checkpoints = []
    fragment_time_checkpoints = []
    escape_rollout_checkpoints = []
    pomcp_rollout_checkpoints = []
    bridge_rollout_checkpoints = []

    map_h, map_w = map.shape
    segmentation = segment_map(fragment, copies)

    # get initial agent pos
    for r in range(map_h):
        for c in range(map_w):
            if map[r, c] == Cell.AGENT.value:
                agent_pos = (r, c)
                map[r, c] = Cell.UNOBSERVED.value


    # pre-computed policy for in-fragment planning
    fragment_subtrees = dict()

    explored_copies = set()
    i = 0
    while copies:
        i += 1
        print("NEW ITERATION")
        if(i == 12):
            break
        # perform bridge search
        bridge_start = time.time()

        remaining_copies = [copy for copy in copies if copy['top left'] not in explored_copies]
        bridge_path, bridge_moves, observation = run_bridge_search(map, agent_pos, fragment, remaining_copies)
        #update_map(map, observation)

        bridge_end = time.time()
        bridge_time_checkpoints.append(bridge_end - bridge_start)

        print("Path to next fragment")
        print(bridge_path)
        agent_path.extend(bridge_path)
        checkpoints.append(len(agent_path))

        agent_pos = bridge_path[-1] # fragment entrance
        copy, base_r, base_c = segmentation[agent_pos]
        
        # mapping from fragment coords to global coords
        coords_mapping = fragment_to_map_coords(fragment, copy)
        
        # perform fragment search
        frag_start = time.time()


        if (base_r, base_c) not in fragment_subtrees.keys():
            fragment_path, fragment_moves, observation = run_fragment_search(fragment_subtrees, fragment, (base_r, base_c))
        else:
            fragment_path = reuse_computed_policy(fragment_subtrees, (base_r, base_c))
        
        #update_map(map, [coords_mapping[obs[0], obs[1]] for obs in observation])
        
        frag_end = time.time()
        fragment_time_checkpoints.append(frag_end - frag_start)

        print("Path to explore current fragment")
        print([coords_mapping[pos[0], pos[1]] for pos in fragment_path])
        agent_path.extend([coords_mapping[pos[0], pos[1]] for pos in fragment_path])
        checkpoints.append(len(agent_path))

        # perform escape search
        fragment_agent_pos = fragment_path[-1] # fragment-relative position to begin escape

        exit_penalty = compute_exit_penalty(fragment, copy, map, coords_mapping, segmentation)# compute exit penalties for current fragment
        
        esc_start = time.time()

        escape_path = run_escape_search(fragment, exit_penalty, fragment_agent_pos)
        
        esc_end = time.time()
        escape_time_checkpoints.append(esc_end - esc_start)

        print("Path to escape current fragment")
        print([coords_mapping[pos[0], pos[1]] for pos in escape_path])
        agent_path.extend([coords_mapping[pos[0], pos[1]] for pos in escape_path])
        checkpoints.append(len(agent_path))

        escape_rollout_checkpoints.append(globals.escape_rollout_count)
        pomcp_rollout_checkpoints.append(globals.fragment_rollout_count)
        bridge_rollout_checkpoints.append(globals.bridge_rollout_count)
        checkpoint_time = time.time()
        time_checkpoints.append(checkpoint_time - start_time)

        fragment_agent_pos = escape_path[-1]

        agent_pos = coords_mapping[fragment_agent_pos[0], fragment_agent_pos[1]]

        explored_copies.add(copy['top left'])
        copies.remove(copy)
        segmentation = segment_map(fragment, copies)

    print(agent_path)
    print(checkpoints)
    return agent_path, checkpoints, escape_rollout_checkpoints, pomcp_rollout_checkpoints,  bridge_rollout_checkpoints, bridge_time_checkpoints, fragment_time_checkpoints, escape_time_checkpoints

