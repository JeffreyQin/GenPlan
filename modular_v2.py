from collections import defaultdict
import numpy as np
from escape_search import EscapeMCTS, EscapeNode
from tree_builder import Cell
from fragment_utils import segment_map, fragment_to_map_coords

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


def run_escape_search(fragment: np.ndarray, agent_pos: tuple[int, int], copy: dict, map: np.ndarray):
    """
    find optimal escape path
    """
    # Compute exit penalties for the fragment
    exit_penalty = compute_exit_penalty(fragment, copy, map)
    
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