from collections import defaultdict, deque
import numpy as np

import globals
from tree_builder import Cell, Action, Node, Tree
from generator import Generator
from pomcp import POMCP
from value_iteration import ValueIteration

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


def fragment_to_map_coords(fragment, copy):
   """
   Returns a dictionary mapping every index of the fragment
   to its global indices in the map. This is essentially an inverse
   of segment_map.
   """
   height, width = len(fragment),len(fragment[0])
   indices = np.indices((height, width))
   if copy["reflect"]:
      # the first dimension is now the index type,
      # so we flip one dimension later than normal.
      indices = np.flip(indices, 2)
   indices = np.rot90(indices, copy["rotations"], (1,2))
   frag_to_map = {}
   tl_i, tl_j = copy["top left"]
   for i in range(indices.shape[1]):
      for j in range(indices.shape[2]):
         frag_to_map[(int(indices[0,i,j]), int(indices[1,i,j]))] = (i+tl_i, j+tl_j)
   return frag_to_map


def get_observation(self, map: list[list[int, int]], pos: tuple[int, int]):

    observations = set()
    (r, c) = pos
    
    # 1st quadrant
    c_left = 0
    
    for r_ in range(r, -1, -1):
        columns = []
        for c_ in range(c, c_left-1, -1):

            if map[r_][c_] == Cell.WALL.value:
                break

            columns.append(c_)

            if map[r_][c_] == Cell.UNOBSERVED.value:
                observations.add((r_, c_))
        
        if not columns:
            break

        c_left = columns[-1]

    # 2nd quadrant
    c_right = map.shape[1] - 1

    for r_ in range(r, -1, -1):
        columns = []
        for c_ in range(c, c_right+1):

            if map[r_][c_] == Cell.WALL.value:
                break

            columns.append(c_)

            if map[r_][c_] == Cell.UNOBSERVED.value:
                observations.add((r_, c_))

        if not columns:
            break

        c_right = columns[-1]

    # 3rd quadrant
    c_left = 0
    
    for r_ in range(r, map.shape[0]):
        columns = []

        for c_ in range(c, c_left-1, -1):

            if map[r_][c_] == Cell.WALL.value:
                break

            columns.append(c_)

            if map[r_][c_] == Cell.UNOBSERVED.value:
                observations.add((r_, c_))
        
        if not columns:
            break

        c_left = columns[-1]

    # 4th quadrant
    c_right = map.shape[1] - 1
    
    for r_ in range(r, map.shape[0]):
        columns = []
        for c_ in range(c, c_right+1):

            if map[r_][c_] == Cell.WALL.value:
                break

            columns.append(c_)

            if map[r_][c_] == Cell.UNOBSERVED.value:
                observations.add((r_, c_))

        if not columns:
            break

        c_right = columns[-1]
        
    return observations


def step_heuristic(tree, segmentation, copies_explored, node_id):
    """
    given closest fragment search tree, return shortest path to closest fragment
    as a sequence of coordinates
    """
    
    steps = tree.nodes[node_id]["steps_from_parent"]

    ## if arrive at fragment entrance
    if tree.nodes[node_id]['pos'] in segmentation.keys():
        copy, _, _ = segmentation[tree.nodes[node_id]['pos']]
        if copy['top left'] not in copies_explored:
            return steps, [tree.nodes[node_id]['pos']]
        
    ## recursively find next cell in the path
    curr_min = float('inf')
    curr_min_path = None
    for child_id in tree.nodes[node_id]['children']:
        subsequent_steps, subsequent_path = step_heuristic(tree, segmentation, copies_explored, child_id)
        if subsequent_steps < curr_min:
            curr_min = subsequent_steps
            curr_min_path = subsequent_path

    return steps + curr_min, [tree.nodes[node_id]['pos']] + curr_min_path


def fragment_planning(subtrees, fragment: list[list[int, int]], agent_pos: tuple[int, int]):
    """
    return the path of exploration within the fragment by pomcp
    """
    
    generator = Generator(fragment)
    #print("HERE IS THE FRAGMENT")
    #print(fragment)
    
    ## initializing root node of subtree if not already exist
    if agent_pos not in subtrees.keys():
        #print(agent_pos)
        init_obs, init_belief = generator.get_init_state(agent_pos)
        #print("DEBUG STATEMENT 1 vvvv ")
        #print( init_obs)
        #print( init_belief)
        subtrees[agent_pos] = Node(agent_pos, init_obs, init_belief, parent_id="", parent_a=0)
    root_node = subtrees[agent_pos]

    pomcp_algorithm = POMCP(generator, discount = 0)
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


def bfs(map, segmentation, copies_explored):

    (height, width) = map.shape
    
    copies_found = set()

    for r in range(height):
        for c in range(width):
            if map[r, c] == Cell.AGENT.value:
                agent_pos = (r, c)
    
    pos_q = deque()
    path_q = deque()

    pos_q.append(agent_pos)
    path_q.append([agent_pos])

    visited = set()

    while pos_q:
        (_r, _c) = pos_q.popleft()
        path = path_q.popleft()

        if (_r, _c) in visited:
            continue
        visited.add((_r, _c))

        if (_r, _c) in segmentation.keys():
            copy, base_r, base_c = segmentation[(_r, _c)]
            if copy['top left'] not in copies_explored:
                return path, agent_pos

        if _r - 1 >= 0 and map[_r - 1, _c] != Cell.WALL.value:
            pos_q.append((_r - 1, _c))
            path_q.append(path + [(_r - 1, _c)])
        if _r + 1 <= height - 1 and map[_r + 1, _c] != Cell.WALL.value:
            pos_q.append((_r + 1, _c))
            path_q.append((path + [(_r + 1, _c)]))
        if _c - 1 >= 0 and map[_r, _c - 1] != Cell.WALL.value:
            pos_q.append((_r, _c - 1))
            path_q.append((path + [(_r, _c - 1)]))
        if _c + 1 <= width - 1 and map[_r, _c + 1] != Cell.WALL.value:
            pos_q.append((_r, _c + 1))
            path_q.append((path + [(_r, _c + 1)]))

    return None, agent_pos


def modular_planning(map, fragment, copies):
    """
    end-to-end planning pipeline of global map
    goal is to explore all fragment in the global map in the order of step heuristic
    """
    segmentation = segment_map(fragment, copies)

    subtrees = dict()
    copies_explored = set()
    checkpoints = []
    agent_positions:list = [] #this is a list that contains the path so that we can visualize
    while len(copies_explored) != len(copies):
        """
        closest_fragment_tree = Tree(map, segmentation, copies_explored, copies)
        
        print("finished tree")
        if len(closest_fragment_tree.copies_found) == 0:
            break
        
        init_pos = closest_fragment_tree.init_pos

        steps, path = step_heuristic(closest_fragment_tree, segmentation, copies_explored, node_id=0)
        
        print("path to closest fragment")
        print(path)
        """
        path, init_pos = bfs(map, segmentation, copies_explored)
        #print(copies_explored)
        if path is None:
            break

        print("path to closest fragment")
        print(path)

        agent_positions.extend(path)

        
        entrance = path[-1]
        copy, base_r, base_c = segmentation[entrance]

        fragment_path = fragment_planning(subtrees, fragment, (base_r, base_c))

        print(f"number of rollouts: {globals.total_rollout}")
        print(f"we are on step: {len(agent_positions)}")

        ## convert in-fragment path to global map coordinates
        fragment_path = [fragment_to_map_coords(fragment, copy)[r, c]
                         for (r, c) in fragment_path]

        print("path inside this fragment")
        print(fragment_path)
        agent_positions.extend(fragment_path)
        checkpoints.append(len(agent_positions))
        copies_explored.add(copy['top left'])

        ## relocate agent position in the global map after fragment exploration
        map[init_pos[0], init_pos[1]] = Cell.UNOBSERVED.value
        map[fragment_path[-1][0], fragment_path[-1][1]] = Cell.AGENT.value

    print('all fragments explored')
    return agent_positions, checkpoints

