from collections import defaultdict
import numpy as np

from tree_builder import Cell, Action, Node, Tree
from generator import Generator
from pomcp import POMCP


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


def step_heuristic(tree, segmentation, node_id):
    steps = tree.nodes[node_id]["steps_from_parent"]

    if tree.nodes[node_id]['pos'] in segmentation.keys():
        return steps, [tree.nodes[node_id]['pos']]
    else:
        curr_min = float('inf')
        curr_min_path = None
        for child_id in tree.nodes[node_id]['children']:
            subsequent_steps, subsequent_path = step_heuristic(tree, segmentation, child_id)
            if subsequent_steps < curr_min:
                curr_min = subsequent_steps
                curr_min_path = subsequent_path

        return steps + curr_min, [tree.nodes[node_id]['pos']] + curr_min_path
    

def fragment_planning(root_node: Node, fragment: list[list[int, int]], agent_pos: tuple[int, int]):
    generator = Generator(fragment)

    pomcp_algorithm = POMCP(generator, discount = 0)
    pomcp_algorithm.search(root_node)

    node_ptr = root_node
    path = list()
    
    while True:
        path.append(node_ptr.agent_pos)
        if len(node_ptr.children) == 0:
            break
        else:
            a_values: list[int] = [node_ptr.action_values[a] for a in range(4)]
            a_optimal: int = a_values.index(max(a_values))
            node_ptr = node_ptr.children[a_optimal]
    
    print("path inside fragment")
    print(path)
    return path[-1]


def modular_planning(map, fragment, copies):

    (height, width) = map.shape
    segmentation = segment_map(fragment, copies)

    subtrees = dict()
    copies_explored = set()

    while len(copies_explored) != len(copies):
        closest_fragment_tree = Tree(map, fragment, segmentation, copies_explored, subtrees)
        init_pos = closest_fragment_tree.init_pos

        steps, path = step_heuristic(closest_fragment_tree, segmentation, node_id=0)

        print("path to closest fragment")
        print(path)

        entrance = path[-1]
        copy, base_r, base_c = segmentation[entrance]
        
        # subtree with this entrance doesn't exist
        if (base_r, base_c) not in subtrees.keys():
            subtrees[(base_r, base_c)] = Node(entrance)

        exit = fragment_planning(subtrees[(base_r, base_c)], fragment, entrance)
        exit = fragment_to_map_coords(fragment, copy)[exit[0], exit[1]]

        copies_explored.add(copy['top left'])

        map[init_pos[0], init_pos[1]] = Cell.UNOBSERVED.value
        map[exit[0], exit[1]] = Cell.AGENT.value

    print('all fragments explored')
        