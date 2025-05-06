
import numpy as np
from tree_builder import Tree

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






from modular_planner import modular_planning

map = np.array([
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

fragment = np.array([
    [0, 2, 0],
    [2, 2, 2],
    [0, 2, 0],
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

fragment2 = np.array([
    [0, 0, 0],
    [2, 2, 0],
    [0, 2, 0],
])

copies2 = [
    {"top left": (0,0), "reflect": False, "rotations": 0},
    {"top left": (0,4), "reflect": False, "rotations": 0},
    {"top left": (4,4), "reflect": False, "rotations": 0}

]
copies = [
    {"top left": (0,4), "reflect": False, "rotations": 0},
    {"top left": (6,3), "reflect": False, "rotations": 0},
    {"top left": (4,6), "reflect": False, "rotations": 0}
]

map3 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 2, 3]
    ])

fragment3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2],
        [0, 0, 0, 0, 0]
    ])

copies3 = [
        {"top left": (0,0), "reflect": False, "rotations": 0},
        {"top left": (3,0), "reflect": False, "rotations": 0}
    ]

modular_planning(map3, fragment3, copies3)