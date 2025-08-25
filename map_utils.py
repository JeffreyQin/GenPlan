import numpy as np
from tree_builder import Cell


def update_map(map: np.ndarray, observation: set):
    for obs in observation:
        map[obs[0], obs[1]] = Cell.OBSERVED.value


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


def get_observation(map: list[list[int, int]], pos: tuple[int, int]):
    """
    Returns a set of coordinates of all new rooms observed from the current pos
    """
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
