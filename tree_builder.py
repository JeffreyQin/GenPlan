from enum import Enum
from collections import defaultdict, deque
import numpy as np

class Cell(Enum):
    WALL = 0
    OBSERVED = 1
    UNOBSERVED = 2
    AGENT = 3
    EXIT = 4
    ENTRANCE = 5

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Node():
    """
    Class Node for pomcp
    """
    def __init__(self, agent_pos, obs, belief, parent_id, parent_a):
        
        self.agent_pos: tuple[int, int] = agent_pos
        self.obs: set[tuple[int, int]] = obs
        self.num_visited: int = 0
        self.value: float = 0.0

        self.belief: set[tuple[int, int]] = belief
        self.children: dict[int, Node] = defaultdict(lambda: None)
        self.action_values: list[float] = [0.0, 0.0, 0.0, 0.0]

        self.id = parent_id + str(parent_a) if parent_id else "#"


class EscapeNode():
    """
    node class for escape search MCTS
    """
    def __init__(self, agent_pos: tuple[int, int], parent_id: str = "", parent_a: int = -1):
        self.agent_pos: tuple[int, int] = agent_pos
        self.num_visited: int = 0
        self.value: float = 0.0

        self.children: dict[int, EscapeNode] = defaultdict(lambda: None)
        self.action_values: list[float] = [0.0, 0.0, 0.0, 0.0]

        self.id = parent_id + str(parent_a) if parent_id else "#"



class Tree():

    def __init__(self, map, segmentation, copies_explored, copies):
        """
        map: current state of the map
        segmentation: mapping of each coordinate (r, c) to its corresponding fragment copy
        copies_explored: set of already explored fragment copies - will be ignored by planning
        subtrees: already constructed subtrees
        
        """

        (self.height, self.width) = map.shape
        self.copies_found = set()

        # determine start position
        num_unobserved = 0
        for r in range(self.height):
            for c in range(self.width):
                if map[r, c] == Cell.AGENT.value:
                    agent_pos = (r, c)
                    self.init_pos = (r, c)
                elif map[r, c] == Cell.UNOBSERVED.value:
                    num_unobserved += 1

        obs = self.get_observation(map, agent_pos)
        updated_map = self.update_map(map, agent_pos, agent_pos)
        num_unobserved -= len(obs)

        self.nodes = {
            0: {
                'pos': agent_pos,
                'remains': num_unobserved,
                'path_from_parent': [],
                'path_from_root': [],
                'steps_from_parent': 0,
                'steps_from_root': 0,
                'path_observation': set(),
                'parent_id': None, # parent node id
                'depth': 0,
                'map': map,
                'children': set()
            }
        }

        # bfs queue
        agenda = deque()
        agenda.append((0, updated_map)) # (node id, current map)

        while agenda:

            node_id, updated_map = agenda.popleft()
            agent_pos = self.nodes[node_id]['pos']
            node_depth = self.nodes[node_id]['depth']

            # if arriving at some (unexplored) fragment entrance, do not explore this node further
            if agent_pos in segmentation.keys():
                copy, base_r, base_c = segmentation[agent_pos]
                if copy['top left'] not in copies_explored:    
                    """
                    if (base_r, base_c) not in subtrees.keys():
                        # fragment subtree for current pos hasn't been created
                        subtrees[(base_r, base_c)] = self.construct_subtree(fragment, agent_pos)
                    """
                    # terminate exploration
                    if copy['top left'] not in self.copies_found:
                        self.copies_found.add(copy['top left'])
                    continue
                    
            # add each path that leads to new observation, or go to some fragment entrance
            for path, path_obs in self.next_path(updated_map, agent_pos, segmentation, copies_explored):
                branch = {
                    'pos': path[-1],
                    'remains': self.nodes[node_id]['remains'] - len(path_obs),
                    'path_from_parent': path,
                    'path_from_root': self.nodes[node_id]['path_from_root'][:-1] + path,
                    'steps_from_parent': len(path) - 1,
                    'steps_from_root': self.nodes[node_id]['steps_from_root'] + len(path) - 1,
                    'path_observation': path_obs,
                    'parent_id': node_id,
                    'depth': node_depth + 1,
                    'map': updated_map, # previous map
                    'children': set()
                }

                new_node_id = max(self.nodes) + 1
                branch_map = self.update_map(updated_map, path[0], path[-1])
                agenda.append((new_node_id, branch_map))

                self.nodes[node_id]['children'].add(new_node_id)
                self.nodes[new_node_id] = branch


    def next_path(self, map: list[list[int, int]], pos: tuple[int, int], segmentation, copies_explored):

        (height, width) = map.shape

        agenda = deque()
        agenda.append([pos])
        visited_pos = set()
        obs = dict()

        while agenda:
            path = agenda.popleft()
            r_, c_ = path[-1]

            if (r_, c_) in visited_pos:
                continue

            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                # clip within bounds
                r, c = max(min(r_ + dir[0], height - 1), 0), max(min(c_ + dir[1], width - 1), 0)

                # ignore if neighbor is wall
                if map[r, c] == Cell.WALL.value or (r, c) in path:
                    continue

                # if new observation is made, then we have a path
                new_obs = self.get_observation(map, (r, c))

                # if (r,c) is an entrance of unexplored copy, we have a path
                is_entrance = False
                if (r, c) in segmentation:
                    copy, base_r, base_c = segmentation[(r, c)]
                    if copy['top left'] not in copies_explored:
                        is_entrance = True

                if new_obs or is_entrance:
                    if (r, c) in obs.get(frozenset(new_obs), set()):
                        continue

                    obs.setdefault(frozenset(new_obs), set()).add((r, c))
                    yield path + [(r, c)], new_obs
                else:
                    agenda.append(path + [(r, c)])

        return []
    
    

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
    

    def update_map(self, map, old_pos, new_pos):

        obs = self.get_observation(map, new_pos)
        
        updated_map = np.array(
            [[Cell.OBSERVED.value if (r, c) in obs else map[r, c]
                    for c in range(self.width)]
                    for r in range(self.height)]
        )
        
        updated_map[old_pos[0], old_pos[1]] = Cell.OBSERVED.value
        
        return updated_map


