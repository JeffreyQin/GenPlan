from enum import Enum
from collections import defaultdict, deque

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
        self.action_values: list[int, int, int, int] = [0,0,0,0]#is it correct to initialize everything to 0?

        self.id = parent_id + str(parent_a)
        self.encode()

    def encode(self):
        """
        encode current node into a string
        
        used for tracking purposes by pomcp
        """
        self.encoding = ""
        self.encoding += str(self.agent_pos[0]) + "," + str(self.agent_pos[1]) + "|"

        obs_list = list(self.obs)
        obs_list = sorted(obs_list)
        for obs in obs_list:
            self.encoding += str(obs[0]) + "," + str(obs[1]) + "|"


class Tree():

    def __init__(self, map, fragment, segmentation):

        self.subtrees = {}
        self.height, self.width = map.shape[0], map.shape[1]
        
        # determine start position
        num_unobserved = 0
        for r in range(self.height):
            for c in range(self.width):
                if map[r, c] == Cell.AGENT.value:
                    agent_pos = (r, c)
                elif map[r, c] == Cell.UNOBSERVED.value:
                    num_unobserved += 1

        obs = self.get_observation(map, agent_pos)
        map = self.update_map(map, agent_pos, agent_pos)
        num_unobserved -= len(obs)

        self.tree = {0: {'pos': agent_pos,
                'remains': num_unobserved,
                'path_from_par': [],
                'path_from_root': [],
                'steps_from_par': 0,
                'steps_from_root': 0,
                'celldistances': set(),
                'children': set(),
                'pid': None,
                'depth': 0,
                'copies_explored': [],
                }}

        # bfs queue
        agenda = deque()
        agenda.append((0, map)) # (node idx, current map)

        while agenda:

            node_idx, updated_map = agenda.popleft()
            agent_pos = self.tree[node_idx]['pos']
            node_depth = self.tree[node_idx]['depth']

            # construct tree node representing current agent pos
            if agent_pos in segmentation.keys():
                copy, base_r, base_c = segmentation[agent_pos]
                if copy['top left'] not in self.tree[node_idx]['copies_explored']:
                    if (base_r, base_c) not in self.subtrees.keys():
                        # fragment subtree for current pos hasn't been created
                        self.subtrees[(base_r, base_c)] = self.construct_subtree(fragment, agent_pos)
                    continue
                    
            
            for path, path_obs in self.next_path(updated_map, agent_pos):
                branch = {  'pos': path[-1],
                        'remains': self.tree[node_idx]['remains'] - len(path_obs),
                        'path_from_par': path,
                        'path_from_root': self.tree[node_idx]['path_from_root'][:-1] + path,
                        'steps_from_par': len(path) - 1,
                        'steps_from_root': self.tree[node_idx]['steps_from_root'] + len(path) - 1,
                        'celldistances': path_obs,
                        'children': set(),
                        'pid': node_idx,
                        'depth': node_depth + 1,
                        'map': updated_map, # map where nid started from!
                        'copies_explored': [corner for corner in self.tree[node_idx]['copies_explored']]
                    }
                
                new_node_idx = max(self.tree) + 1
                agenda.append((new_node_idx, self.update_map(updated_map, path[0], path[-1])))

                self.tree[node_idx]['children'].add(new_node_idx)
                self.tree[new_node_idx] = branch

    
    
    def construct_subtree(self, fragment, init_pos):
        """
        method to construct fragment planning subtree 
        """
        return fragment


    def next_path(self, map: list[list[int, int]], pos: tuple[int, int]):

        (height, width) = map.shape

        agenda = deque()
        agenda.append([[pos]])
        obs = dict()

        while agenda:
            path = agenda.popleft()
            r_, c_ = path[-1]

            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                # clip within bounds
                r, c = max(min(r_ + dir[0], height - 1), 0), max(min(c_ + dir[1], width - 1), 0)

                # ignore if neighbor is wall
                if map[r, c] == Cell.WALL.value or (r, c) in path:
                    continue

                # if new observation is made, then we have a path
                new_obs = self.get_observation(map, (r, c))

                if new_obs:
                    if (r, c) in obs.get(frozenset(obs), set()):
                        continue

                    obs.setdefault(frozenset(new_obs), set()).add(r, c)
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

                if self.map[r_][c_] == Cell.WALL.value:
                    break

                columns.append(c_)

                if self.map[r_][c_] == Cell.UNOBSERVED.value:
                    observations.add((r_, c_))
            
            if not columns:
                break

            c_left = columns[-1]

        # 2nd quadrant
        c_right = self.map_dims[1] - 1

        for r_ in range(r, -1, -1):
            columns = []
            for c_ in range(c, c_right+1):

                if self.map[r_][c_] == Cell.WALL.value:
                    break

                columns.append(c_)

                if self.map[r_][c_] == Cell.UNOBSERVED.value:
                    observations.add((r_, c_))

            if not columns:
                break

            c_right = columns[-1]

        # 3rd quadrant
        c_left = 0
        
        for r_ in range(r, self.map_dims[0]):
            columns = []

            for c_ in range(c, c_left-1, -1):

                if self.map[r_][c_] == Cell.WALL.value:
                    break

                columns.append(c_)

                if self.map[r_][c_] == Cell.UNOBSERVED.value:
                    observations.add((r_, c_))
            
            if not columns:
                break

            c_left = columns[-1]

        # 4th quadrant
        c_right = self.map_dims[1] - 1
        
        for r_ in range(r, self.map_dims[0]):
            columns = []
            for c_ in range(c, c_right+1):

                if self.map[r_][c_] == Cell.WALL.value:
                    break

                columns.append(c_)

                if self.map[r_][c_] == Cell.UNOBSERVED.value:
                    observations.add((r_, c_))

            if not columns:
                break

            c_right = columns[-1]
            
        return observations
    

    def update_map(self, map, old_pos, new_pos):

        obs = self.get_observation(map, new_pos)
        
        updated_map = [[Cell.OBSERVED.value if (r, c) in obs else map[r, c]
                        for c in range(self.width)]
                        for r in range(len(self.height))]
        
        updated_map[old_pos[0], old_pos[1]] = Cell.OBSERVED.value
        
        return updated_map


