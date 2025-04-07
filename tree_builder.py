from enum import Enum

class Cell(Enum):
    WALL = 0
    OBSERVED = 1
    UNOBSERVED = 2
    AGENT = 3
    EXIT = 4

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Node():
    """
    Class Node for pomcp
    """
    def __init__(self, agent_pos, obs, num_visited = 0, value = 0):

        self.agent_pos: tuple[int, int] = agent_pos
        self.obs: set[tuple[int, int]] = obs
        self.num_visited: int = num_visited
        self.value: float = value

        self.belief: set[tuple[int, int]] = dict()
        self.children: list[Node] = list()

        self.encoding: str = ""
        self.encode()

    def encode(self):
        """
        encode current node into a string
        
        used for tracking purposes by pomcp
        """
        self.encoding += str(self.agent_pos[0]) + "," + str(self.agent_pos[1]) + "|"
        self.encoding += str(len(self.obs)) + "|"

        obs_list = list(self.obs)
        obs_list = sorted(obs_list)
        for obs in obs_list:
            self.encoding += str(obs[0]) + "," + str(obs[1]) + "|"
        