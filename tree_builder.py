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

class State():
    def __init__(self, exit: tuple, agent: tuple, observed: set):
        self.exit = exit
        self.agent = agent
        self.observed = observed
        self.exit_found = False


class TreeNode():

    def __init__(self):