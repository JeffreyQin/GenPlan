from enum import Enum
from collections import defaultdict
from copy import Copy


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
    LEFT = 2

class State():
    
    def __init__(self, exit, agent_pos, observed):
        self.exit = exit
        self.agent_pos = agent_pos
        self.observed = observed


class Generator():

    def __init__(self, map, range, reward):
        
        self.map:list[list] = map
        self.map_dims = map.shape
        self.range = range
        self.reward = reward

        # create room index mapping
        self.room_to_idx = defaultdict(lambda: None)
        self.idx_to_room = defaultdict(lambda: None)
        
        curr_idx = 0
        for r in range(self.map_dims[0]):
            for c in range(self.map_dims[1]):
                if map[r, c] == Cell.OBSERVED.value or map[r, c] == Cell.UNOBSERVED.value or map[r, c] == Cell.AGENT.value:
                    self.room_to_idx[(r, c)] = curr_idx
                    self.idx_to_room[curr_idx] = (r, c)
                    curr_idx += 1
                elif map[r, c] == Cell.EXIT.value:
                    self.exit = (r, c)

                    

    def generate(self, state, action):

        if action == Action.UP.value:
            destination = (state.agent_pos[0] - 1, state.agent_pos[1])
        elif action == Action.RIGHT.value:
            destination = (state.agent_pos[0], state.agent_pos[1] + 1)
        elif action == Action.DOWN.value:
            destination = (state.agent_pos[0] + 1, state.agent_pos[1])
        elif action == Action.LEFT.value:
            destination = (state.agent_pos[0], state.agent_pos[1] - 1)

        # if destination is wall or out of bound
        # same state, no observation, no reward
        if self.room_to_idx[destination] is None:
            return state, set(), 0
        
        new_state = State(exit, destination, Copy(state.observed))

        # summarize new observed cells
        observation = set()
        found_exit = False
        for d in range(1, self.range + 1):
            cells = [
                (destination[0] - d, destination[1]),
                (destination[0], destination[1] + d),
                (destination[0] + d, destination[1]),
                (destination[0], destination[1] - d)
            ]
            for cell in cells:
                if self.room_to_idx[cell] is not None and cell not in state.observed:
                    observation.add(cell)
                    new_state.observed.add(cell)
                if cell == self.exit:
                    found_exit = True
            
        if found_exit:
            return new_state, observation, self.reward
        else:
            return new_state, observation, 0


