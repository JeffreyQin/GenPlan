
from collections import defaultdict
from tree_builder import State, Cell, Action
from copy import Copy


class Generator():

    def __init__(self, map: list[list], range: int, reward):
        
        self.map = map
        self.map_dims = map.shape
        self.range = range
        self.reward = reward

        self.rooms = set()
        for r in range(self.map_dims[0]):
            for c in range(self.map_dims[1]):
                if map[r, c] != Cell.WALL.value:
                    self.rooms.add((r, c))


    def is_observable(self, pos: tuple, target: tuple):
        """
        checks if target cell is observable from current cell

        observable iff line connecting the centers of squares
        does not contain a part or corner of a wall
        """
        r1, c1 = pos
        r2, c2 = target

        dr = r2 - r1
        dc = c2 - c1

        num_steps = max(abs(dr), abs(dc))
        r_increment = dr / num_steps
        c_increment = dc / num_steps
        
        r_curr = r1 + 0.5
        c_curr = c1 + 0.5

        for _ in range(num_steps):
            r_curr += r_increment
            c_curr += c_increment

            # check if current square is wall
            r_curr_coord, c_curr_coord = int(r_curr), int(c_curr)
            if self.map[r_curr_coord, c_curr_coord] == Cell.WALL.value:
                return False
            
            # check for pass through wall corner
            r_next = r_curr + r_increment
            c_next = c_curr + c_increment
            r_next_coord, c_next_coord = int(r_next), int(c_next)
            
            if r_next_coord != r_curr_coord and c_next_coord != c_curr_coord:
                if self.map[r_curr_coord, c_next_coord] == Cell.WALL.value or self.map[r_next_coord, c_curr_coord] == Cell.WALL.value:
                    return False
        return True
    

    def generate(self, state: State, action: int):
        """
        runs black box generator by performing input action on input state

        returns new state, observation, and reward
        """

        if action == Action.UP.value:
            dest = (state.agent[0] - 1, state.agent[1])
        elif action == Action.RIGHT.value:
            dest = (state.agent[0], state.agent[1] + 1)
        elif action == Action.DOWN.value:
            dest = (state.agent[0] + 1, state.agent[1])
        elif action == Action.LEFT.value:
            dest = (state.agent[0], state.agent[1] - 1)
        
        new_state = State(exit, dest, Copy(state.observed))
        
        # summarize new observed cells
        observation = set()
        for room in self.rooms:
            if self.is_observable(dest, room):
                if room not in state.observed:
                    new_state.observed.add(room)
                    observation.add(room)
                if room == state.exit:
                    new_state.exit_found = True
        
        if new_state.exit_found:
            return new_state, observation, self.reward
        else:
            # penalize for step
            return new_state, observation, -1


