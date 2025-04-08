
from collections import defaultdict
from tree_builder import Cell, Action

class Generator():

    def __init__(self, map, agent_r, reward, penalty):
        
        self.map: list[list[int]] = map
        self.map_dims: tuple[int, int] = map.shape
        self.agent_r: int = agent_r
        self.reward: float = reward
        self.penalty: float = penalty

        self.rooms: set[tuple[int, int]] = set()
        
        for r in range(self.map_dims[0]):
            for c in range(self.map_dims[1]):
                if map[r, c] != Cell.WALL.value:
                    self.rooms.add((r, c))


    def is_observable(self, pos: tuple[int, int], target: tuple[int, int]):
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
    
    def get_init_obs(self, agent_pos: tuple[int, int]):
        """
        simply returns observable rooms from current agent position

        used for initialization
        """

        obs = set()
        for room in self.rooms:
            if self.is_observable(agent_pos, room):
                obs.add(room)
        return obs
    
    def generate(self, exit_state: tuple[int, int], agent_pos: tuple[int, int], curr_obs: set[tuple[int, int]], action: int):
        """
        runs black box generator by performing input action on current position and observation

        return new position, observation, and reward
        """
        
        if action == Action.UP.value:
            dest = (agent_pos[0] - 1, agent_pos[1])
        elif action == Action.RIGHT.value:
            dest = (agent_pos[0], agent_pos[1] + 1)
        elif action == Action.DOWN.value:
            dest = (agent_pos[0] + 1, agent_pos[1])
        elif action == Action.LEFT.value:
            dest = (agent_pos[0], agent_pos[1] - 1)
        
        # summarize new observed cells
        new_obs = curr_obs.copy()
        exit_found = False
        for room in self.rooms:
            if self.is_observable(dest, room):
                new_obs.add(room)
                if room == exit_state:
                    exit_found = True
        
        if exit_state in new_obs:
            # huge reward if exit found
            return exit_found, dest, new_obs, self.reward
        else:
            # penalize for step
            return exit_found, dest, new_obs, self.penalty


