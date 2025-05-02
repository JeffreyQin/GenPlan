
from collections import defaultdict
from tree_builder import Cell, Action

class Generator():

    def __init__(self, map, agent_r = 5, penalty = -1.0):
        
        self.map: list[list[int]] = map
        self.map_dims: tuple[int, int] = map.shape
        self.agent_r: int = agent_r
        self.penalty: float = penalty

        self.rooms: set[tuple[int, int]] = set()

        for r in range(self.map_dims[0]):
            for c in range(self.map_dims[1]):
                if map[r, c] != Cell.WALL.value:
                    self.rooms.add((r, c))

    
    def get_observation(self, pos: tuple[int, int]):

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

    def get_init_state(self, pos: tuple[int, int]):
        """
        get set of currently observable cells, and unobservable cells (as belief)

        for initialization purposes
        """
        obs = self.get_observation(pos)

        belief = set()
        for room in self.rooms:
            if room not in obs:
                belief.add(room)

        return obs, belief
    

    def generate(self, exit_state: tuple[int, int], agent_pos: tuple[int, int], curr_obs: set[tuple[int, int]], curr_belief: set[tuple[int, int]], action: int):
        """
        runs black box generator by performing input action on current position and observation

        return new position, observation, and reward
        """

        dest = (agent_pos[0], agent_pos[1])
        if action == Action.UP.value:
            dest = (agent_pos[0] - 1, agent_pos[1])
        elif action == Action.RIGHT.value:
            dest = (agent_pos[0], agent_pos[1] + 1)
        elif action == Action.DOWN.value:
            dest = (agent_pos[0] + 1, agent_pos[1])
        elif action == Action.LEFT.value:
            dest = (agent_pos[0], agent_pos[1] - 1)

        # print(f"{agent_pos } this the agent pos")
        # print(action)
        # print( f"{dest} this where the agent gonna be")

        if dest not in self.rooms:
            return False, agent_pos, curr_obs, curr_belief, self.penalty*3 #remember to ask jeffery ab this
        
        # summarize new observed cells
        new_obs = curr_obs.copy()
        new_belief = curr_belief.copy()

        observation = self.get_observation(dest)
        for obs in observation:
            if obs not in new_obs: 
                new_obs.add(obs)
            if obs in new_belief:
                new_belief.remove(obs)


        if exit_state in new_obs:
            return True, dest, new_obs, new_belief, 0.0 #remmeber to change to 0
        else:
            # penalize for step
            return False, dest, new_obs, new_belief, self.penalty


