
from collections import defaultdict
from tree_builder import Cell, Action
import globals
from map_utils import *


class BridgeGenerator():
    """
    POMCP blackbox generator for between-fragment planner
    """

    def __init__(self, map, fragment, copies):

        self.map: list[list[int]] = map
        self.fragment: list[list[int]] = fragment
        self.copies: dict = copies # unexplored copies

        self.segmentation: dict = segment_map(fragment, copies)

        self.map_dims: tuple[int, int] = map.shape
        self.frag_dims: tuple[int, int] = fragment.shape

        self.rooms: set[tuple[int, int]] = set()
        self.observed: set[tuple[int, int]] = set()

        for r in range(self.map_dims[0]):
            for c in range(self.map_dims[1]):
                if map[r, c] != Cell.WALL.value:
                    self.rooms.add((r, c))
                if map[r, c] == Cell.OBSERVED.value:
                    self.observed.add((r, c))
        
        self.num_frag_rooms = np.sum(fragment != Cell.WALL.value)


    def get_fragment_penalty(self, pos: tuple[int, int], curr_obs: set) -> float:
        """
        returns estimated cost of fragment exploration
        computed as the average of pre-exploration and post-exploration stepwise penalty, accumulated across steps
        """

        init_step_penalty = (float(len(curr_obs) / float(len(self.rooms)))) - 1.0
        final_step_penalty = (float(len(curr_obs) + self.num_frag_rooms) / float(len(self.rooms))) - 1.0
        
        accum_step_penalty = self.num_frag_rooms * (init_step_penalty + final_step_penalty) / 2.0
        return accum_step_penalty

    
    def get_penalty(self, curr_obs: set[tuple[int, int]]):
        """
        returns stepwise penalty
        """

        penalty = (float(len(curr_obs) / float(len(self.rooms)))) - 1.0
        return penalty
    

    def is_fragment_border(self, pos: tuple[int, int]) -> bool:
        """
        Check if given position is at a unexplored fragment border
        """
        if pos not in self.segmentation:
            return False

        copy, base_r, base_c = self.segmentation[pos]
        return (base_r == 0 or base_r == self.frag_dims[0] - 1 or
                base_c == 0 or base_c == self.frag_dims[1] - 1)

    
    def get_observation(self, pos: tuple[int, int]):
        """
        returns set of observable cells from current position
        """
        
        observations = set()
        observations.add(pos)
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
        Get initial observation and belief state
        """
        obs = self.get_observation(pos)
        belief = [room for room in self.rooms if room not in obs]
        return obs, belief


    def update_observed(self, pos: tuple[int, int]):
        """
        keep track of observed rooms by adding new obs from current pos
        """
        new_obs = self.get_observation(pos)
        self.observed = self.observed.union(new_obs)


    def generate(self, exit_state: tuple[int, int], agent_pos: tuple[int, int], curr_obs: set[tuple[int, int]], curr_belief: set[tuple[int, int]], action: int):
        """
        Generate next state based on action
        """
        # handle explore action (if applicable)
        if action == Action.EXPLORE.value:
            return True, agent_pos, curr_obs, curr_belief, self.get_fragment_penalty(agent_pos, curr_obs)

        # handle regular actions
        dest = (agent_pos[0], agent_pos[1])
        if action == Action.UP.value:
            dest = (agent_pos[0] - 1, agent_pos[1])
        elif action == Action.RIGHT.value:
            dest = (agent_pos[0], agent_pos[1] + 1)
        elif action == Action.DOWN.value:
            dest = (agent_pos[0] + 1, agent_pos[1])
        elif action == Action.LEFT.value:
            dest = (agent_pos[0], agent_pos[1] - 1)

        if dest not in self.rooms:
            return False, agent_pos, curr_obs, curr_belief, - 1.0

        # update obs and belief
        new_obs = curr_obs.copy()
        new_belief = curr_belief.copy()
        
        observation = self.get_observation(dest)

        num_new_obs = 0
        for obs in observation:
            if obs not in new_obs:
                num_new_obs += 1
                new_obs.add(obs)
            if obs in new_belief:
                new_belief.remove(obs)

        if exit_state in new_obs:
            return True, dest, new_obs, new_belief, 0.0
        else:
            penalty = float(num_new_obs) / float(len(self.rooms)) - 1.0
            return False, dest, new_obs, new_belief, penalty



class Generator():
    """
    POMCP blackbox generator for in-fragment planner
    """

    def __init__(self, map, agent_r = 5):
        
        self.map: list[list[int]] = map
        self.map_dims: tuple[int, int] = map.shape
        self.agent_r: int = agent_r

        self.rooms: set[tuple[int, int]] = set()
        self.observed: set[tuple[int, int]] = set()

        for r in range(self.map_dims[0]):
            for c in range(self.map_dims[1]):
                if map[r, c] != Cell.WALL.value:
                    self.rooms.add((r, c))

    
    def get_observation(self, pos: tuple[int, int]):
        """
        returns set of observable cells from current position
        """

        observations = set()
        observations.add(pos)
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
    
    def update_observed(self, pos: tuple[int, int]):
        """
        keep track of observed rooms by adding new obs from current pos
        """
        new_obs = self.get_observation(pos)
        self.observed = self.observed.union(new_obs)


    def generate(self, exit_state: tuple[int, int], agent_pos: tuple[int, int], curr_obs: set[tuple[int, int]], curr_belief: set[tuple[int, int]], action: int):
        """
        runs black box generator by performing input action on current position and observation

        return new position, observation, and reward
        """

        globals.total_simul_actions += 1

        dest = (agent_pos[0], agent_pos[1])
        if action == Action.UP.value:
            dest = (agent_pos[0] - 1, agent_pos[1])
        elif action == Action.RIGHT.value:
            dest = (agent_pos[0], agent_pos[1] + 1)
        elif action == Action.DOWN.value:
            dest = (agent_pos[0] + 1, agent_pos[1])
        elif action == Action.LEFT.value:
            dest = (agent_pos[0], agent_pos[1] - 1)

        if dest not in self.rooms:
            return False, agent_pos, curr_obs, curr_belief, - 1.0
        
        # summarize new observed cells
        new_obs = curr_obs.copy()
        new_belief = curr_belief.copy()

        observation = self.get_observation(dest)

        num_new_obs = 0
        for obs in observation:
            if obs not in new_obs: 
                num_new_obs += 1
                new_obs.add(obs)
            if obs in new_belief:
                new_belief.remove(obs)
        
        if exit_state in new_obs:
            return True, dest, new_obs, new_belief, 0.0
        else:
            penalty = float(num_new_obs) / float(len(self.rooms)) - 1.0
            return False, dest, new_obs, new_belief, penalty
            

    def get_penalty(self, curr_obs: set[tuple[int, int]]):
        """
        returns stepwise penalty
        """

        penalty = (float(len(curr_obs) / float(len(self.rooms)))) - 1.0
        return penalty