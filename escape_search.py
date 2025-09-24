import random
import math
import numpy as np
import globals
from tree_builder import Cell, EscapeNode
from map_utils import *


class EscapeMCTS():
    """
    Monte Carlo Tree Search for escape search
    Purpose is to find optimal path to a boundary cell that minimizes penalty
        - step penalty
        - boundary penalty: compute efficient approximation of subsequent work to go to other unexplored fragments
    """
    def __init__(self, fragment: np.ndarray, exit_penalty: dict[tuple[int, int], float], exploration: float = math.sqrt(2), depth_limit: int = 20, num_simulations: int = 1000):
        self.fragment = fragment
        self.height, self.width = fragment.shape
        self.exit_penalty = exit_penalty
        self.max_exit_penalty = max([exit_penalty[e] for e in exit_penalty.keys()])

        self.c: float = exploration
        self.depth_limit: int = depth_limit

        # used to track which nodes are currently in the tree
        self.tree: set[str] = set()
        self.step_penalty: float = 0.1
        self.num_simulations = num_simulations

    
    def get_new_position(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """
        Get new position after taking action from current position
        """
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        d_row, d_col = directions[action]
        new_pos = (pos[0] + d_row, pos[1] + d_col)

        if (0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width and self.fragment[new_pos[0], new_pos[1]] != Cell.WALL.value):
            return new_pos
        else: # illegal move
            return pos


    def get_penalty(self, pos: tuple[int, int]) -> float:
        """
        Get penalty for current position
        """
        if pos in self.exit_penalty.keys():
            return - self.exit_penalty[pos]
        else:
            return - self.max_exit_penalty


    def UCB1(self, node: EscapeNode, action: int, depth_diff: int) -> float:
        """
        Action selection using UCB1
        """
        if node.children[action].num_visited == 0:
            return float('inf')
        else:
            depth_diff = max(1, depth_diff)
            return node.action_values[action] / float(depth_diff) + self.c * math.sqrt(
                    math.log(node.num_visited) / node.children[action].num_visited)
        
    
    def rollout(self, node: EscapeNode, depth: int) -> float:
        """
        Rollout function for exploring new actions/states
        """

        globals.escape_rollout_count += 1
        if depth > self.depth_limit:
            return self.get_penalty(node.agent_pos)

        if node.agent_pos in self.exit_penalty.keys():
            return self.get_penalty(node.agent_pos)

        random_action = random.randint(0,3)
        new_pos = self.get_new_position(node.agent_pos, random_action)
        
        temp_node = EscapeNode(new_pos, node.id, random_action)
        penalty = -self.step_penalty + self.rollout(temp_node, depth + 1)
        return penalty


    def simulate(self, node: EscapeNode, depth: int) -> float:
        """
        Simulate function
        """

        globals.escape_rollout_count += 1
        node.num_visited += 1

        if depth > self.depth_limit:
            return self.get_penalty(node.agent_pos)
        
        if node.agent_pos in self.exit_penalty.keys():
            return self.get_penalty(node.agent_pos)

        # if node not currently in tree, expand
        if node.id not in self.tree:
            self.tree.add(node.id)

            for action in range(4):
                new_pos = self.get_new_position(node.agent_pos, action)
                node.children[action] = EscapeNode(new_pos, node.id, action)

            return self.rollout(node, depth)

        # get optimal action based on UCB1
        action_values = list()
        for a in range(4):
            action_values.append(self.UCB1(node, a, self.depth_limit - depth))
        chosen_a = action_values.index(max(action_values))

        penalty = -self.step_penalty + self.simulate(node.children[chosen_a], depth + 1)
        node.action_values[chosen_a] = node.action_values[chosen_a] + (penalty - node.action_values[chosen_a]) / node.children[chosen_a].num_visited

        return penalty

    def search(self, root: EscapeNode) -> int:
        """
        Perform rollouts and get optimal action
        """
        simul_count = 0
        while simul_count < self.num_simulations:
            self.simulate(root, 0)
            simul_count += 1
        
        best_action = root.action_values.index(max(root.action_values))
        return best_action