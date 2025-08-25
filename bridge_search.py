import random
import math
import numpy as np
import globals
from globals import exploration_steps
from tree_builder import BridgeNode, Cell, Action
from generator import BridgeGenerator

class BridgePOMCP():

    def __init__(self, generator, discount = 1.0, exploration = 5.0, epsilon = 0.001, depth = 10, num_simulations = 1000):
        self.generator: BridgeGenerator = generator
        self.discount: float = discount
        self.c: float = exploration
        self.epsilon: float = epsilon
        self.depth_limit: int = depth
        self.num_simulations: int = num_simulations

        # used to track which nodes are currently in the tree
        self.tree: set[str] = set()


    def rollout(self, node: BridgeNode, state: tuple[int, int], depth: int) -> float:

        globals.bridge_rollout_count += 1

        if depth > self.depth_limit:
            return self.generator.get_penalty(node.obs)

        # check if explore option is available
        if self.generator.is_fragment_border(node.agent_pos):
            random_action = random.randint(0, 4)  
        else:
            random_action = random.randint(0, 3)

        exit_found, new_pos, new_obs, new_belief, reward = self.generator.generate(state, node.agent_pos, node.obs, node.belief, random_action)

        if exit_found:
            return 0.0
        else:
            temp_node = BridgeNode(new_pos, new_obs, new_belief, node.id, random_action)
            return reward + self.discount * self.rollout(temp_node, state, depth + 1)
    

    def simulate(self, state: tuple[int, int], node: BridgeNode, depth: int) -> float:

        globals.bridge_rollout_count += 1
        node.num_visited += 1

        if depth > self.depth_limit:
            return self.generator.get_penalty(node.obs)

        if not node.id in self.tree:
            # node not found in tree, expand
            self.tree.add(node.id)

            # determine if "explore" action is available here
            num_actions = 5 if self.generator.is_fragment_border(node.agent_pos) else 4

            for a in range(num_actions):
                _, new_pos, new_obs, new_belief, _ = self.generator.generate(state, node.agent_pos, node.obs, node.belief, a)
                node.children[a] = BridgeNode(new_pos, new_obs, new_belief, node.id, a)

            return self.rollout(node, state, depth)

        else:
            # select optimal action using UCB1
            action_values: list[float] = list()

            # determine if "explore" action is available
            num_actions = 5 if self.generator.is_fragment_border(node.agent_pos) else 4

            for a in range(num_actions):
                action_values.append(self.UCB1(node, a, self.depth_limit - depth))

            chosen_a: int = action_values.index(max(action_values))

            exit_found, new_pos, new_obs, new_belief, reward = self.generator.generate(state, node.agent_pos, node.obs, node.belief, chosen_a)

            if exit_found:
                reward = 0
                node.children[chosen_a].num_visited += 1
            else:
                reward = reward + self.discount * self.simulate(state, node.children[chosen_a], depth + 1)

            node.action_values[chosen_a] = node.action_values[chosen_a] + (reward - node.action_values[chosen_a])/node.children[chosen_a].num_visited

            return reward
    
    def UCB1(self, node: BridgeNode, action: int, depth_diff: int):
        """
        UCB1 formula for action selection
        """
        if node.children[action].num_visited == 0:  # unvisited action
            return float('inf')
        else:
            depth_diff = max(1, depth_diff)
            return node.action_values[action] / float(depth_diff) + self.c * math.sqrt(
                    math.log(node.num_visited) / node.children[action].num_visited)
    
    def best_action_index(self, root: BridgeNode) -> int:
        """
        Find the action with the largest value
        """
        # prevent optimal action to be "explore" if it's not available
        if root.children[4] is None:
            root.action_values[4] = float('-inf')

        best_action: int = root.action_values.index(max(root.action_values))
        return best_action


    def search(self, root: BridgeNode) -> int:

        if len(root.belief) == 0:  # meaning we've seen all empty rooms
            return

        count = 0
        while count < self.num_simulations:
            state: tuple[int, int] = random.choice(list(root.belief))
            self.simulate(state, root, 0)
            count += 1
        
        best_action: int = self.best_action_index(root)
        return best_action
