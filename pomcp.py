import random
import math
import globals
from tree_builder import Node
from generator import Generator 
import time
# #chat let me lock in rq


class POMCP():


    def __init__(self, generator, discount, exploration = 5.0, epsilon = 0.001, depth = 10, num_simulations = 1000): #set depth to high
        """
        generator - black box generator

        discount - discount factor
        exploration - exploration parameter used in UCB
        episilon - cutoff for the discount factor
        """
        
        self.generator: Generator = generator

        self.discount: float = 1.0
        self.c: float = exploration
        self.epsilon: float = epsilon
        self.depth_limit: int = depth
        self.num_simulations: int = num_simulations

        # used to track which nodes are currently in the tree
        self.tree: set[str] = set()

    def rollout(self, node: Node, state: tuple[int, int], depth: int)->float:
        """
        rollout function for exploring new actions/states
        """
        #globals.total_rollout += 1
        globals.simul_rollout_count += 1

        if depth > self.depth_limit or globals.simul_rollout_count >= globals.simul_rollout_limit:
            return self.generator.get_penalty(node.obs)
        
        random_action = random.randint(0,3)

        exit_found, new_pos, new_obs, new_belief, reward = self.generator.generate(state, node.agent_pos, node.obs, node.belief, random_action)

        if exit_found:
            #print("exit found!")
            return 0 #remember to change to 0
        else:
            temp_node = Node(new_pos, new_obs, new_belief, node.id, random_action)
            return reward + self.discount * self.rollout(temp_node, state, depth + 1)
    

    def simulate(self, state: tuple[int, int], node: Node, depth: int) -> float:
        """
        simulate function for searching
        """
        #in_tree, n = self.is_in_tree(node)
        globals.simul_rollout_count += 1
        node.num_visited += 1

        # print(f"\n--- Simulating from state {state} at depth {depth} ---")
        # print(f"Node ID: {node.id}, Visited: {node.num_visited}")
        if(depth > self.depth_limit or globals.simul_rollout_count >= globals.simul_rollout_limit):
            return self.generator.get_penalty(node.obs)

        if not node.id in self.tree:

            self.tree.add(node.id)

            for a in range(4):
                _, new_pos, new_obs, new_belief, _ = self.generator.generate(state, node.agent_pos, node.obs, node.belief, a)
                node.children[a] = Node(new_pos, new_obs, new_belief, node.id, a)

            return self.rollout(node, state, depth)
        else:

            action_values: list[float] = list()
            for a in range(4):
                action_values.append(self.UCB1(node, a, self.depth_limit - depth))

            chosen_a :int = action_values.index(max(action_values)) # a is the action you will take



            exit_found, new_pos, new_obs, new_belief, reward = self.generator.generate(state, node.agent_pos, node.obs, node.belief, chosen_a)

            
            node.children[chosen_a].obs = new_obs
            node.children[chosen_a].agent_pos = new_pos

            # in_tree, n = self.is_in_tree(node.children[chosen_a])
            # if(in_tree):
            #     node.children[chosen_a] = n


            #node.num_visited += 1
            if(exit_found):
                reward = 0
                node.children[chosen_a].num_visited += 1
            else:
                reward = reward + self.discount * self.simulate(state, node.children[chosen_a], depth + 1)

            node.action_values[chosen_a] = node.action_values[chosen_a] + (reward - node.action_values[chosen_a])/node.children[chosen_a].num_visited

            return reward
        
            
    def UCB1(self, node: Node, action: int, depth: int):
        if node.children[action].num_visited == 0: # unvisited action
            return float('inf')
        else:
            return node.action_values[action] / float(depth) + self.c * math.sqrt(
                    math.log(node.num_visited) / node.children[action].num_visited)
        

    def best_action_index(self, root: Node) -> int:
        """
        Small helper function to find the action with the largest value given the node
        """
        # action_values:list[float] = [
        #     root.children[0].value,
        #     root.children[1].value,
        #     root.children[2].value,
        #     root.children[3].value
        # ]
        #print("Action values")
        #print(root.action_values)
        best_action:int = root.action_values.index(max(root.action_values))
        #print("Best action")
        #print(best_action)
        return best_action


    def search(self, root: Node, simul_limit=False, time_limit = False) -> int:
        """
        Search will take a node and return an integer corresponding to the best action
        """

        if(len(root.belief) == 0): #meaing we've seen all empty rooms
            print("all rooms observerd")
            return
        count = 0
        while count < self.num_simulations and globals.simul_rollout_count <= globals.simul_rollout_limit: #REMEMBER TO ASK COLE/MARTA OR JEFF ABOUT THIS #set C to a certain amount
            
            if simul_limit and globals.total_simul_actions >= globals.total_simul_limit:
                break
            if time_limit and time.time() - globals.naive_start_time >= globals.naive_time_limit:
                break

            state: tuple[int, int] = random.choice(list(root.belief))
            self.simulate(state, root, 0)
            count += 1

        best_action:int = self.best_action_index(root)
        
        return best_action

# horizon = depth of a search
# # of rollouts = # of times search is called

# debug simple rollouts to make sure that it works (Rollouts work)
# test different exploration rates (super lost)
# check to see if current node is preserved (FIGURED THIS OUT)
# this weeks tasks