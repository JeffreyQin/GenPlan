import random
import math
import globals
from tree_builder import Node
from generator import Generator 
# #chat let me lock in rq


class POMCP():


    def __init__(self, generator, discount, exploration = 5.0, epsilon = 0.001, depth = 50): #set depth to high
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

        # used to track which nodes are currently in the tree
        self.tree: set[str] = set()
        #self.tree: dict[tuple[int,int], list[Node]] = dict()#surely theres a better way to do this

    # def is_in_tree(self, node: Node):
    #     """
    #     accepts a node and first comapres to see whether the agent position is in the tree then look for nodes w/ the same belief and same agent_pos. Returns false and None if not in the tree, returns the 
    #     node and True if node is found ot be in the tree
    #     """
    #     if(node.agent_pos in self.tree):
    #         belief_checklist = self.tree[node.agent_pos]
    #         for n in belief_checklist:
    #             if(node.obs == n.obs):
    #                 return True, n

    #         return False, None
    #     else:
    #         return False, None

    def rollout(self, node: Node, state: tuple[int, int], depth: int)->float:
        """
        rollout function for exploring new actions/states
        """
        globals.total_rollout += 1
        if depth > self.depth_limit:
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
        node.num_visited += 1

        # print(f"\n--- Simulating from state {state} at depth {depth} ---")
        # print(f"Node ID: {node.id}, Visited: {node.num_visited}")
        if(depth > self.depth_limit):
            return self.generator.get_penalty(node.obs)

        if not node.id in self.tree:
            #print("Node is new, expanding")
            self.tree.add(node.id)
            # if(node.agent_pos in self.tree):
            #     self.tree[node.agent_pos].append(node)
            # else:
            #     self.tree[node.agent_pos] = [node] #add it to the tree

            for a in range(4):
                _, new_pos, new_obs, new_belief, _ = self.generator.generate(state, node.agent_pos, node.obs, node.belief, a)
                node.children[a] = Node(new_pos, new_obs, new_belief, node.id, a)

            return self.rollout(node, state, depth)
        else:
            #node = n
            #print("its not just rollouts dw ")

            action_values: list[float] = list()
            for a in range(4):
                action_values.append(self.UCB1(node, a, self.depth_limit - depth))

            chosen_a :int = action_values.index(max(action_values)) # a is the action you will take

            # print(f"UCB1 values: {action_values}")
            # print(f"Chosen action: {chosen_a}")

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


    def search(self, root: Node) -> int:
        """
        Search will take a node and return an integer corresponding to the best action
        """

        if(len(root.belief) == 0): #meaing we've seen all empty rooms
            print("all rooms observerd")
            return
        count = 0
        while count < 3000: #REMEMBER TO ASK COLE/MARTA OR JEFF ABOUT THIS #set C to a certain amount

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