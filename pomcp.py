import random
import math
from tree_builder import Action
from generator import Generator
# #chat let me lock in rq


class Node():
    """
    Class Node for pomcp
    """
    def __init__(self, agent_pos, obs, num_visited = 0, value = 0):

        self.agent_pos: tuple[int, int] = agent_pos
        self.obs: set[tuple[int, int]] = obs
        self.num_visited: int = num_visited
        self.value: float = value

        self.belief: set(tuple[int, int]) = dict()
        self.children: list[Node] = list()

        self.encoding: str = ""
        self.encode()

    def encode(self):
        """
        encode current node into a string
        
        used for tracking purposes by pomcp
        """
        self.encoding += str(self.agent_pos[0]) + "," + str(self.agent_pos[1]) + "|"
        self.encoding += str(len(self.obs)) + "|"

        obs_list = list(self.obs)
        obs_list = sorted(obs_list)
        for obs in obs_list:
            self.encoding += str(obs[0]) + "," + str(obs[1]) + "|"
        

class POMCP():

    def __init__(self, generator, discount, epsilon = 0.001):
        """
        generator - black box generator

        discount - discount factor
        episilon - cutoff for the discount factor
        """
        
        self.generator: Generator = generator

        self.discount: float = discount
        self.epsilon: float = epsilon

        # used to track which nodes are currently in the tree
        self.tree: set[str] = set()

    def rollout(self, node: Node, state: tuple[int, int], depth: int):
        """
        rollout function for exploring new actions/states
        """
        random_action = random.randint(0,3)
        new_state, new_pos, new_obs, reward = self.generator.generate(state, node.agent_pos, node.obs, random_action)
        return reward + self.rollout(node.children[random_action], new_state, depth + 1)
    

    def simulate(self, state: tuple[int, int], node: Node, depth: int) -> float:
        """
        simulate function for searching
        """
        if not node.encoding in self.tree:
            self.tree.add(node.encoding)
            for a in Action:
                """
                NEED TO CHECK HERE

                when initialize node.children
                if agent pos and observation should be updated
                """
                node.children.append(Node(node.agent_pos, node.obs, 0, 0))
            return self.rollout(node, state, depth)
        else:
            c:float = 0.5
            a0_value:float = node.children[0].value + c * math.sqrt(
                math.log(node.num_visited)/node.children[0].num_visited)
            a1_value:float = node.children[1].value + c * math.sqrt(
                math.log(node.num_visited)/node.children[1].num_visited)
            a2_value:float = node.children[2].value + c * math.sqrt(
                math.log(node.num_visited)/node.children[2].num_visited)
            a3_value:float = node.children[3].value + c * math.sqrt(
                math.log(node.num_visited)/node.children[3].num_visited)
            
            action_values:list[float] = [a0_value, a1_value, a2_value, a3_value]
            a:int = action_values.index(max(action_values)) # a is the action you will take

            new_state, obs, reward = self.generator.generate(state, node, action)

            reward = reward + self.simulate(new_state, node.children[a] ,depth + 1)

            #should belief still stay the same?

            node.num_visited += 1
            node.children[a].num_visited += 1
            node.children[a].value = node.children[a].value + (reward - node.children[a].value)/node.children[a].num_visited

            return reward
            

            
    def best_action_index(self, root: Node) -> int:
        """
        Small helper function to find the action with the largest value given the node
        """
        action_values:list[float] = [
            root.children[0].value, 
            root.children[1].value, 
            root.children[2].value, 
            root.children[3].value
        ]
        best_action:int = action_values.index(max(action_values))
        return best_action


    def search(self, root: Node) -> int:
        """
        Search will take a node and return an integer corresponding to the best action
        """
        depth:int = 5 #REMEMBER TO ASK COLE/MARTA OR JEFF ABOUT THIS
        while(True): #REMEMBER TO ASK COLE/MARTA OR JEFF ABOUT THIS
            state:tuple[int, int] = random.choice(root.belief)
            self.simulate(state, root, depth)
            break #REMEMBER TO REMOVE THIS

        best_action:int = self.best_action_index(root)
        
        return best_action

