from generator import Generator
from pomcp_copy import POMCP
from tree_builder import Node

import numpy as np

input_maps = np.array([
    [0, 2, 0, 0, 2, 0],
    [2, 2, 0, 0, 2, 0],
    [0, 0, 0, 0, 2, 0],
    [2, 2, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [2, 2, 0, 2, 2, 0],
    [0, 2, 0, 0, 2, 0]
])

agent_pos = (3, 2)
range = 5
penalty = -1.0

discount = 0

def run_experiment():
    generator = Generator(input_maps, range, penalty)

    obs, belief = generator.get_init_state(agent_pos)
 
    root_node = Node(agent_pos, obs, belief, parent_id="", parent_a=0)
    pomcp_algorithm = POMCP(generator, discount)

    result = pomcp_algorithm.search(root_node)

if __name__ == "__main__":
    run_experiment()
