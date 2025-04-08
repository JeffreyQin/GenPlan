from generator import Generator
from pomcp import POMCP
from tree_builder import Node

import numpy as np

map = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 0]
])
agent_pos = (0, 0)
range = 5
reward = 100
penalty = -1

discount = 0

def run_experiment():
    generator = Generator(map, range, reward, penalty)
    root_node = Node(agent_pos, set())
    pomcp_algorithm = POMCP(generator, discount)

    

    pomcp_algorithm.search(root_node)


if __name__ == "__main__":
    run_experiment()
