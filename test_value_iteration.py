import numpy as np
from value_iteration import ValueIteration


map = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1]
])

fragment = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
])

partition = [
    {"top left": (0,0), "reflect": False, "rotations": 0},
    {"top left": (3,0), "reflect": False, "rotations": 0}
]


VI = ValueIteration()


entrances, entrance_to_part, part_visited = VI.get_fragment_entrances(map, fragment, partition)

policy_iter_1, num_iter = VI.generate_policy(map, entrances)
print(policy_iter_1)

pos = (5, 6)
VI.perform_search(map, policy_iter_1, pos, entrances, entrance_to_part, part_visited)

policy_iter_2, num_iter = VI.generate_policy(map, entrances)
print(policy_iter_2)

pos = (4, 5)
VI.perform_search(map, policy_iter_2, pos, entrances, entrance_to_part, part_visited)