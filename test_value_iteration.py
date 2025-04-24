import numpy as np
from value_iteration import ValueIteration


map = np.array([
    [5, 1, 0, 5, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [5, 1, 0, 5, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0]
])

VI = ValueIteration()
result, num_iter = VI.generate_policy(map)


print(result)
