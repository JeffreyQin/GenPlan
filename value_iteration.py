import numpy as np

from tree_builder import Cell, Action

# currently assume single possible fragment

class ValueIteration():
    def __init__(self, epsilon = 0.9, gamma = 0.9, tol = 0.01):

        self.epsilon = epsilon
        self.prob_best = epsilon + (1 - epsilon)/4
        self.prob_others = (1 - epsilon) / 4
        self.gamma = gamma
        self.converge_tol = tol

     
    def mark_fragment_entrances(self, map, fragment, partitions):
        
        for par in partitions:
            if par['rotations'] == 1 or par['rotations'] == 3:
                height, width = fragment.shape[1], fragment.shape[0]
            elif par['rotations'] == 0 or par['rotations'] == 2:
                height, width = fragment.shape[0], fragment.shape[1]
        
            top_left = par['top_left']
            bottom_right = (top_left[0] + height - 1, top_left[1] + width - 1)

            for r in range(top_left[0], bottom_right[0] + 1):
                if map[r, top_left[1]] != Cell.WALL.value:
                    map[r, top_left[1]] = Cell.ENTRANCE.value
                elif map[r, bottom_right[1]] != Cell.WALL.value:
                    map[r, bottom_right[1]] = Cell.ENTRANCE.value
            for c in range(top_left[1], bottom_right[1] + 1):
                if map[top_left[0], c] != Cell.WALL.value:
                    map[top_left[0], c] = Cell.ENTRANCE.value
                elif map[bottom_right[0], c] != Cell.WALL.value:
                    map[bottom_right[0], c] = Cell.ENTRANCE.value



    def get_destination(self, map, pos, action):

        dest = pos 
        
        if action == Action.UP.value:
            if pos[0] > 0 and map[pos[0] - 1, pos[1]] != Cell.WALL.value:
                dest = (pos[0] - 1, pos[1])
        elif action == Action.RIGHT.value:
            if pos[1] < map.shape[1] - 1 and map[pos[0], pos[1] + 1] != Cell.WALL.value:
                dest = (pos[0], pos[1] + 1)
        elif action == Action.DOWN.value:
            if pos[0] < map.shape[0] - 1 and map[pos[0] + 1, pos[1]] != Cell.WALL.value:
                dest = (pos[0] + 1, pos[1])
        elif action == Action.LEFT.value:
            if pos[1] > 0 and map[pos[0], pos[1] - 1] != Cell.WALL.value:
                dest = (pos[0], pos[1] - 1)
        
        return dest
    

    def generate_policy(self, map):
    
        height, width = map.shape

        rewards = np.zeros((height, width))
        V = np.zeros((height, width))
        policy = np.zeros((height, width))
        
        for r in range(height):
            for c in range(width):
                if map[r, c] == Cell.WALL.value:
                    rewards[r, c] = -100.0
                    policy[r, c] = None
                elif map[r, c] == Cell.ENTRANCE.value:
                    rewards[r, c] = 100.0
                else:
                    rewards[r, c] = -1.0


        actions = [a for a in range(4)]

        num_iter = 0
        while True:
            oldV = V.copy()
            for r in range(height):
                for c in range(width):
                    if map[r, c] != Cell.WALL.value:
                        Q = dict()
                        for a in actions:
                            (new_r, new_c) = self.get_destination(map, (r, c), a)
                            Q[a] = rewards[r, c] + self.gamma * self.prob_best * oldV[new_r, new_c]

                            for other_a in actions:
                                if other_a != a:
                                    (other_new_r, other_new_c) = self.get_destination(map, (r, c), other_a)
                                    Q[a] += self.gamma * self.prob_others * oldV[other_new_r, other_new_c]
                        
                        V[r, c] = max(Q.values())
                        policy[r, c] = max(Q, key = Q.get)
            num_iter += 1
            if (abs(oldV[:, :] - V[:, :]) < self.converge_tol).all():
                break
        return policy, num_iter
                            
        