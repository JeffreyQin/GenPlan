"""

proof-of-concept value iteration in a non-tree implementation
- does not implement observation mechanism

"""


import numpy as np
from collections import defaultdict

from tree_builder import Cell, Action

# currently assume single possible fragment

class ValueIteration():
    def __init__(self, epsilon = 0.9, gamma = 0.9, tol = 0.01):

        self.epsilon = epsilon
        self.prob_best = epsilon + (1 - epsilon)/4 # probability of choosing best action
        self.prob_others = (1 - epsilon) / 4 # probability of choosing non-optimal action
        self.gamma = gamma
        self.converge_tol = tol

     
    def get_fragment_entrances(self, map, fragment, partitions):
        """
        compute the location of entrances, as well as their corresponding copies
        """
        
        entrances = set()
        entrance_to_part = dict()
        part_visited = defaultdict(lambda: False)

        for part in partitions:
            if part['rotations'] == 1 or part['rotations'] == 3:
                height, width = fragment.shape[1], fragment.shape[0]
            elif part['rotations'] == 0 or part['rotations'] == 2:
                height, width = fragment.shape[0], fragment.shape[1]
        
            top_left = part['top left']
            bottom_right = (top_left[0] + height - 1, top_left[1] + width - 1)

            for r in range(top_left[0], bottom_right[0] + 1):
                if map[r, top_left[1]] != Cell.WALL.value:
                    entrances.add((r, top_left[1]))
                    entrance_to_part[(r, top_left[1])] = part['top left']
                elif map[r, bottom_right[1]] != Cell.WALL.value:
                    entrances.add((r, bottom_right[1]))
                    entrance_to_part[(r, bottom_right[1])] = part['top left']
            for c in range(top_left[1], bottom_right[1] + 1):
                if map[top_left[0], c] != Cell.WALL.value:
                    entrances.add((top_left[0], c))
                    entrance_to_part[(top_left[0], c)] = part['top left']
                elif map[bottom_right[0], c] != Cell.WALL.value:
                    entrances.add((bottom_right[0], c))
                    entrance_to_part[(bottom_right[0], c)] = part['top left']
        
        return entrances, entrance_to_part, part_visited


    def get_destination(self, map, pos, action):
        """
        compute next step destination
        """

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
    

    def generate_policy(self, map, entrances):
    
        height, width = map.shape

        rewards = np.zeros((height, width))
        V = np.zeros((height, width))
        policy = np.zeros((height, width))
        
        for r in range(height):
            for c in range(width):
                if (r, c) in entrances:
                    rewards[r, c] = 100.0 # should ask Cole/Marta/Albert
                elif map[r, c] == Cell.WALL.value:
                    rewards[r, c] = -100.0 # should ask Cole/Marta/Albert
                    policy[r, c] = None
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
                            
    

    def perform_search(self, map, policy, pos, entrances, entrance_to_part, part_visited):
        """
        move agent according to policy generated by value iteration
        """
        
        while True:
            new_pos = self.get_destination(map, pos, policy[pos[0]][pos[1]])

            if new_pos == pos:
                break
            elif new_pos in entrances: # if arrived at entrance
                part = entrance_to_part[new_pos]

                to_be_removed = set()
                for entr in entrances:
                    if entrance_to_part[entr] == part:
                        to_be_removed.add(entr)
                for entr in to_be_removed:
                    entrances.remove(entr)

                part_visited[part] = True
                print("arrived at entrance " + str(new_pos))
                break
            
            pos = new_pos
                
