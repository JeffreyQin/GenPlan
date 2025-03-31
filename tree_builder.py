from enum import Enum

class TreeNode():

    def __init__(self, generator, node_idx, parent_idx = -1):

        self.generator = generator

        self.node_idx = node_idx
        self.parent_idx = parent_idx
        self.is_root = True if parent_idx == -1 else False
        
        self.N = 
        self.V = 
        self.B = 
        self.children = dict()
        