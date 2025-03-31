import numpy as np
#from map_completion_prompt import Prompt

# Main code 

prompt_id = 2

# LLM prompt has two parts, system prompt and user prompt
system_prompt = {"role": "system", 
                 "content": '''You are a designer's assistant, skilled in noticing patterns, combining fragmets into a patterns, and extrapolating them. 
     You are skilled in identifying the underlying structure of a pattern and generating new fragments that fit the pattern. 
     You are also skilled at writing Python code.'''}

# The user prompt includes an input pattern, whcih is a 2D array loaded from file
# This input represents an observed map
# Currently the input is constrained to {1, 0} values, to simplify tha task.
# In the next iteration, will be extending input to integers in {0,..9}

# TBD
# There may be many fragments that can be used to regenerate the input pattern, but here we ask to generate just one.
# TBD 1: sample multiple fragments to achieve several versions of pattern completions
# in this prototype, the goal is to have GPT understand a simple pattern like mirror symmetry
# TBD 2: use follow-up prompts, for code refinement.

user_prompt_1 =  ''' There are two steps to this task. 
In Step 1, you will be given a pattern and asked to identify its constituent fragments. 
The pattern is given by an input matrix, elements of which can take values 1 and 0. 
Your task is to identify a repeating fragment in this input.
Example 1. Given input: [
  [1, 0, 1, 0], 
  [0, 1, 0, 1], 
  [1, 0, 1, 0], 
  [0, 1, 0, 1] ], 
The repeating fragment is : [
   [1, 0], 
   [0, 1]
]

Example 2. Given input: [
  [0,0,1,0,0,1],
  [0,0,1,0,0,1],
  [1,0,1,1,0,1],
  [0,0,0,0,0,0],
  [1,0,1,1,0,1],
  [0,0,1,0,0,1],
  [0,0,1,0,0,1]] 
The repeating fragment is: [
   [0,0,1], 
   [0,0,1], 
   [1,0,1]]

 To be considered a repeating fragment, the fragment does not have to tile the space exactly, but it should be repeated at least twice.
 The fragment instances may be flipped horizontally or vertically, translated horizontally or vertically, rotated 90 degrees, and may partly overlap.

In Step 2, you will write a function that attempts to identify all occurrences of the fragment in the input. 
Return a list containing the indexical locations of the top left corner for each copy, along with whether to reflect the copy horizontally and
the number of 90 degree counter-clockwise rotations (these operations together generate the dihedral group D4).

For instance, in the examples above,

Example 1.

def partition():
   return [
      {"top left": (0,0), "reflect": False, "rotations": 0},
      {"top left": (0,2), "reflect": False, "rotations": 0},
      {"top left": (2,0), "reflect": False, "rotations": 0},
      {"top left": (2,2), "reflect": False, "rotations": 0},
   ]

An alternative, more structured solution is

def partition():
   copies = []
   for tl_i, tl_j in [(0,0),(0,2),(2,0),(2,2)]:
      copies.append(
         {"top left": (tl_i, tl_j), "reflect": False, "rotations": 0},
      )
   return copies

Example 2.

def partition():
   return [
      {"top left": (0,0), "reflect": False, "rotations": 0},
      {"top left": (0,3), "reflect": False, "rotations": 0},
      {"top left": (4,0), "reflect": True, "rotations": 2},
      {"top left": (4,3), "reflect": True, "rotations": 2},
   ]

An alternative, more structured solution is

def partition():
   copies = []
   for tl_i, tl_j in [(0,0),(0,3)]:
      copies.append(
         {"top left": (tl_i, tl_j), "reflect": False, "rotation": 0},
      )
   for tl_i, tl_j in [(4,0),(4,3)]:
      copies.append(
         {"top left": (tl_i, tl_j), "reflect": True, "rotations": 2},
      )
   return corners

Given your fragment and partition, the user will attempt to reconstruct the input using the following function:

def construct_copy(fragment, reflect, rotations):
    if reflect:
        fragment = np.flip(fragment, 1)
    fragment = np.rot90(fragment, rotations)
    return fragment

def regenerate_pattern(fragment, copies, input_dims):
   pattern = -1 * np.ones(shape=input_dims)
   for copy in copies:
      transformed = construct_copy(
         fragment,
         copy["reflect"],
         copy["rotations"],
      )
      height, width = transformed.shape
      tl_i, tl_j = copy["top left"]
      try:
         pattern[tl_i : tl_i+height, tl_j : tl_j+width] = transformed
      except:
         pass
   return pattern

We can test the success of this regeneration with

input_map = np.array(input_map)
output = regenerate_pattern(
    fragment,
    partition(),
    input_map.shape,
)
print(f"Errors and omissions: {(input_map.shape[0]*input_map.shape[1]) - np.sum(input_map == output)}")


Now is your turn. Propose a fragment that can be used to reconstruct the given input.  
Respond by completing the following Python code: 

input_map = ''' 

user_prompt_2 = '''

# make sure to define all arrays as numpy arrays
import numpy as np
input_map = np.array(input_map)

fragment = [ ... ]
fragment = np.array(fragment)

def partition():
   copies = []

   # Place your code here. Let's think step by step

   return copies

Please include only code in you response, no text.'''

class PartitionPrompt:
       
   def __init__(self):
         super().__init__()
         self.n_completions = 5
         self.system_prompt:int = system_prompt
         self.prompt_id:str = prompt_id
         self.user_prompt_1:str = user_prompt_1
         self.user_prompt_2:str = user_prompt_2


   '''
   # functions given in the prompt 
   def construct_copy(self, fragment, reflect, rotations):
      if reflect:
        fragment = np.flip(fragment, 1)
      fragment = np.rot90(fragment, rotations)
      return fragment

   def regenerate_pattern(self, fragment, copies, input_dims):
    pattern = 0.5 * np.ones(shape=input_dims)
    for copy in copies:
        transformed = self.construct_copy(
            fragment,
            copy["reflect"],
            copy["rotations"],
        )
        height, width = transformed.shape
        tl_i, tl_j = copy["top left"]
        try:
         pattern[tl_i : tl_i+height, tl_j : tl_j+width] = transformed
        except:
         pass
    return pattern
    '''

   def segment_map(self, fragment, copies):
      """
      Creates a dictionary mapping each index pair to its corresponding copy.
      Only indices inside a copy appear in the keys.
      """
      segmentation = {}
      height, width = len(fragment), len(fragment[0])
      for copy in copies:
         tl_i, tl_j = copy["top left"]
         if copy["rotations"] % 2 == 1:
            cp_height = width
            cp_width = height
         else:
            cp_height = height
            cp_width = width
         for del_i in range(cp_height):
            for del_j in range(cp_width):
               i, j = tl_i + del_i, tl_j + del_j
               # Would need overall size...
               # if i >= height or j >= width:
               #    continue

               # TODO: Optimize by creating and transforming
               # the full index matrix at once.
               
               # map del_i, del_j to offsets in original fragment

               mask = np.zeros((cp_height, cp_width))
               mask[del_i,del_j] = 1
               mask = np.rot90(mask, -copy["rotations"])
               if copy["reflect"]:
                  mask = np.flip(mask, 1)
               base_i,base_j = np.where(mask)
               base_i,base_j = int(base_i[0]),int(base_j[0])
               segmentation[(i,j)] = (copy, base_i, base_j)
      return segmentation
   

def construct_copy(fragment, reflect, rotations):
    if reflect:
        fragment = np.flip(fragment, 1)
    fragment = np.rot90(fragment, rotations)
    return fragment

def regenerate_pattern(fragment, copies, input_dims):
   pattern = 0.5 * np.ones(shape=input_dims)
   for copy in copies:
      transformed = construct_copy(
         fragment,
         copy["reflect"],
         copy["rotations"],
      )
      height, width = transformed.shape
      tl_i, tl_j = copy["top left"]
      try:
         pattern[tl_i : tl_i+height, tl_j : tl_j+width] = transformed
      except:
         pass
   return pattern

def fragment_to_map_coords(fragment, copy):
   """
   Returns a dictionary mapping every index of the fragment
   to its global indices in the map. This is essentially an inverse
   of segment_map.
   """
   height, width = len(fragment),len(fragment[0])
   indices = np.indices((height, width))
   if copy["reflect"]:
      # the first dimension is now the index type,
      # so we flip one dimension later than normal.
      indices = np.flip(indices, 2)
   indices = np.rot90(indices, copy["rotations"], (1,2))
   frag_to_map = {}
   tl_i, tl_j = copy["top left"]
   for i in range(indices.shape[1]):
      for j in range(indices.shape[2]):
         frag_to_map[(int(indices[0,i,j]), int(indices[1,i,j]))] = (i+tl_i, j+tl_j)
   return frag_to_map