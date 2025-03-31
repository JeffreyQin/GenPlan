# Helper utils - keeping the main code readable

import matplotlib.pyplot as plt
import numpy as np
import os
import re

# @TODO: This is a cludge: currently there is 
# - one code used by pattern editor
# - another used by the tree builder
# - and finally grayscale for the plotting utilities
def to_grayscale(pattern):
    grayscale_vals = {
        0: 0.5, # Unobserved cells should be gray
        0.5: 0.25, # Ambiguous values (according to map synthesis) will be dark gray
        1: 0.0, # Walls should be black
        2: 1.0, # Observed cells should be white
    }
    gray_pattern = [[grayscale_vals[entry] for entry in row] for row in pattern]
    return gray_pattern

# plotting - showing one image
def plot_pattern(pattern, title):
      plt.imshow(to_grayscale(pattern), cmap='gray')
      plt.title(title)
      plt.show()

# plotting a list of patterns with titles, helper function
# optionally, save the plot to image_logs/{save_image}.png to use later in slides
def plot_patterns(patterns, titles, figure_title = "", save_image="", show_plots=True):

    fig, axs = plt.subplots(1, len(patterns), figsize=(15, 5))
    for i, pattern in enumerate(patterns):
        axs[i].imshow(to_grayscale(pattern), cmap='gray')
        axs[i].set_title(titles[i])
    
    plt.suptitle(figure_title)
    if (len(save_image) > 0):
        plt.savefig(f"image_logs/{save_image}.png")

    if (show_plots):
        plt.show()

    plt.close()

# plotting input, and an array of proposed fragments
def plot_input_response(input_image, fragments, save_image="", show_plots=True):
    plot_patterns([input_image] + fragments, 
                  ["Input Pattern"] + [f"Fragment {i}" for i in range(len(fragments))], 
                  "Input and Returned Fragment", 
                  save_image, 
                  show_plots)

# plotting input, fragment, and output
def plot_input_fragment_output(input_image, fragment, output, figure_title = "", save_image="", show_plots=True):
   plot_patterns([input_image] + [fragment] + [output], 
                 ["Input Pattern"] + ["Fragment"] + ["Output"], 
                 figure_title, save_image, show_plots)

def generate_completion_plot(input_map, f, title, plot_filename, show_plots=True):
    p = partition(input_map, f)
    output = regenerate_pattern(f, p, input_map.shape)
    plot_input_fragment_output(input_map, f, output, title, plot_filename, show_plots=show_plots)
      

# log prompts into file prompts_log.json 
# prompts_log.json contains prompt ID, promt text, and the input pattern (which is also part of the prompt)
def log_prompt(prompt_id, system_prompt, user_prompt, input_id):
    
    # to avoid producing a corrupt .json file, check if prompt_id already exists in the .json file
    prompt_saved = False
    with open("prompts_log.json", "r") as f:
        for line in f:
            if f'"prompt_id": {prompt_id}' in line:
                prompt_saved = True

        if prompt_saved:
            print("prompt_id already exists in prompts_log.json - skipping log entry. Make sure to manually update prompt_id if this is a new version.")
        else:
            with open("prompts_log.json", "a") as f:
                f.write(f'{{"prompt_id": {prompt_id}, "system_prompt": "{system_prompt}", "user_prompt": "{user_prompt}", "input_pattern": "{input_id}"}}\n')

    
# converting a nested list to a string, for example:
# [[1, 0], [0, 1]] -> 
# '[
# [1, 0], [0, 1]
# ]'
# This is useful when reading a pattern from a file and passing it as a string in the prompt
def array_to_string(array):
    sublists_as_strings = ['[' + ','.join(map(str, sublist)) + ']' for sublist in array]
    result = '[\n\t' + ',\n\t'.join(sublists_as_strings) + '\n]'
    return result

# compare the regenerated pattern to the input pattern
# The similarity between the input and output patterns is a measure of how well the function has regenerated the input 
# pattern using the provided fragments. A similarity of 1.0 would indicate an exact match, while a similarity of 0.0 would 
# indicate no match. 
# This is just a simple element-wise comparison of the two patterns.
# tbd:is still  used?
""" def compare_patterns_simple(pattern1, pattern2):
    pattern1 = np.array(pattern1)
    pattern2 = np.array(pattern2)

    if pattern1.shape != pattern2.shape:
        return 0
    
    return np.sum(pattern1 == pattern2) / pattern1.size
# compare patterns, with a decreased penalty for undefined elements in the output
# tbd: move to parent prompt class
def compare_patterns(pattern1, pattern2):
    pattern1 = np.array(pattern1)
    pattern2 = np.array(pattern2)

    # are the arrays the same size?
    if pattern1.shape != pattern2.shape:
        return 0
    
    total_matches = np.sum(pattern1 == pattern2)
    return(total_matches + 0.5 *  np.sum(pattern2 == 0.5))/ pattern1.size 
"""

# file names are log.py, log1.py, log2.py, ... find the next available name
# get the last log file name in code_synthesis_log directory
def get_log_name():
   files = os.listdir("code_synthesis_log")
   
   # select only python files
   files = [f for f in files if f.endswith(".py")]

   if len(files) == 0:
       return "log0"
   else:
       max_log_number = max([int(re.search(r'\d+', log).group()) for log in files])
       return f"log{max_log_number+1}" 
   
def log_completion_text(completion_text, filename):
    with open(f"code_synthesis_log/{filename}.txt", "w") as f:
        f.write(completion_text)


# save code returned by gpt to a log file
def log_completion(input_map, pattern_code):
    str_map = array_to_string(input_map)
    logf = get_log_name()  # get the next available log file name, and save the code there
    print(f"Saving code to code_synthesis_log/{logf}.py")

    with open(f"code_synthesis_log/{logf}.py", "w") as f:
        f.write('input_map=' + str_map + '\n')
        f.write(pattern_code)
        
    return logf
   
# Calculate the number of errors and omissions in the output pattern relative to the input pattern
#tbd, move to Prompt
def get_errors_and_omissions(input_map, output):
    input_dims = (len(input_map), len(input_map[0]))
    errors_and_omissions = (input_dims[0]*input_dims[1]) - np.sum(input_map == output)
    omissions = np.sum(output == 0.5)
    errors = errors_and_omissions - omissions
    return errors, omissions


# Calculate the mdl score of the solution -- moving out here, as we are using this function in several prompts
# tbd: move to Prompt 
def structural_mdl_score(fragment, copies, errors, omissions):
   """
   Calculates an mdl score relying only on the fragment chosen and the placement of its copies (structure),
   taking into account any errors and omissions relative to the original map. 
   This ignores any compressibility of the structure by scoring it directly instead of scoring the partition function
   which generated it.
   """
   height, width = len(fragment), len(fragment[0])
   def num_encoding(n):
      if n == 0:
         return 1
      return 2*np.ceil(np.log2(n)) + 1
   coordinate_cost = num_encoding(height) + num_encoding(width)
   # specify height and width then specify each element of the fragment
   fragment_cost = coordinate_cost + height * width
   # specify corner position, then 1 bit for reflection and 2 bits for 2^2 possible rotations
   copy_cost = coordinate_cost + 1 + 2
   # A self-delimiting code must specify the number of (copy) entries, then include each of them
   structure_cost = num_encoding(len(copies)) + copy_cost * len(copies)
   # The cost of correcting each error (by specifying its positions)
   error_cost = num_encoding(errors) + coordinate_cost * errors 
   # Assuming the original input size is known, the number of omissions can be derived
   # from the fragment and its copies. Therefore it is only necessary to list the correct value
   # at each omitted coordinate. 
   omission_cost = omissions
   return fragment_cost + structure_cost + error_cost + omission_cost


# Cole's helper function, given explicitly to gpt in prompts 2 and 3, but used by all prompts
def construct_copy(fragment, reflect, rotations):
        if reflect:
            fragment = np.flip(fragment, 1)
        fragment = np.rot90(fragment, rotations)
        return fragment

# Cole's helper function, given explicitly to gpt in prompts 2 and 3, but used by all prompts
def regenerate_pattern(fragment, copies, input_dims):
    
    if (type(fragment) == list):
        fragment = np.array(fragment)

    pattern = 0.5 * np.ones(shape=input_dims)
    for copy in copies:
        transformed = construct_copy( fragment, copy["reflect"], copy["rotations"])
        height, width = transformed.shape
        tl_i, tl_j = copy["top left"]
        try:
            pattern[tl_i : tl_i+height, tl_j : tl_j+width] = transformed
        except:
            pass
    return pattern

# returns the similarity between two patterns
def similarity(input_map, output):

    # are the arrays the same size?
    if input_map.shape != output.shape:
        return 0
        
    total_matches = np.sum(input_map == output)
    return(total_matches + 0.5 * np.sum(output == -1))/input_map.size

   

# Generalized solution, where various orientations of a fragment are matched to the map.
# This partition can provides an acceptable solution, but it has two issues: 
# 1. It can not handle noise. If even one pixel differs between input and fragnmet, we will match nothing
#    TBD: Maybe may replace fragment pixels that do not match output as undefined, to allow partial match
# 2. We only process vertical reflection, 
#    Horizontal reflection represented as two rotations map to a higher MDL score than they should be

def partition(input_map, fragment):
    copies = []

    # sometimes fragment and input_map still come in as a list, 
    # this must be coming from exec(code), where LLM code calls partition from here

    if (type(fragment) == list):
        fragment = np.array(fragment)

    if (type(input_map) == list):
        input_map = np.array(input_map)

    fragment_height, fragment_width = fragment.shape
    input_height, input_width = input_map.shape
    for tl_i in range(0,input_height): # moving vertically
        for tl_j in range(0,input_width): # moving horizontally
            window = input_map[tl_i:tl_i+fragment_height, tl_j:tl_j+fragment_width]
            
            if window.shape != fragment.shape:
                continue
            
            found = False
            for reflect in [False, True]: # vertical reflection along Y-axis 
                for rotations in range(4):
                  transformed = construct_copy(fragment, reflect, rotations)

                  if np.array_equal(transformed, window): 
                     copies.append({ "top left": (tl_i, tl_j), "reflect": reflect, "rotations": rotations})
                     found = True
                     break
                  
                if found:
                    break
    return copies

# Extract python code from a string that may contain mixed text and python code
def extract_python(s):

    # does s contain "```python"  ?
    if "```python" not in s and "```Python" not in s:
        return extract_python_code_from_possibly_mixed_output(s)
    
    # if it does, attempt to remove the text and return only python 
    lines = re.compile(r'```python(.*?)```', re.DOTALL | re.IGNORECASE).findall(s)
    
    # Lines is a list of strings. We need a single string 
    resp = ""
    for line in lines:
        resp += line
        
    return resp
    

def fragment_id(f):
    return hash(str(f.tolist()))


# idk if this will work, so for now just logging lines that should be processed 
# the goal here is to fix inappropriate gpt-4 output, where text and python is mixed
def extract_python_code_from_possibly_mixed_output(mixed_text_and_code):

    # the dubmest way to do this is to look for what seems to be Python blocks,
    # as any stray text will be unindented

    # any unindented text should either start with "#" or end with one of ":", ")", "]", "}", ",", "[", "(", "{"
    # if these conditions are not met, it is probably garbadge

    resp = ""
    lines = mixed_text_and_code.split("\n")
    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == " " or line[0] == "\t":
            #print("Ignoring indented line:", line)
            resp += line + "\n"
        else:
            line = line.strip()
            if (
                (line[-1] in [")", "]", "}", ",", "[", "(", "{"]) or 
                (line[0] == "#") or 
                (line[0:4] == "def ") or 
                (line[0:4] == "class ") or
                (line[0:6] == "import") or
                (line[0:4] == "from") or
                (line[0:3] == "for") or
                (line[0:5] == "while") or
                (line[0:2] == "if") or
                (line[0:4] == "else") or
                (line[0:3] == "try") or
                (line[0:5] == "except") or
                (line[0:6] == "return") or
                (line[0:5] == "print")
                ):
                resp += line+ "\n"
            else:
                print("Suspected misformed comment:", line)

    return resp