import os, re
from map_utils import *


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

# extract python code from mixed input
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
    

    # save code returned by gpt to a log file
def log_completion(input_map, pattern_code):
    str_map = map_to_string(input_map)
    logf = get_log_name()  # get the next available log file name, and save the code there
    print(f"Saving code to code_synthesis_log/{logf}.py")

    with open(f"code_synthesis_log/{logf}.py", "w") as f:
        f.write('input_map=' + str_map + '\n')
        f.write(pattern_code)
            
    return logf
    