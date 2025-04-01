
import utils as ut
import numpy as np
from openai import OpenAI
# import secret_keys (comment this out and hope it works)
# import pattern_editor
from utils import regenerate_pattern
from utils import similarity
from utils import construct_copy
import traceback

class MapCompletionPrompt:
 
    def __init__(self):
        self.model = "gpt-4" #you might need to change this jeff

        # to be overriden in derived classes
        self.n_completions      = 5
        self.system_prompt      = ""
        self.prompt_id          = 0
        self.user_prompt_1      = ""
        self.user_prompt_2      = ""
        self.input_map          = None
        self.input_id          = ""

        # to initialize the OpenAI client here we need to have a file secret_keys.py that contains an API key
        self.client = OpenAI(api_key="") #please put your API key here
    

    def get_user_prompt(self):
        #returns the full prompt to be used for chatgpt
        if (self.input_map is None):
            print("Input map is not set, call set_input_map to specify a map")
            raise Exception("Input map is not set, call set_input_map to specify a map")
        
        str_map = ut.array_to_string(self.input_map)    
        return (self.user_prompt_1 + str_map + self.user_prompt_2)
    
    # input_map is a 2D array
    # n_completions - we may ask for more completions for harder testcases, and fewer for simple ones
    def set_input_map(self, input_id, n_completions):
        self.input_id = input_id
        self.input_map = [] #fill this  map up
        self.input_map = np.array(self.input_map)
        self.n_completions = n_completions
   

    # This is where we interact with the LLM, or simulator.
    # We send the prompt, get the completions, and return a dictionary of map completions 
    # id - fragmment hash, 
    # fragment, 
    # reconstructed map, 
    # mdl, 
    # similarity
    #
    # Explanation for the unusual exception handling:
    # locals() does not have variables defined in exec(code), but local_vars does
    # If exec(code) declared any global variables, they will go to local_vars
    # If exec(code) defined any function trying to use its own global variables, that function will fail with NameError

    def process_map_completion_from_response(self, code, log_file_name="", show_plots=True, console_logs = False):
    
        map_completions = {}
        local_vars = {'self': self, 'np': np}

        # this matters for some test-cases, although it seems that doing this in NameError block should be enough
        globals()['input_map'] = self.input_map

        try:
            if(console_logs): print("Running code..\n", code)
            exec(code, globals(), local_vars)
        
        except NameError as e:
            print(f"LLM-returned code failed with Error: {e}")
            traceback.print_exc()

            print("Adding from local_vars to globals(): ")
            added_vars = ['input_map']
            
            for k, v in local_vars.items():
                # add to global any local variables that do not begin with "__"
                if k not in globals() and k != 'self' and k[:2] != '__':
                    globals()[k] = v
                    added_vars.append(k)
                    print(k)

            try:
                exec(code, globals(), local_vars)

                # clean up globals()
                for k in added_vars: del globals()[k] 
            except Exception as e:
                print(f"LLM-returned code failed second time, Error: {e}")
                traceback.print_exc()

                # clean up globals()
                for k in added_vars: del globals()[k] 
                return {}
        except Exception as e:
            print(f"LLM-returned code failed with Error: {e}")
            traceback.print_exc()
            return {}

        # if code defines input_map, make sure it is a numpy array
        if code.find("input_map") != -1: 
            input_map = local_vars.get('input_map', [])
            input_map = np.array(input_map)

        # code might contain an array of fragments, or one fragment
        fragments = local_vars.get('fragments', [])
        if len(fragments) == 0:
            fragments = [local_vars.get('fragment', [])]
            print("One fragment returned", fragments)
            
        # make sure all fragments are numpy arrays
        fragments = [np.array(f) for f in fragments]
            
        # remove any one-dimentional fragmetns
        fragments = [f for f in fragments if len(f.shape) == 2]

        # remove any that are too small for planning
        fragments = [f for f in fragments if f.shape[0] > 2 and f.shape[1] > 2]

        # ensure fragments are smaller than the input
        fragments = [f for f in fragments if f.shape != self.input_map.shape]

        if (len(fragments) == 0):
            print("No valid fragments returned")
            return {}
            
        ut.plot_input_response(self.input_map, fragments, save_image=log_file_name, show_plots=show_plots)

        try:
            for f in fragments:
                partition = ut.partition(self.input_map, f)
                output = regenerate_pattern(f, partition, self.input_map.shape)
                sim = round(similarity(self.input_map, output),2)

                if (console_logs): 
                    print("fragment:\n ", f, "\n partition: \n", partition, "\n output: \n", output, "\n similarity: ", sim)

                map_completions[ut.fragment_id(f)] = {"fragment": f, "log_file": log_file_name, "reconstructed_map": output, 
                                    "mdl": ut.structural_mdl_score(f,partition, *ut.get_errors_and_omissions(self.input_map, output)), 
                                    "similarity": sim}
         
        except Exception as e:
            print(f"Reconstruction failed with Error: {e}")

        return (map_completions)
    
    # sending the prompt to the OpenAI API
    # DebugPrompt will override this function
    def get_completions(self):
        print("Sending prompt..")
        completion = self.client.chat.completions.create(n=self.n_completions, 
                        model=self.model,  
                        messages=[ self.system_prompt, 
                        {"role": "user", "content":  self.get_user_prompt()} ] )
        
        # return a dictionary where i are keys, and response text are values
        responses = {}
        for i, completion in enumerate(completion.choices):
            responses[i] = completion.message['content']
        
        return responses
    
          

    def send_prompt(self, input_id, n_completions, show_plots = True, debug_mode = False, console_logs = False):

        self.set_input_map(input_id, n_completions)
        maps = {}
        log_files = log_file_name = ""

        # sending prompt, and getting responces
        completions = self.get_completions()
        
        # aggregating maps from all completions
        for i, resp in completions.items():
            print(f"************************************ Completion {i}:\n")

            code = ut.extract_python(resp)
    
            if (not debug_mode):
                log_file_name = ut.log_completion(self.input_map, code)  # log what we think is Python code part
                ut.log_completion_text(resp, log_file_name)              # log full text
                ut.log_prompt(self.prompt_id, self.system_prompt, self.get_user_prompt(), self.input_id) # log prompt
                log_files += log_file_name
            else:
                # get the part of sring i from "/" until the end of line - this is the log file name
                log_files = i[i.find("/")+1:]

            maps.update( self.process_map_completion_from_response(code, log_file_name, show_plots, console_logs) )


        # return all map completions sorted by similarity ( or MDL )
        if (len(maps) == 0):
            print("No completions returned")
            if (not debug_mode):
                with open("resp_log.csv", "a") as f:
                    f.write(f'{self.prompt_id}, "{self.input_id}", {self.n_completions}, {0.0}, {0.0}, "{log_files}", False\n')

            return {}
        else:

            # plot the best MDL and best similarity completions 
            best_mdl = sorted(maps.items(), key=lambda item: item[1]['mdl'], reverse=True)[0]
            sorted_maps = sorted(maps.items(), key=lambda item: item[1]['similarity'], reverse=True)
            best_similarity = sorted_maps[0]

            if (not debug_mode):
                if (best_mdl[0] != best_similarity[0]):
                    ut.generate_completion_plot(self.input_map, best_similarity[1].get("fragment"), "Best Similarity completion", f"out_similarity_{log_files}", show_plots)
                    ut.generate_completion_plot(self.input_map, best_mdl[1].get("fragment"), "Best MDL completion", f"out_mdl_{log_files}", show_plots)
                else:
                    ut.generate_completion_plot(self.input_map, best_similarity[1].get("fragment"), "Best completion", f"out_{log_files}", show_plots)
            
            print(f"{len(sorted_maps)} completions returned")

            if (not debug_mode):
                with open("resp_log.csv", "a") as f:
                    f.write(f'{self.prompt_id}, "{self.input_id}", {self.n_completions}, {best_similarity[1].get("similarity")}, {best_mdl[1].get("mdl")}, "{log_files}", True\n')

            return sorted_maps

    