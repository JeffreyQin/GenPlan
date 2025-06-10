
import numpy as np
from openai import OpenAI
from map_utils import *
from plot_utils import *
from other_utils import *
from maps import input_maps
import traceback
import os, re
from dotenv import load_dotenv

class MapCompletionAgent:
 
    def __init__(self):
        
        load_dotenv()

        self.model = "gpt-4"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.system_prompt = ""
        self.user_prompt = ""
        self.prompt_id = 0
        self.n_completions = 0
        self.input_map = ""
        self.input_id = 0

    def get_user_prompt(self):
        #returns the full prompt to be used for chatgpt
        if (self.input_map is None):
            print("Input map is not set, call set_input_map to specify a map")
            raise Exception("Input map is not set, call set_input_map to specify a map")
        
        str_map = map_to_string(self.input_map)
        return (self.user_prompt[0] + str_map + self.user_prompt[1])
    
    # input_map is a 2D array
    # n_completions - we may ask for more completions for harder testcases, and fewer for simple ones
    def set_input_map(self, input_id, n_completions):
        self.input_id = input_id
        self.input_map = np.array(input_maps[input_id])
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
            
        #plot_input_response(self.input_map, fragments, save_image=log_file_name, show_plots=show_plots)

        try:
            for f in fragments:

                partition = find_map_partition(self.input_map, f)
                output = generate_from_partition(f, partition, self.input_map.shape)
                if output is False:
                    print("FAILURE")
                    continue
                else:
                    print("SUCCESS")
                sim = round(similarity_score(self.input_map, output), 2)
                errors_and_omissions = compute_errors_and_omissions(self.input_map, output)
                mdl_score = structural_mdl_score(f, partition, errors_and_omissions[0], errors_and_omissions[1])
                
                # plot_input_response(output, fragments, save_image=log_file_name, show_plots=show_plots)

                if (console_logs): 
                    print("fragment:\n ", f, "\n partition: \n", partition, "\n output: \n", output, "\n similarity: ", sim)

                map_completions[get_fragment_id(f)] = {
                    "fragment": f, 
                    "log_file": log_file_name,
                    "original_map": self.input_map, 
                    "reconstructed_map": output, 
                    "fragment": fragments,
                    "mdl": mdl_score,
                    "similarity": sim
                }
         
        except Exception as e:
            print(f"Reconstruction failed with Error: {e}")

        return map_completions
    

    def get_completions(self):
        print("Sending prompt..")
        completion = self.client.chat.completions.create(
            n=self.n_completions, 
            model=self.model,  
            messages=[ self.system_prompt, 
            {"role": "user", "content":  self.get_user_prompt()} ]
        )

        responses = {}
        for i, completion in enumerate(completion.choices):
            responses[i] = completion.message.content

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

            code = extract_python(resp)

            print(code)

            """
            if (not debug_mode):
                log_file_name = log_completion(self.input_map, code)  # log what we think is Python code part
                log_completion_text(resp, log_file_name)              # log full text
                log_prompt(self.prompt_id, self.system_prompt, self.get_user_prompt(), self.input_id) # log prompt
                log_files += log_file_name
            else:
                # get the part of sring i from "/" until the end of line - this is the log file name
                log_files = i[i.find("/")+1:]
            """

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

            plot_input_response(best_similarity[1]['original_map'], best_similarity[1]['fragment'], "Best Similarity: original map", save_image=log_file_name, show_plots=show_plots)
            plot_input_response(best_similarity[1]['reconstructed_map'], best_similarity[1]['fragment'], "Best Similarity: reconstructed map", save_image=log_file_name, show_plots=show_plots)
            """
            if (not debug_mode):
                if (best_mdl[0] != best_similarity[0]):
                    generate_completion_plot(self.input_map, best_similarity[1].get("fragment"), "Best Similarity completion", f"out_similarity_{log_files}", show_plots)
                    generate_completion_plot(self.input_map, best_mdl[1].get("fragment"), "Best MDL completion", f"out_mdl_{log_files}", show_plots)
                else:
                    generate_completion_plot(self.input_map, best_similarity[1].get("fragment"), "Best completion", f"out_{log_files}", show_plots)
            """

            print(f"{len(sorted_maps)} completions returned")

            if (not debug_mode):
                with open("resp_log.csv", "a") as f:
                    f.write(f'{self.prompt_id}, "{self.input_id}", {self.n_completions}, {best_similarity[1].get("similarity")}, {best_mdl[1].get("mdl")}, "{log_files}", True\n')

            return sorted_maps

