import numpy as np
from openai import OpenAI
import google.generativeai as genai
from groq import Groq
from map_utils import *
from plot_utils import *
from other_utils import *
import traceback
import re
import os
from dotenv import load_dotenv


class PartialObservationAgent:

    def __init__(self):

        load_dotenv()

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-5"

        self.n_unit_completions = 1
        self.n_recon_completions = 1 

        self.prompt_id = 0
        self.input_id = 1

    def set_maps(self, partial_map, complete_map, zero_one_only=True):
        if zero_one_only:
            partial_map = np.where(np.array(partial_map) > 0, 1, 0)
            complete_map = np.where(np.array(complete_map) > 0, 1, 0)

        self.partial_map = np.array(partial_map)
        self.complete_map = np.array(complete_map)
        self.input_map = np.array(complete_map)

    def set_prompts(self,
        system_prompt,
        unit_prompt,
        recon_prompt
    ):
        self.unit_system_prompt = system_prompt
        self.unit_user_prompt = unit_prompt

        self.recon_system_prompt = system_prompt
        self.recon_user_prompt = recon_prompt


    def get_unit_prompt(self):
        return self.unit_user_prompt[0] + map_to_string(self.partial_map)

    def get_reconstruction_prompt(self, unit):
        
        return self.recon_user_prompt[0] + map_to_string(self.complete_map) + self.recon_user_prompt[1] + unit + self.recon_user_prompt[2]

    
    def get_groq_unit_completions(self):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        responses = {}

        for i in range(1):
            result = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",  # Default recommended model
                messages=[
                    self.unit_system_prompt,
                    {"role": "user", "content": self.get_unit_prompt()}
                ],
                temperature=1.0,
                max_tokens=4096,
                stream=False
            )

            responses[i] = result.choices[0].message.content

        return responses

    def get_gemini_unit_completions(self):

        print("sending unit prompt...")

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel('gemini-1.5-flash')  # or another Gemini model

        response = model.generate_content(
            self.get_unit_prompt(),
            generation_config={"candidate_count": self.n_unit_completions}
        )

        out = {}
        for i, candidate in enumerate(response.candidates):
            out[i] = candidate.content.parts[0].text

        return out
    

    def get_unit_completions(self):
    
        print("sending unit prompt...")

        response = self.client.chat.completions.create(
            n=self.n_completions,
            model='gpt-5',
            messages=[
                self.unit_system_prompt,
                {"role": "user", "content": self.get_unit_prompt()}
            ]
        )
        out = {}
        for i, c in enumerate(response.choices):
            out[i] = c.message.content

        return out

    def get_groq_reconstruction_completion(self, unit):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        responses = {}

        for i in range(1):
            result = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",  # Default recommended model
                messages=[
                    self.recon_system_prompt,
                    {"role": "user", "content": self.get_reconstruction_prompt(unit)}
                ],
                temperature=1.0,
                max_tokens=4096,
                stream=False
            )

            responses[i] = result.choices[0].message.content

        return responses

    def get_gemini_reconstruction_completion(self, unit):

        print("sending reconstruction prompt...")

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel('gemini-2.0-flash')  # or another Gemini model

        response = model.generate_content(
            self.get_reconstruction_prompt(unit),
            generation_config={"candidate_count": self.n_recon_completions}
        )

        out = {}
        for i, candidate in enumerate(response.candidates):
            out[i] = candidate.content.parts[0].text

        return out

    def get_reconstruction_completion(self, unit):
    
        print("sending reconstruction prompt...")

        response = self.client.chat.completions.create(
            n=self.n_recon_completions,
            model='gpt-5',
            messages=[
                self.recon_system_prompt,
                {"role": "user", "content": self.get_reconstruction_prompt(unit)}
            ]
        )
        out = {}
        for i, c in enumerate(response.choices):
            out[i] = c.message.content

        return out
    

    def run(self, n_completions, show_plots=True, console_logs=False, debug_mode=False):

        self.n_completions = n_completions
        maps = {}
        log_files = log_file_name = ""

        # get unit candidates
        unit_candidates = self.get_gemini_unit_completions()
        
        for i, unit_str in unit_candidates.items():

            print("\n=============== unit candidate", i, "================\n")

            print(unit_candidates)
            recon_resp = self.get_groq_reconstruction_completion(unit_str)
            raw_recon_text = recon_resp[0]

            recon_code = extract_python(raw_recon_text)

            maps.update( self.process_map_completion_from_response(recon_code, unit_str, log_file_name, show_plots, console_logs) )

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



    def process_map_completion_from_response(self, code, fragment, log_file_name="", show_plots=True, console_logs = False):
    
        map_completions = {}
        local_vars = {'self': self, 'np': np}

        # this matters for some test-cases, although it seems that doing this in NameError block should be enough
        globals()['input_map'] = self.input_map
        
        import ast
        globals()['fragment'] = np.array(ast.literal_eval(fragment))

        print("ARRIVED")
        print(code)
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

                partition2 = find_map_partition(self.input_map, f)
                partition = local_vars['result']

                print(partition)
                print(partition2)

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
    