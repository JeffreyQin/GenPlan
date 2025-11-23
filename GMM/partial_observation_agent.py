import numpy as np
from openai import OpenAI
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
        self.model = "gpt-4"

        self.n_unit_completions = 0
        self.n_recon_completions = 1 

    def set_maps(self, partial_map, complete_map, zero_one_only=True):
        if zero_one_only:
            partial_map = np.where(np.array(partial_map) > 0, 1, 0)
            complete_map = np.where(np.array(complete_map) > 0, 1, 0)

        self.partial_map = np.array(partial_map)
        self.complete_map = np.array(complete_map)

    def set_prompts(self,
        unit_system_prompt,
        unit_prompt,
        recon_system_prompt,
        recon_prompt
    ):
        self.unit_system_prompt = {"role": "system", "content": unit_system_prompt}
        self.unit_user_prompt = unit_prompt

        self.recon_system_prompt = {"role": "system", "content": recon_system_prompt}
        self.recon_user_prompt = recon_prompt


    def get_unit_prompt(self):
        return self.unit_user_prompt[0] + map_to_string(self.partial_map) + self.unit_user_prompt[1]

    def get_reconstruction_prompt(self, unit):
        return self.recon_user_prompt[0] + map_to_string(self.complete_map) + self.recon_user_prompt[1] + unit


    def get_unit_completions(self):
    
        print("sending unit prompt...")

        response = self.client.chat.completions.create(
            n=self.n_completions,
            model=self.model,
            messages=[
                self.unit_system_prompt,
                {"role": "user", "content": self.get_unit_prompt()}
            ]
        )
        out = {}
        for i, c in enumerate(response.choices):
            out[i] = c.message.content

        return out


    def get_reconstruction_completion(self, unit):

        print("sending reconstruction prompt...")

        response = self.client.chat.completions.create(
            n=self.n_recon_completions,
            model=self.model,
            messages=[
                self.recon_system_prompt,
                {"role": "user", "content": self.get_unit_prompt()}
            ]
        )
        out = {}
        for i, c in enumerate(response.choices):
            out[i] = c.message.content

        return out
    

    def run(self, n_completions, show_plots=True, console_logs=False):

        self.n_completions = n_completions
        all_results = {}

        # get unit candidates
        unit_candidates = self.get_unit_completions()

        for i, unit_str in unit_candidates.items():

            print("\n=============== unit candidate", i, "================\n")

            print("UNIT")
            print(unit_str)

            plot_input_response(self.partial_map, unit_str, "Unit candidate i")

            # get reconstruction code
            recon_resp = self.get_reconstruction_completion(unit_str)
            raw_recon_text = recon_resp[0]

            recon_code = extract_python(raw_recon_text)
            print("RECONSTRUCTION CODE")
            print(recon_code)

            # PROCESS RECONSTRUCTION RESULTS (your existing logic)
            recon_results = self.process_reconstruction_code(
                recon_code,
                show_plots=show_plots,
                console_logs=console_logs
            )

            # aggregate
            for k, v in recon_results.items():
                all_results[k] = v

        if len(all_results) == 0:
            print("No reconstruction succeeded.")
            return {}

        ranked = sorted(all_results.items(), key=lambda x: x[1]["similarity"], reverse=True)

        return ranked

    def process_reconstruction_code(self, code, show_plots=True, console_logs=False):


        map_completions = {}
        local_vars = {'np': np}

        globals()['input_map'] = self.complete_map

        try:
            exec(code, globals(), local_vars)

        except Exception as e:
            print("Reconstruction code failed:", e)
            traceback.print_exc()
            return {}

        fragment = local_vars.get("fragment", None)
        if fragment is None:
            return {}

        fragment = np.array(fragment)

        try:
            partition = find_map_partition(self.complete_map, fragment)
            output = generate_from_partition(fragment, partition, self.complete_map.shape)
            if output is False:
                return {}

            sim = round(similarity_score(self.complete_map, output), 2)
            errors, omissions = compute_errors_and_omissions(self.complete_map, output)
            mdl = structural_mdl_score(fragment, partition, errors, omissions)

            map_completions[get_fragment_id(fragment)] = {
                "fragment": fragment,
                "original_map": self.complete_map,
                "reconstructed_map": output,
                "mdl": mdl,
                "similarity": sim,
            }

        except Exception as e:
            print("Evaluation failed:", e)
            traceback.print_exc()

        return map_completions
