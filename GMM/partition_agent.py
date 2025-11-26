from prompts import system_prompt, user_prompt, unit_prompt, recon_prompt
#from map_completion_agent import MapCompletionAgent
from partial_observation_agent import PartialObservationAgent

from map_completion_agent_orig import MapCompletionAgentOrig


from maps import input_maps

#class PartitionAgent(MapCompletionAgent):
#       
##    def __init__(self):
 #        super().__init__()
 ##        self.system_prompt: int = system_prompt
  #       self.user_prompt: str = user_prompt

#agent = PartitionAgent()

def crop(matrix):
    return [row[:11] for row in matrix]

input_map = input_maps[25]
partial_map = crop(input_map)

print(partial_map)
print(input_map)

partial_agent = PartialObservationAgent()
partial_agent.set_maps(partial_map, input_map, True)
partial_agent.set_prompts(system_prompt, unit_prompt, recon_prompt)

partial_agent.run(1)
partial_agent.input_id = 26
exit()
#try:
#    agent.send_prompt(3, n_completions=3, show_plots=True)
#except Exception as e:
#   print(f"error sending prompt")


