from prompts import system_prompt, user_prompt
from map_completion_agent_orig import MapCompletionAgentOrig

class PartitionAgent(MapCompletionAgentOrig):
       
    def __init__(self):
         super().__init__()
         self.system_prompt: int = system_prompt
         self.user_prompt: str = user_prompt


agent = PartitionAgent()
try:
    agent.send_prompt(1, n_completions=1, show_plots=True)
except Exception as e:
    print(f"{e}")


