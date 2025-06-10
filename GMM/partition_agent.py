from prompts import system_prompt, user_prompt
from map_completion_agent import MapCompletionAgent

class PartitionAgent(MapCompletionAgent):
       
    def __init__(self):
         super().__init__()
         self.system_prompt: int = system_prompt
         self.user_prompt: str = user_prompt


agent = PartitionAgent()
try:
    agent.send_prompt(3, n_completions=3, show_plots=True)
except Exception as e:
    print(f"error sending prompt")


