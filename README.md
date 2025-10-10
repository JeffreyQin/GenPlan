# GenPlan

Codebase associated with paper * *Planning with Generative Cognitive Maps* *. We propose a novel planning framework that integrates

1. a Generative Map Module (GMM) that infers generative compositional structure and
2. a Structure-Based Planner (SBP) that exploits structural redundancies to reduce planning costs.

In particular, we show that the framework models human planning better than existing models via human experimentation, and demonstrate its computational efficiency via quantitative comparison with a Naive POMCP planner.

## Navigation

Relevant code for the Generative Map Module is found in ```/GMM/```.

Environment maps used in either experiments can be found in ```/maps/```.

Plots for experimental results can be found in ```/plots/```

Our implementation of the structure-based planner (SBP) consists of the fragment search, escape search, and bridge search modules. Fragment search plans a local policy for within the unit, which is reused whenever a similar unit is encountered. Escape search plans a path to exit the current unit upon completion, and tries to minimize the approximated remaining costs of exploration. Bridge search plans a global policy that encourages the agent to look for "unit" structures while exploring between-unit areas and minimizing anticipated costs.

## Use

### Setup

Install dependencies with
```
pip install -r requirements.txt
```

To run Gen-POMCP, run
```
python test_sbp_planner.py
```

To run Naive-POMCP (for comparison purposes), run
```
python test_naive_planner.py
```

On completion, a window will pop up to visualize the path the agent has taken, and can be controlled by pressing the "enter" key.





