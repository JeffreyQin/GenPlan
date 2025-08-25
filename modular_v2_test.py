from modular_v2 import *

"""
global_map = np.array([
    [0, 0, 0, 2, 0, 0, 0],
    [2, 2, 0, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 2, 0],
    [2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0],
    [3, 2, 2, 0, 0, 0, 0],
])
fragment = np.array([
        [0, 0, 0],
        [2, 2, 0],
        [0, 2, 0],
])

agent_pos = (6, 0)


import collections
path = run_fragment_search(dict(), global_map, agent_pos)
print(path)
"""
# Test code
if __name__ == "__main__":
    # Test run_modular function
    print("Testing run_modular function...")
    
    global_map = np.array(
    [
        [0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    ]
    )

    fragment = np.array([
        [2, 2, 2, 2, 2, 2, 2],
        [2, 0, 0, 2, 0, 2, 2],
        [2, 0, 0, 2, 0, 2, 2],
        [2, 2, 2, 2, 0, 2, 2],
    ])

    copies = [
        {"top left": (0,2), "reflect": False, "rotations": 0},
        {"top left": (0,15), "reflect": False, "rotations": 0},
        {"top left": (2,21), "reflect": True, "rotations": 0},
        {"top left": (6,1), "reflect": True, "rotations": 0},
        {"top left": (6,16), "reflect": False, "rotations": 0},
        {"top left": (11,1), "reflect": False, "rotations": 2},
        {"top left": (15,1), "reflect": True, "rotations": 0},
        {"top left": (15,7), "reflect": True, "rotations": 0},
        {"top left": (11,16), "reflect": True, "rotations": 2},
    ]
    
    print(f"Fragment shape: {fragment.shape}")
    print(f"Global map shape: {global_map.shape}")
    print(f"Number of copies: {len(copies)}")
    print(f"Initial agent position: (4, 4)")
    
    
    # Test the modular planning
    print("\nStarting modular planning...")
    result = run_modular(global_map, fragment, copies)
    print("Modular planning completed successfully!")

    
    print("\nTest completed!")
    