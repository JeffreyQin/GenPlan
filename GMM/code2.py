# START OF CODE
def partition():
    # The partition size should be equivalent to the size of the fragment
    fragment_shape = fragment.shape
    # Initialize an empty list to hold the pattern positions
    copies = []
    # Iterate through the input_map with the same step size as the fragment's shape in both dimensions
    for i in range(0, input_map.shape[0] - fragment_shape[0] + 1):
        for j in range(0, input_map.shape[1] - fragment_shape[1] + 1):
            # Extract a sub matrix of the same size as the fragment, from the current position in the input_map
            sub = input_map[i:i+fragment_shape[0], j:j+fragment_shape[1]]

            # Initialize variables for reflection and rotation
            reflection = False
            rotation = 0

            # Check if the fragment matches in normal, reflected and rotated conditions
            for reflect in [True, False]:
                for rotation in range(4):
                    transformed = construct_copy(fragment, reflect, rotation)
                    # Check if the sub part matches the transformed fragment
                    if np.array_equal(sub, transformed):
                        reflection = reflect
                        rotations = rotation
                        # Append the current position and transformation information to the list of matches
                        copies.append({"top left": (i, j), "reflect": reflection, "rotations": rotations})
    return copies
result = partition()