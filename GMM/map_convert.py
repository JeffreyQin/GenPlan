import numpy as np
import os
from glob import glob

def load_and_process(filename):
    print(filename)
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Ignore the first two lines (height and width)
    data_lines = lines[2:]

    # Parse characters into int matrix
    array = np.array([[int(char) for char in line] for line in data_lines], dtype=int)

    # Apply mapping: 3 → 1, 0/5/6 → 0, everything else → 0
    processed_array = np.where(array == 3, 1, 0)
    return processed_array

def load_all_maps(directory):
    all_files = sorted(glob(os.path.join(directory, '*.txt')))
    all_arrays = []
    filenames = []

    for f in all_files:
        try:
            filenames.append(f)
            arr = load_and_process(f)
            all_arrays.append(arr)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return all_arrays, filenames
# Example usage
directory = 'maps'
result, filenames = load_all_maps(directory)

maps = {idx: arr.tolist() for idx, arr in enumerate(result)}
f_maps = {idx: file for idx, file in enumerate(filenames)}

# Save to a .py file
with open("maps.py", "w") as f:
    f.write("maps = {\n")
    for k, v in maps.items():
        f.write(f"    {k}: {v},\n")
    f.write("}\n")

with open("files.py", "w") as f:
    f.write("files = {\n")
    for k, v in f_maps.items():
        f.write(f"    {k}: {v},\n")
    f.write("}\n")