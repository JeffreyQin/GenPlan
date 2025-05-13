import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def parse_grid_from_string(grid_str):
    """
    Parses a multiline string where each line is digits into a 2D list of ints.
    """
    grid = []
    for line in grid_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        row = [int(ch) for ch in line]
        grid.append(row)
    return grid


def map_values(grid, mapping):
    """
    Maps each cell value in the grid according to the provided mapping dict.
    """
    return [[mapping.get(cell, cell) for cell in row] for row in grid]

def visualize_array(arr):
    """
    Visualizes a 2D numpy array with distinct colors per integer value and grid lines.
    Now also labels rows and columns with numbers.
    """
    unique = sorted(set(arr.flatten()))
    # choose a colormap
    if len(unique) <= 20:
        cmap = plt.get_cmap('tab20', len(unique))
        colors = [cmap(i) for i in range(len(unique))]
    else:
        colors = [plt.cm.hsv(i / len(unique)) for i in range(len(unique))]
    # map values to indices
    idx_map = {val: i for i, val in enumerate(unique)}
    indexed = np.vectorize(lambda x: idx_map[x])(arr)
    custom_cmap = ListedColormap(colors)

    rows, cols = arr.shape
    fig, ax = plt.subplots(figsize=(cols * 0.3, rows * 0.3))
    ax.imshow(indexed, cmap=custom_cmap, interpolation='nearest')

    # draw grid lines
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)

    # set major ticks for labeling
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticklabels(np.arange(rows))

    # rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', va='center', rotation_mode='anchor')

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_ticks_position('top')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Convert and visualize numeric grid.')
    parser.add_argument('--file', '-f', help='Path to text file grid', default=None)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            grid_str = f.read()
    else:
        # <<< PASTE YOUR GRID BELOW >>>
        grid_str = """
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
33333333333333333333333333333333333333333333333333333333333
"""

    # parse and map
    grid = parse_grid_from_string(grid_str)
    mapping = {5: 3, 0: 2, 3: 0, 6: 2}
    mapped = map_values(grid, mapping)
    arr = np.array(mapped)

    # print array for copy-paste
    print("np.array(")
    print(repr(arr.tolist()).replace('],', '],\n'))
    print(")")

    # visualize
    visualize_array(arr)

if __name__ == '__main__':
    main()
