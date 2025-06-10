from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os, re

class Cell(Enum):
    ROOM = 0
    WALL = 1
    UNDEFINED = 2

class Cost(Enum):
    REFLECT = 1
    ROTATE = 2

    
def get_fragment_id(f):
    return hash(str(f.tolist()))


def reformat(raw_map):
    formatted_map = [[Cell(num) for num in row] for row in range(len(raw_map))]
    return np.array(formatted_map)


def map_to_string(map_array):
    sublists_as_strings = ['[' + ','.join(map(str, sublist)) + ']' for sublist in map_array]
    result = '[\n\t' + ',\n\t'.join(sublists_as_strings) + '\n]'
    return result


def compute_errors_and_omissions(input_map, output_map):
    height, width = input_map.shape

    matches = np.sum(input_map == output_map)
    omissions = np.sum(output_map == Cell.UNDEFINED.value)
    errors = height * width - matches - omissions
    return omissions, errors


def similarity_score(input_map, output_map):
    if input_map.shape != output_map.shape:
        return 0
    
    num_matches = np.sum(input_map == output_map)
    num_undefined = np.sum(output_map == Cell.UNDEFINED.value)

    # whole penalty for mismatch and half penalty for undefined
    return (num_matches + 0.5 * num_undefined) / input_map.size


def structural_mdl_score(fragment, copies, errors, omissions):
    def num_encoding(n):
        return 2 * np.ceil(np.log2(n)) + 1 if n != 0 else 1
    
    height, width = fragment.shape

    coordinate_cost = num_encoding(height) + num_encoding(width)
    # specify height, width, and each cell in fragment
    fragment_cost = coordinate_cost + height * width
    # specify corner position, reflection, and rotation
    copy_cost = coordinate_cost + Cost.REFLECT.value + Cost.ROTATE.value
    # a self-delimiting code must specify the number of (copy) entries, then include each of them
    structure_cost = num_encoding(len(copies)) + copy_cost * len(copies)
    # cost of correcting each error by specifying its position
    error_cost = num_encoding(errors) + coordinate_cost * errors
    omission_cost = omissions
    
    return fragment_cost + structure_cost + error_cost + omission_cost


def transform_fragment(fragment, reflection, rotation):
    if reflection:
        fragment = np.flip(fragment, 1)
    fragment = np.rot90(fragment, rotation)
    return fragment


# Generalized solution, where various orientations of a fragment are matched to the map.
# This partition can provides an acceptable solution, but it has two issues: 
# 1. It can not handle noise. If even one pixel differs between input and fragnmet, we will match nothing
#    TBD: Maybe may replace fragment pixels that do not match output as undefined, to allow partial match
# 2. We only process vertical reflection, 
#    Horizontal reflection represented as two rotations map to a higher MDL score than they should be

def find_map_partition(input_map, fragment):

    if type(input_map) == list:
        input_map = np.array(input_map)
    if type(fragment) == list:
        fragment = np.array(fragment)

    partition = []

    input_height, input_width = input_map.shape
    fragment_height, fragment_width = fragment.shape

    for row in range(input_height):
        for col in range(input_width):
            sub_map = input_map[row:row + fragment_height, col:col + fragment_width]

            if sub_map.shape != fragment.shape:
                continue

            # find a transformed fragment that matches submap
            # prioritize no reflection / rotation
            found = False
            for reflection in [False, True]:
                for rotation in range(4):
                    transformed = transform_fragment(fragment, reflection, rotation)
                    if np.array_equal(transformed, sub_map):
                        partition.append({
                            "top_left": (row, col),
                            "reflection": reflection,
                            "rotation": rotation
                        })
                        found = True
                        break
                if found:
                    break
    return partition
                


def generate_from_partition(fragment, partition, input_dims):
    if type(fragment) == list:
        fragment = np.array(fragment)
    
    generated_map = np.full(input_dims, Cell.UNDEFINED.value)
    
    for p in partition:
        transformed = transform_fragment(fragment, p["reflection"], p["rotation"])
        height, width = transformed.shape
        tl_row, tl_col = p["top_left"]

        # Check if placement is within bounds
        if tl_row + height > input_dims[0] or tl_col + width > input_dims[1]:
            return False

        # Check for overlap
        current_slice = generated_map[tl_row:tl_row + height, tl_col:tl_col + width]
        overlap = (current_slice != Cell.UNDEFINED.value) & (transformed != Cell.UNDEFINED.value)
        if np.any(overlap):
            return False

        generated_map[tl_row:tl_row + height, tl_col:tl_col + width] = transformed

    return generated_map