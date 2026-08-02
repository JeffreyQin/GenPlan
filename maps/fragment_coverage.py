"""Report how much of each map in map_set2 is covered by fragments."""

import json
from pathlib import Path

import numpy as np

import map_set2


OUTPUT_PATH = Path(__file__).with_name("fragment_coverage.json")


MAPS = {
    map_number: (
        getattr(map_set2, f"map_{map_number}"),
        (
            [
                (map_set2.fragment_9_a, map_set2.copies_9_a),
                (map_set2.fragment_9_b, map_set2.copies_9_b),
            ]
            if map_number == 9
            else [
                (
                    getattr(map_set2, f"fragment_{map_number}"),
                    getattr(map_set2, f"copies_{map_number}"),
                )
            ]
        ),
    )
    for map_number in range(1, 11)
}


def transform_fragment(fragment: np.ndarray, copy: dict) -> np.ndarray:
    """Apply a copy's reflection and counter-clockwise quarter rotations."""
    transformed = np.fliplr(fragment) if copy["reflect"] else fragment
    return np.rot90(transformed, k=copy["rotations"] % 4)


def calculate_coverage(
    map_data: np.ndarray,
    fragment_groups: list[tuple[np.ndarray, list[dict]]],
) -> tuple[float, int, int, float, int, int]:
    """Return open-cell and whole-grid fragment coverage.

    Overlapping fragment copies are counted only once in both measurements.
    """
    covered_open_cells_mask = np.zeros(map_data.shape, dtype=bool)
    covered_all_cells_mask = np.zeros(map_data.shape, dtype=bool)

    for fragment, copies in fragment_groups:
        for copy in copies:
            transformed = transform_fragment(fragment, copy)
            row, column = copy["top left"]
            height, width = transformed.shape
            bottom, right = row + height, column + width

            if row < 0 or column < 0 or bottom > map_data.shape[0] or right > map_data.shape[1]:
                raise ValueError(
                    f"Fragment copy at {(row, column)} with shape {transformed.shape} "
                    f"falls outside map shape {map_data.shape}"
                )

            # A fragment covers its open cells, not the walls in its
            # rectangular bounding box.
            covered_open_cells_mask[row:bottom, column:right] |= transformed != 0
            covered_all_cells_mask[row:bottom, column:right] = True

    open_cells = map_data != 0
    covered_open_cells = int(np.count_nonzero(covered_open_cells_mask & open_cells))
    total_open_cells = int(np.count_nonzero(open_cells))
    open_cells_percentage = 100.0 * covered_open_cells / total_open_cells

    covered_all_cells = int(np.count_nonzero(covered_all_cells_mask))
    total_all_cells = int(map_data.size)
    all_cells_percentage = 100.0 * covered_all_cells / total_all_cells

    return (
        open_cells_percentage,
        covered_open_cells,
        total_open_cells,
        all_cells_percentage,
        covered_all_cells,
        total_all_cells,
    )


def main() -> None:
    results = {}
    for map_number, (map_data, fragment_groups) in MAPS.items():
        (
            open_cells_percentage,
            covered_open_cells,
            total_open_cells,
            all_cells_percentage,
            covered_all_cells,
            total_all_cells,
        ) = calculate_coverage(map_data, fragment_groups)
        results[str(map_number)] = {
            "open_cells": {
                "coverage_percentage": round(open_cells_percentage, 2),
                "covered_cells": covered_open_cells,
                "total_cells": total_open_cells,
            },
            "all_cells": {
                "coverage_percentage": round(all_cells_percentage, 2),
                "covered_cells": covered_all_cells,
                "total_cells": total_all_cells,
            },
        }
        print(
            f"Map {map_number:2d}: open cells = {open_cells_percentage:6.2f}%, "
            f"all cells = {all_cells_percentage:6.2f}%"
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")

    print(f"\nCoverage results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
