"""
Fragment-coverage vs SBP-rollout experiment for map_set1 / map_set2.

While at least 2 fragment copies remain:
  1. measure all-cells fragment coverage
  2. run SBP over remaining fragment copies, then naive POMCP on the
     leftover unobserved map; record total rollouts
  3. if more than 2 copies remain, replace one random copy's footprint
     with a hand-authored corrupted fragment and drop it from copies

map_set1 is the smaller smoke-test suite; map_set2 is the full suite.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MAPS_DIR = ROOT / "maps"
if str(MAPS_DIR) not in sys.path:
    sys.path.insert(0, str(MAPS_DIR))

import globals
import map_set1
import map_set2
from fragment_coverage import calculate_coverage, transform_fragment
from fragment_search import FragmentPOMCP
from generator import Generator
from structure_based_planner import run_sbp_planner
from tree_builder import Cell, Node

DEFAULT_MAP_SET = 2
DEFAULT_MAP = 8
OUTPUT_DIR = ROOT / "experiment_results"

# map_set1: skip 1 (no copies), 10/11 (large), 12 (empty fragment).
ALL_MAPS_SET1 = [2, 3, 4, 5, 6, 7, 8]
# map_set2: map 9 has two fragment templates; SBP takes one, so skip it.
ALL_MAPS_SET2 = [1, 2, 3, 4, 5, 6, 7, 8, 10]

MAP_SETS: dict[int, tuple[ModuleType, list[int]]] = {
    1: (map_set1, ALL_MAPS_SET1),
    2: (map_set2, ALL_MAPS_SET2),
}


def load_map(
    map_module: ModuleType,
    map_number: int,
    map_set: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    if map_set == 2 and map_number == 9:
        raise ValueError(
            "map_set2 map 9 has two fragment types (9_a / 9_b); "
            "this experiment supports single-fragment maps only."
        )

    map_data = getattr(map_module, f"map_{map_number}").copy()
    fragment = getattr(map_module, f"fragment_{map_number}").copy()
    corrupted_name = f"fragment_{map_number}_corrupted"
    if not hasattr(map_module, corrupted_name):
        raise ValueError(
            f"Missing {corrupted_name} in map_set{map_set}.py. "
            "Add a corrupted fragment before running this experiment."
        )
    if not hasattr(map_module, f"copies_{map_number}"):
        raise ValueError(f"Missing copies_{map_number} in map_set{map_set}.py")

    corrupted_fragment = getattr(map_module, corrupted_name).copy()
    copies = deepcopy(getattr(map_module, f"copies_{map_number}"))
    if len(copies) < 2:
        raise ValueError(
            f"map_set{map_set} map {map_number} has {len(copies)} copies; need at least 2."
        )
    return map_data, fragment, corrupted_fragment, copies


def find_start(map_data: np.ndarray) -> tuple[int, int]:
    matches = np.argwhere(map_data == Cell.AGENT.value)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one agent cell, found {len(matches)}")
    row, column = matches[0]
    return int(row), int(column)


def substitute_corrupted_copy(
    map_data: np.ndarray,
    corrupted_fragment: np.ndarray,
    copy: dict,
) -> None:
    """Paste the corrupted fragment into the copy footprint, preserving the agent cell."""
    transformed = transform_fragment(corrupted_fragment, copy)
    row0, column0 = copy["top left"]
    height, width = transformed.shape
    agent_pos = find_start(map_data)

    map_data[row0 : row0 + height, column0 : column0 + width] = transformed

    agent_row, agent_column = agent_pos
    if row0 <= agent_row < row0 + height and column0 <= agent_column < column0 + width:
        map_data[agent_row, agent_column] = Cell.AGENT.value


def reset_rollout_counters() -> None:
    globals.escape_rollout_count = 0
    globals.bridge_rollout_count = 0
    globals.fragment_rollout_count = 0
    globals.simul_rollout_count = 0


def observations_along_path(map_data: np.ndarray, path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Accumulate quadrant observations from every position on a path."""
    generator = Generator(map_data)
    observed: set[tuple[int, int]] = set()
    for position in path:
        observed |= generator.get_observation(position)
    return observed


def run_naive_remainder(
    map_data: np.ndarray,
    start_pos: tuple[int, int],
    prior_observed: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """Explore remaining unobserved open cells with naive FragmentPOMCP.

    Starts from ``start_pos`` with ``prior_observed`` already known, so planning
    focuses on the leftover (e.g. corrupted / non-fragment) regions.
    """
    generator = Generator(map_data)
    prior_observed = set(prior_observed) | generator.get_observation(start_pos)
    belief = {room for room in generator.rooms if room not in prior_observed}

    if not belief:
        return [start_pos], 0

    rollouts_before = globals.fragment_rollout_count
    pomcp = FragmentPOMCP(generator, depth=len(generator.rooms))
    root = Node(start_pos, prior_observed, belief, parent_id="", parent_a=0)

    path = [start_pos]
    max_steps = len(generator.rooms) * 10
    for _ in range(max_steps):
        if len(root.belief) == 0:
            break

        best_action = pomcp.search(root)
        if best_action is None or len(root.children) == 0:
            break

        root = root.children[best_action]
        path.append(root.agent_pos)
        generator.update_observed(root.agent_pos)

    return path, globals.fragment_rollout_count - rollouts_before


def run_sbp_then_naive(
    map_data: np.ndarray,
    fragment: np.ndarray,
    copies: list[dict],
) -> dict[str, int]:
    """Run SBP on remaining copies, then naive POMCP on the unplanned remainder."""
    reset_rollout_counters()
    working_map = map_data.copy()
    agent_path, *_ = run_sbp_planner(working_map, fragment.copy(), deepcopy(copies))

    sbp_rollouts = (
        globals.bridge_rollout_count
        + globals.fragment_rollout_count
        + globals.escape_rollout_count
    )

    if not agent_path:
        raise RuntimeError("SBP returned an empty agent path")

    prior_observed = observations_along_path(map_data, agent_path)
    _, naive_rollouts = run_naive_remainder(map_data, agent_path[-1], prior_observed)

    return {
        "sbp_rollouts": sbp_rollouts,
        "naive_rollouts": naive_rollouts,
        "total_rollouts": sbp_rollouts + naive_rollouts,
    }


def all_cells_coverage(map_data: np.ndarray, fragment: np.ndarray, copies: list[dict]) -> float:
    *_, all_cells_percentage, _, _ = calculate_coverage(map_data, [(fragment, copies)])
    return float(all_cells_percentage)


def plot_results(
    results: list[dict],
    map_set: int,
    map_number: int,
    output_path: Path,
) -> None:
    labels = [f"{point['coverage_percentage']:.2f}%" for point in results]
    rollouts = [point["total_rollouts"] for point in results]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(results)), rollouts, color="#3b6d8c")
    axis.set_xticks(range(len(results)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_xlabel("Fragment coverage (all cells)")
    axis.set_ylabel("Total rollouts (SBP + naive cleanup)")
    axis.set_title(f"map_set{map_set} map {map_number}: rollouts vs fragment coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def append_iteration_log(log_path: Path, iteration: int, point: dict, note: str = "") -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"iteration={iteration}\t"
            f"remaining_copies={point['remaining_copies']}\t"
            f"coverage_percentage={point['coverage_percentage']:.2f}\t"
            f"sbp_rollouts={point['sbp_rollouts']}\t"
            f"naive_rollouts={point['naive_rollouts']}\t"
            f"total_rollouts={point['total_rollouts']}"
        )
        if note:
            log_file.write(f"\t{note}")
        log_file.write("\n")
        log_file.flush()


def run_experiment(
    map_module: ModuleType,
    map_set: int,
    map_number: int,
    seed: int,
    log_path: Path,
) -> list[dict]:
    rng = random.Random(seed)
    map_data, fragment, corrupted_fragment, copies = load_map(map_module, map_number, map_set)
    results: list[dict] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"map_set={map_set}\n")
        log_file.write(f"map={map_number}\n")
        log_file.write(f"seed={seed}\n")
        log_file.write(f"starting_copies={len(copies)}\n")
        log_file.write("corruption=substitute_fragment_N_corrupted\n")
        log_file.write("planner=SBP then naive POMCP cleanup\n")
        log_file.write("---\n")

    print(f"Running experiment on map_set{map_set} map {map_number} with seed {seed}")
    print(f"Starting with {len(copies)} fragment copies")
    print(f"Logging each iteration to {log_path}")

    iteration = 0
    while len(copies) >= 2:
        iteration += 1
        coverage = all_cells_coverage(map_data, fragment, copies)
        print(
            f"\n[{len(copies)} copies] coverage = {coverage:.2f}% — "
            "running SBP + naive cleanup..."
        )
        rollout_stats = run_sbp_then_naive(map_data, fragment, copies)
        point = {
            "remaining_copies": len(copies),
            "coverage_percentage": round(coverage, 2),
            **rollout_stats,
        }
        results.append(point)
        print(
            f"[{len(copies)} copies] "
            f"sbp={rollout_stats['sbp_rollouts']} "
            f"naive={rollout_stats['naive_rollouts']} "
            f"total={rollout_stats['total_rollouts']}"
        )
        append_iteration_log(log_path, iteration, point)

        if len(copies) == 2:
            break

        index = rng.randrange(len(copies))
        chosen = copies.pop(index)
        note = f"substituted_corrupted_copy_top_left={chosen['top left']}"
        print(
            f"Substituting corrupted fragment at {chosen['top left']} "
            f"({len(copies)} copies remain)"
        )
        substitute_corrupted_copy(map_data, corrupted_fragment, chosen)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{note}\tremaining_copies_after={len(copies)}\n")
            log_file.flush()

    return results


def run_one_map(map_set: int, map_number: int, seed: int) -> Path:
    """Run the full experiment for one map; write txt/json/png under experiment_results/."""
    map_module, _ = MAP_SETS[map_set]
    OUTPUT_DIR.mkdir(exist_ok=True)
    prefix = f"set{map_set}_map_{map_number}"
    txt_path = OUTPUT_DIR / f"{prefix}_coverage_rollouts.txt"
    json_path = OUTPUT_DIR / f"{prefix}_coverage_rollouts.json"
    plot_path = OUTPUT_DIR / f"{prefix}_coverage_rollouts.png"

    results = run_experiment(map_module, map_set, map_number, seed, txt_path)

    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "map_set": map_set,
                "map": map_number,
                "seed": seed,
                "points": results,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")

    plot_results(results, map_set, map_number, plot_path)
    print(f"\nWrote {txt_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    return txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-set",
        type=int,
        choices=sorted(MAP_SETS),
        default=DEFAULT_MAP_SET,
        help="1 = small smoke-test maps (map_set1), 2 = full suite (map_set2). Default: 2",
    )
    parser.add_argument(
        "--map",
        type=int,
        default=None,
        help="map number within the chosen map set",
    )
    parser.add_argument(
        "--all-maps",
        action="store_true",
        help="run every supported map in the chosen map set",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for which copy to remove")
    args = parser.parse_args()

    if args.all_maps and args.map is not None:
        parser.error("use either --map or --all-maps, not both")

    _, supported = MAP_SETS[args.map_set]
    if args.all_maps:
        map_numbers = supported
    elif args.map is not None:
        map_numbers = [args.map]
    else:
        map_numbers = [DEFAULT_MAP if args.map_set == 2 else supported[0]]

    print(
        f"map_set={args.map_set} maps={map_numbers} seed={args.seed} "
        f"(supported for --all-maps: {supported})"
    )
    for map_number in map_numbers:
        print("\n" + "=" * 60)
        print(f"map_set{args.map_set} MAP {map_number}")
        print("=" * 60)
        run_one_map(args.map_set, map_number, args.seed)

    print("\nAll requested map experiments finished.")


if __name__ == "__main__":
    main()
