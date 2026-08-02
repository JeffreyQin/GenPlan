"""
Fragment-coverage vs SBP-rollout experiment for a single map_set2 map.

While at least 2 fragment copies remain:
  1. measure all-cells fragment coverage
  2. run SBP over remaining fragment copies, then naive POMCP on the
     leftover unobserved map; record total rollouts
  3. if more than 2 copies remain, replace one random copy's footprint
     with a hand-authored corrupted fragment and drop it from copies
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MAPS_DIR = ROOT / "maps"
if str(MAPS_DIR) not in sys.path:
    sys.path.insert(0, str(MAPS_DIR))

import globals
import map_set2
from fragment_coverage import calculate_coverage, transform_fragment
from fragment_search import FragmentPOMCP
from generator import Generator
from structure_based_planner import run_sbp_planner
from tree_builder import Cell, Node

# Map 8: ~89% all-cell / 100% open-cell fragment coverage, 6 copies.
DEFAULT_MAP = 8
OUTPUT_DIR = ROOT / "experiment_results"


def load_map(map_number: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    if map_number == 9:
        raise ValueError("Map 9 has two fragment types; pick a single-fragment map for this experiment.")

    map_data = getattr(map_set2, f"map_{map_number}").copy()
    fragment = getattr(map_set2, f"fragment_{map_number}").copy()
    corrupted_name = f"fragment_{map_number}_corrupted"
    if not hasattr(map_set2, corrupted_name):
        raise ValueError(
            f"Missing {corrupted_name} in map_set2.py. "
            "Add a hand-authored corrupted fragment before running this experiment."
        )
    corrupted_fragment = getattr(map_set2, corrupted_name).copy()
    copies = deepcopy(getattr(map_set2, f"copies_{map_number}"))
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


def plot_results(results: list[dict], map_number: int, output_path: Path) -> None:
    labels = [f"{point['coverage_percentage']:.2f}%" for point in results]
    rollouts = [point["total_rollouts"] for point in results]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(results)), rollouts, color="#3b6d8c")
    axis.set_xticks(range(len(results)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_xlabel("Fragment coverage (all cells)")
    axis.set_ylabel("Total rollouts (SBP + naive cleanup)")
    axis.set_title(f"Map {map_number}: rollouts vs fragment coverage")
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


def run_experiment(map_number: int, seed: int, log_path: Path) -> list[dict]:
    rng = random.Random(seed)
    map_data, fragment, corrupted_fragment, copies = load_map(map_number)
    results: list[dict] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"map={map_number}\n")
        log_file.write(f"seed={seed}\n")
        log_file.write(f"starting_copies={len(copies)}\n")
        log_file.write("corruption=substitute_fragment_N_corrupted\n")
        log_file.write("planner=SBP then naive POMCP cleanup\n")
        log_file.write("---\n")

    print(f"Running experiment on map {map_number} with seed {seed}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=int, default=DEFAULT_MAP, help="map_set2 map number")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for which copy to remove")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    txt_path = OUTPUT_DIR / f"map_{args.map}_coverage_rollouts.txt"
    results = run_experiment(args.map, args.seed, txt_path)

    json_path = OUTPUT_DIR / f"map_{args.map}_coverage_rollouts.json"
    plot_path = OUTPUT_DIR / f"map_{args.map}_coverage_rollouts.png"

    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "map": args.map,
                "seed": args.seed,
                "points": results,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")

    plot_results(results, args.map, plot_path)
    print(f"\nWrote {txt_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
