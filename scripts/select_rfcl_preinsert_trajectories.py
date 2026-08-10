#!/usr/bin/env python3
"""Select long, seed-balanced RFCL successes starting in pre-insert motion."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREINSERT_TAG = "move_usb_to_play_pre_insert"


@dataclass(frozen=True)
class Candidate:
    path: str
    episode: int
    demo_index: int
    seed: int
    state_index: int
    insertion_start_index: int
    length: int
    initial_x_mm: float
    initial_y_mm: float
    initial_z_mm: float
    relative_path_length_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select balanced long RFCL trajectories from pre-insert motion."
    )
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=200)
    return parser.parse_args()


def load_stage_metadata(
    manifest_path: Path,
) -> tuple[dict[int, int], dict[tuple[int, int], str], dict[int, int]]:
    manifest = json.loads(manifest_path.read_text())
    seeds = {}
    tags = {}
    insertion_starts = {}
    for demo_index, demo in enumerate(manifest["demos"]):
        seeds[demo_index] = int(demo["seed"])
        rows = [manifest["snapshots"][snapshot_id] for snapshot_id in demo["snapshot_ids"]]
        for state_index, row in enumerate(rows):
            tags[(demo_index, state_index)] = str(row["atom_tag"])
        insertion_starts[demo_index] = min(
            state_index
            for state_index, row in enumerate(rows)
            if row["atom_tag"] == "insert_USB_into_slot"
        )
    return seeds, tags, insertion_starts


def load_candidate(
    path: Path,
    *,
    seeds: dict[int, int],
    tags: dict[tuple[int, int], str],
    insertion_starts: dict[int, int],
) -> Candidate | None:
    with np.load(path, allow_pickle=False) as data:
        demo_index = int(data["demo_index"])
        state_index = int(data["state_index"])
        if tags[(demo_index, state_index)] != PREINSERT_TAG:
            return None
        states = np.asarray(data["states"], dtype=np.float64)
        next_states = np.asarray(data["next_states"], dtype=np.float64)
        if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] < 46:
            raise ValueError(f"Invalid states in {path}: {states.shape}")
        if next_states.shape != states.shape:
            raise ValueError(f"Transition arrays disagree in {path}")
        episode = int(data["episode"])
    relative_path = np.concatenate((states[:1, 39:42], next_states[:, 39:42]), axis=0)
    relative_path_mm = relative_path * 1000.0
    path_length = float(
        np.linalg.norm(np.diff(relative_path_mm, axis=0), axis=1).sum()
    )
    return Candidate(
        path=str(path),
        episode=episode,
        demo_index=demo_index,
        seed=seeds[demo_index],
        state_index=state_index,
        insertion_start_index=insertion_starts[demo_index],
        length=int(states.shape[0]),
        initial_x_mm=float(relative_path_mm[0, 0]),
        initial_y_mm=float(relative_path_mm[0, 1]),
        initial_z_mm=float(relative_path_mm[0, 2]),
        relative_path_length_mm=path_length,
    )


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if count >= len(values):
        return list(values)
    raw_indices = np.linspace(0, len(values) - 1, count)
    indices = []
    for raw_index in raw_indices:
        index = int(round(float(raw_index)))
        if index not in indices:
            indices.append(index)
    if len(indices) < count:
        indices.extend(index for index in range(len(values)) if index not in indices)
    return [values[index] for index in indices[:count]]


def select_group(candidates: list[Candidate], quota: int) -> list[Candidate]:
    if len(candidates) < quota:
        raise ValueError(f"Only {len(candidates)} candidates available for quota {quota}")
    by_state: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_state.setdefault(candidate.state_index, []).append(candidate)
    representatives = {
        state_index: max(rows, key=lambda row: (row.length, row.relative_path_length_mm))
        for state_index, rows in by_state.items()
    }
    states = evenly_spaced(sorted(representatives), min(quota, len(representatives)))
    selected = [representatives[state_index] for state_index in states]
    selected_paths = {candidate.path for candidate in selected}
    remaining = sorted(
        (candidate for candidate in candidates if candidate.path not in selected_paths),
        key=lambda row: (row.length, row.relative_path_length_mm),
        reverse=True,
    )
    selected.extend(remaining[: quota - len(selected)])
    return sorted(selected, key=lambda row: (row.seed, row.state_index, -row.length))


def write_csv(path: Path, candidates: list[Candidate]) -> None:
    rows = [asdict(candidate) for candidate in candidates]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_selection(output: Path, selected: list[Candidate]) -> None:
    seeds = sorted({candidate.seed for candidate in selected})
    colors = dict(zip(seeds, plt.cm.tab10(np.linspace(0.0, 1.0, len(seeds)))))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for seed in seeds:
        rows = [candidate for candidate in selected if candidate.seed == seed]
        axes[0, 0].scatter(
            [row.state_index for row in rows],
            [row.length for row in rows],
            color=colors[seed],
            label=str(seed),
            alpha=0.8,
        )
        axes[0, 1].scatter(
            [row.initial_x_mm for row in rows],
            [row.initial_y_mm for row in rows],
            color=colors[seed],
            label=str(seed),
            alpha=0.8,
        )
        for row in rows:
            with np.load(row.path, allow_pickle=False) as data:
                states = np.asarray(data["states"], dtype=np.float64)
                next_states = np.asarray(data["next_states"], dtype=np.float64)
            path = np.concatenate((states[:1, 39:42], next_states[:, 39:42]), axis=0) * 1000.0
            axes[1, 0].plot(path[:, 0], path[:, 2], color=colors[seed], alpha=0.18)
            axes[1, 1].plot(path[:, 1], path[:, 2], color=colors[seed], alpha=0.18)
    axes[0, 0].set(
        title="Balanced long pre-insert successes",
        xlabel="Start state index",
        ylabel="RL trajectory length",
    )
    axes[0, 0].legend(title="Seed", ncol=2, fontsize=8)
    axes[0, 1].set(
        title="Initial USB-in-slot XY",
        xlabel="Relative X (mm)",
        ylabel="Relative Y (mm)",
    )
    axes[0, 1].legend(title="Seed", ncol=2, fontsize=8)
    axes[1, 0].set(
        title="Successful pre-insert + insertion paths: X-Z",
        xlabel="Relative X (mm)",
        ylabel="Relative Z (mm)",
    )
    axes[1, 1].set(
        title="Successful pre-insert + insertion paths: Y-Z",
        xlabel="Relative Y (mm)",
        ylabel="Relative Z (mm)",
    )
    fig.tight_layout()
    fig.savefig(output / "preinsert_longest_balanced_200.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, selected: list[Candidate], total_candidates: int) -> None:
    lines = [
        "# Balanced Long Pre-Insert RFCL Trajectories",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        f"- Eligible successful pre-insert trajectories: {total_candidates}",
        f"- Selected trajectories: {len(selected)}",
        f"- Stage requirement: start tag is exactly `{PREINSERT_TAG}`",
        "- Selection: equal Seed quota, evenly covered unique starts, then longest remaining trajectories",
        "",
        "| Seed | Count | Unique starts | Min length | Mean length | Max length | State range |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted({candidate.seed for candidate in selected}):
        rows = [candidate for candidate in selected if candidate.seed == seed]
        lengths = np.asarray([row.length for row in rows])
        states = [row.state_index for row in rows]
        lines.append(
            f"| {seed} | {len(rows)} | {len(set(states))} | {lengths.min()} | "
            f"{lengths.mean():.1f} | {lengths.max()} | {min(states)}–{max(states)} |"
        )
    all_lengths = np.asarray([candidate.length for candidate in selected])
    lines.extend(
        (
            "",
            f"Overall length: min `{all_lengths.min()}`, mean `{all_lengths.mean():.1f}`, max `{all_lengths.max()}` RL steps.",
            "",
            "Every selected rollout starts before the insertion atom and ends in task success.",
            "The files contain privileged states and actions, not RGB or tactile frames.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.total <= 0:
        raise ValueError("--total must be positive")
    seeds, tags, insertion_starts = load_stage_metadata(args.manifest)
    demo_indices = sorted(seeds)
    if args.total % len(demo_indices) != 0:
        raise ValueError("--total must be divisible by the number of demos")
    quota = args.total // len(demo_indices)
    candidates = []
    skipped = []
    for path in sorted(args.trajectory_dir.glob("*.npz")):
        try:
            candidate = load_candidate(
                path,
                seeds=seeds,
                tags=tags,
                insertion_starts=insertion_starts,
            )
        except (OSError, ValueError, KeyError) as error:
            skipped.append({"path": str(path), "error": repr(error)})
            continue
        if candidate is not None:
            candidates.append(candidate)
    selected = []
    for demo_index in demo_indices:
        group = [candidate for candidate in candidates if candidate.demo_index == demo_index]
        selected.extend(select_group(group, quota))
    if len(selected) != args.total:
        raise RuntimeError(f"Selected {len(selected)} trajectories, expected {args.total}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "selected_trajectories.csv", selected)
    (args.output / "selected_trajectories.txt").write_text(
        "\n".join(candidate.path for candidate in selected) + "\n"
    )
    plot_selection(args.output, selected)
    write_report(args.output / "REPORT.md", selected, len(candidates))
    summary = {
        "eligible": len(candidates),
        "selected": len(selected),
        "quota_per_seed": quota,
        "length_min": min(candidate.length for candidate in selected),
        "length_mean": float(np.mean([candidate.length for candidate in selected])),
        "length_max": max(candidate.length for candidate in selected),
        "unique_start_pairs": len(
            {(candidate.demo_index, candidate.state_index) for candidate in selected}
        ),
        "skipped": skipped,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
