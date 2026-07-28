#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pinocchio as pin

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


SUCCESS_COLOR = "#2dd4bf"
SUCCESS_MEAN_COLOR = "#00ffd5"
FAILURE_COLOR = "#fb7185"
FAILURE_MEAN_COLOR = "#ff365d"
TARGET_COLOR = "#fbbf24"
OBJECT_COLOR = "#60a5fa"
BACKGROUND_COLOR = "#07111f"
PANEL_COLOR = "#0b1728"
GRID_COLOR = "#334155"
TEXT_COLOR = "#e2e8f0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize successful and failed RL end-effector trajectories."
    )
    parser.add_argument("evaluation_json", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--object-mesh", type=Path)
    parser.add_argument("--frame", default="panda_hand")
    parser.add_argument("--gripper-center-offset", type=float, default=0.131)
    parser.add_argument("--resample-points", type=int, default=120)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    triangles = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:]:
                vertex_index = int(token.split("/")[0])
                if vertex_index < 0:
                    vertex_index = len(vertices) + vertex_index
                else:
                    vertex_index -= 1
                indices.append(vertex_index)
            for triangle_index in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[triangle_index], indices[triangle_index + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def quaternion_wxyz_to_rotation(quaternion: list[float]) -> np.ndarray:
    quaternion_array = np.asarray(quaternion, dtype=np.float64)
    quaternion_array /= np.linalg.norm(quaternion_array)
    scalar, x_value, y_value, z_value = quaternion_array
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y_value**2 + z_value**2),
                2.0 * (x_value * y_value - z_value * scalar),
                2.0 * (x_value * z_value + y_value * scalar),
            ],
            [
                2.0 * (x_value * y_value + z_value * scalar),
                1.0 - 2.0 * (x_value**2 + z_value**2),
                2.0 * (y_value * z_value - x_value * scalar),
            ],
            [
                2.0 * (x_value * z_value - y_value * scalar),
                2.0 * (y_value * z_value + x_value * scalar),
                1.0 - 2.0 * (x_value**2 + y_value**2),
            ],
        ]
    )


def trace_to_configuration(model: pin.Model, qpos: list[float]) -> np.ndarray:
    configuration = pin.neutral(model)
    arm_values = np.asarray(qpos[:7], dtype=np.float64)
    configuration[: len(arm_values)] = arm_values
    if model.nq >= 9 and len(qpos) >= 8:
        configuration[7] = float(qpos[7])
        configuration[8] = float(qpos[7])
    return configuration


def compute_gripper_center_trajectory(
    model: pin.Model,
    frame_id: int,
    trace: list[dict],
    gripper_center_offset: float,
) -> np.ndarray:
    model_data = model.createData()
    local_offset = np.asarray([0.0, 0.0, gripper_center_offset])
    positions = []
    for trace_step in trace:
        configuration = trace_to_configuration(model, trace_step["qpos"])
        pin.forwardKinematics(model, model_data, configuration)
        pin.updateFramePlacements(model, model_data)
        placement = model_data.oMf[frame_id]
        center_position = placement.translation + placement.rotation @ local_offset
        positions.append(np.asarray(center_position, dtype=np.float64).copy())
    return np.asarray(positions)


def resample_trajectory(positions: np.ndarray, points: int) -> np.ndarray:
    source_progress = np.linspace(0.0, 1.0, len(positions))
    target_progress = np.linspace(0.0, 1.0, points)
    return np.column_stack(
        [
            np.interp(target_progress, source_progress, positions[:, axis_index])
            for axis_index in range(3)
        ]
    )


def pairwise_trajectory_diversity(
    trajectories: list[dict],
    resample_points: int,
) -> float:
    if len(trajectories) < 2:
        return 0.0
    resampled = [
        resample_trajectory(item["positions"], resample_points)
        for item in trajectories
    ]
    distances = []
    for first_index, second_index in combinations(range(len(resampled)), 2):
        pointwise_distance = np.linalg.norm(
            resampled[first_index] - resampled[second_index], axis=1
        )
        distances.append(float(np.sqrt(np.mean(pointwise_distance**2))))
    return float(np.mean(distances))


def endpoint_dispersion(trajectories: list[dict]) -> float:
    if not trajectories:
        return 0.0
    endpoints = np.asarray([item["positions"][-1] for item in trajectories])
    mean_endpoint = endpoints.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((endpoints - mean_endpoint) ** 2, axis=1))))


def group_summary(trajectories: list[dict], resample_points: int) -> dict:
    path_lengths = [item["path_length_m"] for item in trajectories]
    net_displacements = [item["net_displacement_m"] for item in trajectories]
    return {
        "episodes": len(trajectories),
        "mean_path_length_m": float(np.mean(path_lengths)) if path_lengths else 0.0,
        "std_path_length_m": float(np.std(path_lengths)) if path_lengths else 0.0,
        "mean_net_displacement_m": (
            float(np.mean(net_displacements)) if net_displacements else 0.0
        ),
        "pairwise_trajectory_diversity_m": pairwise_trajectory_diversity(
            trajectories, resample_points
        ),
        "endpoint_dispersion_m": endpoint_dispersion(trajectories),
    }


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique_points = sorted(set(map(tuple, points.tolist())))
    if len(unique_points) <= 1:
        return np.asarray(unique_points)

    def cross(origin: tuple, first: tuple, second: tuple) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def add_object_meshes_3d(
    axis: plt.Axes,
    trajectories: list[dict],
    object_triangles: np.ndarray,
) -> None:
    if not len(object_triangles):
        return
    available = [
        item for item in trajectories if item["object_world_vertices"] is not None
    ]
    if not available:
        return
    centers = np.asarray(
        [item["object_world_vertices"].mean(axis=0) for item in available]
    )
    mean_center = centers.mean(axis=0)
    representative_index = int(
        np.argmin(np.linalg.norm(centers - mean_center, axis=1))
    )
    representative = available[representative_index]
    triangle_vertices = representative["object_world_vertices"][object_triangles]
    collection = Poly3DCollection(
        triangle_vertices,
        facecolor=OBJECT_COLOR,
        edgecolor="#bfdbfe",
        linewidth=0.25,
        alpha=0.34,
        label=f"Target scale / 50×25×40 mm / seed {representative['seed']}",
    )
    axis.add_collection3d(collection)


def add_object_footprints_2d(axis: plt.Axes, trajectories: list[dict]) -> None:
    label_added = False
    for item in trajectories:
        world_vertices = item["object_world_vertices"]
        if world_vertices is None:
            continue
        footprint = convex_hull_2d(world_vertices[:, :2])
        patch = Polygon(
            footprint,
            closed=True,
            facecolor=OBJECT_COLOR,
            edgecolor="#93c5fd",
            linewidth=0.5,
            alpha=0.09,
            label="Target footprint" if not label_added else None,
        )
        axis.add_patch(patch)
        label_added = True


def extract_trajectories(
    evaluation: dict,
    metadata: dict,
    model: pin.Model,
    frame_id: int,
    gripper_center_offset: float,
    object_vertices: np.ndarray,
) -> list[dict]:
    trajectories = []
    for result in evaluation["results"]:
        seed = int(result["seed"])
        positions = compute_gripper_center_trajectory(
            model,
            frame_id,
            result["trace"],
            gripper_center_offset,
        )
        differences = np.diff(positions, axis=0)
        path_length = float(np.linalg.norm(differences, axis=1).sum())
        net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        episode_metadata = metadata.get(str(seed), {})
        success_diagnostics = episode_metadata.get("success_diagnostics", {})
        target_pose = success_diagnostics.get("target_initial_pose")
        target_position = (
            np.asarray(target_pose[:3], dtype=np.float64)
            if target_pose is not None
            else None
        )
        object_world_vertices = None
        if target_pose is not None and len(object_vertices):
            object_rotation = quaternion_wxyz_to_rotation(target_pose[3:7])
            object_world_vertices = (
                object_vertices @ object_rotation.T + target_position
            )
        trajectories.append(
            {
                "seed": seed,
                "success": bool(result["success"]),
                "reward": float(result["reward"]),
                "actions": int(result["actions"]),
                "positions": positions,
                "target_position": target_position,
                "object_world_vertices": object_world_vertices,
                "lifted_height": float(result["metrics"]["lifted_height"]),
                "max_lifted_height": float(result["metrics"]["max_lifted_height"]),
                "path_length_m": path_length,
                "net_displacement_m": net_displacement,
            }
        )
    return trajectories


def configure_3d_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL_COLOR)
    axis.xaxis.pane.set_facecolor(PANEL_COLOR)
    axis.yaxis.pane.set_facecolor(PANEL_COLOR)
    axis.zaxis.pane.set_facecolor(PANEL_COLOR)
    axis.xaxis.pane.set_edgecolor(GRID_COLOR)
    axis.yaxis.pane.set_edgecolor(GRID_COLOR)
    axis.zaxis.pane.set_edgecolor(GRID_COLOR)
    axis.grid(True, color=GRID_COLOR, alpha=0.28)
    axis.tick_params(colors=TEXT_COLOR, labelsize=8)
    axis.set_xlabel("X / m", color=TEXT_COLOR, labelpad=10)
    axis.set_ylabel("Y / m", color=TEXT_COLOR, labelpad=10)
    axis.set_zlabel("Z / m", color=TEXT_COLOR, labelpad=10)


def configure_2d_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL_COLOR)
    axis.grid(True, color=GRID_COLOR, alpha=0.35, linewidth=0.7)
    axis.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(GRID_COLOR)


def set_3d_limits(axis: plt.Axes, trajectories: list[dict]) -> None:
    position_cloud = np.concatenate(
        [item["positions"] for item in trajectories], axis=0
    )
    target_positions = [
        item["target_position"]
        for item in trajectories
        if item["target_position"] is not None
    ]
    if target_positions:
        position_cloud = np.vstack([position_cloud, np.asarray(target_positions)])
    object_vertices = [
        item["object_world_vertices"]
        for item in trajectories
        if item["object_world_vertices"] is not None
    ]
    if object_vertices:
        position_cloud = np.vstack([position_cloud, *object_vertices])
    minimum = position_cloud.min(axis=0)
    maximum = position_cloud.max(axis=0)
    spans = np.maximum(maximum - minimum, 0.02)
    padding = spans * 0.08
    axis.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    axis.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
    axis.set_zlim(minimum[2] - padding[2], maximum[2] + padding[2])
    axis.set_box_aspect(spans)


def plot_trajectory_group(
    axis: plt.Axes,
    trajectories: list[dict],
    color: str,
    mean_color: str,
    label: str,
    resample_points: int,
    annotate_endpoints: bool,
) -> None:
    for item_index, item in enumerate(trajectories):
        positions = item["positions"]
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            alpha=0.48,
            linewidth=1.35,
            label=label if item_index == 0 else None,
        )
        axis.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            color=color,
            s=12,
            alpha=0.8,
        )
        axis.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            color=color,
            marker="D",
            s=22,
            edgecolor="white",
            linewidth=0.35,
        )
        if annotate_endpoints:
            axis.text(
                positions[-1, 0],
                positions[-1, 1],
                positions[-1, 2],
                str(item["seed"]),
                color=TEXT_COLOR,
                fontsize=6,
                alpha=0.82,
            )
    if trajectories:
        resampled = np.stack(
            [
                resample_trajectory(item["positions"], resample_points)
                for item in trajectories
            ]
        )
        mean_path = resampled.mean(axis=0)
        axis.plot(
            mean_path[:, 0],
            mean_path[:, 1],
            mean_path[:, 2],
            color=mean_color,
            linewidth=4.0,
            alpha=0.95,
            label=f"{label} mean",
        )


def save_3d_figure(
    trajectories: list[dict],
    output_path: Path,
    resample_points: int,
    object_triangles: np.ndarray,
) -> None:
    successful = [item for item in trajectories if item["success"]]
    failed = [item for item in trajectories if not item["success"]]
    figure = plt.figure(figsize=(14, 11), facecolor=BACKGROUND_COLOR)
    axis = figure.add_subplot(111, projection="3d")
    configure_3d_axis(axis)
    plot_trajectory_group(
        axis,
        successful,
        SUCCESS_COLOR,
        SUCCESS_MEAN_COLOR,
        "Success",
        resample_points,
        True,
    )
    add_object_meshes_3d(axis, trajectories, object_triangles)
    plot_trajectory_group(
        axis,
        failed,
        FAILURE_COLOR,
        FAILURE_MEAN_COLOR,
        "Failure",
        resample_points,
        True,
    )
    target_positions = [
        item["target_position"]
        for item in trajectories
        if item["target_position"] is not None
    ]
    if target_positions:
        targets = np.asarray(target_positions)
        axis.scatter(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            color=TARGET_COLOR,
            marker="*",
            s=55,
            alpha=0.75,
            label="Target initial pose",
        )
    set_3d_limits(axis, trajectories)
    axis.view_init(elev=28, azim=-58)
    axis.set_title(
        f"SAC End-Effector Trajectories  |  {len(successful)} success / {len(failed)} failure",
        color=TEXT_COLOR,
        fontsize=16,
        pad=24,
        fontweight="bold",
    )
    legend = axis.legend(
        loc="upper left",
        frameon=True,
        facecolor=PANEL_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )
    legend.get_frame().set_alpha(0.88)
    figure.text(
        0.5,
        0.035,
        "Circle = start   Diamond = end   Star = target initial pose   Thick line = group mean",
        ha="center",
        color="#94a3b8",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def save_dashboard(
    trajectories: list[dict],
    output_path: Path,
    resample_points: int,
    object_triangles: np.ndarray,
) -> None:
    successful = [item for item in trajectories if item["success"]]
    failed = [item for item in trajectories if not item["success"]]
    figure = plt.figure(figsize=(18, 10), facecolor=BACKGROUND_COLOR)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.42, 1.0),
        height_ratios=(1.0, 1.0),
        hspace=0.24,
        wspace=0.16,
    )
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_xy = figure.add_subplot(grid[0, 1])
    axis_z = figure.add_subplot(grid[1, 1])
    configure_3d_axis(axis_3d)
    configure_2d_axis(axis_xy)
    configure_2d_axis(axis_z)
    plot_trajectory_group(
        axis_3d,
        successful,
        SUCCESS_COLOR,
        SUCCESS_MEAN_COLOR,
        "Success",
        resample_points,
        False,
    )
    add_object_meshes_3d(axis_3d, trajectories, object_triangles)
    add_object_footprints_2d(axis_xy, trajectories)
    plot_trajectory_group(
        axis_3d,
        failed,
        FAILURE_COLOR,
        FAILURE_MEAN_COLOR,
        "Failure",
        resample_points,
        False,
    )
    target_positions = [
        item["target_position"]
        for item in trajectories
        if item["target_position"] is not None
    ]
    if target_positions:
        targets = np.asarray(target_positions)
        axis_3d.scatter(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            color=TARGET_COLOR,
            marker="*",
            s=45,
            alpha=0.7,
        )
        axis_xy.scatter(
            targets[:, 0],
            targets[:, 1],
            color=TARGET_COLOR,
            marker="*",
            s=34,
            alpha=0.65,
            label="Target",
        )
    set_3d_limits(axis_3d, trajectories)
    axis_3d.view_init(elev=26, azim=-58)
    axis_3d.set_title("3D gripper-center paths", color=TEXT_COLOR, fontsize=12, pad=18)
    for group, color, label in (
        (successful, SUCCESS_COLOR, "Success"),
        (failed, FAILURE_COLOR, "Failure"),
    ):
        for item_index, item in enumerate(group):
            positions = item["positions"]
            axis_xy.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                alpha=0.5,
                linewidth=1.25,
                label=label if item_index == 0 else None,
            )
            progress = np.linspace(0.0, 100.0, len(positions))
            axis_z.plot(
                progress,
                positions[:, 2],
                color=color,
                alpha=0.5,
                linewidth=1.25,
                label=label if item_index == 0 else None,
            )
    axis_xy.set_title("Top view / XY", color=TEXT_COLOR, fontsize=12)
    axis_xy.set_xlabel("X / m", color=TEXT_COLOR)
    axis_xy.set_ylabel("Y / m", color=TEXT_COLOR)
    axis_xy.set_aspect("equal", adjustable="datalim")
    axis_z.set_title("Vertical motion over rollout", color=TEXT_COLOR, fontsize=12)
    axis_z.set_xlabel("Normalized rollout progress / %", color=TEXT_COLOR)
    axis_z.set_ylabel("Gripper-center Z / m", color=TEXT_COLOR)
    for axis in (axis_xy, axis_z):
        legend = axis.legend(
            loc="best",
            frameon=True,
            facecolor=PANEL_COLOR,
            edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
        )
        legend.get_frame().set_alpha(0.85)
    figure.suptitle(
        f"SFT-Regularized SAC Rollout Geometry  |  {len(successful)}/20 successful",
        color=TEXT_COLOR,
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.savefig(output_path, dpi=210, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def trajectory_rows(trajectories: list[dict]) -> list[dict]:
    rows = []
    for item in trajectories:
        positions = item["positions"]
        target_position = item["target_position"]
        rows.append(
            {
                "seed": item["seed"],
                "success": item["success"],
                "reward": item["reward"],
                "actions": item["actions"],
                "path_length_m": item["path_length_m"],
                "net_displacement_m": item["net_displacement_m"],
                "start_x_m": positions[0, 0],
                "start_y_m": positions[0, 1],
                "start_z_m": positions[0, 2],
                "end_x_m": positions[-1, 0],
                "end_y_m": positions[-1, 1],
                "end_z_m": positions[-1, 2],
                "min_z_m": positions[:, 2].min(),
                "max_z_m": positions[:, 2].max(),
                "lifted_height_m": item["lifted_height"],
                "max_lifted_height_m": item["max_lifted_height"],
                "target_x_m": target_position[0] if target_position is not None else "",
                "target_y_m": target_position[1] if target_position is not None else "",
                "target_z_m": target_position[2] if target_position is not None else "",
            }
        )
    return rows


def save_metrics_csv(trajectories: list[dict], output_path: Path) -> None:
    rows = trajectory_rows(trajectories)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plotly_trajectory_traces(
    trajectories: list[dict],
    object_triangles: np.ndarray,
) -> list[dict]:
    plot_traces = []
    status_seen = {True: False, False: False}
    for item in trajectories:
        success = item["success"]
        positions = item["positions"]
        color = SUCCESS_COLOR if success else FAILURE_COLOR
        status = "Success" if success else "Failure"
        hover_text = [
            (
                f"seed={item['seed']}<br>status={status}<br>step={step_index}"
                f"<br>x={position[0]:.4f} m<br>y={position[1]:.4f} m"
                f"<br>z={position[2]:.4f} m<br>reward={item['reward']:.3f}"
            )
            for step_index, position in enumerate(positions)
        ]
        plot_traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": positions[:, 0].tolist(),
                "y": positions[:, 1].tolist(),
                "z": positions[:, 2].tolist(),
                "text": hover_text,
                "hoverinfo": "text",
                "name": status,
                "legendgroup": status.lower(),
                "showlegend": not status_seen[success],
                "line": {"color": color, "width": 4},
                "opacity": 0.62,
            }
        )
        status_seen[success] = True
        plot_traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": [float(positions[-1, 0])],
                "y": [float(positions[-1, 1])],
                "z": [float(positions[-1, 2])],
                "text": [
                    f"seed={item['seed']} end<br>status={status}<br>reward={item['reward']:.3f}"
                ],
                "hoverinfo": "text",
                "name": f"seed {item['seed']} end",
                "legendgroup": status.lower(),
                "showlegend": False,
                "marker": {
                    "color": color,
                    "size": 4.5,
                    "symbol": "diamond",
                    "line": {"color": "#ffffff", "width": 1},
                },
            }
        )
    target_items = [
        item for item in trajectories if item["target_position"] is not None
    ]
    if target_items:
        plot_traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": [float(item["target_position"][0]) for item in target_items],
                "y": [float(item["target_position"][1]) for item in target_items],
                "z": [float(item["target_position"][2]) for item in target_items],
                "text": [f"seed={item['seed']} target" for item in target_items],
                "hoverinfo": "text",
                "name": "Target initial pose",
                "marker": {"color": TARGET_COLOR, "size": 5.5, "symbol": "cross"},
            }
        )
    if len(object_triangles):
        available = [
            item
            for item in trajectories
            if item["object_world_vertices"] is not None
        ]
        if available:
            centers = np.asarray(
                [item["object_world_vertices"].mean(axis=0) for item in available]
            )
            mean_center = centers.mean(axis=0)
            representative_index = int(
                np.argmin(np.linalg.norm(centers - mean_center, axis=1))
            )
            item = available[representative_index]
            world_vertices = item["object_world_vertices"]
            plot_traces.append(
                {
                    "type": "mesh3d",
                    "x": world_vertices[:, 0].tolist(),
                    "y": world_vertices[:, 1].tolist(),
                    "z": world_vertices[:, 2].tolist(),
                    "i": object_triangles[:, 0].tolist(),
                    "j": object_triangles[:, 1].tolist(),
                    "k": object_triangles[:, 2].tolist(),
                    "color": OBJECT_COLOR,
                    "opacity": 0.4,
                    "flatshading": True,
                    "name": "Target object / 50×25×40 mm",
                    "legendgroup": "target-object",
                    "showlegend": True,
                    "text": f"seed={item['seed']} scale reference<br>50 × 25 × 40 mm",
                    "hoverinfo": "text",
                }
            )
    return plot_traces


def save_interactive_html(
    trajectories: list[dict],
    summary: dict,
    output_path: Path,
    object_triangles: np.ndarray,
    object_size_mm: np.ndarray,
) -> None:
    plot_traces = plotly_trajectory_traces(trajectories, object_triangles)
    metric_rows = trajectory_rows(trajectories)
    table_rows = "".join(
        (
            f"<tr class=\"{'success' if row['success'] else 'failure'}\">"
            f"<td>{row['seed']}</td>"
            f"<td>{'SUCCESS' if row['success'] else 'FAILURE'}</td>"
            f"<td>{row['reward']:.3f}</td>"
            f"<td>{row['path_length_m']:.3f}</td>"
            f"<td>{row['max_z_m']:.3f}</td>"
            f"<td>{row['lifted_height_m']:.3f}</td></tr>"
        )
        for row in metric_rows
    )
    html = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
  <title>SAC End-Effector Trajectories</title>
  <script src=\"plotly-2.35.2.min.js\" onerror=\"this.onerror=null;this.src='https://cdn.plot.ly/plotly-2.35.2.min.js';\"></script>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0b1728; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --ok:{SUCCESS_COLOR}; --bad:{FAILURE_COLOR}; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#11243d 0,var(--bg) 48%); color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; }}
    main {{ width:min(1500px,96vw); margin:24px auto 44px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(24px,3vw,42px); letter-spacing:-.03em; }}
    .subtitle {{ color:var(--muted); margin-bottom:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-bottom:14px; }}
    .card,.panel {{ background:rgba(11,23,40,.9); border:1px solid var(--line); border-radius:16px; box-shadow:0 18px 55px rgba(0,0,0,.22); }}
    .card {{ padding:16px 18px; }}
    .card span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .card strong {{ display:block; font-size:28px; margin-top:5px; }}
    .panel {{ overflow:hidden; }}
    #trajectory-plot {{ width:100%; height:min(76vh,900px); }}
    .note {{ padding:12px 16px; color:var(--muted); border-top:1px solid var(--line); }}
    .table-wrap {{ margin-top:14px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ padding:10px 12px; text-align:right; border-bottom:1px solid rgba(51,65,85,.65); font-variant-numeric:tabular-nums; }}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    th {{ position:sticky; top:0; background:#0f1d31; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    tr.success td:nth-child(2) {{ color:var(--ok); }}
    tr.failure td:nth-child(2) {{ color:var(--bad); }}
    @media (max-width:800px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  </style>
</head>
<body>
<main>
  <h1>SAC 末端轨迹对比</h1>
  <div class=\"subtitle\">20 个确定性 held-out rollout · 成功与失败分色 · 鼠标拖动旋转，滚轮缩放，悬停查看 seed</div>
  <section class=\"cards\">
    <div class=\"card\"><span>Success</span><strong style=\"color:var(--ok)\">{summary['success']['episodes']} / 20</strong></div>
    <div class=\"card\"><span>Success diversity</span><strong>{summary['success']['pairwise_trajectory_diversity_m'] * 1000:.1f} mm</strong></div>
    <div class=\"card\"><span>Failure diversity</span><strong>{summary['failure']['pairwise_trajectory_diversity_m'] * 1000:.1f} mm</strong></div>
    <div class=\"card\"><span>Frame</span><strong>Gripper center</strong></div>
    <div class=\"card\"><span>Target size</span><strong>{object_size_mm[0]:.0f}×{object_size_mm[1]:.0f}×{object_size_mm[2]:.0f} mm</strong></div>
  </section>
  <section class=\"panel\">
    <div id=\"trajectory-plot\"></div>
    <div class=\"note\">青色 = 成功，粉红 = 失败，半透明蓝色实体 = 50×25×40 mm 目标物体，黄色十字 = 物体原点。点击图例可整组隐藏。</div>
  </section>
  <section class=\"panel table-wrap\">
    <table>
      <thead><tr><th>Seed</th><th>Result</th><th>Reward</th><th>Path / m</th><th>Max Z / m</th><th>Object lift / m</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>
</main>
<script>
const trajectoryData = {json.dumps(plot_traces, ensure_ascii=False)};
const layout = {{
  paper_bgcolor:'#0b1728', plot_bgcolor:'#0b1728', font:{{color:'#e2e8f0'}},
  margin:{{l:0,r:0,t:24,b:0}}, showlegend:true,
  legend:{{x:0.02,y:0.98,bgcolor:'rgba(7,17,31,.72)',bordercolor:'#334155',borderwidth:1,groupclick:'togglegroup'}},
  scene:{{
    aspectmode:'data', bgcolor:'#0b1728',
    camera:{{eye:{{x:1.45,y:-1.65,z:1.15}}}},
    xaxis:{{title:'X / m',gridcolor:'#334155',zerolinecolor:'#475569',backgroundcolor:'#0b1728'}},
    yaxis:{{title:'Y / m',gridcolor:'#334155',zerolinecolor:'#475569',backgroundcolor:'#0b1728'}},
    zaxis:{{title:'Z / m',gridcolor:'#334155',zerolinecolor:'#475569',backgroundcolor:'#0b1728'}}
  }}
}};
Plotly.newPlot('trajectory-plot',trajectoryData,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = load_json(args.evaluation_json)
    metadata = load_json(args.metadata_json) if args.metadata_json else {}
    if args.object_mesh:
        object_vertices, object_triangles = load_obj_mesh(args.object_mesh)
    else:
        object_vertices = np.empty((0, 3), dtype=np.float64)
        object_triangles = np.empty((0, 3), dtype=np.int32)
    object_size_mm = (
        np.ptp(object_vertices, axis=0) * 1000.0
        if len(object_vertices)
        else np.zeros(3, dtype=np.float64)
    )
    model = pin.buildModelFromUrdf(str(args.urdf))
    frame_id = model.getFrameId(args.frame)
    if frame_id >= len(model.frames):
        raise ValueError(f"Frame not found in URDF: {args.frame}")
    trajectories = extract_trajectories(
        evaluation,
        metadata,
        model,
        frame_id,
        args.gripper_center_offset,
        object_vertices,
    )
    successful = [item for item in trajectories if item["success"]]
    failed = [item for item in trajectories if not item["success"]]
    summary = {
        "episodes": len(trajectories),
        "success_rate": len(successful) / len(trajectories),
        "coordinate_frame": args.frame,
        "gripper_center_offset_m": args.gripper_center_offset,
        "urdf": str(args.urdf),
        "object_mesh": str(args.object_mesh) if args.object_mesh else None,
        "object_size_mm": object_size_mm.tolist(),
        "success": group_summary(successful, args.resample_points),
        "failure": group_summary(failed, args.resample_points),
    }
    save_3d_figure(
        trajectories,
        args.output_dir / "ee_trajectory_3d.png",
        args.resample_points,
        object_triangles,
    )
    save_dashboard(
        trajectories,
        args.output_dir / "ee_trajectory_dashboard.png",
        args.resample_points,
        object_triangles,
    )
    save_metrics_csv(trajectories, args.output_dir / "trajectory_metrics.csv")
    with (args.output_dir / "trajectory_summary.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
    save_interactive_html(
        trajectories,
        summary,
        args.output_dir / "ee_trajectory_interactive.html",
        object_triangles,
        object_size_mm,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
