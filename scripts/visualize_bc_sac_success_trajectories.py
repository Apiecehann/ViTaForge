#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pinocchio as pin

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_rl_ee_trajectories import (
    BACKGROUND_COLOR,
    GRID_COLOR,
    OBJECT_COLOR,
    PANEL_COLOR,
    TARGET_COLOR,
    TEXT_COLOR,
    add_object_footprints_2d,
    add_object_meshes_3d,
    configure_2d_axis,
    configure_3d_axis,
    extract_trajectories,
    group_summary,
    load_json,
    load_obj_mesh,
    plot_trajectory_group,
    resample_trajectory,
    set_3d_limits,
)


BC_COLOR = "#60a5fa"
BC_MEAN_COLOR = "#bfdbfe"
SAC_COLOR = "#2dd4bf"
SAC_MEAN_COLOR = "#00ffd5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare successful BC and SAC end-effector trajectories."
    )
    parser.add_argument("bc_evaluation_json", type=Path)
    parser.add_argument("sac_evaluation_json", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--frame", default="panda_hand")
    parser.add_argument("--gripper-center-offset", type=float, default=0.131)
    parser.add_argument("--resample-points", type=int, default=120)
    parser.add_argument("--common-success-only", action="store_true")
    return parser.parse_args()


def success_by_seed(trajectories: list[dict]) -> dict[int, dict]:
    return {item["seed"]: item for item in trajectories if item["success"]}


def paired_metrics(
    bc_success: list[dict],
    sac_success: list[dict],
    resample_points: int,
) -> dict:
    bc_by_seed = success_by_seed(bc_success)
    sac_by_seed = success_by_seed(sac_success)
    common_seeds = sorted(set(bc_by_seed) & set(sac_by_seed))
    trajectory_distances = []
    endpoint_distances = []
    per_seed = []
    for seed in common_seeds:
        bc_positions = resample_trajectory(
            bc_by_seed[seed]["positions"], resample_points
        )
        sac_positions = resample_trajectory(
            sac_by_seed[seed]["positions"], resample_points
        )
        pointwise_distance = np.linalg.norm(bc_positions - sac_positions, axis=1)
        trajectory_distance = float(np.sqrt(np.mean(pointwise_distance**2)))
        endpoint_distance = float(
            np.linalg.norm(bc_positions[-1] - sac_positions[-1])
        )
        trajectory_distances.append(trajectory_distance)
        endpoint_distances.append(endpoint_distance)
        per_seed.append(
            {
                "seed": seed,
                "trajectory_rms_distance_m": trajectory_distance,
                "endpoint_distance_m": endpoint_distance,
            }
        )
    return {
        "common_success_seeds": common_seeds,
        "common_success_count": len(common_seeds),
        "mean_trajectory_rms_distance_m": (
            float(np.mean(trajectory_distances)) if trajectory_distances else 0.0
        ),
        "std_trajectory_rms_distance_m": (
            float(np.std(trajectory_distances)) if trajectory_distances else 0.0
        ),
        "mean_endpoint_distance_m": (
            float(np.mean(endpoint_distances)) if endpoint_distances else 0.0
        ),
        "per_seed": per_seed,
    }


def select_success_sets(
    bc_trajectories: list[dict],
    sac_trajectories: list[dict],
    common_success_only: bool,
) -> tuple[list[dict], list[dict], list[int]]:
    bc_by_seed = success_by_seed(bc_trajectories)
    sac_by_seed = success_by_seed(sac_trajectories)
    common_seeds = sorted(set(bc_by_seed) & set(sac_by_seed))
    if common_success_only:
        return (
            [bc_by_seed[seed] for seed in common_seeds],
            [sac_by_seed[seed] for seed in common_seeds],
            common_seeds,
        )
    return list(bc_by_seed.values()), list(sac_by_seed.values()), common_seeds


def add_target_markers(axis: plt.Axes, scene_trajectories: list[dict]) -> None:
    targets = np.asarray(
        [
            item["target_position"]
            for item in scene_trajectories
            if item["target_position"] is not None
        ]
    )
    if not len(targets):
        return
    axis.scatter(
        targets[:, 0],
        targets[:, 1],
        targets[:, 2],
        color=TARGET_COLOR,
        marker="*",
        s=48,
        alpha=0.62,
        label="Target initial pose",
    )


def save_3d_figure(
    bc_success: list[dict],
    sac_success: list[dict],
    scene_trajectories: list[dict],
    object_triangles: np.ndarray,
    output_path: Path,
    resample_points: int,
    common_success_only: bool,
) -> None:
    figure = plt.figure(figsize=(14, 11), facecolor=BACKGROUND_COLOR)
    axis = figure.add_subplot(111, projection="3d")
    configure_3d_axis(axis)
    plot_trajectory_group(
        axis,
        bc_success,
        BC_COLOR,
        BC_MEAN_COLOR,
        "BC success",
        resample_points,
        True,
    )
    plot_trajectory_group(
        axis,
        sac_success,
        SAC_COLOR,
        SAC_MEAN_COLOR,
        "SAC success",
        resample_points,
        True,
    )
    add_object_meshes_3d(axis, scene_trajectories, object_triangles)
    add_target_markers(axis, scene_trajectories)
    set_3d_limits(axis, bc_success + sac_success)
    axis.view_init(elev=28, azim=-58)
    subset_label = "common-success seeds" if common_success_only else "all successful rollouts"
    axis.set_title(
        f"BC vs SAC End-Effector Trajectories  |  {subset_label}",
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
        "Only successful trajectories are shown   Diamond = endpoint   Thick line = policy mean",
        ha="center",
        color="#94a3b8",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def save_dashboard(
    bc_success: list[dict],
    sac_success: list[dict],
    scene_trajectories: list[dict],
    object_triangles: np.ndarray,
    output_path: Path,
    resample_points: int,
    common_success_only: bool,
) -> None:
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
        bc_success,
        BC_COLOR,
        BC_MEAN_COLOR,
        "BC success",
        resample_points,
        False,
    )
    plot_trajectory_group(
        axis_3d,
        sac_success,
        SAC_COLOR,
        SAC_MEAN_COLOR,
        "SAC success",
        resample_points,
        False,
    )
    add_object_meshes_3d(axis_3d, scene_trajectories, object_triangles)
    add_object_footprints_2d(axis_xy, scene_trajectories)
    targets = np.asarray(
        [
            item["target_position"]
            for item in scene_trajectories
            if item["target_position"] is not None
        ]
    )
    if len(targets):
        axis_3d.scatter(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            color=TARGET_COLOR,
            marker="*",
            s=42,
            alpha=0.58,
        )
        axis_xy.scatter(
            targets[:, 0],
            targets[:, 1],
            color=TARGET_COLOR,
            marker="*",
            s=32,
            alpha=0.58,
            label="Target",
        )
    set_3d_limits(axis_3d, bc_success + sac_success)
    axis_3d.view_init(elev=26, azim=-58)
    axis_3d.set_title("Successful gripper-center paths", color=TEXT_COLOR, fontsize=12, pad=18)
    for group, color, label in (
        (bc_success, BC_COLOR, "BC success"),
        (sac_success, SAC_COLOR, "SAC success"),
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
    subset_label = "8 paired common-success seeds" if common_success_only else "successful rollouts only"
    figure.suptitle(
        f"BC vs SFT-Regularized SAC  |  {subset_label}",
        color=TEXT_COLOR,
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.savefig(output_path, dpi=210, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def comparison_rows(policy: str, trajectories: list[dict]) -> list[dict]:
    rows = []
    for item in trajectories:
        positions = item["positions"]
        rows.append(
            {
                "policy": policy,
                "seed": item["seed"],
                "reward": item["reward"],
                "actions": item["actions"],
                "path_length_m": item["path_length_m"],
                "net_displacement_m": item["net_displacement_m"],
                "end_x_m": positions[-1, 0],
                "end_y_m": positions[-1, 1],
                "end_z_m": positions[-1, 2],
                "lifted_height_m": item["lifted_height"],
            }
        )
    return rows


def save_metrics_csv(
    bc_success: list[dict],
    sac_success: list[dict],
    output_path: Path,
) -> None:
    rows = comparison_rows("BC", bc_success) + comparison_rows("SAC", sac_success)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plotly_policy_traces(
    policy: str,
    trajectories: list[dict],
    color: str,
) -> list[dict]:
    traces = []
    for item_index, item in enumerate(trajectories):
        positions = item["positions"]
        hover_text = [
            (
                f"policy={policy}<br>seed={item['seed']}<br>step={step_index}"
                f"<br>x={position[0]:.4f} m<br>y={position[1]:.4f} m"
                f"<br>z={position[2]:.4f} m<br>reward={item['reward']:.3f}"
            )
            for step_index, position in enumerate(positions)
        ]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": positions[:, 0].tolist(),
                "y": positions[:, 1].tolist(),
                "z": positions[:, 2].tolist(),
                "text": hover_text,
                "hoverinfo": "text",
                "name": f"{policy} success",
                "legendgroup": policy.lower(),
                "showlegend": item_index == 0,
                "line": {"color": color, "width": 4},
                "opacity": 0.62,
            }
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": [float(positions[-1, 0])],
                "y": [float(positions[-1, 1])],
                "z": [float(positions[-1, 2])],
                "text": [f"policy={policy}<br>seed={item['seed']} endpoint"],
                "hoverinfo": "text",
                "name": f"{policy} endpoint",
                "legendgroup": policy.lower(),
                "showlegend": False,
                "marker": {
                    "color": color,
                    "size": 4.5,
                    "symbol": "diamond",
                    "line": {"color": "#ffffff", "width": 1},
                },
            }
        )
    return traces


def representative_object_trace(
    trajectories: list[dict],
    object_triangles: np.ndarray,
) -> dict | None:
    available = [
        item for item in trajectories if item["object_world_vertices"] is not None
    ]
    if not available or not len(object_triangles):
        return None
    centers = np.asarray(
        [item["object_world_vertices"].mean(axis=0) for item in available]
    )
    mean_center = centers.mean(axis=0)
    representative_index = int(
        np.argmin(np.linalg.norm(centers - mean_center, axis=1))
    )
    item = available[representative_index]
    world_vertices = item["object_world_vertices"]
    return {
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
        "text": f"seed={item['seed']} scale reference<br>50 × 25 × 40 mm",
        "hoverinfo": "text",
    }


def save_interactive_html(
    bc_success: list[dict],
    sac_success: list[dict],
    scene_trajectories: list[dict],
    object_triangles: np.ndarray,
    summary: dict,
    output_path: Path,
) -> None:
    plot_traces = plotly_policy_traces("BC", bc_success, BC_COLOR)
    plot_traces.extend(plotly_policy_traces("SAC", sac_success, SAC_COLOR))
    object_trace = representative_object_trace(scene_trajectories, object_triangles)
    if object_trace:
        plot_traces.append(object_trace)
    target_items = [
        item for item in scene_trajectories if item["target_position"] is not None
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
    html = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
  <title>BC vs SAC Successful Trajectories</title>
  <script src=\"plotly-2.35.2.min.js\" onerror=\"this.onerror=null;this.src='https://cdn.plot.ly/plotly-2.35.2.min.js';\"></script>
  <style>
    :root {{ color-scheme:dark; --bg:#07111f; --panel:#0b1728; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --bc:{BC_COLOR}; --sac:{SAC_COLOR}; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#11243d 0,var(--bg) 48%); color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; }}
    main {{ width:min(1500px,96vw); margin:24px auto 44px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(24px,3vw,42px); letter-spacing:-.03em; }}
    .subtitle {{ color:var(--muted); margin-bottom:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-bottom:14px; }}
    .card,.panel {{ background:rgba(11,23,40,.9); border:1px solid var(--line); border-radius:16px; box-shadow:0 18px 55px rgba(0,0,0,.22); }}
    .card {{ padding:16px 18px; }} .card span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .card strong {{ display:block; font-size:27px; margin-top:5px; }} .panel {{ overflow:hidden; }}
    #trajectory-plot {{ width:100%; height:min(78vh,920px); }}
    .note {{ padding:12px 16px; color:var(--muted); border-top:1px solid var(--line); }}
    @media (max-width:850px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  </style>
</head>
<body><main>
  <h1>BC vs SAC 成功轨迹</h1>
  <div class=\"subtitle\">失败轨迹已完全过滤 · 点击图例切换策略 · 悬停查看 policy、seed 和末端坐标</div>
  <section class=\"cards\">
    <div class=\"card\"><span>BC success</span><strong style=\"color:var(--bc)\">{summary['bc']['episodes']}</strong></div>
    <div class=\"card\"><span>SAC success</span><strong style=\"color:var(--sac)\">{summary['sac']['episodes']}</strong></div>
    <div class=\"card\"><span>Common success</span><strong>{summary['paired']['common_success_count']}</strong></div>
    <div class=\"card\"><span>Paired distance</span><strong>{summary['paired']['mean_trajectory_rms_distance_m'] * 1000:.1f} mm</strong></div>
    <div class=\"card\"><span>Target size</span><strong>50×25×40 mm</strong></div>
  </section>
  <section class=\"panel\"><div id=\"trajectory-plot\"></div>
    <div class=\"note\">蓝色 = BC 成功轨迹，青色 = SAC 成功轨迹；图中不包含任何失败轨迹。</div>
  </section>
</main>
<script>
const trajectoryData={json.dumps(plot_traces, ensure_ascii=False)};
const layout={{paper_bgcolor:'#0b1728',plot_bgcolor:'#0b1728',font:{{color:'#e2e8f0'}},margin:{{l:0,r:0,t:24,b:0}},showlegend:true,
legend:{{x:0.02,y:0.98,bgcolor:'rgba(7,17,31,.72)',bordercolor:'#334155',borderwidth:1,groupclick:'togglegroup'}},
scene:{{aspectmode:'data',bgcolor:'#0b1728',camera:{{eye:{{x:1.45,y:-1.65,z:1.15}}}},xaxis:{{title:'X / m',gridcolor:'#334155'}},yaxis:{{title:'Y / m',gridcolor:'#334155'}},zaxis:{{title:'Z / m',gridcolor:'#334155'}}}}}};
Plotly.newPlot('trajectory-plot',trajectoryData,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});
</script></body></html>"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bc_evaluation = load_json(args.bc_evaluation_json)
    sac_evaluation = load_json(args.sac_evaluation_json)
    metadata = load_json(args.metadata_json)
    object_vertices, object_triangles = load_obj_mesh(args.object_mesh)
    model = pin.buildModelFromUrdf(str(args.urdf))
    frame_id = model.getFrameId(args.frame)
    bc_trajectories = extract_trajectories(
        bc_evaluation,
        metadata,
        model,
        frame_id,
        args.gripper_center_offset,
        object_vertices,
    )
    sac_trajectories = extract_trajectories(
        sac_evaluation,
        metadata,
        model,
        frame_id,
        args.gripper_center_offset,
        object_vertices,
    )
    selected_bc, selected_sac, common_seeds = select_success_sets(
        bc_trajectories,
        sac_trajectories,
        args.common_success_only,
    )
    scene_trajectories = selected_sac
    paired = paired_metrics(
        [item for item in bc_trajectories if item["success"]],
        [item for item in sac_trajectories if item["success"]],
        args.resample_points,
    )
    summary = {
        "common_success_only": args.common_success_only,
        "source_success_rates": {
            "bc": float(bc_evaluation["success_rate"]),
            "sac": float(sac_evaluation["success_rate"]),
        },
        "bc": group_summary(selected_bc, args.resample_points),
        "sac": group_summary(selected_sac, args.resample_points),
        "paired": paired,
        "failed_trajectories_included": 0,
        "object_size_mm": (np.ptp(object_vertices, axis=0) * 1000.0).tolist(),
    }
    prefix = "bc_vs_sac_common_success" if args.common_success_only else "bc_vs_sac_success_only"
    save_3d_figure(
        selected_bc,
        selected_sac,
        scene_trajectories,
        object_triangles,
        args.output_dir / f"{prefix}_3d.png",
        args.resample_points,
        args.common_success_only,
    )
    save_dashboard(
        selected_bc,
        selected_sac,
        scene_trajectories,
        object_triangles,
        args.output_dir / f"{prefix}_dashboard.png",
        args.resample_points,
        args.common_success_only,
    )
    save_metrics_csv(
        selected_bc,
        selected_sac,
        args.output_dir / f"{prefix}_metrics.csv",
    )
    with (args.output_dir / f"{prefix}_summary.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
    save_interactive_html(
        selected_bc,
        selected_sac,
        scene_trajectories,
        object_triangles,
        summary,
        args.output_dir / f"{prefix}_interactive.html",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
