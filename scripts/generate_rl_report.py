from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


NAVY = "#17324D"
BLUE = "#2E6F9E"
CYAN = "#53A7B8"
ORANGE = "#E99052"
RED = "#C84C4C"
GREEN = "#4C956C"
LIGHT = "#EFF4F7"
GRID = "#D8E1E7"
TEXT = "#263238"


def configure_style() -> None:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            family = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update(
        {
            "axes.edgecolor": NAVY,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "font.size": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def hdf5_paths(dataset_root: Path) -> list[Path]:
    root = dataset_root / "hdf5" if (dataset_root / "hdf5").exists() else dataset_root
    candidates = sorted(root.glob("*.hdf5"), key=lambda path: int(path.stem))
    readable = []
    for path in candidates:
        try:
            with h5py.File(path, "r") as handle:
                if "embodiment/joint" in handle and "phase/id" in handle:
                    readable.append(path)
        except (BlockingIOError, OSError):
            continue
    return readable


def decode_jpeg(value) -> np.ndarray:
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode an HDF5 JPEG frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def tactile_path(handle: h5py.File, side: str) -> str:
    for candidate in (
        f"tactile/{side}_tactile/rgb_marker",
        f"tactile/{side}_gsmini/rgb_marker",
    ):
        if candidate in handle:
            return candidate
    raise KeyError(f"Missing {side} rgb_marker")


def collect_dataset_statistics(paths: list[Path]) -> dict:
    frames = []
    policy_frames = []
    policy_pairs = []
    seeds = []
    phase_versions = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            frame_count = len(handle["embodiment/joint"])
            phase = np.asarray(handle["phase/id"]) if "phase/id" in handle else np.zeros(frame_count)
            frames.append(frame_count)
            policy_frames.append(int(np.sum(phase == 1)))
            policy_pairs.append(int(np.sum((phase[:-1] == 1) & (phase[1:] == 1))))
            seeds.append(int(path.stem))
            phase_versions.append(int(handle["phase"].attrs.get("schema_version", 1)))
    return {
        "episodes": len(paths),
        "frames": frames,
        "policy_frames": policy_frames,
        "policy_pairs": policy_pairs,
        "seeds": seeds,
        "schema_versions": sorted(set(phase_versions)),
        "total_frames": int(sum(frames)),
        "total_policy_pairs": int(sum(policy_pairs)),
        "total_bytes": int(sum(path.stat().st_size for path in paths)),
    }


def example_modalities(path: Path) -> tuple[dict[str, np.ndarray], int]:
    with h5py.File(path, "r") as handle:
        phase = np.asarray(handle["phase/id"])
        indices = np.flatnonzero(phase == 1)
        left_path = tactile_path(handle, "left")
        right_path = tactile_path(handle, "right")
        if len(indices):
            reference_left = decode_jpeg(handle[left_path][int(indices[0])]).astype(np.float32)
            reference_right = decode_jpeg(handle[right_path][int(indices[0])]).astype(np.float32)
            scores = []
            for index in indices:
                left = decode_jpeg(handle[left_path][int(index)]).astype(np.float32)
                right = decode_jpeg(handle[right_path][int(index)]).astype(np.float32)
                scores.append(float(np.mean(np.abs(left - reference_left)) + np.mean(np.abs(right - reference_right))))
            frame_index = int(indices[int(np.argmax(scores))])
        else:
            frame_index = len(phase) // 2
        images = {
            "Head RGB": decode_jpeg(handle["observation/head/rgb"][frame_index]),
            "Wrist RGB": decode_jpeg(handle["observation/wrist/rgb"][frame_index]),
            "Left GelSight rgb_marker": decode_jpeg(handle[left_path][frame_index]),
            "Right GelSight rgb_marker": decode_jpeg(handle[right_path][frame_index]),
        }
    return images, frame_index


def read_bc_history(run_root: Path) -> list[dict]:
    return read_json(run_root / "bc_phase" / "training_history.json", [])


def read_monitor(run_root: Path) -> dict[str, np.ndarray]:
    candidates = list((run_root / "sac").glob("*monitor.csv*"))
    if not candidates:
        return {"reward": np.array([]), "length": np.array([]), "success": np.array([])}
    rows = []
    with candidates[0].open("r", encoding="utf-8") as stream:
        lines = [line for line in stream if not line.startswith("#")]
    for row in csv.DictReader(lines):
        rows.append(row)
    def values(key: str) -> np.ndarray:
        parsed = []
        for row in rows:
            value = row.get(key, "")
            if value in ("", None):
                parsed.append(np.nan)
            elif str(value).lower() in ("true", "false"):
                parsed.append(float(str(value).lower() == "true"))
            else:
                parsed.append(float(value))
        return np.asarray(parsed, dtype=float)
    return {"reward": values("r"), "length": values("l"), "success": values("success")}


def rolling_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) == 0:
        return values
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = np.nanmean(values[start : index + 1])
    return result


def video_frame(path: Path, fraction: float = 0.72) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_count - 1, int(frame_count * fraction))))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def new_page(title: str, subtitle: str | None = None):
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.09)
    figure.text(0.07, 0.93, title, color=NAVY, fontsize=20, fontweight="bold")
    if subtitle:
        figure.text(0.07, 0.895, subtitle, color="#60717D", fontsize=9)
    figure.add_artist(plt.Line2D([0.07, 0.95], [0.88, 0.88], color=CYAN, linewidth=2.5))
    return figure


def add_footer(figure, page_number: int) -> None:
    figure.text(0.07, 0.035, "UniVTAC | GelSight multimodal residual SAC", color="#75858F", fontsize=7)
    figure.text(0.95, 0.035, str(page_number), color="#75858F", fontsize=7, ha="right")


def rounded_box(axis, xy, width, height, title, body, color=BLUE):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width * 0.05, xy[1] + height * 0.68, title, color=color, fontsize=11, fontweight="bold")
    axis.text(xy[0] + width * 0.05, xy[1] + height * 0.26, body, color=TEXT, fontsize=8, va="center")


def page_overview(pdf: PdfPages, stats: dict, page: int) -> None:
    figure = new_page(
        "GelSight 多模态 BC + Residual SAC",
        "grasp_in_clutter | 20K policy timesteps | 最终 20 集独立评估",
    )
    axis = figure.add_axes([0.07, 0.12, 0.88, 0.70])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    cards = [
        ("示范数据", f"{stats['episodes']} episodes\nsave_frequency = 2", BLUE),
        ("BC", "ResNet18 multimodal\n30 epochs, patience 5", CYAN),
        ("SAC", "Residual action\n20,000 policy steps", ORANGE),
        ("评估", "20 held-out seeds\n逐集保留成功/失败标签", GREEN),
    ]
    for index, (title, body, color) in enumerate(cards):
        x = 0.015 + index * 0.245
        rounded_box(axis, (x, 0.62), 0.205, 0.22, title, body, color)
        if index < len(cards) - 1:
            axis.annotate("", xy=(x + 0.245, 0.73), xytext=(x + 0.212, 0.73), arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.4})
    axis.text(0.02, 0.46, "实验约束", color=NAVY, fontsize=13, fontweight="bold")
    constraints = [
        "规划器只负责 Reset 与 Pre_Move；Policy 仅控制明确标注的 Action phase。",
        "触觉输入固定为左右 GelSight rgb_marker，不修改任何 Sensor 渲染或 Marker 实现。",
        "本任务的抓取闭合位于 POLICY phase，因此 SAC 学习包含夹爪在内的 8 维受限残差。",
        "第一版不引入 Diversity Reward，优先验证 BC 初始化与任务奖励的有效性。",
    ]
    for index, text in enumerate(constraints):
        y = 0.39 - index * 0.085
        axis.add_patch(plt.Circle((0.035, y + 0.012), 0.012, color=CYAN))
        axis.text(0.065, y, text, fontsize=9.5, color=TEXT)
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def page_inputs(pdf: PdfPages, paths: list[Path], stats: dict, page: int) -> None:
    figure = new_page("输入模态与 HDF5 边界", "所有图像均直接解码自本次示范 HDF5")
    if not paths:
        figure.text(0.5, 0.5, "No HDF5 episodes found", ha="center", color=RED)
    else:
        images, frame_index = example_modalities(paths[len(paths) // 2])
        grid = figure.add_gridspec(2, 4, left=0.07, right=0.95, top=0.83, bottom=0.30, wspace=0.06, hspace=0.15)
        for index, (title, image) in enumerate(images.items()):
            axis = figure.add_subplot(grid[:, index])
            axis.imshow(image)
            axis.set_title(title, fontsize=9, pad=7)
            axis.axis("off")
        figure.text(0.07, 0.23, "Schema", color=NAVY, fontsize=11, fontweight="bold")
        figure.text(
            0.07,
            0.13,
            f"schema_version = {stats['schema_versions']}     episode = {paths[len(paths)//2].stem}     policy frame = {frame_index}\n"
            "phase/id: PRE_MOVE=0, POLICY=1, TERMINAL=2     phase/policy_step 与每帧同步保存\n"
            "训练样本仅保留相邻两帧均为 POLICY 的动作对，防止 Pre_Move 泄漏到学习动作。",
            color=TEXT,
            fontsize=9.5,
            linespacing=1.6,
        )
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def page_method(pdf: PdfPages, page: int) -> None:
    figure = new_page("策略结构与奖励", "BC 提供可行轨迹先验，SAC 只优化受限残差")
    axis = figure.add_axes([0.07, 0.12, 0.88, 0.70])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    rounded_box(axis, (0.02, 0.66), 0.23, 0.20, "多模态编码器", "Head + Wrist RGB\nLeft + Right rgb_marker\n8D qpos + policy_step", BLUE)
    rounded_box(axis, (0.38, 0.66), 0.23, 0.20, "BC action head", "512D fused feature\nMLP: 256 - 256 - 8\n预测 Δq_BC", CYAN)
    rounded_box(axis, (0.74, 0.66), 0.23, 0.20, "Residual SAC", "冻结 BC encoder\nMLP: 256 - 256\n输出 8D [-1, 1]", ORANGE)
    axis.annotate("", xy=(0.38, 0.76), xytext=(0.25, 0.76), arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.6})
    axis.annotate("shared feature", xy=(0.74, 0.76), xytext=(0.61, 0.76), arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.6})
    axis.text(0.5, 0.53, r"$q_{cmd}=q_{cmd}^{t-1}+\Delta q_{BC}+0.5\,\sigma_{\Delta q}\odot a_{SAC}$", ha="center", fontsize=16, color=NAVY)
    axis.text(0.5, 0.45, "动作在安全范围内裁剪，每个策略动作线性插值执行 2 个仿真步；本任务同步控制夹爪。", ha="center", fontsize=9.5, color=TEXT)
    axis.text(0.02, 0.31, "Task reward", color=NAVY, fontsize=12, fontweight="bold")
    axis.text(
        0.04,
        0.20,
        r"$r_t=5\Delta\hat h+25\,clip(\Delta d,-0.02,0.02)+2\Delta g$"
        "\n" r"$\qquad +0.02(p+g)+0.05\,clip(\hat h_t,-1,1.25)-10^{-3}\,mean(a_t^2)$"
        "\n" r"$\qquad +10\,\mathbb{1}[success]-5\,\mathbb{1}[dropped]$",
        fontsize=14,
        color=TEXT,
        linespacing=1.6,
    )
    axis.text(0.04, 0.08, "p=目标-夹爪 proximity，g=p×夹爪闭合度。成功为抬升 9-11 cm；只有曾抬升后明显回落/脱离才判掉落。", fontsize=9.5, color=TEXT)
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def page_dataset(pdf: PdfPages, stats: dict, page: int) -> None:
    figure = new_page("示范数据统计", "100 条成功规划轨迹；统计仅报告实际落盘内容")
    grid = figure.add_gridspec(2, 2, left=0.08, right=0.94, top=0.82, bottom=0.13, hspace=0.38, wspace=0.30)
    axis = figure.add_subplot(grid[0, 0])
    axis.hist(stats["frames"], bins=min(16, max(4, len(stats["frames"]) // 5)), color=BLUE, edgecolor="white")
    axis.set_title("每条轨迹保存帧数")
    axis.set_xlabel("frames / episode")
    axis.set_ylabel("episodes")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis = figure.add_subplot(grid[0, 1])
    axis.hist(stats["policy_pairs"], bins=min(16, max(4, len(stats["policy_pairs"]) // 5)), color=CYAN, edgecolor="white")
    axis.set_title("每条轨迹有效 POLICY 动作对")
    axis.set_xlabel("action pairs / episode")
    axis.set_ylabel("episodes")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis = figure.add_subplot(grid[1, 0])
    axis.scatter(stats["seeds"], stats["policy_pairs"], s=18, color=ORANGE, alpha=0.75)
    axis.set_title("Seed 与有效动作长度")
    axis.set_xlabel("successful seed")
    axis.set_ylabel("policy pairs")
    axis.grid(color=GRID, linewidth=0.7)
    axis = figure.add_subplot(grid[1, 1])
    axis.axis("off")
    size_gb = stats["total_bytes"] / 1024**3
    lines = [
        ("Episodes", f"{stats['episodes']}"),
        ("Total frames", f"{stats['total_frames']:,}"),
        ("POLICY pairs", f"{stats['total_policy_pairs']:,}"),
        ("Dataset size", f"{size_gb:.2f} GiB"),
        ("Schema", ", ".join(map(str, stats["schema_versions"]))),
    ]
    for index, (name, value) in enumerate(lines):
        y = 0.90 - index * 0.18
        axis.text(0.02, y, name, color="#6B7B85", fontsize=9)
        axis.text(0.98, y, value, color=NAVY, fontsize=14, fontweight="bold", ha="right")
        axis.plot([0.02, 0.98], [y - 0.06, y - 0.06], color=GRID, lw=0.8)
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def page_training(pdf: PdfPages, bc_history: list[dict], monitor: dict, page: int) -> None:
    figure = new_page("训练过程", "BC validation early stopping + SAC 20K policy timesteps")
    grid = figure.add_gridspec(1, 2, left=0.08, right=0.94, top=0.82, bottom=0.15, wspace=0.28)
    axis = figure.add_subplot(grid[0, 0])
    if bc_history:
        epochs = [row["epoch"] for row in bc_history]
        axis.plot(epochs, [row["training_loss"] for row in bc_history], marker="o", ms=3, color=BLUE, label="train")
        axis.plot(epochs, [row["validation_loss"] for row in bc_history], marker="o", ms=3, color=ORANGE, label="validation")
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "BC history unavailable", ha="center", color="#75858F")
    axis.set_title("BC Smooth L1 loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalized delta loss")
    axis.grid(color=GRID, linewidth=0.7)
    axis = figure.add_subplot(grid[0, 1])
    rewards = monitor["reward"]
    if len(rewards):
        episodes = np.arange(1, len(rewards) + 1)
        axis.plot(episodes, rewards, color=BLUE, alpha=0.25, lw=0.8, label="episode reward")
        axis.plot(episodes, rolling_mean(rewards), color=BLUE, lw=2.0, label="rolling mean (20)")
        if np.isfinite(monitor["success"]).any():
            secondary = axis.twinx()
            secondary.plot(episodes, rolling_mean(monitor["success"]), color=GREEN, lw=1.8, label="success rate")
            secondary.set_ylabel("rolling success rate", color=GREEN)
            secondary.set_ylim(-0.02, 1.02)
        axis.legend(frameon=False, loc="upper left")
    else:
        axis.text(0.5, 0.5, "SAC monitor unavailable", ha="center", color="#75858F")
    axis.set_title("Residual SAC rollout")
    axis.set_xlabel("training episode")
    axis.set_ylabel("episode reward")
    axis.grid(color=GRID, linewidth=0.7)
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def page_evaluation(pdf: PdfPages, bc_eval: dict, sac_eval: dict, page: int) -> None:
    figure = new_page("最终 20 集评估", "固定 seeds 20000-20019；BC 与 SAC 使用相同初始条件")
    grid = figure.add_gridspec(2, 2, left=0.08, right=0.94, top=0.82, bottom=0.13, hspace=0.38, wspace=0.30)
    axis = figure.add_subplot(grid[0, 0])
    labels = ["BC", "BC + SAC"]
    rates = [float(bc_eval.get("success_rate", 0.0)), float(sac_eval.get("success_rate", 0.0))]
    bars = axis.bar(labels, rates, color=[CYAN, ORANGE], width=0.58)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("success rate")
    axis.set_title("Success rate")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    for bar, value in zip(bars, rates):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.0%}", ha="center", color=NAVY, fontweight="bold")
    axis = figure.add_subplot(grid[0, 1])
    mean_lifts = []
    for evaluation in (bc_eval, sac_eval):
        lifts = [
            float(row.get("metrics", {}).get("lifted_height", 0.0)) * 100.0
            for row in evaluation.get("results", [])
        ]
        mean_lifts.append(float(np.mean(lifts)) if lifts else 0.0)
    bars = axis.bar(labels, mean_lifts, color=[CYAN, ORANGE], width=0.58)
    axis.set_title("Mean final lift")
    axis.set_ylabel("lifted height (cm)")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    for bar, value in zip(bars, mean_lifts):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f" {value:.2f}", ha="center", va="bottom" if value >= 0 else "top", color=NAVY)
    axis = figure.add_subplot(grid[1, :])
    results = sac_eval.get("results", [])
    if results:
        episode = np.asarray([row["episode"] + 1 for row in results])
        reward = np.asarray([row["reward"] for row in results], dtype=float)
        success = np.asarray([row["success"] for row in results], dtype=bool)
        colors = np.where(success, GREEN, RED)
        axis.bar(episode, reward, color=colors, width=0.72)
        axis.set_xticks(episode)
        axis.set_xticklabels([str(value) for value in episode], fontsize=7)
        for x, ok in zip(episode, success):
            axis.text(x, axis.get_ylim()[0], "S" if ok else "F", ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")
    else:
        axis.text(0.5, 0.5, "Evaluation results unavailable", ha="center", color="#75858F")
    axis.set_title("SAC 每集回报（绿色=成功，红色=失败）")
    axis.set_xlabel("evaluation episode")
    axis.set_ylabel("reward")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def page_qualitative(pdf: PdfPages, sac_eval: dict, video_dir: Path, page: int) -> None:
    figure = new_page("定性评估帧", "从最终 SAC 评估视频读取；文件名保留 success / failed 真实标签")
    videos = sorted(video_dir.glob("*.mp4"))
    results_by_seed = {int(row["seed"]): row for row in sac_eval.get("results", [])}
    selected = []
    if videos:
        successes = [path for path in videos if "success" in path.stem]
        failures = [path for path in videos if "failed" in path.stem or "fail" in path.stem]
        selected.extend(successes[:3])
        selected.extend(failures[:1])
        for path in videos:
            if path not in selected and len(selected) < 4:
                selected.append(path)
    grid = figure.add_gridspec(2, 2, left=0.07, right=0.95, top=0.82, bottom=0.12, hspace=0.22, wspace=0.10)
    for index in range(4):
        axis = figure.add_subplot(grid[index // 2, index % 2])
        if index < len(selected):
            path = selected[index]
            frame = video_frame(path)
            if frame is not None:
                axis.imshow(frame)
            try:
                seed = int(path.stem.split("_")[0])
            except ValueError:
                seed = -1
            result = results_by_seed.get(seed, {})
            status = "SUCCESS" if result.get("success", "success" in path.stem) else "FAILED"
            color = GREEN if status == "SUCCESS" else RED
            axis.set_title(f"seed {seed} | {status} | reward {result.get('reward', float('nan')):.2f}", color=color, fontsize=9)
        else:
            axis.text(0.5, 0.5, "Video unavailable", ha="center", color="#75858F")
        axis.axis("off")
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def page_reproducibility(pdf: PdfPages, stats: dict, bc_eval: dict, sac_eval: dict, page: int) -> None:
    figure = new_page("配置与结论", "可复现参数清单与结果边界")
    axis = figure.add_axes([0.07, 0.12, 0.88, 0.70])
    axis.axis("off")
    parameters = [
        ("Task", "grasp_in_clutter"),
        ("Sensor", "GelSight Mini, left/right rgb_marker"),
        ("Sampling", "100 successful episodes, save_frequency=2"),
        ("BC", "ResNet18, image=128, batch=32, max=30 epochs, patience=5"),
        ("SAC", "20K steps, horizon=120, 8D incl. gripper, action_repeat=2"),
        ("Reward", "proximity + grasp proxy + lift progress + success - drop"),
        ("Evaluation", "20 episodes, seeds 20000-20019, deterministic actions"),
    ]
    for index, (name, value) in enumerate(parameters):
        y = 0.92 - index * 0.095
        axis.text(0.01, y, name, color=BLUE, fontsize=9, fontweight="bold")
        axis.text(0.23, y, value, color=TEXT, fontsize=9)
        axis.plot([0.01, 0.99], [y - 0.035, y - 0.035], color=GRID, lw=0.7)
    bc_rate = float(bc_eval.get("success_rate", 0.0))
    sac_rate = float(sac_eval.get("success_rate", 0.0))
    axis.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.15, boxstyle="round,pad=0.015", facecolor=LIGHT, edgecolor=CYAN))
    axis.text(0.04, 0.145, "Result statement", color=NAVY, fontsize=11, fontweight="bold")
    axis.text(
        0.04,
        0.075,
        f"BC success = {bc_rate:.0%}; BC + SAC success = {sac_rate:.0%}; delta = {sac_rate - bc_rate:+.0%}. "
        f"Dataset = {stats['episodes']} episodes / {stats['total_policy_pairs']:,} policy pairs.\n"
        "该结果只代表当前任务、当前奖励和固定 20 个 seeds；失败轨迹保留用于误差分析，不计为成功。",
        color=TEXT,
        fontsize=9.5,
    )
    add_footer(figure, page)
    pdf.savefig(figure)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the GelSight BC + SAC experiment report.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    configure_style()
    paths = hdf5_paths(args.dataset_root)
    stats = collect_dataset_statistics(paths)
    bc_history = read_bc_history(args.run_root)
    monitor = read_monitor(args.run_root)
    evaluation_root = args.run_root / "evaluation_final"
    bc_eval = read_json(evaluation_root / "bc" / "evaluation.json")
    sac_eval = read_json(evaluation_root / "sac" / "evaluation.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        page_overview(pdf, stats, 1)
        page_inputs(pdf, paths, stats, 2)
        page_method(pdf, 3)
        page_dataset(pdf, stats, 4)
        page_training(pdf, bc_history, monitor, 5)
        page_evaluation(pdf, bc_eval, sac_eval, 6)
        page_qualitative(pdf, sac_eval, evaluation_root / "sac" / "video", 7)
        page_reproducibility(pdf, stats, bc_eval, sac_eval, 8)

    report_data = {
        "dataset": stats,
        "bc_evaluation": {key: bc_eval.get(key) for key in ("episodes", "successes", "success_rate", "mean_reward")},
        "sac_evaluation": {key: sac_eval.get(key) for key in ("episodes", "successes", "success_rate", "mean_reward")},
        "report": str(args.output),
    }
    with (args.output.parent / "report_data.json").open("w", encoding="utf-8") as stream:
        json.dump(report_data, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report_data, ensure_ascii=False))


if __name__ == "__main__":
    main()
