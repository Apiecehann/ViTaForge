"""Train the privileged InsertUSB policy with the RFCL SAC pilot."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Total episode target, including episodes already present on resume.",
    )
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument(
        "--snapshot-sync-steps",
        type=int,
        default=0,
        help="Action-free snapshot settling steps; target/velocity demos use 0.",
    )
    parser.add_argument(
        "--demo-horizon-to-max-steps-ratio",
        type=float,
        default=1.0,
        help="RFCL demo-horizon divisor; 1.0 permits complete stride=1 replay.",
    )
    parser.add_argument("--reverse-step-size", type=int, default=2)
    parser.add_argument("--per-demo-buffer-size", type=int, default=3)
    parser.add_argument("--geometric-p", type=float, default=0.5)
    parser.add_argument("--minimum-episode-horizon", type=int, default=16)
    parser.add_argument("--action-scale-margin", type=float, default=1.05)
    parser.add_argument(
        "--bootstrap-handoff",
        action="store_true",
        help=(
            "For each sampled seed, run the scripted grasp/pre-insert handoff "
            "in the live process before restoring its RFCL suffix snapshot."
        ),
    )
    parser.add_argument(
        "--demo-block-size",
        type=int,
        default=1,
        help=(
            "Train this many consecutive episodes on one sampled demo. "
            "With --bootstrap-handoff, successful episodes reuse the live "
            "gripper contact; a failure always forces a fresh handoff."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--demo-pretrain-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument(
        "--auto-alpha",
        action="store_true",
        help="Learn SAC temperature instead of keeping --alpha fixed.",
    )
    parser.add_argument(
        "--backup-entropy",
        action="store_true",
        help="Include the entropy term in the bootstrapped target Q value.",
    )
    parser.add_argument("--num-qs", type=int, default=2)
    parser.add_argument("--num-min-qs", type=int, default=None)
    parser.add_argument("--initial-log-std", type=float, default=-3.0)
    parser.add_argument("--demo-fraction", type=float, default=0.5)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument(
        "--save-success-trajectories",
        action="store_true",
        help="Persist successful online RFCL rollouts as privileged .npz files.",
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=25)
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Number of numbered full checkpoints to retain (latest.pt is a symlink).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Full RFCL checkpoint to resume; use OUTPUT/latest.pt for the latest.",
    )
    parser.add_argument(
        "--stop-when-all-solved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop once every demo frontier reaches state 0 and passes its success window.",
    )
    parser.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def _bc_pretrain(trainer, replay, *, steps: int, batch_size: int) -> None:
    if int(steps) <= 0:
        return
    for _ in range(int(steps)):
        batch = replay.sample(batch_size, demo_fraction=1.0)
        states = trainer._states(np.stack([x.state for x in batch]))
        actions = torch.as_tensor(
            np.stack([x.action for x in batch]),
            dtype=torch.float32,
            device=trainer.device,
        )
        mean, _ = trainer.actor(states)
        loss = torch.nn.functional.mse_loss(torch.tanh(mean), actions)
        trainer.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), 10.0)
        trainer.actor_optimizer.step()


def _save_success_trajectory(
    output: Path,
    *,
    episode: int,
    demo_index: int,
    state_index: int,
    transitions: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]],
    action_scale: np.ndarray,
    action_mode: str,
) -> Path:
    """Save one successful privileged rollout for later sensor re-recording."""
    if not transitions:
        raise ValueError("Cannot save an empty RFCL trajectory")
    trajectory_dir = output / "success_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    path = trajectory_dir / (
        f"episode_{int(episode):06d}_demo_{int(demo_index):06d}_"
        f"state_{int(state_index):06d}.npz"
    )
    np.savez_compressed(
        path,
        states=np.stack([item[0] for item in transitions]).astype(np.float32),
        actions=np.stack([item[1] for item in transitions]).astype(np.float32),
        rewards=np.asarray([item[2] for item in transitions], dtype=np.float32),
        next_states=np.stack([item[3] for item in transitions]).astype(np.float32),
        terminated=np.asarray([item[4] for item in transitions], dtype=np.bool_),
        episode=np.asarray(int(episode), dtype=np.int64),
        demo_index=np.asarray(int(demo_index), dtype=np.int64),
        state_index=np.asarray(int(state_index), dtype=np.int64),
        action_scale=np.asarray(action_scale, dtype=np.float32),
        action_mode=np.asarray(str(action_mode)),
    )
    return path


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: dict[str, object], *, restore_cuda: bool = False
) -> None:
    np.random.set_state(tuple(state["numpy"]))
    torch.set_rng_state(state["torch"])
    # Isaac/UIPC owns the active CUDA context.  Resetting the process-global
    # CUDA RNG after the simulator is created can terminate Kit natively, so
    # CUDA RNG is recorded for diagnostics but intentionally not restored here.
    if restore_cuda and torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _truncate_resume_outputs(output: Path, *, last_episode: int) -> None:
    """Discard rows/files newer than the durable checkpoint before appending."""

    log_path = output / "metrics.jsonl"
    if log_path.exists():
        retained = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["episode"]) <= int(last_episode):
                retained.append(line)
        temporary = log_path.with_name(f".{log_path.name}.tmp")
        temporary.write_text(
            "".join(f"{line}\n" for line in retained), encoding="utf-8"
        )
        os.replace(temporary, log_path)
    trajectory_dir = output / "success_trajectories"
    if trajectory_dir.exists():
        for path in trajectory_dir.glob("episode_*.npz"):
            try:
                episode = int(path.name.split("_", 2)[1])
            except (IndexError, ValueError):
                continue
            if episode > int(last_episode):
                path.unlink()


def _update_latest_checkpoint(output: Path, checkpoint: Path) -> None:
    latest = output / "latest.pt"
    temporary = output / ".latest.pt.tmp"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(checkpoint.name)
    os.replace(temporary, latest)


def _prune_checkpoints(output: Path, *, keep: int) -> None:
    checkpoints = sorted(output.glob("rfcl_sac_episode_*.pt"))
    for path in checkpoints[: max(0, len(checkpoints) - int(keep))]:
        path.unlink()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.demo_block_size <= 0:
        raise ValueError("--demo-block-size must be positive")
    if args.demo_fraction < 0.0 or args.demo_fraction > 1.0:
        raise ValueError("--demo-fraction must be in [0, 1]")
    if args.replay_capacity <= 0:
        raise ValueError("--replay-capacity must be positive")
    if args.checkpoint_frequency <= 0:
        raise ValueError("--checkpoint-frequency must be positive")
    if args.keep_checkpoints <= 0:
        raise ValueError("--keep-checkpoints must be positive")
    if args.resume == Path("latest"):
        args.resume = args.output / "latest.pt"
    if args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
    if args.resume is None and (args.output / "metrics.jsonl").exists():
        raise FileExistsError(
            f"{args.output} already contains a run; pass --resume or choose a new output"
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[rfcl-train] received signal {signum}; saving after this episode",
            flush=True,
        )

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    task = None
    env = None
    try:
        from policy.RL.rfcl_env import RFCLPrivilegedEnv
        from policy.RL.rfcl import (
            MixedReplayBuffer,
            RFCLTransition,
            RoundRobinDemoScheduler,
        )
        from policy.RL.rfcl_sac import (
            RFCLSACTrainer,
            add_demo_transitions,
            demo_state_statistics,
        )
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.snapshot_root,
            video_frequency=0,
            step_limit=args.step_limit,
            mode="eval_test",
            save_pre_move=False,
            device=args.device,
        )
        print("[rfcl-train] task created", flush=True)
        env = RFCLPrivilegedEnv(
            task,
            args.snapshot_root,
            action_repeat=args.action_repeat,
            snapshot_sync_steps=args.snapshot_sync_steps,
            reverse_step_size=args.reverse_step_size,
            per_demo_buffer_size=args.per_demo_buffer_size,
            geometric_p=args.geometric_p,
            demo_horizon_to_max_steps_ratio=args.demo_horizon_to_max_steps_ratio,
            minimum_episode_horizon=args.minimum_episode_horizon,
            action_scale_margin=args.action_scale_margin,
            seed=args.seed,
        )
        if env.action_mode == "target_pos_vel_force":
            if not args.bootstrap_handoff:
                raise ValueError(
                    "target_pos_vel_force RFCL training requires "
                    "--bootstrap-handoff to preserve gripper contact"
                )
            if args.snapshot_sync_steps != 0:
                raise ValueError(
                    "target_pos_vel_force RFCL training requires "
                    "--snapshot-sync-steps 0"
                )
        print(
            f"[rfcl-train] env created demos={len(env.curriculum.demos)} "
            f"action_scale={env.action_scale.tolist()}",
            flush=True,
        )
        demos = list(env.curriculum.demos)
        replay = MixedReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
        add_demo_transitions(replay, demos)
        state_mean, state_std = demo_state_statistics(demos)
        trainer = RFCLSACTrainer(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=args.device,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
            backup_entropy=args.backup_entropy,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
            initial_log_std=args.initial_log_std,
            state_mean=state_mean,
            state_std=state_std,
        )
        scheduler = RoundRobinDemoScheduler(
            len(demos), block_size=args.demo_block_size
        )
        log_path = args.output / "metrics.jsonl"
        start_episode = 0
        total_steps = 0
        elapsed_before_resume = 0.0
        if args.resume is None:
            _bc_pretrain(
                trainer,
                replay,
                steps=args.demo_pretrain_steps,
                batch_size=args.batch_size,
            )
            print("[rfcl-train] demo pretrain complete", flush=True)
        else:
            print(f"[rfcl-train] loading checkpoint {args.resume}", flush=True)
            payload = trainer.load(args.resume)
            print("[rfcl-train] SAC checkpoint loaded", flush=True)
            extra = payload.get("extra", {})
            if extra.get("runner_schema") != "rfcl_runner_checkpoint_v1":
                raise ValueError(
                    "This checkpoint lacks full runner/replay state and cannot be "
                    "used for exact resume"
                )
            if Path(extra["snapshot_root"]).resolve() != args.snapshot_root.resolve():
                raise ValueError("Resume snapshot_root does not match the command line")
            if str(extra.get("action_mode")) != env.action_mode:
                raise ValueError("Resume checkpoint action_mode does not match the dataset")
            if not np.allclose(extra.get("action_scale"), env.action_scale):
                raise ValueError("Resume checkpoint action_scale does not match the dataset")
            resume_expected = {
                "step_limit": int(args.step_limit),
                "action_repeat": int(args.action_repeat),
                "snapshot_sync_steps": int(args.snapshot_sync_steps),
                "bootstrap_handoff": bool(args.bootstrap_handoff),
                "batch_size": int(args.batch_size),
                "gradient_steps": int(args.gradient_steps),
                "replay_capacity": int(args.replay_capacity),
            }
            for name, value in resume_expected.items():
                if extra.get(name) != value:
                    raise ValueError(
                        f"Resume {name} mismatch: checkpoint={extra.get(name)!r}, "
                        f"current={value!r}"
                    )
            if not np.isclose(float(extra["demo_fraction"]), args.demo_fraction):
                raise ValueError("Resume demo_fraction does not match the command line")
            env.curriculum.load_state_dict(extra["curriculum"])
            print("[rfcl-train] curriculum restored", flush=True)
            replay.load_state_dict(extra["replay"])
            print("[rfcl-train] replay restored", flush=True)
            scheduler.load_state_dict(extra["scheduler"])
            print("[rfcl-train] scheduler restored", flush=True)
            start_episode = int(extra["episode"]) + 1
            total_steps = int(extra["total_steps"])
            elapsed_before_resume = float(extra.get("elapsed_s", 0.0))
            # Do not reset process-global NumPy/CPU RNG after Isaac has created
            # its native CUDA/UIPC context.  The curriculum and replay each
            # restore their own RNG exactly; actor exploration is intentionally
            # re-seeded by the fresh simulator process.
            _truncate_resume_outputs(
                args.output, last_episode=int(extra["episode"])
            )
            print(
                f"[rfcl-train] resumed checkpoint={args.resume} "
                f"next_episode={start_episode} total_steps={total_steps}",
                flush=True,
            )
        if args.stop_when_all_solved and bool(env.curriculum.solved.all()):
            print("[rfcl-train] all demos are already solved", flush=True)
            return
        if start_episode >= args.episodes:
            print(
                f"[rfcl-train] nothing to do: checkpoint already reached "
                f"episode {start_episode - 1}, target={args.episodes}",
                flush=True,
            )
            return
        start_time = time.perf_counter()
        bootstrap_required = True
        with log_path.open("a" if args.resume is not None else "w", encoding="utf-8") as log_file:
            for episode in range(start_episode, args.episodes):
                print(f"[rfcl-train] episode={episode} reset", flush=True)
                # In the default mode solved demos are removed from the
                # schedule.  A fixed-length data-collection run can opt out
                # of early stopping and keep cycling through every demo.
                scheduled_solved = (
                    env.curriculum.solved
                    if args.stop_when_all_solved
                    else np.zeros_like(env.curriculum.solved)
                )
                demo_index, new_demo_block = scheduler.select_demo(
                    scheduled_solved
                )
                _, state_index = env.curriculum.sample_checkpoint(
                    demo_id=demo_index
                )
                if new_demo_block:
                    bootstrap_required = True
                if args.bootstrap_handoff:
                    demo_seed = env.dataset.demos[demo_index].get("seed")
                    bootstrap_handoff_ran = bool(bootstrap_required)
                    if bootstrap_required:
                        task.reset(
                            seed=None if demo_seed is None else int(demo_seed)
                        )
                        prepare_handoff = getattr(
                            task, "_prepare_usb_standard", None
                        )
                        if not callable(prepare_handoff):
                            raise RuntimeError(
                                "--bootstrap-handoff requires an InsertUSB task "
                                "with _prepare_usb_standard()"
                            )
                        prepare_handoff()
                    state, reset_info = env.reset(
                        options={
                            "demo_index": int(demo_index),
                            "state_index": int(state_index),
                            "skip_task_reset": True,
                        }
                    )
                else:
                    bootstrap_handoff_ran = False
                    state, reset_info = env.reset(
                        options={
                            "demo_index": int(demo_index),
                            "state_index": int(state_index),
                        }
                    )
                print(
                    f"[rfcl-train] episode={episode} reset_done "
                    f"demo={reset_info['demo_index']} state={reset_info['state_index']}",
                    flush=True,
                )
                episode_reward = 0.0
                episode_steps = 0
                success = False
                episode_transitions = []
                while True:
                    action = trainer.act(state, deterministic=False)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    transition = env.pop_transition()
                    if transition is None:
                        raise RuntimeError("RFCL environment did not expose a transition")
                    episode_transitions.append(transition)
                    replay.add(
                        RFCLTransition(
                            state=transition[0],
                            action=transition[1],
                            reward=transition[2],
                            next_state=transition[3],
                            terminated=transition[4],
                            demo_id=None,
                            timestep=episode_steps,
                            source="online",
                        )
                    )
                    for _ in range(args.gradient_steps):
                        if len(replay) >= args.batch_size:
                            metrics = trainer.update(
                                replay,
                                batch_size=args.batch_size,
                                demo_fraction=args.demo_fraction,
                            )
                    state = next_state
                    episode_reward += float(reward)
                    episode_steps += 1
                    total_steps += 1
                    success = success or bool(info.get("success", False))
                    if terminated or truncated:
                        break
                scheduler.record_episode(demo_index)
                if args.bootstrap_handoff:
                    # A failed rollout can destroy the live gripper contact.
                    # Rebuild it before restoring another suffix checkpoint.
                    bootstrap_required = not bool(success)
                trajectory_path = None
                if success and args.save_success_trajectories:
                    trajectory_path = _save_success_trajectory(
                        args.output,
                        episode=episode,
                        demo_index=int(reset_info["demo_index"]),
                        state_index=int(reset_info["state_index"]),
                        transitions=episode_transitions,
                        action_scale=env.action_scale,
                        action_mode=env.action_mode,
                    )
                row = {
                    "episode": episode,
                    "total_steps": total_steps,
                    "episode_steps": episode_steps,
                    "episode_reward": episode_reward,
                    "success": bool(success),
                    "demo_index": reset_info["demo_index"],
                    "state_index": reset_info["state_index"],
                    "bootstrap_handoff_ran": bootstrap_handoff_ran,
                    "demo_block_size": int(args.demo_block_size),
                    "frontiers": env.curriculum_state()["frontiers"].tolist(),
                    "solved": env.curriculum_state()["solved"].tolist(),
                    "demo_visit_counts": scheduler.visit_counts.tolist(),
                    "replay_source_counts": replay.source_counts(),
                    "elapsed_s": elapsed_before_resume + time.perf_counter() - start_time,
                }
                if trajectory_path is not None:
                    row["success_trajectory"] = str(trajectory_path)
                if "metrics" in locals():
                    row.update(
                        {
                            "critic_loss": metrics.critic_loss,
                            "actor_loss": metrics.actor_loss,
                            "alpha": metrics.alpha,
                            "q_mean": metrics.q_mean,
                            "batch_source_counts": trainer.last_batch_source_counts,
                        }
                    )
                log_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                log_file.flush()
                all_solved = bool(env.curriculum.solved.all())
                should_stop = bool(
                    stop_requested
                    or (args.stop_when_all_solved and all_solved)
                )
                should_checkpoint = bool(
                    (episode + 1) % args.checkpoint_frequency == 0
                    or episode == args.episodes - 1
                    or should_stop
                )
                if should_checkpoint:
                    elapsed_s = elapsed_before_resume + time.perf_counter() - start_time
                    checkpoint_path = (
                        args.output / f"rfcl_sac_episode_{episode:06d}.pt"
                    )
                    trainer.save(
                        checkpoint_path,
                        extra={
                            "runner_schema": "rfcl_runner_checkpoint_v1",
                            "snapshot_root": str(args.snapshot_root),
                            "action_scale": env.action_scale.tolist(),
                            "action_scale_margin": float(args.action_scale_margin),
                            "action_mode": env.action_mode,
                            "step_limit": int(args.step_limit),
                            "action_repeat": int(args.action_repeat),
                            "snapshot_sync_steps": int(args.snapshot_sync_steps),
                            "bootstrap_handoff": bool(args.bootstrap_handoff),
                            "reverse_step_size": int(args.reverse_step_size),
                            "per_demo_buffer_size": int(args.per_demo_buffer_size),
                            "geometric_p": float(args.geometric_p),
                            "minimum_episode_horizon": int(args.minimum_episode_horizon),
                            "demo_fraction": float(args.demo_fraction),
                            "batch_size": int(args.batch_size),
                            "gradient_steps": int(args.gradient_steps),
                            "replay_capacity": int(args.replay_capacity),
                            "curriculum": env.curriculum_state(),
                            "replay": replay.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "rng_state": _capture_rng_state(),
                            "episode": episode,
                            "total_steps": total_steps,
                            "elapsed_s": elapsed_s,
                        },
                    )
                    _update_latest_checkpoint(args.output, checkpoint_path)
                    _prune_checkpoints(
                        args.output, keep=args.keep_checkpoints
                    )
                print(json.dumps(row, ensure_ascii=True), flush=True)
                if should_stop:
                    reason = "all_demos_solved" if all_solved else "signal"
                    print(
                        f"[rfcl-train] stopped reason={reason} episode={episode} "
                        f"frontiers={env.curriculum.frontiers.tolist()} "
                        f"visits={scheduler.visit_counts.tolist()}",
                        flush=True,
                    )
                    break
    finally:
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
