import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from policy.RL.bc import MultiModalBC
from policy.RL.dataset import (
    ActionPhaseDataset,
    compute_joint_statistics,
    split_episode_paths,
)


def move_batch(observation, action, device):
    observation = {
        key: value.to(device, non_blocking=True)
        for key, value in observation.items()
    }
    return observation, action.to(device, non_blocking=True)


def action_target(model, observation, action):
    target_delta = action - observation["qpos"]
    if hasattr(model, "action_scale"):
        return torch.clamp(
            (target_delta - model.delta_mean) / model.action_scale,
            min=-1.0,
            max=1.0,
        )
    return (target_delta - model.delta_mean) / model.delta_std


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for observation, action in loader:
            observation, action = move_batch(observation, action, device)
            target = action_target(model, observation, action)
            prediction = model.forward_policy_action(observation)
            losses.append(nn.functional.smooth_l1_loss(prediction, target).item())
    return float(sum(losses) / max(len(losses), 1))


def main():
    parser = argparse.ArgumentParser(description="Train action-phase multimodal BC.")
    parser.add_argument("dataset_root")
    parser.add_argument("output_dir")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--policy-step-stride", type=int, default=2)
    parser.add_argument(
        "--camera-keys",
        nargs="*",
        default=["cam_high", "cam_wrist"],
    )
    parser.add_argument(
        "--tactile-keys",
        nargs="*",
        default=["tac_left", "tac_right"],
    )
    parser.add_argument("--visual-backbone", default="act_resnet18")
    parser.add_argument("--tactile-backbone", default="act_resnet18")
    parser.add_argument("--visual-pretrained", action="store_true")
    parser.add_argument("--tactile-pretrained", action="store_true")
    parser.add_argument("--visual-checkpoint", default=None)
    parser.add_argument("--tactile-checkpoint", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths, validation_paths = split_episode_paths(
        args.dataset_root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    statistics = compute_joint_statistics(
        train_paths,
        policy_step_stride=args.policy_step_stride,
    )
    dataset_kwargs = {
        "image_size": args.image_size,
        "camera_keys": args.camera_keys,
        "tactile_keys": args.tactile_keys,
        "require_phase": True,
        "policy_step_stride": args.policy_step_stride,
    }
    train_dataset = ActionPhaseDataset(train_paths, **dataset_kwargs)
    validation_dataset = ActionPhaseDataset(validation_paths, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model_config = {
        "qpos_dim": 8,
        "policy_step_dim": 1,
        "camera_keys": args.camera_keys,
        "tactile_keys": args.tactile_keys,
        "feature_dim": 512,
        "visual_backbone": args.visual_backbone,
        "tactile_backbone": args.tactile_backbone,
        "visual_pretrained": args.visual_pretrained,
        "tactile_pretrained": args.tactile_pretrained,
        "visual_checkpoint": args.visual_checkpoint,
        "tactile_checkpoint": args.tactile_checkpoint,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiModalBC(model_config, statistics).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_losses = []
        for observation, action in train_loader:
            observation, action = move_batch(observation, action, device)
            target = action_target(model, observation, action)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model.forward_policy_action(observation)
                loss = nn.functional.smooth_l1_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(loss.item())
        training_loss = float(sum(training_losses) / max(len(training_losses), 1))
        validation_loss = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(json.dumps(record))
        metadata = {
            "epoch": epoch,
            "train_episodes": len(train_paths),
            "validation_episodes": len(validation_paths),
            "train_pairs": len(train_dataset),
            "validation_pairs": len(validation_dataset),
            "train_episode_ids": [path.stem for path in train_paths],
            "validation_episode_ids": [path.stem for path in validation_paths],
            "policy_step_stride": args.policy_step_stride,
        }
        torch.save(model.checkpoint(metadata), output_dir / "bc_last.pt")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.checkpoint(metadata), output_dir / "bc_best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(
                json.dumps(
                    {
                        "early_stop_epoch": epoch,
                        "best_validation_loss": best_validation_loss,
                    }
                )
            )
            break
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)


if __name__ == "__main__":
    main()
