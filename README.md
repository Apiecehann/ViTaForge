<h1 align="center">ViTaForge</h1>

<p align="center">
  An extended UniVTAC workspace for XenseWS visuo-tactile manipulation,
  reproducible data collection, and BC/SAC policy learning.
</p>

ViTaForge is developed from [UniVTAC](https://github.com/univtac/UniVTAC),
which is built on NVIDIA Isaac Lab and the TacEx UIPC tactile simulator. This
branch keeps the original UniVTAC workflow while adding:

- XenseWS RGB, marker, depth, and pose observations;
- adaptive tactile grasping and task-specific contact handling;
- eight XenseWS manipulation tasks with a shared final configuration;
- dataset validation, behavior cloning (BC/SFT), SAC/PPO fine-tuning, policy
  evaluation, trajectory visualization, and report generation.

Upstream resources: [paper](https://arxiv.org/abs/2602.10093) |
[project website](https://univtac.github.io/) |
[dataset](https://huggingface.co/datasets/byml/UniVTAC)

## Environment

The recommended project environment is a Conda environment named `UniVTAC`.
The repository contains modified TacEx/UIPC sources under `third_party/TacEx`;
install those local sources instead of replacing them with the public TacEx
package.

The current all-in-one installer reproduces the upstream-tested stack with
Python 3.10, Isaac Sim 4.5, and Isaac Lab 2.1.1:

```bash
git clone --branch minnan https://github.com/Apiecehann/ViTaForge.git
cd ViTaForge
conda activate base
bash scripts/install.sh
conda activate UniVTAC
```

See [docs/Installation.md](docs/Installation.md) for the manual TacEx, UIPC,
cuRobo, and Isaac Lab setup.

### Isaac Sim 5.1

Isaac Sim 5.1 can coexist in a separate Python 3.11 Conda prefix. Do not mix
its `site-packages` or shared libraries with the Python 3.10/Isaac Sim 4.5
environment. To use 5.1, install a compatible Isaac Lab version and rebuild
the editable TacEx/UIPC packages with the 5.1 interpreter. The current
`scripts/install.sh` does not automate that migration yet.

Before a long run, verify the active interpreter and package versions:

```bash
python --version
python -c "import importlib.metadata as m; print(m.version('isaacsim')); print(m.version('torch'))"
```

## XenseWS Task Suite

The final shared configuration is `task_config/xense_8tasks.yml`.

| Task module | Manipulation objective |
|---|---|
| `grasp_half_cylinder_in_clutter` | Grasp a target among distractors |
| `insert_half_cylinder_into_box` | Insert a shape into the matching opening |
| `insert_usb` | Grasp and insert a USB connector |
| `place_wooden_cube_on_yellow_area` | Place a cube in the target area |
| `pour_ball_to_cup` | Grasp, carry, and pour into a target cup |
| `pull_drawer` | Grasp and pull a drawer |
| `swap_cup_order` | Rearrange colored cups |
| `turn_gear_pair` | Grasp and rotate the gear pair |

The associated USD/OBJ assets are stored in `assets/objects/task_0724/`.
The final configuration enables XenseWS tactile `rgb`, `rgb_marker`, `marker`,
`depth`, and `pose` observations and writes generated episodes below
`data_xense_8tasks/`.

## Data Collection

Activate the environment and run one task from the repository root. The final
three arguments below select start seed 0, maximum seed 0, and one episode:

```bash
conda activate UniVTAC
bash collect_data.sh grasp_half_cylinder_in_clutter \
  xense_8tasks 0 0 0 1
```

Use the same config with any module in the table. Parallel collection accepts
an explicit worker count and episode target:

```bash
python scripts/parallel_collect_data.py \
  grasp_half_cylinder_in_clutter xense_8tasks \
  --workers 2 --episodes 8 --gpu 0
```

Keep the worker count conservative for XenseWS/UIPC runs: every process owns
an Isaac Sim application and tactile solver. Set `render_frequency: 1` in the
YAML only when an interactive Isaac Sim window is needed.

The original UniVTAC collection details and HDF5 layout are described in
[docs/Collection.md](docs/Collection.md). Generated datasets, videos,
checkpoints, logs, and diagnostic frames are intentionally ignored by Git.

## BC And RL

Validate a collected HDF5 directory before training:

```bash
python scripts/validate_rl_dataset.py data_xense_8tasks/hdf5 \
  --expected-episodes 100 --output dataset_validation.json
```

Train a behavior-cloning policy:

```bash
python scripts/train_bc.py data_xense_8tasks/hdf5 policy/RL/runs/xense_bc \
  --epochs 30 --batch-size 32 --image-size 128 \
  --visual-pretrained --tactile-pretrained
```

Fine-tune the BC checkpoint with SAC and offline/online BC regularization:

```bash
python scripts/train_rl.py \
  grasp_half_cylinder_in_clutter xense_8tasks \
  policy/RL/runs/xense_bc/bc_best.pt policy/RL/runs/xense_sac \
  --algorithm sac --total-timesteps 20000 \
  --bc-dataset-root data_xense_8tasks \
  --control-mode direct --action-repeat 2 --step-limit 120 \
  --control-gripper --force-control
```

Evaluate SFT/BC, SAC, or PPO with held-out seeds:

```bash
python scripts/eval_rl.py \
  grasp_half_cylinder_in_clutter xense_8tasks \
  policy/RL/runs/xense_bc/bc_best.pt policy/RL/runs/xense_eval \
  --algorithm sac --model-path policy/RL/runs/xense_sac/sac/final_model.zip \
  --episodes 20 --start-seed 20000 --save-traces \
  --control-mode direct --action-repeat 2 --step-limit 120 \
  --control-gripper --force-control
```

`scripts/run_grasp_rl_pipeline.sh` is the end-to-end reference pipeline used
for the 97-episode GelSight grasp dataset. It validates the dataset, trains BC,
applies train/held-out success gates, fine-tunes SAC, evaluates the policy, and
generates a report. Override its machine-specific inputs with environment
variables:

```bash
SOURCE_DATASET_ROOT=/path/to/grasp_dataset \
RUN_ROOT=/path/to/output/run \
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
bash scripts/run_grasp_rl_pipeline.sh 20000 20
```

## Original Policy Baselines

The upstream ACT, Ablation, and ViTAL baselines remain under `policy/`. Their
shared evaluation entry points are:

```bash
bash eval_policy.sh <task_name> <task_config> <policy_config> <gpu_id>
bash parallel_eval.sh <task_name> <task_config> <policy_config> <gpu_id> \
  [num_processes] [total_num]
```

See [docs/Deploy.md](docs/Deploy.md) for custom policy deployment.

## Repository Hygiene

Do not commit collected data, videos, checkpoints, diagnostic frames, or local
`.codex_*` work directories. Keep reusable source, task YAML, and compact
assets in Git; publish large datasets and trained models through an artifact
store.

## Citation And License

ViTaForge inherits UniVTAC's MIT license. If this code is useful in your work,
please cite the original UniVTAC project:

```bibtex
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```

See [LICENSE](LICENSE) for details.
