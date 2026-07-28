import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize available policy evaluation results.")
    parser.add_argument("run_root")
    parser.add_argument("--evaluation-dir", default="evaluation")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    comparison = {}
    for algorithm in ("bc", "sac", "ppo"):
        result_path = run_root / args.evaluation_dir / algorithm / "evaluation.json"
        if not result_path.is_file():
            continue
        with open(result_path, "r", encoding="utf-8") as result_file:
            result = json.load(result_file)
        comparison[algorithm] = {
            "episodes": result["episodes"],
            "successes": result["successes"],
            "success_rate": result["success_rate"],
            "mean_reward": result["mean_reward"],
        }
    if not comparison:
        raise FileNotFoundError(
            f"No evaluation results found under {run_root / args.evaluation_dir}"
        )
    best = max(
        comparison,
        key=lambda name: (
            comparison[name]["success_rate"],
            comparison[name]["mean_reward"],
        ),
    )
    summary = {"best_success_rate": best, "comparison": comparison}
    output_path = run_root / "comparison.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
