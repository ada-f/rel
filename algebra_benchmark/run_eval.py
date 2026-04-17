#
# Run evaluation on algebra_benchmark numerical datasets.
# Loads config + dataset, builds prompts, calls model (stub or callable), computes accuracy.
# Compatible with the rpmllm mamba environment (Python 3.10).
#

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from loader import load_config_and_dataset
from rpm_numeric import build_query
from solver_pred import guard_answer, majority_vote, text2num


def _stub_model(prompt: str, query: str) -> str:
    """Stub model: returns 'Answer 0' for every query (for testing pipeline)."""
    return "Answer 0"


def run_eval(
    config_path: str | Path,
    model_fn: Callable[[str, str], str] | None = None,
    *,
    prompt: str = "Complete the Raven's progressive matrix:",
    prefix: str = (
        "Complete the Raven's progressive matrix. Only return the missing panel index (0-7)!\n"
    ),
    n_return: int = 1,
) -> dict[str, Any]:
    """
    Load config and dataset, run model on each sample, compute accuracy.

    model_fn(prompt, query) -> response string. If None, uses stub that returns "Answer 0".
    Returns dict with keys: correct, total, accuracy, predictions (list of pred indices).
    """
    cfg, samples = load_config_and_dataset(config_path)
    data = cfg["data"]
    ntest = min(data.get("ntest", len(samples)), len(samples))
    samples = samples[:ntest]
    model_fn = model_fn or _stub_model
    n_attr = data.get("nattr", 1)

    correct = 0
    predictions = []
    for i, sample in enumerate(samples):
        query = build_query(sample, prefix=prefix)
        response = model_fn(prompt, query)
        preds = text2num(response, n_attr, n_return=n_return)
        pred_idx = guard_answer(majority_vote(preds))
        if isinstance(pred_idx, list):
            pred_idx = pred_idx[0] if pred_idx else 0
        predictions.append(pred_idx)
        target = sample["target"]
        if pred_idx == target:
            correct += 1

    total = len(samples)
    accuracy = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "predictions": predictions,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Run evaluation on algebra_benchmark dataset.")
    p.add_argument("config", type=str, help="Path to config.yml (or config.json).")
    p.add_argument("--prompt", type=str, default="Complete the Raven's progressive matrix:", help="System/main prompt.")
    p.add_argument(
        "--prefix",
        type=str,
        default=(
            "Complete the Raven's progressive matrix. "
            "Only return the missing panel index (0-7)!\n"
        ),
        help="Query prefix.",
    )
    p.add_argument("--stub", action="store_true", help="Use stub model (returns Answer 0) for testing.")
    args = p.parse_args()

    model_fn = _stub_model if args.stub else None
    result = run_eval(
        args.config,
        model_fn=model_fn,
        prompt=args.prompt,
        prefix=args.prefix,
    )
    print(f"Correct: {result['correct']} / {result['total']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
