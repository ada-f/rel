#!/usr/bin/env python3
"""
Build algebra benchmark dataset in unified JSONL format.
Generates REL-A1, REL-A2, REL-A3, and REL-A4 Raven's Progressive Matrix questions.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Any

from algebra_benchmark.generators import generate_dataset
from algebra_benchmark.rpm_numeric import build_query

QUESTION_PREFIX = (
    "Complete the Raven's progressive matrix. Only return the missing panel index (0-7)!\n"
)


def convert_to_unified_format(sample: dict, task: str, index: int) -> dict:
    """Convert an algebra sample to unified JSONL format."""
    # Build question text using the same method as the benchmark
    prefix = QUESTION_PREFIX
    question_text = build_query(sample, prefix=prefix)
    
    unified = {
        "id": f"algebra_{task}_{index:05d}",
        "domain": "algebra",
        "task": task,
        "question": question_text,
        "answer": {
            "target": sample.get("target", 0)
        },
        "metadata": {
            "panels": sample.get("panels", []),
            "choices": sample.get("choices", [])
        }
    }
    
    return unified


def main():
    ap = argparse.ArgumentParser(description="Build algebra benchmark in unified JSONL format")
    ap.add_argument("--out", type=str, default="REL/algebra/REL-A1.jsonl", help="Output JSONL file path")
    ap.add_argument("--task", type=str, required=True, help="Task: REL-A1, REL-A2, REL-A3, or REL-A4")
    ap.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    ap.add_argument("--gridsize", type=int, default=3, help="Matrix size (default: 3)")
    ap.add_argument("--maxval", type=int, default=1000, help="Max value for numeric entries (default: 1000)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = ap.parse_args()
    
    # Validate task
    valid_tasks = ["REL-A1", "REL-A2", "REL-A3", "REL-A4"]
    if args.task not in valid_tasks:
        raise ValueError(f"Task must be one of {valid_tasks}, got {args.task}")
    
    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Generating {args.num_samples} samples for {args.task}...")
    print(f"[INFO] Gridsize: {args.gridsize}, Maxval: {args.maxval}")
    
    # Generate samples
    samples = generate_dataset(
        args.task,
        args.num_samples,
        args.gridsize,
        args.maxval,
        args.seed,
    )
    
    print(f"[INFO] Generated {len(samples)} samples")
    
    # Convert to unified format and write
    print(f"[INFO] Converting to unified format and writing to {out_path}...")
    count = 0
    errors = 0
    
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for i, sample in enumerate(samples):
            try:
                unified = convert_to_unified_format(sample, args.task, i)
                f_out.write(json.dumps(unified, ensure_ascii=False) + '\n')
                count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to convert sample {i}: {e}")
                errors += 1
                continue
    
    print(f"[INFO] Wrote {count} records to {out_path}")
    if errors > 0:
        print(f"[WARN] {errors} records failed to convert")


if __name__ == "__main__":
    main()
