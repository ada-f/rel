#!/usr/bin/env python3
"""
Test script for build_benchmark.py
Generates 3 questions of each type (REL-C1, REL-C2, REL-C3) and validates format.

To run this test:
    From parent directory: python -m chem_benchmark.test_build_benchmark
    Or: cd .. && python -m chem_benchmark.test_build_benchmark
"""

import json
import os
import random
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Ensure parent directory is in path for package imports
parent_dir = Path(__file__).parent.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import directly from submodules to avoid relative import issues
try:
    from chem_benchmark.isomer_sources import BUILTIN_ISOMER_UNIVERSES
    from chem_benchmark.molecule_bank import BankIndex, load_bank
    from chem_benchmark.tasks import (
        generate_q1_instance,
        generate_q2_instance,
        generate_q3_instance,
        generate_q4_instance,
    )
    
    def build_universe_by_formula(**kwargs):
        """Build universe by formula - reimplemented here to avoid import issues."""
        universe = {}
        if kwargs.get('include_builtin', True):
            for f, u in BUILTIN_ISOMER_UNIVERSES.items():
                universe[f] = list(u)
        return universe
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("\nPlease run this test from the parent directory using:")
    print("  python -m chem_benchmark.test_build_benchmark")
    print("\nOr ensure you're in the relational_reasoning directory and run:")
    print("  python -m chem_benchmark.test_build_benchmark")
    sys.exit(1)


def validate_unified_format(record, expected_task):
    """Validate that a record matches the unified format specification."""
    errors = []
    
    # Required fields
    required_fields = ["id", "domain", "task", "question", "answer"]
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Check domain
    if record.get("domain") != "chemistry":
        errors.append(f"Expected domain='chemistry', got '{record.get('domain')}'")
    
    # Check task
    if record.get("task") != expected_task:
        errors.append(f"Expected task='{expected_task}', got '{record.get('task')}'")
    
    # Check answer structure
    answer = record.get("answer", {})
    if not isinstance(answer, dict):
        errors.append("Answer must be a dictionary")
    
    # Check that molecules are in answer
    if "molecules" not in answer:
        errors.append("Answer must contain 'molecules' field")
    
    # Task-specific answer validation
    if expected_task == "REL-C1":
        if "label" not in answer:
            errors.append("REL-C1 answer must contain 'label' field")
        if answer.get("label") not in ["Yes", "No"]:
            errors.append(f"REL-C1 label must be 'Yes' or 'No', got '{answer.get('label')}'")
    
    elif expected_task == "REL-C2":
        if "smiles" not in answer:
            errors.append("REL-C2 answer must contain 'smiles' field")
    
    elif expected_task == "REL-C3":
        if "missing_smiles" not in answer:
            errors.append("REL-C3 answer must contain 'missing_smiles' field")
        if not isinstance(answer.get("missing_smiles"), list):
            errors.append("REL-C3 missing_smiles must be a list")

    elif expected_task == "REL-C4":
        if "selected_molecule_indices" not in answer:
            errors.append("REL-C4 answer must contain 'selected_molecule_indices' field")
        if not isinstance(answer.get("selected_molecule_indices"), list):
            errors.append("REL-C4 selected_molecule_indices must be a list")
        if "selected_motifs" not in answer:
            errors.append("REL-C4 answer must contain 'selected_motifs' field")
        if not isinstance(answer.get("selected_motifs"), dict):
            errors.append("REL-C4 selected_motifs must be a dict")

    return errors


def test_generate_samples():
    """Generate 3 samples of each task type and validate format."""
    print("=" * 60)
    print("Testing Chemistry Benchmark Generation")
    print("=" * 60)
    
    rng = random.Random(42)
    
    # Setup: Load molecule bank from chem_data
    # Try multiple possible paths
    possible_paths = [
        Path("chem_benchmark/molecule_bank_chembl_xlarge.json"),  # From project root
        Path(__file__).parent / "molecule_bank_chembl_xlarge.json",  # Relative to test file
    ]
    
    bank_path = None
    for path in possible_paths:
        if path.exists():
            bank_path = path
            break
    
    if bank_path and bank_path.exists():
        print(f"[INFO] Loading existing bank from {bank_path}")
        bank = load_bank(bank_path)
    else:
        print(f"[WARN] Molecule bank not found. Tried: {possible_paths}")
        print("[WARN] Skipping Q1a generation (requires molecule bank)")
        bank = []
    
    if len(bank) < 10:
        print("[WARN] Bank too small for testing. Skipping Q1a generation.")
        bank_index = None
    else:
        bank_index = BankIndex(bank)
    
    # Build isomer universe
    universe_by_formula = build_universe_by_formula(include_builtin=True)
    
    results = {
        "REL-C1": [],
        "REL-C2": [],
        "REL-C3": [],
        "REL-C4": [],
    }
    
    # Generate REL-C1 (Q2) samples
    print("\n[TEST] Generating 3 REL-C1 samples...")
    for i in range(3):
        try:
            inst = generate_q1_instance(
                instance_id=f"test_q1_{i:03d}",
                universe_by_formula=universe_by_formula,
                n_molecules=5,
                rng=rng,
                want_yes=(i % 2 == 0),
            )
            
            # Convert to unified format
            unified = {
                "id": inst.id,
                "domain": "chemistry",
                "task": "REL-C1",
                "question": inst.prompt,
                "answer": {
                    "label": inst.answer["label"],
                    "molecules": inst.molecules,
                },
                "metadata": inst.metadata.copy() if inst.metadata else {},
            }
            unified["metadata"]["original_task"] = inst.task
            
            # Validate
            errors = validate_unified_format(unified, "REL-C1")
            if errors:
                print(f"  [ERROR] Sample {i} validation failed:")
                for error in errors:
                    print(f"    - {error}")
            else:
                print(f"  [OK] Sample {i} validated successfully")
                results["REL-C1"].append(unified)
        except Exception as e:
            print(f"  [ERROR] Failed to generate REL-C1 sample {i}: {e}")
    
    # Generate REL-C2 (Q1a) samples
    if bank_index is not None:
        print("\n[TEST] Generating 3 REL-C2 (Q2) samples...")
        for i in range(3):
            try:
                inst = generate_q2_instance(
                    instance_id=f"test_q2_{i:03d}",
                    bank_index=bank_index,
                    n_molecules=5,
                    rng=rng,
                )
                
                # Convert to unified format
                unified = {
                    "id": inst.id,
                    "domain": "chemistry",
                    "task": "REL-C2",
                    "question": inst.prompt,
                    "answer": {
                        "smiles": inst.answer["smiles"],
                        "molecules": inst.molecules,
                    },
                    "metadata": inst.metadata.copy() if inst.metadata else {},
                }
                unified["metadata"]["original_task"] = inst.task
                
                # Validate
                errors = validate_unified_format(unified, "REL-C2")
                if errors:
                    print(f"  [ERROR] Sample {i} validation failed:")
                    for error in errors:
                        print(f"    - {error}")
                else:
                    print(f"  [OK] Sample {i} validated successfully")
                    results["REL-C2"].append(unified)
            except Exception as e:
                print(f"  [ERROR] Failed to generate REL-C2 sample {i}: {e}")
    else:
        print("\n[SKIP] Skipping REL-C2 generation (bank too small)")
    
    # Generate REL-C3 (Q3) samples
    print("\n[TEST] Generating 3 REL-C3 (Q3) samples...")
    for i in range(3):
        try:
            inst = generate_q3_instance(
                instance_id=f"test_q3_{i:03d}",
                universe_by_formula=universe_by_formula,
                n_molecules=5,
                rng=rng,
            )
            
            # Convert to unified format
            unified = {
                "id": inst.id,
                "domain": "chemistry",
                "task": "REL-C3",
                "question": inst.prompt,
                "answer": {
                    "missing_smiles": inst.answer["missing_smiles"],
                    "molecules": inst.molecules,
                },
                "metadata": inst.metadata.copy() if inst.metadata else {},
            }
            unified["metadata"]["original_task"] = inst.task
            
            # Validate
            errors = validate_unified_format(unified, "REL-C3")
            if errors:
                print(f"  [ERROR] Sample {i} validation failed:")
                for error in errors:
                    print(f"    - {error}")
            else:
                print(f"  [OK] Sample {i} validated successfully")
                results["REL-C3"].append(unified)
        except Exception as e:
            print(f"  [ERROR] Failed to generate REL-C3 sample {i}: {e}")

    # Generate REL-C4 (Q4) samples
    if bank_index is not None:
        print("\n[TEST] Generating 3 REL-C4 samples...")
        constraint_types = ["total_carboxylic_acids", "total_aromatic_rings", "total_primary_amines"]
        for i in range(3):
            try:
                constraint_type = constraint_types[i % len(constraint_types)]

                inst = generate_q4_instance(
                    instance_id=f"test_q4_{i:03d}",
                    bank_index=bank_index,
                    n_molecules=5,
                    k_molecules=3,
                    constraint_type=constraint_type,
                    target_value=None,  # Will be determined during generation
                    min_motif_atoms=6,
                    rng=rng,
                    max_attempts=100,
                )

                # Convert to unified format
                unified = {
                    "id": inst.id,
                    "domain": "chemistry",
                    "task": "REL-C4",
                    "question": inst.prompt,
                    "answer": {
                        "selected_molecule_indices": inst.answer["selected_molecule_indices"],
                        "selected_motifs": inst.answer["selected_motifs"],
                        "molecules": inst.molecules,
                    },
                    "metadata": inst.metadata.copy() if inst.metadata else {},
                }
                unified["metadata"]["original_task"] = inst.task

                # Validate
                errors = validate_unified_format(unified, "REL-C4")
                if errors:
                    print(f"  [ERROR] Sample {i} validation failed:")
                    for error in errors:
                        print(f"    - {error}")
                else:
                    print(f"  [OK] Sample {i} validated successfully")
                    results["REL-C4"].append(unified)
            except Exception as e:
                print(f"  [ERROR] Failed to generate REL-C4 sample {i}: {e}")
    else:
        print("\n[SKIP] Skipping REL-C4 generation (bank too small)")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for task, samples in results.items():
        print(f"{task}: {len(samples)}/3 samples generated and validated")
    
    # Write test output to temporary file
    with NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for task_samples in results.values():
            for sample in task_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"\n[INFO] Test output written to: {f.name}")
    
    # Check if all tests passed
    all_passed = all(len(samples) == 3 for samples in results.values() if results.get("REL-C2") or len(results["REL-C2"]) == 0)
    if not all_passed:
        c1_ok = len(results["REL-C1"]) == 3
        c2_ok = len(results["REL-C2"]) == 3 or bank_index is None
        c3_ok = len(results["REL-C3"]) == 3
        c4_ok = len(results["REL-C4"]) == 3 or bank_index is None
        all_passed = c1_ok and c2_ok and c3_ok and c4_ok
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(test_generate_samples())
