from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from .molecule_bank import (
    BankIndex,
    clean_chembl_records,
    fetch_chembl_max_phase_smiles,
    load_bank,
    save_bank,
    select_diverse_subset_maxmin,
)
from .isomer_sources import BUILTIN_ISOMER_UNIVERSES, get_isomer_universe
from .tasks import (
    generate_q1_instance,
    generate_q2_instance,
    generate_q3_instance,
    generate_q4_instance,
)


def build_universe_by_formula(
    *,
    include_builtin: bool = True,
    pubchem_formulas: List[str] | None = None,
    use_pubchem: bool = False,
    cache_dir: Path = Path("cache/isomers"),
    refresh: bool = False,
) -> Dict[str, List[str]]:
    universe: Dict[str, List[str]] = {}

    if include_builtin:
        for f, u in BUILTIN_ISOMER_UNIVERSES.items():
            universe[f] = list(u)

    if use_pubchem and pubchem_formulas:
        for f in pubchem_formulas:
            try:
                u = get_isomer_universe(
                    f,
                    source="pubchem",
                    cache_dir=cache_dir,
                    refresh=refresh,
                )
                # Keep only manageable universes by default; you can override later in task config
                if len(u) >= 5:
                    universe[f] = u
            except Exception as e:
                print(f"[WARN] PubChem universe fetch failed for {f}: {e}")

    return universe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/dataset.jsonl")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--bank_path", type=str, default="chem_data/molecule_bank_chembl_xlarge.json")
    ap.add_argument("--bank_size", type=int, default=200)
    ap.add_argument("--rebuild_bank", action="store_true")
    ap.add_argument("--chembl_max_records", type=int, default=500)

    ap.add_argument("--n_values", type=int, nargs="+", default=[5, 10, 30])

    ap.add_argument("--q1_per_n", type=int, default=10)
    ap.add_argument("--q2_per_n", type=int, default=10)
    ap.add_argument("--q3_per_n", type=int, default=10)
    ap.add_argument("--q4_per_n", type=int, default=10)

    ap.add_argument("--use_pubchem", action="store_true")
    ap.add_argument(
        "--pubchem_formulas",
        type=str,
        nargs="+",
        default=["C4H8O", "C4H10O", "C5H10O", "C5H10O2", "C6H12O", "C6H12O2"],
    )
    ap.add_argument("--pubchem_refresh", action="store_true")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bank_path = Path(args.bank_path)
    if args.rebuild_bank or (not bank_path.exists()):
        print("[INFO] Fetching ChEMBL phase 1-4 molecules...")
        raw = fetch_chembl_max_phase_smiles(max_records=args.chembl_max_records)
        print(f"[INFO] Raw fetched: {len(raw)}")

        cleaned = clean_chembl_records(raw, min_heavy_atoms=15, max_heavy_atoms=60)
        print(f"[INFO] Cleaned drug-like molecules: {len(cleaned)}")

        diverse = select_diverse_subset_maxmin(cleaned, args.bank_size, seed=args.seed)
        print(f"[INFO] Selected diverse subset: {len(diverse)}")
        save_bank(bank_path, diverse)
        bank = diverse
    else:
        print("[INFO] Loading existing molecule bank...")
        bank = load_bank(bank_path)
        print(f"[INFO] Loaded bank size: {len(bank)}")

    bank_index = BankIndex(bank)

    # Print scaffold statistics to verify complexity filtering
    print(f"[INFO] Scaffold statistics:")
    print(f"       Total molecules in bank: {len(bank)}")
    print(f"       Scaffold families (complex only): {len(bank_index.scaffold_to_indices)}")
    if bank_index.scaffold_to_indices:
        family_sizes = [len(idxs) for idxs in bank_index.scaffold_to_indices.values()]
        print(f"       Largest scaffold family: {max(family_sizes)} molecules")
        print(f"       Average family size: {sum(family_sizes) / len(family_sizes):.1f} molecules")
        print(f"       Families with >=50 molecules: {sum(1 for s in family_sizes if s >= 50)}")
        print(f"       Families with >=25 molecules: {sum(1 for s in family_sizes if s >= 25)}")
        print(f"       Families with >=10 molecules: {sum(1 for s in family_sizes if s >= 10)}")
        print(f"       Families with >=5 molecules: {sum(1 for s in family_sizes if s >= 5)}")

    universe_by_formula = build_universe_by_formula(
        include_builtin=True,
        use_pubchem=args.use_pubchem,
        pubchem_formulas=args.pubchem_formulas,
        refresh=args.pubchem_refresh,
    )
    print(f"[INFO] Isomer universes available: {sorted(universe_by_formula.keys())}")

    instances = []
    counter = 0

    # Track statistics for each task type
    q1_attempted = 0
    q1_created = 0
    q2_attempted = 0
    q2_created = 0
    q3_attempted = 0
    q3_created = 0
    q4_attempted = 0
    q4_created = 0

    # Track unique molecule combinations to avoid duplicates
    seen_q1_combos = set()
    seen_q2_combos = set()
    seen_q3_combos = set()
    seen_q4_combos = set()

    # Q1/Q2/Q3/Q4 generation stratified by n_values
    for n in args.n_values:
        # Q2 generation 
        max_retries = 10  # Retry up to 10 times to find unique combination
        for _ in range(args.q2_per_n):
            counter += 1
            q2_attempted += 1

            # Try to generate a unique instance
            for retry in range(max_retries):
                try:
                    inst = generate_q2_instance(
                        instance_id=f"q2_n{n}_{counter:05d}",
                        bank_index=bank_index,
                        n_molecules=n,
                        rng=rng,
                    )

                    # Check for uniqueness (use sorted tuple for order-independent comparison)
                    combo_key = tuple(sorted(inst.molecules))
                    if combo_key in seen_q2_combos:
                        if retry < max_retries - 1:
                            continue  # Try again
                        else:
                            print(f"[WARN] Q2 instance q2_n{n}_{counter:05d}: Could not find unique combination after {max_retries} retries")
                            break

                    # Unique combination found!
                    seen_q2_combos.add(combo_key)
                    instances.append(inst)
                    q2_created += 1
                    break  # Success, move to next instance

                except (RuntimeError, ValueError) as e:
                    if retry == max_retries - 1:
                        print(f"[WARN] Failed to generate Q2 instance q2_n{n}_{counter:05d}: {e}")
                    # Otherwise try again

        # Q1 generation
        for q1_idx in range(args.q1_per_n):
            counter += 1
            q1_attempted += 1
            want_yes = q1_idx % 2 == 0

            # Try to generate a unique instance
            for retry in range(max_retries):
                try:
                    inst = generate_q1_instance(
                        instance_id=f"q1_n{n}_{counter:05d}",
                        universe_by_formula=universe_by_formula,
                        n_molecules=n,
                        rng=rng,
                        want_yes=want_yes,
                    )

                    # Check for uniqueness
                    combo_key = tuple(sorted(inst.molecules))
                    if combo_key in seen_q1_combos:
                        if retry < max_retries - 1:
                            continue  # Try again
                        else:
                            print(f"[WARN] Q1 instance q1_n{n}_{counter:05d}: Could not find unique combination after {max_retries} retries")
                            break

                    # Unique combination found!
                    seen_q1_combos.add(combo_key)
                    instances.append(inst)
                    q1_created += 1
                    break  # Success, move to next instance

                except (RuntimeError, ValueError) as e:
                    if retry == max_retries - 1:
                        print(f"[WARN] Failed to generate Q1 instance q1_n{n}_{counter:05d}: {e}")
                    # Otherwise try again

        # For Q3, "n" here means number of GIVEN molecules (universe must be larger)
        for _ in range(args.q3_per_n):
            counter += 1
            q3_attempted += 1

            # Try to generate a unique instance
            for retry in range(max_retries):
                try:
                    inst = generate_q3_instance(
                        instance_id=f"q3_given{n}_{counter:05d}",
                        universe_by_formula=universe_by_formula,
                        n_molecules=n,
                        rng=rng,
                    )

                    # Check for uniqueness
                    combo_key = tuple(sorted(inst.molecules))
                    if combo_key in seen_q3_combos:
                        if retry < max_retries - 1:
                            continue  # Try again
                        else:
                            print(f"[WARN] Q3 instance q3_given{n}_{counter:05d}: Could not find unique combination after {max_retries} retries")
                            break

                    # Unique combination found!
                    seen_q3_combos.add(combo_key)
                    instances.append(inst)
                    q3_created += 1
                    break  # Success, move to next instance

                except (RuntimeError, ValueError) as e:
                    if retry == max_retries - 1:
                        print(f"[WARN] Failed to generate Q3 instance q3_given{n}_{counter:05d}: {e}")
                    # Otherwise try again

    # Q4 generation (constraint satisfaction)
    print(f"[INFO] Starting Q4 generation")

    # Define constraint types
    constraint_types = [
        "total_carboxylic_acids",
        "total_aromatic_rings",
        "total_primary_amines",
        "total_alcohols",
        "total_ketones",
    ]

    for n in args.n_values:
        # k is approximately 50% of n
        k = round(n * 0.5)

        for q4_idx in range(args.q4_per_n):
            counter += 1
            q4_attempted += 1

            # Cycle through constraint types
            constraint_type = constraint_types[q4_idx % len(constraint_types)]

            # Try to generate a unique instance
            for retry in range(max_retries):
                try:
                    inst = generate_q4_instance(
                        instance_id=f"q4_n{n}_k{k}_{counter:05d}",
                        bank_index=bank_index,
                        n_molecules=n,
                        k_molecules=k,
                        constraint_type=constraint_type,
                        target_value=None,  # Will be determined during generation
                        min_motif_atoms=6,
                        rng=rng,
                        max_attempts=100,
                    )

                    if inst is None:
                        continue  # Failed to generate, try again

                    # Check for uniqueness
                    combo_key = tuple(sorted(inst.molecules))
                    if combo_key in seen_q4_combos:
                        if retry < max_retries - 1:
                            continue  # Try again
                        else:
                            print(f"[WARN] Q4 instance q4_n{n}_k{k}_{counter:05d}: Could not find unique combination after {max_retries} retries")
                            break

                    # Unique combination found!
                    seen_q4_combos.add(combo_key)
                    instances.append(inst)
                    q4_created += 1
                    break  # Success, move to next instance

                except (RuntimeError, ValueError) as e:
                    if retry == max_retries - 1:
                        print(f"[WARN] Failed to generate Q4 instance q4_n{n}_k{k}_{counter:05d}: {e}")
                    # Otherwise try again

    # Print statistics
    print(f"[INFO] Q1 instances: {q1_created} created out of {q1_attempted} attempted (unique combinations: {len(seen_q1_combos)})")
    print(f"[INFO] Q2 instances: {q2_created} created out of {q2_attempted} attempted (unique combinations: {len(seen_q2_combos)})")
    print(f"[INFO] Q3 instances: {q3_created} created out of {q3_attempted} attempted (unique combinations: {len(seen_q3_combos)})")
    print(f"[INFO] Q4 instances: {q4_created} created out of {q4_attempted} attempted (unique combinations: {len(seen_q4_combos)})")
    total_attempted = q1_attempted + q2_attempted + q3_attempted + q4_attempted
    total_unique = len(seen_q1_combos) + len(seen_q2_combos) + len(seen_q3_combos) + len(seen_q4_combos)
    print(f"[INFO] Total instances: {len(instances)} created out of {total_attempted} attempted")
    print(f"[INFO] Total unique molecule combinations: {total_unique}")

    # Task name mapping to unified format
    TASK_MAPPING = {
        "q2_isomer_set_yes_no": "REL-C1",
        "q1_largest_common_motif_chembl": "REL-C2",
        "q3_missing_isomers": "REL-C3",
        "q4_constraint_satisfaction_selection": "REL-C4",
    }

    def convert_to_unified_format(inst):
        """Convert a BenchmarkInstance to unified JSONL format."""
        # Map task name to REL-C* format
        unified_task = TASK_MAPPING.get(inst.task, inst.task)

        # Build unified answer structure
        unified_answer = {}

        # Add molecules to answer (present in most chemistry tasks)
        if inst.molecules:
            unified_answer["molecules"] = inst.molecules

        # Add task-specific answer fields
        if "label" in inst.answer:
            unified_answer["label"] = inst.answer["label"]
        if "smiles" in inst.answer:
            unified_answer["smiles"] = inst.answer["smiles"]
        if "missing_smiles" in inst.answer:
            unified_answer["missing_smiles"] = inst.answer["missing_smiles"]
        if "selected_molecule_indices" in inst.answer:
            unified_answer["selected_molecule_indices"] = inst.answer["selected_molecule_indices"]
        if "selected_motifs" in inst.answer:
            unified_answer["selected_motifs"] = inst.answer["selected_motifs"]

        # Build unified record
        unified_record = {
            "id": inst.id,
            "domain": "chemistry",
            "task": unified_task,
            "question": inst.prompt,
            "answer": unified_answer,
            "metadata": inst.metadata.copy() if inst.metadata else {},
        }

        # Add original task name to metadata
        unified_record["metadata"]["original_task"] = inst.task

        return unified_record

    # Write JSONL in unified format
    with out_path.open("w", encoding="utf-8") as f:
        for inst in instances:
            unified_record = convert_to_unified_format(inst)
            f.write(
                json.dumps(unified_record, ensure_ascii=False) + "\n"
            )

    print(f"[INFO] Wrote {len(instances)} instances to {out_path} in unified format")


if __name__ == "__main__":
    main()
