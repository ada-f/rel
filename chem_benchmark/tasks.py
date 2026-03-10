from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS

from .molecule_bank import BankIndex, MoleculeRecord
from .solvers import solve_q1_largest_common_motif, solve_q2_is_constitutional_isomer_set, solve_q3_missing_isomers
from .rdkit_utils import canonical_smiles, canonical_smiles_from_smiles, mol_from_smiles, mol_formula


@dataclass(frozen=True)
class BenchmarkInstance:
    id: str
    task: str
    n_molecules: int
    molecules: List[str]
    prompt: str
    answer: Dict[str, Any]
    metadata: Dict[str, Any]


def _format_smiles_list(smiles: Sequence[str]) -> str:
    lines = []
    for i, s in enumerate(smiles, start=1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def verify_motif_in_all_molecules(motif_smiles: str, molecules: Sequence[str]) -> bool:
    """
    Verify that the motif is actually a substructure of all molecules.
    Returns True if valid, False otherwise.
    """
    motif_mol = mol_from_smiles(motif_smiles)
    if motif_mol is None:
        return False

    for mol_smiles in molecules:
        mol = mol_from_smiles(mol_smiles)
        if mol is None:
            return False

        # Check if motif is a substructure
        if not mol.HasSubstructMatch(motif_mol):
            return False

    return True


def build_q1_prompt(smiles: Sequence[str]) -> str:
    # Mirrors your Q1 description: "largest continuous common chemical motif" and answer as <smiles>.
    return (
        "Given the following list of SMILES, what is the largest *connected* common chemical motif "
        "(maximum common substructure) present in every molecule?\n"
        "Rules:\n"
        "- The motif must be a single connected fragment.\n"
        "- Do NOT tautomerize molecules.\n"
        "- Ignore stereochemistry unless it is explicitly encoded and required.\n\n"
        "SMILES:\n"
        f"{_format_smiles_list(smiles)}\n\n"
        "Return your final answer as a single SMILES wrapped exactly like:\n"
        "<smiles>YOUR_SMILES_HERE</smiles>\n"
        "No explanation."
    )


def build_q2_prompt(smiles: Sequence[str]) -> str:
    return (
        "Is this list of molecules a set of *constitutional isomers* (same molecular formula, different connectivity)?\n\n"
        "SMILES:\n"
        f"{_format_smiles_list(smiles)}\n\n"
        "Return exactly one of:\n"
        "<Yes>\n"
        "or\n"
        "<No>\n"
        "No explanation."
    )


def build_q3_prompt(given: Sequence[str]) -> str:
    return (
        "Given the following list of constitutional isomers, complete the set by identifying the missing constitutional isomers.\n\n"
        "Given SMILES:\n"
        f"{_format_smiles_list(given)}\n\n"
        "Return the missing molecules as SMILES, one per line, each wrapped exactly like:\n"
        "<smiles>YOUR_SMILES_HERE</smiles>\n"
        "No explanation."
    )


def generate_q1a_instance(
    *,
    instance_id: str,
    bank_index: BankIndex,
    n_molecules: int,
    rng,
    min_mcs_atoms: int = 8,
    max_attempts: int = 200,
    mcs_timeout_s: int = 15,
) -> BenchmarkInstance:
    """
    Q1a: Similarity-based sampling from ChEMBL molecules.
    Uses Tanimoto similarity (0.35-0.90) to find related molecules.

    Suitable for smaller n (5-20) where diverse molecules can still share meaningful motifs.
    """
    # Adjust parameters for larger molecule sets
    adaptive_timeout = max(mcs_timeout_s, n_molecules * 2)

    # Scale min_mcs_atoms for larger n
    if n_molecules <= 10:
        adaptive_min_atoms = min_mcs_atoms
    elif n_molecules <= 20:
        adaptive_min_atoms = min_mcs_atoms
    else:
        adaptive_min_atoms = min_mcs_atoms

    # More attempts for larger n
    adaptive_max_attempts = max_attempts if n_molecules <= 10 else max_attempts * 2

    for attempt in range(adaptive_max_attempts):
        group = bank_index.sample_similar_group(n_molecules, rng=rng, force_scaffold_sampling=False)
        mcs = solve_q1_largest_common_motif(group, timeout_s=adaptive_timeout)
        if mcs is None:
            continue
        if mcs.num_atoms < adaptive_min_atoms:
            continue

        # Verify that the motif is actually present in all molecules
        motif_smiles = mcs.motif_smiles
        num_atoms = mcs.num_atoms
        num_bonds = mcs.num_bonds
        corrected = False

        if not verify_motif_in_all_molecules(motif_smiles, group):
            continue
           
        # Check if (possibly corrected) motif still meets minimum size
        if num_atoms < adaptive_min_atoms:
            continue

        prompt = build_q1_prompt(group)
        return BenchmarkInstance(
            id=instance_id,
            task="q1a_largest_common_motif_chembl",
            n_molecules=n_molecules,
            molecules=list(group),
            prompt=prompt,
            answer={"smiles": motif_smiles},
            metadata={
                "mcs_num_atoms": num_atoms,
                "mcs_num_bonds": num_bonds,
                "attempt": attempt,
                "sampling_strategy": "similarity",
                "adaptive_min_atoms": adaptive_min_atoms,
                "corrected": corrected,
            },
        )
    raise RuntimeError(f"Failed to generate a valid Q1a instance after {adaptive_max_attempts} attempts (min_mcs_atoms={adaptive_min_atoms})")


def generate_q1b_instance(
    *,
    instance_id: str,
    bank_index: BankIndex,
    n_molecules: int,
    rng,
    min_mcs_atoms: int = 6,
    max_attempts: int = 200,
    mcs_timeout_s: int = 15,
) -> BenchmarkInstance:
    """
    Q1b: Scaffold-based sampling from ChEMBL molecules.
    Samples molecules sharing a common Murcko scaffold.

    Suitable for larger n (20-50) where scaffold ensures meaningful shared core.
    Default min_mcs_atoms=6 because scaffolds typically give 5-10 atom cores.
    """
    # Adjust parameters for larger molecule sets
    if n_molecules <= 20:
        adaptive_timeout = max(30, n_molecules * 2)
    else:
        adaptive_timeout = max(60, n_molecules * 3)

    # Scale min_mcs_atoms: scaffold core size varies by family
    if n_molecules <= 20:
        adaptive_min_atoms = min_mcs_atoms  # Expect 6+ atoms
    elif n_molecules <= 35:
        adaptive_min_atoms = max(5, min_mcs_atoms - 1)
    else:  # n > 35
        adaptive_min_atoms = 5  # Large families may have smaller cores

    # More attempts for larger n
    if n_molecules <= 20:
        adaptive_max_attempts = max_attempts
    else:
        adaptive_max_attempts = max_attempts * 2

    for attempt in range(adaptive_max_attempts):
        group = bank_index.sample_similar_group(n_molecules, rng=rng, force_scaffold_sampling=True)
        mcs = solve_q1_largest_common_motif(group, timeout_s=adaptive_timeout)
        if mcs is None:
            continue
        if mcs.num_atoms < adaptive_min_atoms:
            continue

        # Verify that the motif is actually present in all molecules
        motif_smiles = mcs.motif_smiles
        num_atoms = mcs.num_atoms
        num_bonds = mcs.num_bonds
        corrected = False

        if not verify_motif_in_all_molecules(motif_smiles, group):
            continue

        # Check if (possibly corrected) motif still meets minimum size
        if num_atoms < adaptive_min_atoms:
            continue

        prompt = build_q1_prompt(group)
        return BenchmarkInstance(
            id=instance_id,
            task="q1b_largest_common_motif_scaffold",
            n_molecules=n_molecules,
            molecules=list(group),
            prompt=prompt,
            answer={"smiles": motif_smiles},
            metadata={
                "mcs_num_atoms": num_atoms,
                "mcs_num_bonds": num_bonds,
                "attempt": attempt,
                "sampling_strategy": "scaffold",
                "adaptive_min_atoms": adaptive_min_atoms,
                "corrected": corrected,
            },
        )
    raise RuntimeError(f"Failed to generate a valid Q1b instance after {adaptive_max_attempts} attempts (min_mcs_atoms={adaptive_min_atoms})")


def generate_q2_instance(
    *,
    instance_id: str,
    universe_by_formula: Dict[str, List[str]],
    n_molecules: int,
    rng,
    want_yes: bool = True,
) -> BenchmarkInstance:
    """
    Generates a Q2 instance. For 'Yes': sample N from one formula universe.
    For 'No': sample N-1 from one formula universe and 1 from a different formula universe.
    """
    formulas = [f for f, u in universe_by_formula.items() if len(u) >= n_molecules]
    if not formulas:
        raise ValueError(f"No formula universe has >= {n_molecules} isomers")

    if want_yes:
        f = formulas[rng.randrange(len(formulas))]
        u = universe_by_formula[f]
        chosen = rng.sample(u, n_molecules)
        label = solve_q2_is_constitutional_isomer_set(chosen)
        assert label == "Yes", f"Failed to solve Q2 instance {instance_id} for formula {f}"
        # Should be Yes; if not, fallback to strict label from solver
        if label is None:
            label = "No"
        prompt = build_q2_prompt(chosen)
        return BenchmarkInstance(
            id=instance_id,
            task="q2_isomer_set_yes_no",
            n_molecules=n_molecules,
            molecules=chosen,
            prompt=prompt,
            answer={"label": label},
            metadata={"formula": f, "constructed_label": "Yes"},
        )

    # No case: mix formulas but keep N the same
    # Pick a base formula for N-1, and a different formula for 1 molecule
    f1 = formulas[rng.randrange(len(formulas))]
    u1 = universe_by_formula[f1]
    base = rng.sample(u1, n_molecules - 1)  # sample WITHOUT replacement
    base_set = set(base)

    other_formulas = [f for f in universe_by_formula.keys() if f != f1 and len(universe_by_formula[f]) >= 1]
    if not other_formulas:
        # If only one formula exists, sample one more from the same formula
        # (without replacement) to get n_molecules total, ensuring all different SMILES
        available = [mol for mol in u1 if mol not in base_set]
        if available:
            chosen = base + [available[rng.randrange(len(available))]]
        else:
            # Fallback: if we've exhausted the universe, duplicate (shouldn't happen with buffer)
            raise ValueError(f"Formula {f1} universe exhausted when trying to generate 'No' case")
    else:
        # Sample from a different formula, ensuring no duplicate SMILES
        f2 = other_formulas[rng.randrange(len(other_formulas))]
        u2 = universe_by_formula[f2]
        # Try to find a molecule from u2 that's not already in base
        available_u2 = [mol for mol in u2 if mol not in base_set]
        if available_u2:
            chosen = base + [available_u2[rng.randrange(len(available_u2))]]
        else:
            # Unlikely case where all molecules in u2 are already in base
            # Just pick any molecule from u2 (will create duplicate)
            chosen = base + [u2[rng.randrange(len(u2))]]

    # Shuffle
    rng.shuffle(chosen)

    # Verify no duplicates
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"Generated Q2 'No' instance has duplicate SMILES: {chosen}")

    label = solve_q2_is_constitutional_isomer_set(chosen) or "No"
    prompt = build_q2_prompt(chosen)
    return BenchmarkInstance(
        id=instance_id,
        task="q2_isomer_set_yes_no",
        n_molecules=n_molecules,
        molecules=chosen,
        prompt=prompt,
        answer={"label": label},
        metadata={"constructed_label": "No", "base_formula": f1},
    )


def generate_q3_instance(
    *,
    instance_id: str,
    universe_by_formula: Dict[str, List[str]],
    n_molecules: int,
    rng,
    max_universe_size: int = 100,
    min_universe_size: int = 8,
) -> BenchmarkInstance:
    """
    Q3 requires a "complete set" universe. We choose a formula universe of manageable size,
    then provide n_molecules as the *given* subset and expect the remaining as the answer.
    """
    # Allow reasonable buffer above n_molecules for larger n values
    # For n=50, this allows universes up to 100, giving ~50 missing molecules to identify
    effective_max_size = max(max_universe_size, n_molecules + max(10, n_molecules // 2))
    
    candidate_formulas = [
        f for f, u in universe_by_formula.items()
        if (min_universe_size <= len(u) <= effective_max_size) and (len(u) > n_molecules)
    ]
    if not candidate_formulas:
        raise ValueError(
            f"No universe found with size in [{min_universe_size},{effective_max_size}] and > given size {n_molecules}"
        )

    f = candidate_formulas[rng.randrange(len(candidate_formulas))]
    universe = universe_by_formula[f]

    given = rng.sample(universe, n_molecules)
    missing = solve_q3_missing_isomers(given, universe)
    if missing is None:
        raise RuntimeError("Failed to compute missing isomers for Q3")

    prompt = build_q3_prompt(given)
    return BenchmarkInstance(
        id=instance_id,
        task="q3_missing_isomers",
        n_molecules=n_molecules,
        molecules=given,
        prompt=prompt,
        answer={"missing_smiles": missing},
        metadata={"formula": f, "universe_size": len(universe), "given_size": len(given)},
    )


def build_q4_prompt(
    group_combined: List[str],
    scaffold_smiles: List[str],
) -> str:
    """
    Build prompt for Q4 - identify linker fragment that connects scaffolds.


    Args:
        group_combined: List of molecules containing all scaffolds connected by linker
        scaffold_smiles: List of scaffold SMILES to exclude from answer

    Returns:
        Formatted prompt string for Q4 task
    """
    combined_list = _format_smiles_list(group_combined)
    scaffold_list = _format_smiles_list(scaffold_smiles)
    n_scaffolds = len(scaffold_smiles)

    return (
        f"CHEMISTRY PROBLEM: Identify the Linker Fragment\n\n"
        f"CONCEPT:\n"
        f"The molecules below are built from {n_scaffolds} different scaffolds connected by a common linker.\n"
        f"Your task is to identify this linker - the fragment that:\n"
        f"  • Connects the scaffolds together\n"
        f"  • Is present in every molecule\n"
        f"  • Does NOT include any scaffold itself\n\n"
        f"INPUT MOLECULES:\n"
        f"{combined_list}\n\n"
        f"SCAFFOLDS (to exclude from your answer):\n"
        f"{scaffold_list}\n\n"
        f"WHAT YOU NEED TO FIND:\n"
        f"The LARGEST connected fragment that:\n"
        f"  1. Is a substructure of every molecule above\n"
        f"  2. Does NOT overlap with any of the {n_scaffolds} scaffolds\n"
        f"  3. Is the connecting piece between scaffolds\n\n"
        f"VALIDATION CHECKLIST:\n"
        f"Before submitting your answer, verify:\n"
        f"□ Is it present in molecule 1? molecule 2? ... all molecules?\n"
        f"□ Does it match or contain scaffold 1? scaffold 2? ... any scaffold? (should be NO)\n"
        f"□ Is the SMILES complete and valid? (all rings properly closed)\n"
        f"□ Is it reasonably sized? (typically 5-15 atoms for a linker)\n\n"
        f"EXAMPLES OF LINKERS:\n"
        f"  • Alkyl chains: CCCC, CCCCC, CC(C)CC\n"
        f"  • Cyclic ethers: C1CCOCC1 (tetrahydropyran)\n"
        f"  • Ether chains: CCOC(C)OCC\n"
        f"  • Functional linkers: CCCC(O)CCC (with hydroxyl)\n\n"
        f"OUTPUT FORMAT:\n"
        f"<smiles>YOUR_ANSWER_HERE</smiles>\n"
        f"No explanation."
    )


def generate_q4_instance(
    *,
    instance_id: str,
    scaffolds: List[Dict[str, str]],
    linker: str,
    linker_category: str,
    molecules_per_group: int,
    min_answer_atoms: int,
    rng,
    max_attempts: int = 50,
) -> Optional[BenchmarkInstance]:
    """
    Q4: Scaffold avoidance - generalized for N scaffolds.

    Creates N+1 groups:
    - Group i (i=1..N): scaffold_i + linker + decoration
    - Group N+1: scaffold_1 + linker + scaffold_2 + ... + scaffold_N + decoration

    Args:
        scaffolds: List of dicts with keys "scaffold" (SMILES) and "name" (str)
        molecules_per_group: Number of molecules per group
    """
    from .scaffold_operations import remove_substructure_get_largest_fragment

    N = len(scaffolds)
    if N < 2:
        raise ValueError("Need at least 2 scaffolds for Q4")

    # Extract scaffold SMILES and names
    scaffold_smiles = [s["scaffold"] for s in scaffolds]
    scaffold_names = [s["name"] for s in scaffolds]

    # Decorations for variety
    decorations = ["C", "CC", "CCC", "C(C)C", "CCCC", "C(C)CC"]

    # Generate individual groups (Groups 1 to N)
    groups = [[] for _ in range(N)]
    max_mol_attempts = 20

    for group_idx in range(N):
        attempts = 0
        while len(groups[group_idx]) < molecules_per_group and attempts < max_mol_attempts:
            dec = rng.choice(decorations)
            mol_smiles = f"{scaffold_smiles[group_idx]}{linker}{dec}"
            mol = mol_from_smiles(mol_smiles)
            if mol:
                can_smi = canonical_smiles(mol)
                if can_smi not in groups[group_idx]:  # Avoid duplicates
                    groups[group_idx].append(can_smi)
            attempts += 1

        # Check if generation succeeded
        if len(groups[group_idx]) < molecules_per_group:
            return None

    # Generate combined group (Group N+1) with all scaffolds chained
    group_combined = []
    attempts = 0
    while len(group_combined) < molecules_per_group and attempts < max_mol_attempts:
        dec = rng.choice(decorations)
        # Chain all scaffolds with linker between each
        mol_smiles = linker.join(scaffold_smiles) + dec
        mol = mol_from_smiles(mol_smiles)
        if mol:
            can_smi = canonical_smiles(mol)
            if can_smi not in group_combined:
                group_combined.append(can_smi)
        attempts += 1

    # Verify combined group has correct size
    if len(group_combined) < molecules_per_group:
        return None

    # Compute expected answer by removing all scaffolds from combined group
    filtered_fragments = []
    for mol_smiles in group_combined:
        frag = remove_substructure_get_largest_fragment(
            mol_smiles,
            scaffold_smiles  # List of all scaffolds
        )
        if frag:
            filtered_fragments.append(frag)

    if len(filtered_fragments) != len(group_combined):
        return None  # Some molecules didn't produce valid fragments

    # Find MCS of filtered fragments
    mcs_result = solve_q1_largest_common_motif(filtered_fragments)
    if mcs_result is None:
        return None

    # Check if MCS meets minimum size
    if mcs_result.num_atoms < min_answer_atoms:
        return None

    # Build the instance
    # Use simplified prompt (achieves 80% accuracy vs 0% with old complex prompt)
    prompt = build_q4_prompt(group_combined, scaffold_smiles)

    all_molecules = [mol for group in groups for mol in group] + group_combined

    # Build metadata with dynamic group storage
    metadata = {
        "n_groups": N,
        "molecules_per_group": molecules_per_group,
        "scaffolds": scaffold_smiles,
        "scaffold_names": scaffold_names,
        "linker_template": linker,
        "answer_num_atoms": mcs_result.num_atoms,
        "answer_num_bonds": mcs_result.num_bonds,
        "linker_category": linker_category,
        "group_combined": group_combined,
    }

    # Store individual groups dynamically
    for i in range(N):
        metadata[f"group_{i+1}"] = groups[i]

    return BenchmarkInstance(
        id=instance_id,
        task="q4_avoid_scaffolds",
        n_molecules=len(all_molecules),
        molecules=all_molecules,
        prompt=prompt,
        answer={"smiles": mcs_result.motif_smiles},
        metadata=metadata,
    )


def build_q5_prompt(
    molecules: List[str],
    constraint_type: str,
    target_value: int,
    k_molecules: int,
    min_motif_atoms: int,
) -> str:
    """Build prompt for Q5 (constraint satisfaction with selection)."""
    constraint_name_map = {
        "aromatic_ring": "aromatic rings (e.g., benzene c1ccccc1, pyridine)",
        "carboxylic_acid": "carboxylic acid groups (-COOH, written as C(=O)O in SMILES)",
        "alcohol": "alcohol groups (-OH attached to sp3 carbon)",
        "primary_amine": "primary amine groups (-NH2 attached to carbon)",
        "ketone": "ketone groups (C=O bonded to two carbons, NOT in acids/esters/amides)",
        "ether": "ether linkages (C-O-C where O connects two carbons)",
        "fluoride": "fluorine atoms (F)",
        "chloride": "chlorine atoms (Cl)",
        "double_bond": "carbon-carbon double bonds (C=C)",
        "nitrile": "nitrile groups (-C≡N)",
    }

    constraint_description = constraint_name_map.get(constraint_type, constraint_type)
    smiles_list = _format_smiles_list(molecules)

    return (
        f"Given the following {len(molecules)} molecules, identify one continuous motif from EACH molecule.\n\n"
        f"TASK:\n"
        f"1. From EACH of the {k_molecules} molecules below, extract one continuous motif (substructure)\n"
        f"2. Ensure the total count of {constraint_description} across ALL motifs equals {target_value}\n\n"
        f"CONSTRAINTS:\n"
        f"- Each motif must be a VALID SMILES string (complete, parseable by RDKit)\n"
        f"- Each motif must be a substructure that actually exists in its parent molecule\n"
        f"- Each motif must contain at least {min_motif_atoms} heavy atoms (non-hydrogen)\n"
        f"- The sum of {constraint_description} across all selected motifs = {target_value}\n\n"
        f"CRITICAL VALIDATION RULES:\n"
        f"- SMILES must be COMPLETE - do NOT truncate or abbreviate\n"
        f"- RINGS MUST BE CLOSED: Every ring opening digit (1-9) must have a matching closing digit\n"
        f"  WRONG: 'CC12CCC(=O)C=C1' (ring 2 never closes) - INVALID SMILES\n"
        f"  RIGHT: 'CC12CCC(=O)C=C1CC2' (both rings 1 and 2 close properly)\n"
        f"- Each motif MUST be a continuous fragment that exists exactly as written in its parent molecule\n"
        f"- When extracting from complex fused rings, use simpler motifs if needed\n"
        f"- Count {constraint_description} carefully - be specific about what counts\n"
        f"- Verify your sum equals {target_value} before submitting\n\n"
        f"MOLECULES:\n{smiles_list}\n\n"
        f"STEP-BY-STEP APPROACH:\n"
        f"1. For EACH molecule, identify potential motifs (≥{min_motif_atoms} atoms)\n"
        f"2. Count {constraint_description} in each motif candidate\n"
        f"3. Select one motif from each molecule such that the sum equals {target_value}\n"
        f"4. Note: Some motifs may have 0 {constraint_description} - this is OK!\n"
        f"5. Extract the EXACT substructure from the parent - copy it precisely\n"
        f"6. Ensure SMILES is complete: all rings must be properly closed (e.g., c1ccccc1)\n"
        f"7. Final check: motif exists in parent AND sum equals {target_value}\n\n"
        f"FUNCTIONAL GROUP EXAMPLES (for reference):\n"
        f"- Ketone: C(=O)C or CC(=O)CC (carbonyl between two carbons)\n"
        f"- Carboxylic acid: C(=O)O or CC(=O)O\n"
        f"- Ester: C(=O)OC or CC(=O)OC\n"
        f"- Aldehyde: C(=O) at chain end\n"
        f"- Primary amine: CNH2 or CCN\n"
        f"- Alcohol: CO (hydroxyl on sp3 carbon)\n"
        f"- Aromatic ring: c1ccccc1 (benzene)\n\n"
        f"OUTPUT FORMAT (indices are 0-indexed, must include ALL molecules):\n"
        f"<indices>0,1,2</indices>\n"
        f"<motif_0>CCCCCC</motif_0>\n"
        f"<motif_1>c1ccccc1</motif_1>\n"
        f"<motif_2>CC(=O)O</motif_2>\n\n"
        f"FORMAT RULES:\n"
        f"- List ALL molecule indices in <indices> tag (0 through {k_molecules-1}), comma-separated\n"
        f"- For each index, provide complete motif SMILES in <motif_N> tag\n"
        f"- Do NOT use <smiles> tags - use <motif_N> where N is the molecule index\n"
        f"- SMILES must be COMPLETE (e.g., 'c1ccccc1' not 'c1ccc')\n\n"
        f"CRITICAL: To find {target_value} {constraint_description}:\n"
        f"- You must provide a motif for EVERY molecule (all {k_molecules} molecules)\n"
        f"- Some motifs may have 0 {constraint_description} - balance is key!\n"
        f"- Adjust motif selections so total = {target_value}\n\n"
        f"BEFORE SUBMITTING - VERIFY:\n"
        f"✓ Provided motif for ALL {k_molecules} molecules (indices 0 through {k_molecules-1})\n"
        f"✓ Each SMILES is complete and valid (all rings closed)\n"
        f"✓ Each motif exists in its parent molecule\n"
        f"✓ Count {constraint_description} in each motif\n"
        f"✓ Sum of {constraint_description} = {target_value} (NOT more, NOT less)\n\n"
        f"Provide ONLY the formatted answer above. No explanation."
    )


def generate_q5_instance(
    *,
    instance_id: str,
    bank_index: BankIndex,
    n_molecules: int,
    k_molecules: int,
    constraint_type: str,
    target_value: Optional[int],
    min_motif_atoms: int,
    rng,
    max_attempts: int = 100,
) -> Optional[BenchmarkInstance]:
    """
    Q5: Constraint satisfaction - select motifs from ALL molecules satisfying constraint.

    NEW DESIGN:
    - Selects a motif from EVERY molecule (k_molecules = n_molecules)
    - Uses DP-based subset sum to find achievable targets
    - Target defaults to closest achievable sum to k/2
    - Much faster and guaranteed to find solutions

    Uses ChEMBL sampling with DP solver to ensure solvability.
    If target_value is None, it will be automatically determined as closest to k/2.
    """
    import logging
    from .motif_extraction import enumerate_motifs_with_functional_groups

    logger = logging.getLogger(__name__)

    # NEW: k_molecules must equal n_molecules (select motif from every molecule)
    k_molecules = n_molecules

    for attempt in range(max_attempts):
        logger.info(f"Q5 generation attempt {attempt + 1}/{max_attempts}")

        # Sample molecules using similarity-based sampling
        try:
            molecules = bank_index.sample_similar_group(
                n_molecules,
                rng=rng,
                min_similarity=0.35,
                max_similarity=0.90,
            )
        except ValueError:
            logger.debug("Molecule sampling failed, retrying")
            continue

        # Enumerate motifs for each molecule (with caching)
        logger.info(f"Enumerating motifs for {n_molecules} molecules...")
        molecules_with_motifs = []
        failed_molecules = 0

        for i, mol_smiles in enumerate(molecules):
            motifs = enumerate_motifs_with_functional_groups(
                mol_smiles,
                min_atoms=min_motif_atoms,
                max_motifs=30,  # Reduced from 100 for better performance
                use_cache=True,
            )
            if not motifs:
                logger.debug(f"Molecule {i} has no valid motifs (min_atoms={min_motif_atoms})")
                failed_molecules += 1
                break  # This molecule has no valid motifs
            molecules_with_motifs.append((mol_smiles, motifs))

        if len(molecules_with_motifs) != n_molecules:
            logger.debug(f"Failed: {failed_molecules} molecules had no valid motifs")
            continue  # Some molecules didn't have valid motifs

        logger.info(f"Motif enumeration complete. Solving constraint satisfaction...")

        # Use DP-based solver to find achievable targets and select best one
        solution = _solve_q5_with_dp(
            molecules_with_motifs,
            constraint_type,
            target_value,
            k_molecules,
            rng,
        )

        if solution:
            # Success! Build the instance
            actual_target = solution["target_value"]
            logger.info(f"Solution found! Target={actual_target}, Total={solution['total']}")

            prompt = build_q5_prompt(
                molecules,
                constraint_type,
                actual_target,
                k_molecules,
                min_motif_atoms
            )

            return BenchmarkInstance(
                id=instance_id,
                task="q5_constraint_satisfaction_selection",
                n_molecules=n_molecules,
                molecules=list(molecules),
                prompt=prompt,
                answer={
                    "selected_molecule_indices": solution["selected_molecule_indices"],
                    "selected_motifs": solution["selected_motifs"],
                },
                metadata={
                    "constraint_type": constraint_type,
                    "target_value": actual_target,
                    "actual_total": solution["total"],
                    "k_molecules": k_molecules,
                    "min_motif_atoms": min_motif_atoms,
                    "attempt": attempt,
                    "values": solution["values"],  # Keep for debugging
                },
            )
        else:
            logger.debug(f"No solution found on attempt {attempt + 1}")

    # Failed to find solvable instance
    logger.warning(f"Failed to generate Q5 instance after {max_attempts} attempts")
    return None


def _solve_q5_with_dp(
    molecules_with_motifs: List[Tuple[str, List[Dict]]],
    constraint_type: str,
    target_value: Optional[int],
    k_molecules: int,
    rng,
) -> Optional[Dict]:
    """
    DP-based solver for Q5 constraint satisfaction.

    NEW APPROACH:
    1. Compute all achievable target sums using DP
    2. If target_value is None, pick the achievable sum closest to k/2
    3. Use DP to reconstruct the solution

    Much faster than backtracking (O(n*k*max_sum) vs O(2^n)).

    Returns dict with:
    - selected_molecule_indices: List[int]
    - selected_motifs: Dict[str, str] (index -> motif SMILES)
    - values: List[int]
    - total: int
    - target_value: int (the actual target used)
    """
    import logging
    logger = logging.getLogger(__name__)

    n = len(molecules_with_motifs)

    # Step 1: Build value matrix - for each molecule, list all possible values from its motifs
    molecule_values = []  # List[List[Tuple[int, str, Dict]]] - value, motif_smiles, motif_data

    for mol_idx, (mol_smiles, motifs) in enumerate(molecules_with_motifs):
        motif_values = []
        for motif_data in motifs:
            # Extract constraint value
            if constraint_type == "total_aromatic_rings":
                value = motif_data.get("num_aromatic_rings", 0)
            elif constraint_type.startswith("total_"):
                fg_key = constraint_type[6:]  # Remove "total_"
                if fg_key.endswith("s"):
                    fg_key = fg_key[:-1]  # Remove trailing "s"
                value = motif_data["functional_groups"].get(fg_key, 0)
            else:
                value = motif_data["functional_groups"].get(constraint_type, 0)

            if value is None:
                value = 0

            motif_values.append((value, motif_data["motif_smiles"], motif_data))

        molecule_values.append(motif_values)

    # Step 2: Use DP to find all achievable sums
    # dp[i][s] = True if we can achieve sum s using first i molecules
    # We also track which motif was used: parent[i][s] = (motif_idx, prev_sum)

    max_possible_sum = sum(max(vals, key=lambda x: x[0])[0] for vals in molecule_values)
    max_sum = min(max_possible_sum, 200)  # Cap to avoid memory issues

    # Initialize DP table
    dp = [{} for _ in range(n + 1)]
    dp[0][0] = True  # Base case: 0 molecules, sum 0

    parent = [{} for _ in range(n + 1)]  # Track which motif was chosen

    # Fill DP table
    for i in range(n):
        for prev_sum in dp[i]:
            if not dp[i][prev_sum]:
                continue

            # Try each motif from molecule i
            for motif_idx, (value, motif_smiles, motif_data) in enumerate(molecule_values[i]):
                new_sum = prev_sum + value
                if new_sum <= max_sum:
                    if new_sum not in dp[i + 1]:
                        dp[i + 1][new_sum] = True
                        parent[i + 1][new_sum] = (motif_idx, prev_sum)

    # Step 3: Find achievable sums after using all n molecules
    achievable_sums = [s for s in dp[n] if dp[n][s]]

    if not achievable_sums:
        logger.warning("No achievable sums found (DP solver)")
        return None

    # Step 4: Choose target - closest to k/2 if not specified
    if target_value is None:
        ideal_target = k_molecules / 2.0
        target_value = min(achievable_sums, key=lambda s: abs(s - ideal_target))
        logger.info(f"Auto-selected target={target_value} (closest to k/2={ideal_target:.1f})")
        logger.info(f"Achievable sums: {sorted(achievable_sums)}")
    elif target_value not in achievable_sums:
        logger.warning(f"Target {target_value} not achievable. Achievable: {sorted(achievable_sums)}")
        return None

    # Step 5: Reconstruct solution by backtracking through DP table
    selected_molecule_indices = list(range(n))  # All molecules are selected
    selected_motifs_dict = {}
    values = []

    current_sum = target_value
    for i in range(n, 0, -1):
        if current_sum not in parent[i]:
            logger.error(f"DP reconstruction failed at molecule {i-1}, sum {current_sum}")
            return None

        motif_idx, prev_sum = parent[i][current_sum]
        mol_idx = i - 1

        value, motif_smiles, motif_data = molecule_values[mol_idx][motif_idx]

        selected_motifs_dict[str(mol_idx)] = motif_smiles
        values.insert(0, value)
        current_sum = prev_sum

    return {
        "selected_molecule_indices": selected_molecule_indices,
        "selected_motifs": selected_motifs_dict,
        "values": values,
        "total": target_value,
        "target_value": target_value,
    }


def _solve_q5_backtracking(
    molecules_with_motifs: List[Tuple[str, List[Dict]]],
    constraint_type: str,
    target_value: int,
    k_molecules: int,
) -> Optional[Dict]:
    """
    Backtracking solver for Q5 constraint satisfaction.

    Returns dict with:
    - selected_molecule_indices: List[int]
    - selected_motifs: Dict[str, str] (index -> motif SMILES)
    - values: List[int]
    - total: int
    """
    n = len(molecules_with_motifs)

    def backtrack(
        mol_idx: int,
        selected_count: int,
        current_sum: int,
        selected_indices: List[int],
        selected_motifs_dict: Dict[str, str],
        values: List[int],
    ):
        # Base case: processed all molecules
        if mol_idx == n:
            if selected_count == k_molecules and current_sum == target_value:
                return {
                    "selected_molecule_indices": selected_indices[:],
                    "selected_motifs": selected_motifs_dict.copy(),
                    "values": values[:],
                    "total": current_sum,
                }
            return None

        remaining_molecules = n - mol_idx

        # Pruning: impossible to reach k molecules
        if selected_count + remaining_molecules < k_molecules:
            return None

        # Pruning: already selected k molecules, skip remaining
        if selected_count == k_molecules:
            return backtrack(
                mol_idx + 1,
                selected_count,
                current_sum,
                selected_indices,
                selected_motifs_dict,
                values,
            )

        mol_smiles, motifs = molecules_with_motifs[mol_idx]

        # Option 1: Skip this molecule
        result = backtrack(
            mol_idx + 1,
            selected_count,
            current_sum,
            selected_indices,
            selected_motifs_dict,
            values,
        )
        if result:
            return result

        # Option 2: Select a motif from this molecule
        for motif_data in motifs:
            # Get constraint value
            # Map constraint_type to the actual key in functional_groups or num_aromatic_rings
            if constraint_type == "total_aromatic_rings":
                value = motif_data.get("num_aromatic_rings", 0)
            elif constraint_type.startswith("total_"):
                # Remove "total_" prefix and handle singular/plural
                fg_key = constraint_type[6:]  # Remove "total_"
                # Convert plural to singular for functional_groups dict
                if fg_key.endswith("s"):
                    fg_key = fg_key[:-1]  # Remove trailing "s"
                value = motif_data["functional_groups"].get(fg_key, 0)
            else:
                # Fallback: use constraint_type directly
                value = motif_data["functional_groups"].get(constraint_type, 0)

            # Ensure value is not None
            if value is None:
                value = 0

            # Pruning: check if we can still reach target
            needed = k_molecules - selected_count - 1
            if current_sum + value > target_value + needed * 20:
                continue  # Too high

            # Recurse
            selected_indices.append(mol_idx)
            selected_motifs_dict[str(mol_idx)] = motif_data["motif_smiles"]
            values.append(value)

            result = backtrack(
                mol_idx + 1,
                selected_count + 1,
                current_sum + value,
                selected_indices,
                selected_motifs_dict,
                values,
            )

            if result:
                return result

            # Backtrack
            selected_indices.pop()
            del selected_motifs_dict[str(mol_idx)]
            values.pop()

        return None

    return backtrack(0, 0, 0, [], {}, [])
