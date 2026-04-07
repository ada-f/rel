"""
Evaluation functions for chemistry benchmarks.

Provides evaluate_response function that takes question, answer, and response
and returns a score dictionary for each chemistry task (REL-C1, REL-C2, REL-C3, REL-C4).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import rdFMCS

from .llm_runner import extract_all_smiles_tags, extract_first_smiles_tag, extract_yesno_tag, extract_indices_and_motifs
from .rdkit_utils import are_isomorphic_smiles, canonical_smiles_from_smiles, mol_from_smiles


def is_substructure(query_smiles: str, target_smiles: str) -> bool:
    """
    Check if query is a substructure of target.
    
    Args:
        query_smiles: SMILES string of the query molecule
        target_smiles: SMILES string of the target molecule
        
    Returns:
        True if query is a substructure of target, False otherwise
    """
    query_mol = mol_from_smiles(query_smiles)
    target_mol = mol_from_smiles(target_smiles)
    
    if query_mol is None or target_mol is None:
        return False
    
    return target_mol.HasSubstructMatch(query_mol)


def calculate_overlap_metric(response_smiles: str, correct_smiles: str) -> float:
    """
    Calculate atom overlap metric: N_overlap / max(N_response, N_correct).
    
    Args:
        response_smiles: SMILES string from model response
        correct_smiles: SMILES string of correct answer
        
    Returns:
        Overlap metric between 0.0 and 1.0
    """
    response_mol = mol_from_smiles(response_smiles)
    correct_mol = mol_from_smiles(correct_smiles)
    
    if response_mol is None or correct_mol is None:
        return 0.0
    
    n_response = response_mol.GetNumHeavyAtoms()
    n_correct = correct_mol.GetNumHeavyAtoms()
    
    if correct_mol.HasSubstructMatch(response_mol):
        n_overlap = n_response
    elif response_mol.HasSubstructMatch(correct_mol):
        n_overlap = n_correct
    else:
        mcs = rdFMCS.FindMCS([response_mol, correct_mol], timeout=5)
        if mcs.smartsString:
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            if mcs_mol:
                n_overlap = mcs_mol.GetNumHeavyAtoms()
            else:
                n_overlap = 0
        else:
            n_overlap = 0
    
    max_atoms = max(n_response, n_correct)
    if max_atoms == 0:
        return 0.0
    
    return n_overlap / max_atoms


def evaluate_c1_response(question: str, answer: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Evaluate response for REL-C1 (isomer set yes/no question).
    
    Args:
        question: The question text
        answer: Answer dict containing "label" ("Yes" or "No")
        response: Model response text
        
    Returns:
        Dictionary with evaluation results:
        - correct: bool
        - pred: str or None
        - gold: str
    """
    pred = extract_yesno_tag(response)
    gold_label = answer.get("label")
    
    if gold_label is None:
        return {"correct": False, "pred": pred, "gold": None, "error": "Missing label in answer"}
    
    correct = (pred is not None) and (pred == gold_label)
    return {"correct": correct, "pred": pred, "gold": gold_label}


def evaluate_c2_response(question: str, answer: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Evaluate response for REL-C2 (largest common motif identification).
    
    Response is considered correct only if the SMILES represent the same molecule
    (checked via isomorphic matching to handle non-canonical SMILES).
    
    Args:
        question: The question text
        answer: Answer dict containing "smiles" (correct SMILES string)
        response: Model response text
        
    Returns:
        Dictionary with evaluation results:
        - correct: bool (True if SMILES represent the same molecule)
        - pred: str or None (extracted SMILES from response)
        - gold: str (correct SMILES)
        - response_is_substructure_of_correct: bool
        - correct_is_substructure_of_response: bool
        - overlap_metric: float
    """
    pred_smiles = extract_first_smiles_tag(response)
    gold_smiles = answer.get("smiles")
    
    if gold_smiles is None:
        return {
            "correct": False,
            "pred": pred_smiles,
            "gold": None,
            "response_is_substructure_of_correct": False,
            "correct_is_substructure_of_response": False,
            "overlap_metric": 0.0,
            "error": "Missing smiles in answer",
        }
    
    if pred_smiles is None:
        return {
            "correct": False,
            "pred": None,
            "gold": gold_smiles,
            "response_is_substructure_of_correct": False,
            "correct_is_substructure_of_response": False,
            "overlap_metric": 0.0,
        }
    
    # Check if SMILES represent the same molecule (isomorphic matching)
    correct = are_isomorphic_smiles(pred_smiles, gold_smiles)
    
    # Also compute substructure relationships for informational purposes
    response_is_sub = is_substructure(pred_smiles, gold_smiles)
    correct_is_sub = is_substructure(gold_smiles, pred_smiles)
    
    # Calculate overlap metric
    overlap = calculate_overlap_metric(pred_smiles, gold_smiles)
    
    return {
        "correct": correct,
        "pred": pred_smiles,
        "gold": gold_smiles,
        "response_is_substructure_of_correct": response_is_sub,
        "correct_is_substructure_of_response": correct_is_sub,
        "overlap_metric": overlap,
    }


def evaluate_c3_response(question: str, answer: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Evaluate response for REL-C3 (missing isomers completion).
    
    Uses isomorphic SMILES matching to handle non-canonical SMILES.
    
    Args:
        question: The question text
        answer: Answer dict containing "missing_smiles" (list of SMILES strings)
        response: Model response text
        
    Returns:
        Dictionary with evaluation results:
        - correct: bool (True if exact match)
        - pred: list[str] (extracted SMILES from response)
        - gold: list[str] (correct SMILES)
        - tp: int (true positives)
        - fp: int (false positives)
        - fn: int (false negatives)
        - precision: float
        - recall: float
        - f1: float
    """
    pred_list = extract_all_smiles_tags(response)
    gold_list = answer.get("missing_smiles", [])
    
    if not isinstance(gold_list, list):
        return {
            "correct": False,
            "pred": pred_list,
            "gold": [],
            "tp": 0,
            "fp": len(pred_list),
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": "Missing or invalid missing_smiles in answer",
        }
    
    # Match predicted SMILES to gold SMILES using isomorphic matching
    # This handles non-canonical SMILES
    matched_gold_indices = set()
    matched_pred_indices = set()
    
    for i, pred_smiles in enumerate(pred_list):
        for j, gold_smiles in enumerate(gold_list):
            if j in matched_gold_indices:
                continue
            if are_isomorphic_smiles(pred_smiles, gold_smiles):
                matched_gold_indices.add(j)
                matched_pred_indices.add(i)
                break
    
    tp = len(matched_gold_indices)
    fp = len(pred_list) - len(matched_pred_indices)
    fn = len(gold_list) - len(matched_gold_indices)
    
    # Calculate metrics
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    # Exact match if all predicted match all gold
    exact = (tp == len(gold_list)) and (fp == 0) and (fn == 0)
    
    return {
        "correct": exact,
        "pred": pred_list,
        "gold": gold_list,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def evaluate_c4_response(question: str, answer: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Evaluate response for REL-C4 (constraint satisfaction with motif selection).

    Checks if the model selected the correct set of molecule indices and corresponding
    motifs that satisfy the constraint.

    Args:
        question: The question text
        answer: Answer dict containing:
            - "selected_molecule_indices": list[int]
            - "selected_motifs": dict[str, str] (molecule_idx -> motif_smiles)
            - "molecules": list[str] (input molecules)
        response: Model response text

    Returns:
        Dictionary with evaluation results:
        - correct: bool (True if both indices and motifs match exactly)
        - indices_correct: bool (True if indices match)
        - motifs_correct: bool (True if all motifs match)
        - pred_indices: list[int] or None
        - gold_indices: list[int]
        - pred_motifs: dict[str, str] or None
        - gold_motifs: dict[str, str]
    """
    # Extract predicted indices and motifs from response
    pred_indices, pred_motifs = extract_indices_and_motifs(response)

    # Get gold answers
    gold_indices = answer.get("selected_molecule_indices", [])
    gold_motifs = answer.get("selected_motifs", {})

    if not isinstance(gold_indices, list):
        return {
            "correct": False,
            "indices_correct": False,
            "motifs_correct": False,
            "pred_indices": pred_indices,
            "gold_indices": [],
            "pred_motifs": pred_motifs,
            "gold_motifs": {},
            "error": "Missing or invalid selected_molecule_indices in answer",
        }

    # Check if indices match (order-independent)
    indices_correct = (
        pred_indices is not None and
        set(pred_indices) == set(gold_indices)
    )

    # Check if motifs match (use isomorphic matching for SMILES)
    motifs_correct = False
    if pred_motifs is not None and indices_correct:
        # Check that all predicted indices have motifs
        if set(pred_motifs.keys()) == set(str(i) for i in gold_indices):
            # Check each motif using isomorphic matching
            all_match = True
            for idx_str in gold_motifs.keys():
                gold_motif_smiles = gold_motifs[idx_str]
                pred_motif_smiles = pred_motifs.get(idx_str)

                if pred_motif_smiles is None:
                    all_match = False
                    break

                if not are_isomorphic_smiles(pred_motif_smiles, gold_motif_smiles):
                    all_match = False
                    break

            motifs_correct = all_match

    # Overall correct if both indices and motifs are correct
    correct = indices_correct and motifs_correct

    return {
        "correct": correct,
        "indices_correct": indices_correct,
        "motifs_correct": motifs_correct,
        "pred_indices": pred_indices,
        "gold_indices": gold_indices,
        "pred_motifs": pred_motifs,
        "gold_motifs": gold_motifs,
    }


def evaluate_response(question: str, answer: Dict[str, Any], response: str, task: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a model response for a chemistry benchmark question.

    Routes to task-specific evaluators based on task type. Task can be inferred
    from answer structure if not provided.

    Args:
        question: The question text
        answer: Answer dict containing task-specific fields:
            - REL-C1: "label" ("Yes" or "No")
            - REL-C2: "smiles" (SMILES string)
            - REL-C3: "missing_smiles" (list of SMILES strings)
            - REL-C4: "selected_molecule_indices" (list[int]), "selected_motifs" (dict)
        response: Model response text
        task: Optional task identifier (REL-C1, REL-C2, REL-C3, REL-C4, or legacy names)
               If not provided, inferred from answer structure

    Returns:
        Dictionary with evaluation results (structure depends on task)
    """
    # Infer task from answer structure if not provided
    if task is None:
        if "label" in answer:
            task = "REL-C1"
        elif "missing_smiles" in answer:
            task = "REL-C3"
        elif "selected_molecule_indices" in answer:
            task = "REL-C4"
        elif "smiles" in answer:
            # Need to distinguish between REL-C2 and REL-C4
            # Check metadata for task type if available
            task = "REL-C2"  # Default to C2
        else:
            return {"error": "Could not infer task from answer structure", "correct": False}

    # Normalize task name (handle legacy names)
    task_upper = task.upper()
    if task_upper == "REL-C1":
        return evaluate_c1_response(question, answer, response)
    elif task_upper == "REL-C2":
        return evaluate_c2_response(question, answer, response)
    elif task_upper == "REL-C3":
        return evaluate_c3_response(question, answer, response)
    elif task_upper == "REL-C4":
        return evaluate_c4_response(question, answer, response)
    else:
        return {"error": f"Unknown task: {task}", "correct": False}
