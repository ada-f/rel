from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolDescriptors

from .rdkit_utils import canonical_smiles, canonical_smiles_from_smiles, mol_from_smiles, mol_formula


@dataclass(frozen=True)
class MCSResult:
    motif_smiles: str
    num_atoms: int
    num_bonds: int
    smarts_pattern: Optional[str] = None


def solve_q2_largest_common_motif(
    smiles_list: Sequence[str],
    *,
    timeout_s: int = 15,
    match_chiral: bool = False,
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
) -> Optional[MCSResult]:
    """
    Q1: Largest connected common substructure (MCS), returned as a *concrete* SMILES fragment
    extracted from the first molecule (to avoid query-atom SMARTS artifacts).
    """
    mols = []
    for s in smiles_list:
        mol = mol_from_smiles(s)
        if mol is None:
            return None
        mols.append(mol)

    params = rdFMCS.MCSParameters()
    params.Timeout = int(timeout_s)
    params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    params.BondCompare = rdFMCS.BondCompare.CompareOrder
    params.MatchValences = False
    params.MatchChiralTag = bool(match_chiral)
    params.RingMatchesRingOnly = bool(ring_matches_ring_only)
    params.CompleteRingsOnly = bool(complete_rings_only)
    params.MaximizeBonds = True
    params.Connected = True

    res = rdFMCS.FindMCS(mols, params)
    if not res or not res.smartsString:
        return None
    if getattr(res, "canceled", False):
        return None

    q = Chem.MolFromSmarts(res.smartsString)
    if q is None:
        return None

    # Extract a "real" fragment SMILES from one of the molecules (prefer first)
    match = mols[0].GetSubstructMatch(q)
    if not match:
        for m in mols[1:]:
            match = m.GetSubstructMatch(q)
            if match:
                mols = [m] + [x for x in mols if x is not m]
                break
    if not match:
        return None

    frag_smiles = Chem.MolFragmentToSmiles(
        mols[0],
        atomsToUse=list(match),
        canonical=True,
        isomericSmiles=bool(match_chiral),
    )
    frag_mol = mol_from_smiles(frag_smiles)
    if frag_mol is None:
        return None

    motif = canonical_smiles(frag_mol, isomeric=bool(match_chiral))
    return MCSResult(
        motif_smiles=motif,
        num_atoms=int(res.numAtoms),
        num_bonds=int(res.numBonds),
        smarts_pattern=res.smartsString,
    )


def solve_q1_is_constitutional_isomer_set(smiles_list: Sequence[str]) -> Optional[str]:
    """
    Q2: Return 'Yes' if:
      - all molecules parse
      - all molecules have the same molecular formula
      - and there are at least 2 unique connectivities (ignoring stereochem)
    else 'No'.
    """
    mols = []
    for s in smiles_list:
        m = mol_from_smiles(s)
        if m is None:
            return None
        mols.append(m)

    formulas = [rdMolDescriptors.CalcMolFormula(m) for m in mols]
    if len(set(formulas)) != 1:
        return "No"

    # Require at least two distinct constitutional structures (ignore stereo)
    canon = []
    for m in mols:
        canon.append(Chem.MolToSmiles(m, canonical=True, isomericSmiles=False))
    if len(set(canon)) < 2:
        return "No"

    return "Yes"


def solve_q3_missing_isomers(
    given_smiles: Sequence[str],
    universe_smiles: Sequence[str],
) -> Optional[List[str]]:
    """
    Q3: universe_smiles defines the "complete set".
    Return the missing ones (canonical set difference), sorted.
    """
    given_canon = set()
    for s in given_smiles:
        cs = canonical_smiles_from_smiles(s, isomeric=False)
        if cs is None:
            return None
        given_canon.add(cs)

    univ_canon = set()
    for s in universe_smiles:
        cs = canonical_smiles_from_smiles(s, isomeric=False)
        if cs is None:
            return None
        univ_canon.add(cs)

    missing = sorted(list(univ_canon - given_canon))
    return missing
