# REL Benchmark Dataset

REL is a relational reasoning benchmark spanning three scientific domains — **algebra**, **biology**, and **chemistry** — totaling **12,952 questions** across 17 task files. All tasks are formatted as multiple-choice or open-ended questions in JSONL.

---

## Algebra (1,000 questions)

All algebra tasks are **Raven's Progressive Matrices**: complete a 3×3 or 9×9 grid of number tuples by identifying the missing panel from 8 candidates. The matrix sizes differ in context length and difficulty.

| Task | RC | RC min | RC max | Size | Questions | Assesses |
|------|----|--------|--------|------|-----------|---------|
| REL-A1 | 2 | 2 | 2 | 3×3 and 9×9 | 125 each | **Constant rows** — each row is identical; tests whether the model recognizes invariance |
| REL-A2 | 2 | 2 | 2 | 3×3 and 9×9 | 125 each | **Arithmetic progression** — values shift by a fixed increment across rows; tests detection of linear trends |
| REL-A3 | $n$ where $n\times n$ is the matrix size | 3 | 9 | 3×3 and 9×9 | 125 each | **Permutation** — each row is a cyclic permutation of the same values; tests recognition of reordering patterns |
| REL-A4 | $n$ where $n\times n$ is the matrix size | 3 | 9 | 3×3 and 9×9 | 125 each | **Row-Sum** — the final value in each row is the sum of all other entries in the same row multiplied by either $\pm1$, depending on the column. |

---

## Biology (7,952 questions)

All biology tasks are **REL-B1**, split across 12 files by phylogenetic tree/alignment instance. Each question presents a multiple sequence alignment (FASTA) and a phylogenetic tree, and asks whether **structured homoplasy** is present and which taxa are involved.

| Task | RC | RC min | RC max | Files | Total questions | Assesses |
|------|----|--------|--------|-------|-----------------|---------|
| REL-B1 | Number of homoplastic taxa | 2 | 25 | 001–012 | 7,952 | **Homoplasy detection** — identifying convergent evolution where distantly related taxa independently share nucleotide motifs across many alignment columns more than expected by chance; tests relational pattern recognition over long biological sequences |

---

## Chemistry (4,016 questions)

| Task | RC | RC min | RC max | Questions | Assesses |
|------|----|--------|--------|-----------|---------|
| REL-C1 | 2 | 2 | 2 | 1,000 | **Constitutional isomer identification** — given a set of SMILES, determine whether all molecules share the same molecular formula (same atoms, different connectivity); tests recognition of molecular equivalence relations |
| REL-C2 | 2 | 2 | 2 | 1,016 | **Maximum common substructure (MCS)** — find the largest single connected chemical motif present in every molecule in the set; tests relational reasoning over graph structure |
| REL-C3 | Number of isomers | 1 | 92 | 1,000 | **Isomer set completion** — given a partial set of constitutional isomers, identify all missing members; tests exhaustive enumeration of a structural equivalence class |
| REL-C4 | Number of molecules | 5 | 50 | 1,000 | **Constrained motif extraction** — extract one substructure from each of the molecules such that a specified functional group count sums to a target value; tests joint constraint satisfaction over molecular substructures |
