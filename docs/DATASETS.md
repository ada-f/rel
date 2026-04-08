# Dataset Layout

REL uses a unified JSONL record format across chemistry, biology, and algebra.

## Record schema

Each row contains:

```json
{
  "id": "unique_identifier",
  "domain": "chemistry | biology | algebra",
  "task": "task_name",
  "question": "prompt shown to the model",
  "answer": {},
  "metadata": {}
}
```

Required fields:
- `id`
- `domain`
- `task`
- `question`
- `answer`

## Domain-specific answers

### Algebra

Tasks:
- `REL-A1`
- `REL-A2`
- `REL-A3`
- `REL-A4`

Answer shape:

```json
{
  "target": 2
}
```

### Biology

Task:
- `REL-B1`

Answer shape:

```json
{
  "label": "yes",
  "taxa": [15, 49, 18]
}
```

### Chemistry

Tasks:
- `REL-C1`
- `REL-C2`
- `REL-C3`
- `REL-C4`

Answer shapes:

For `REL-C1`:
```json
{
  "label": "Yes",
  "molecules": ["..."]
}
```

For `REL-C2`:
```json
{
  "smiles": "O=CC1CCC(=O)N1",
  "molecules": ["..."]
}
```

For `REL-C3`:
```json
{
  "missing_smiles": ["..."],
  "molecules": ["..."]
}
```

For `REL-C4`:
```json
{
  "selected_molecule_indices": [0, 1, 2],
  "selected_motifs": {
    "0": "...",
    "1": "...",
    "2": "..."
  },
  "molecules": ["..."]
}
```
Multiple answers are allowed for `REL-C4`, the evaluator will check if the selected indices and motifs satisfy the question constraint.

## Where files live

- Builder scripts default to writing unified outputs under `REL/chemistry`, `REL/biology`, and `REL/algebra`.
- The current repository also includes example chemistry datasets under `data/` and `chem_data/`.

Use [docs/DEVELOPMENT.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/docs/DEVELOPMENT.md) for generation commands.
