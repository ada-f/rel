# Evaluation Guide

How to scoring model outputs on REL.

## Run tests

Evaluator examples:

```bash
python -m chem_benchmark.test_evaluation
python -m bio_benchmark.test_evaluation
python -m algebra_benchmark.test_evaluation
```

## Chemistry: full runner

The chemistry benchmark has an end-to-end runner that:
- loads a JSONL dataset
- calls the model API
- scores each response
- writes a `.summary.json` file

Example:

```bash
export OPENAI_API_KEY=...

uv run --python "$UV_PROJECT_ENVIRONMENT/bin/python" python -m chem_benchmark.llm_runner \
  --provider openai \
  --model gpt-4o-mini \
  --dataset data/dataset.jsonl \
  --out runs/my_model.jsonl \
  --test_mode 1
```

Useful flags:
- `--provider`: `openai`, `claude`, or `gemini`
- `--task_filter`: limit to specific chemistry tasks
- `--test_mode N`: sample `N` examples per task before running the full set
- `--max_tokens`: response budget for the API call

The runner writes:
- `runs/*.jsonl`: per-example responses and scores
- `runs/*.summary.json`: overall accuracy and per-task accuracy

## Scoring your own responses

If you already have a model output and just want to score it, call the domain-specific evaluation function directly.

### Chemistry

```python
from chem_benchmark.evaluation import evaluate_response

question = "..."
answer = {"label": "Yes", "molecules": ["CCO", "C(C)O"]}
response = "<Yes>"

result = evaluate_response(question, answer, response, task="REL-C1")
print(result)
```

Expected response formats:
- `REL-C1`: `<Yes>` or `<No>`
- `REL-C2`: `<smiles>...</smiles>`
- `REL-C3`: one or more `<smiles>...</smiles>` lines
- `REL-C4`: list of indicies and the motifs for each index in the format `<indices>0,1,2</indices>\n<motif_0>CCCCCC</motif_0>\n<motif_1>c1ccccc1</motif_1>\n<motif_2>CC(=O)O</motif_2>\n\n`

### Biology

```python
from bio_benchmark.evaluation import evaluate_response

question = "..."
answer = {"label": "yes", "taxa": [15, 49, 18]}
response = "Yes. The taxa are 15, 49, and 18."

result = evaluate_response(question, answer, response, task="REL-B1")
print(result)
```

Expected response format:
- include `yes` or `no`
- if `yes`, include the taxa identifiers

### Algebra

```python
from algebra_benchmark.evaluation import evaluate_response

question = "..."
answer = {"target": 2}
response = "Answer 3"

result = evaluate_response(question, answer, response, task="REL-A1")
print(result)
```

Expected response format:
- either `Answer N` or just `N`
- valid choices are `1` through `8`

## Sanity checks

These example scripts are useful to confirm the evaluators are parsing outputs as expected:

```bash
python -m chem_benchmark.test_evaluation
python -m bio_benchmark.test_evaluation
python -m algebra_benchmark.test_evaluation
```
