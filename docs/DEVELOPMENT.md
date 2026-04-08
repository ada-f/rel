# Development Guide

How to generate new benchmark questions.

## Run tests

Dataset builder smoke tests:

```bash
python -m chem_benchmark.test_build_benchmark
python -m bio_benchmark.test_build_benchmark
python -m algebra_benchmark.test_build_benchmark
```

## Generate new benchmark questions

### Chemistry

```bash
python -m chem_benchmark.build_benchmark \
  --out REL/chemistry/REL-questions-test.jsonl \
  --seed 42 \
  --n_values 5 10 30 \
  --q1_per_n 30 \
  --q2_per_n 30 \
  --q3_per_n 10 \
  --q4_per_n 10
```

### Biology

```bash
python -m bio_benchmark.build_benchmark \
  --out REL/biology/REL-B1.jsonl \
  --seed 42 \
  --num_yes 100 \
  --num_no 100 \
  --seq_len 300 \
  --num_leaves 50
```

### Algebra

```bash
python -m algebra_benchmark.build_benchmark \
  --task REL-A1 \
  --num_samples 100 \
  --gridsize 3 \
  --maxval 1000 \
  --out REL/algebra/REL-A1.jsonl \
  --seed 42
```

## Domain-specific notes

- [chem_benchmark/README.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/chem_benchmark/README.md)
- [bio_benchmark/README.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/bio_benchmark/README.md)
- [algebra_benchmark/README.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/algebra_benchmark/README.md)
