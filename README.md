# Evaluating Relational Reasoning in LLMs with REL

<p align="center">
  <a href="https://arxiv.org/abs/2604.12176">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/YOUR_DATASET_PATH">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  </a>
  <a href="https://zitniklab.hms.harvard.edu/REL/">
    <img src="https://img.shields.io/badge/Project-Website-0A7A5A?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Website">
  </a>
</p>

<p align="center">
  <img src="assets/fig1.png" alt="REL benchmarks for chemistry, biology, and algebra." width="800">
</p>
<p align="center">Figure 1: REL benchmarks for chemistry, biology, and algebra.</p>

<p align="center">
  <img src="assets/fig2.png" alt="Example questions from REL." width="800">
</p>
<p align="center">Figure 2: Example questions from REL.</p>


**Authors**: Lukas Fesser\*, Yasha Ektefaie\*, Ada Fang\*, Sham M. Kakakde, Marinka Zitnik
\* indicates equal contribution


## Setup

This repo is configured around `uv` and the local helper script in [setup_uv_env.sh](setup_uv_env.sh).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source setup_uv_env.sh
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

If you want the full environment notes, including cache locations and troubleshooting, see [UV_SETUP.md](UV_SETUP.md).

## Running the benchmark

The questions are provided in `REL/` or you can download them from Hugging Face with `hf download ada-f/rel --repo-type dataset --local-dir .`.
Run your LLM on the questions and evaluate the responses with the domain evaluators. Examples of how to run fronteir LLMs (Claude, Gemini, GPT-5) are provided in `chem_benchmark/llm_runner.py`.

### Evaluate responses from your own pipeline

If you already have model responses and just want scoring, use the domain evaluators directly:

- Chemistry: `chem_benchmark.evaluation`
- Biology: `bio_benchmark.evaluation`
- Algebra: `algebra_benchmark.evaluation`

The expected answer formats and minimal examples are in [docs/EVALUATION.md](docs/EVALUATION.md).

## More Docs

- [docs/EVALUATION.md](docs/EVALUATION.md): scoring details and example commands
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): tests and how to generate new benchmark questions
- [docs/DATASETS.md](docs/DATASETS.md): unified dataset format and task layout

## Citation

```bibtex
@article{fesser2026rel,
  title         = {Evaluating Relational Reasoning in LLMs with REL},
  author        = {Lukas Fesser and Yasha Ektefaie and Ada Fang and Sham M. Kakade and Marinka Zitnik},
  year          = {2026},
  journal       = {arXiv preprint arXiv:2604.12176},
  eprint        = {2604.12176},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.12176}
}
```
