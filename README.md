# REL: Relational Reasoning Benchmarks

<p align="center">
  <a href="https://arxiv.org/abs/YOUR_ARXIV_ID">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/YOUR_DATASET_PATH">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
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

This repo is configured around `uv` and the local helper script in [setup_uv_env.sh](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/setup_uv_env.sh).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source setup_uv_env.sh
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

If you want the full environment notes, including cache locations and troubleshooting, see [UV_SETUP.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/UV_SETUP.md).

## Questions

Run the questions provided in `REL/` with your model or download them from Hugging Face with `hf download ada-f/rel --repo-type dataset --local-dir .`.

### Evaluate responses from your own pipeline

If you already have model responses and just want scoring, use the domain evaluators directly:

- Chemistry: `chem_benchmark.evaluation`
- Biology: `bio_benchmark.evaluation`
- Algebra: `algebra_benchmark.evaluation`

The expected answer formats and minimal examples are in [docs/EVALUATION.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/docs/EVALUATION.md).

## More Docs

- [docs/EVALUATION.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/docs/EVALUATION.md): scoring details and example commands
- [docs/DEVELOPMENT.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/docs/DEVELOPMENT.md): tests, dataset generation, and validation workflows
- [docs/DATASETS.md](/n/holylabs/LABS/mzitnik_lab/Users/afang/relational_reasoning/docs/DATASETS.md): unified dataset format and task layout

## Citation

```bibtex
@article{fesser2026rel,
  title={Exploring Relational Reasoning Capabilities in LLMs with REL},
  author={Fesser, Lukas and Ektefaie, Yasha and Fang, Ada and Zitnik, Marinka},
  journal={},
  year={2026}
}
```
