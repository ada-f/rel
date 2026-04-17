#
# Numerical RPM sample generators for algebra_benchmark (REL-A1 … REL-A4).
# Rule logic matches I-RAVEN-X (see iravenx_task.py in raven-large-language-models):
# Constant, Progression, Distribute_Three, Arithmetic, unbiased multi-choice
# construction, and the same row-major flattening of the n×n attribute grids.
#
# Output format: each context/choice "panel" is a 1×3 matrix [[Type, Size, Color]]
# (floats) for one grid cell — aligned with I-RAVEN-X center_single attributes.
# Context panel count is n*n - 1 (missing bottom-right cell); choices are always 8.
#
# Parameters gridsize (= n) and maxval follow I-RAVEN-X semantics (see iravenx_task).
#

from __future__ import annotations

import random
from typing import Any

import numpy as np

from algebra_benchmark.format import validate_sample
from algebra_benchmark.tasks import RULE_TO_TASK, TASK_TO_RULE

N_CHOICES = 8
LOWVAL = 0


def _np_rng_from_py_rng(py_rng: random.Random) -> np.random.Generator:
    """Derive an independent NumPy Generator from Python's Random (one draw)."""
    return np.random.default_rng(py_rng.randrange(1 << 63))


def _triple_panel(t: int | float, s: int | float, c: int | float) -> list[list[float]]:
    """One grid cell as a 1×3 matrix (Type, Size, Color)."""
    return [[float(t), float(s), float(c)]]


def _copy_matrix(M: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in M]


def _constant(n: int, maxval: int, np_rng: np.random.Generator) -> np.ndarray:
    """I-RAVEN-X Constant(n, maxval): tile random column across n×n (same as iravenx_task)."""
    return np.tile(np_rng.integers(low=LOWVAL, high=maxval, size=n, dtype=np.int64), (n, 1)).T


def _progression(n: int, maxval: int, np_rng: np.random.Generator, py_rng: random.Random) -> np.ndarray:
    """I-RAVEN-X Progression(n, maxval); delta in {-2, -1, 1, 2}."""
    delta = int(py_rng.choice([-2, -1, 1, 2]))
    context = np.zeros((n, n), dtype=np.int64)
    if delta > 0:
        context[:, 0] = np_rng.integers(
            low=LOWVAL, high=max(LOWVAL + 1, maxval - (n - 1) * delta), size=n, dtype=np.int64
        )
        for col in range(1, n):
            context[:, col] = context[:, col - 1] + delta
    elif delta < 0:
        context[:, -1] = np_rng.integers(
            low=LOWVAL, high=max(LOWVAL + 1, maxval + ((n - 1) * delta)), size=n, dtype=np.int64
        )
        for col in range(n - 2, -1, -1):
            context[:, col] = context[:, col + 1] - delta
    return context


def _non_repeating_list(n: int, maxval: int, py_rng: random.Random) -> np.ndarray:
    """I-RAVEN-X non_repeating_list (inclusive maxval on random.randint)."""
    random_list: list[int] = []
    while len(random_list) < n:
        r = py_rng.randint(LOWVAL, maxval)
        if r not in random_list:
            random_list.append(r)
    return np.array(random_list, dtype=np.int64)


def _distribute_three(n: int, maxval: int, py_rng: random.Random) -> np.ndarray:
    """I-RAVEN-X Distribute_Three(n, maxval)."""
    context = np.zeros((n, n), dtype=np.int64)
    context[0, :] = _non_repeating_list(n, maxval, py_rng)
    delta = int(py_rng.choice([-1, 1]))
    for row in range(1, n):
        context[row, :] = np.roll(context[row - 1, :], delta)
    return context


def _sample_arithmetic(n: int, maxval: int, np_rng: np.random.Generator) -> np.ndarray:
    """I-RAVEN-X sample_arithmetic (operand vector length n)."""
    context = np.zeros(n, dtype=np.int64)
    target_sum = int(np_rng.integers(low=max(1, maxval // 2), high=maxval + 1))
    curr_maxval = target_sum + 1
    for col in range(n):
        context[col] = int(np_rng.integers(low=LOWVAL, high=max(LOWVAL + 1, curr_maxval)))
        curr_maxval = target_sum - int(np.sum(context)) + 1
    return context


def _generate_arithmetic_shuffle(n: int, maxval: int, np_rng: np.random.Generator) -> np.ndarray:
    """I-RAVEN-X generate_arithmetic_shuffle."""
    context = np.zeros((n, n - 1), dtype=np.int64)
    for row in range(n):
        while np.sum(context[row]) == 0:
            context[row] = _sample_arithmetic(n - 1, maxval, np_rng)
    context_permuted = np_rng.permuted(context, axis=1).copy()
    for row in range(n):
        if context_permuted[row, -1] == LOWVAL:
            non_zero_idx = np.nonzero(context_permuted[row, :] - LOWVAL)[0]
            context_permuted[row, -1] = context_permuted[row, non_zero_idx[0]]
            context_permuted[row, non_zero_idx[0]] = LOWVAL
    return context_permuted


def _arithmetic(
    n: int,
    maxval: int,
    arithmetic_strategy: str,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> np.ndarray:
    """I-RAVEN-X Arithmetic(n, maxval, arithmetic_strategy)."""
    sign = int(py_rng.choice([-1, 1]))
    context = np.zeros((n, n), dtype=np.int64)
    if arithmetic_strategy == "uniform":
        high_op = max(1, int(maxval / max(1, (n - 1))))
        context_sum_operands = np_rng.integers(low=LOWVAL, high=high_op, size=(n, n - 1), dtype=np.int64)
    elif arithmetic_strategy == "shuffle":
        context_sum_operands = _generate_arithmetic_shuffle(n, maxval, np_rng)
    else:
        raise ValueError(f"Unknown arithmetic_strategy: {arithmetic_strategy!r} (use 'shuffle' or 'uniform')")

    if sign > 0:
        context[:, :-1] = context_sum_operands
        context[:, -1] = np.sum(context, axis=1)
    else:
        context[:, 1:] = context_sum_operands
        context[:, 0] = np.sum(context, axis=1)
    return context


def _rule_matrix(
    rule: str,
    n: int,
    maxval: int,
    arithmetic_strategy: str,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> np.ndarray:
    if rule == "Constant":
        return _constant(n, maxval, np_rng)
    if rule == "Progression":
        return _progression(n, maxval, np_rng, py_rng)
    if rule == "Distribute_Three":
        return _distribute_three(n, maxval, py_rng)
    if rule == "Arithmetic":
        return _arithmetic(n, maxval, arithmetic_strategy, np_rng, py_rng)
    raise ValueError(f"Unknown I-RAVEN-X rule: {rule}")


def _unbiased_candidates(
    context_panels: np.ndarray,
    maxval: int,
    np_rng: np.random.Generator,
    py_rng: random.Random,
    strategy: str = "existent",
) -> tuple[np.ndarray, int]:
    """
    I-RAVEN-X unbiased_candidates (Stratified Rule-Aware style).
    context_panels: shape (3, n, n) — Type, Size, Color grids.
    Returns candidates (8, 3) and target index in [0, 7].
    """
    answer = context_panels[:, -1, -1].astype(np.int64, copy=False)
    wvals: list[int] = []
    for i in range(3):
        wval = int(answer[i])
        while wval == int(answer[i]):
            if strategy == "random":
                wval = py_rng.randint(LOWVAL, maxval)
            elif strategy == "existent":
                wval = int(np_rng.choice(np.unique(context_panels)))
            elif strategy == "existent_att":
                wval = int(np_rng.choice(np.unique(context_panels[i])))
            else:
                raise ValueError("Strategy not implemented")

        wvals.append(wval)

    def recursive_tweak(i: int, panel: np.ndarray) -> np.ndarray:
        if i == 3:
            return panel[None, :]
        tweaked = np.copy(panel)
        tweaked[i] = wvals[i]
        return np.concatenate((recursive_tweak(i + 1, panel), recursive_tweak(i + 1, tweaked)), axis=0)

    candidates = np_rng.permutation(recursive_tweak(0, answer.astype(np.float64)))
    target = int(np.where((candidates == answer).all(axis=1))[0][0])
    return candidates.astype(np.int64), target


def _get_iravenx_sample(
    n: int,
    maxval: int,
    rule: str,
    arithmetic_strategy: str,
    py_rng: random.Random,
    np_rng: np.random.Generator,
) -> dict[str, Any]:
    """
    One I-RAVEN-X center_single sample (fixed rule on Type, Size, Color).
    Returns panels (n*n-1), choices (8), target, each panel 1×3.
    """
    type_val = _rule_matrix(rule, n, maxval, arithmetic_strategy, np_rng, py_rng)
    size_val = _rule_matrix(rule, n, maxval, arithmetic_strategy, np_rng, py_rng)
    color_val = _rule_matrix(rule, n, maxval, arithmetic_strategy, np_rng, py_rng)

    stacked = np.stack((type_val, size_val, color_val))
    candidates, target = _unbiased_candidates(stacked, maxval, np_rng, py_rng, strategy="existent")

    size_val = np.concatenate([size_val.flatten()[:-1], candidates[:, 1]])
    type_val = np.concatenate([type_val.flatten()[:-1], candidates[:, 0]])
    color_val = np.concatenate([color_val.flatten()[:-1], candidates[:, 2]])

    n_ctx = n * n - 1
    panels = [
        _triple_panel(int(type_val[j]), int(size_val[j]), int(color_val[j])) for j in range(n_ctx)
    ]
    choices = [
        _triple_panel(
            int(type_val[n_ctx + k]),
            int(size_val[n_ctx + k]),
            int(color_val[n_ctx + k]),
        )
        for k in range(N_CHOICES)
    ]
    sample = {"panels": panels, "target": target, "choices": choices}
    validate_sample(sample, n_choices=N_CHOICES)
    return sample


def generate_constant_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
    *,
    arithmetic_strategy: str = "shuffle",
) -> dict[str, Any]:
    """REL-A1 / Constant — I-RAVEN-X definition."""
    return _get_iravenx_sample(gridsize, maxval, "Constant", arithmetic_strategy, rng, _np_rng_from_py_rng(rng))


def generate_progression_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
    *,
    arithmetic_strategy: str = "shuffle",
) -> dict[str, Any]:
    """REL-A2 / Progression — I-RAVEN-X definition."""
    return _get_iravenx_sample(
        gridsize, maxval, "Progression", arithmetic_strategy, rng, _np_rng_from_py_rng(rng)
    )


def generate_distribute_three_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
    *,
    arithmetic_strategy: str = "shuffle",
) -> dict[str, Any]:
    """REL-A3 / Distribute_Three — I-RAVEN-X definition."""
    return _get_iravenx_sample(
        gridsize, maxval, "Distribute_Three", arithmetic_strategy, rng, _np_rng_from_py_rng(rng)
    )


def generate_arithmetic_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
    *,
    arithmetic_strategy: str = "shuffle",
) -> dict[str, Any]:
    """REL-A4 / Arithmetic — I-RAVEN-X definition (shuffle or uniform operand sampling)."""
    return _get_iravenx_sample(
        gridsize, maxval, "Arithmetic", arithmetic_strategy, rng, _np_rng_from_py_rng(rng)
    )


def generate_sample(
    task: str,
    gridsize: int,
    maxval: int,
    rng: random.Random,
    *,
    arithmetic_strategy: str = "shuffle",
) -> dict[str, Any]:
    """
    Generate one sample for the given task.

    For REL-A1 … REL-A4 (matrix / I-RAVEN-X rules), ``gridsize`` is the panel grid
    size ``n`` and ``maxval`` is the exclusive upper bound for NumPy integer draws
    and the inclusive upper bound for Python ``randint`` draws, matching the
    original I-RAVEN-X script mixture.

    ``arithmetic_strategy`` is only used for REL-A4 (``'shuffle'`` or ``'uniform'``).
    """
    if gridsize < 2:
        raise ValueError("gridsize (n) must be >= 2 for I-RAVEN-X center_single samples")
    if maxval < 1:
        raise ValueError("maxval must be >= 1")

    tid = task if task.startswith("REL-") else RULE_TO_TASK.get(task)
    rule = TASK_TO_RULE.get(tid, task) if tid else task

    if rule == "constant" or tid == "REL-A1":
        return generate_constant_sample(gridsize, maxval, rng, arithmetic_strategy=arithmetic_strategy)
    if rule == "progression" or tid == "REL-A2":
        return generate_progression_sample(gridsize, maxval, rng, arithmetic_strategy=arithmetic_strategy)
    if task == "permutation":
        return generate_placeholder_sample(gridsize, maxval, rng)
    if rule == "distribute-three" or tid == "REL-A3":
        return generate_distribute_three_sample(gridsize, maxval, rng, arithmetic_strategy=arithmetic_strategy)
    if rule == "arithmetic" or tid == "REL-A4":
        return generate_arithmetic_sample(gridsize, maxval, rng, arithmetic_strategy=arithmetic_strategy)
    return generate_placeholder_sample(gridsize, maxval, rng)


def generate_placeholder_sample(
    gridsize: int,
    maxval: int,
    rng: random.Random,
) -> dict[str, Any]:
    """
    Placeholder sample (valid structure). Used for REL-A5…REL-A7 and permutation.
    """
    np_rng = _np_rng_from_py_rng(rng)

    def _rand_matrix() -> list[list[float]]:
        return [[float(np_rng.uniform(0, maxval)) for _ in range(gridsize)] for _ in range(gridsize)]

    n_panels = 8
    m = _rand_matrix()
    panels = [_copy_matrix(m) for _ in range(n_panels)]
    target = int(rng.randint(0, N_CHOICES - 1))
    choices: list[Any] = [None] * N_CHOICES
    choices[target] = _copy_matrix(m)
    for i in range(N_CHOICES):
        if choices[i] is None:
            choices[i] = _rand_matrix()
    sample = {"panels": panels, "target": target, "choices": choices}
    validate_sample(sample, n_choices=N_CHOICES)
    return sample


def generate_dataset(
    task: str,
    num_samples: int,
    gridsize: int,
    maxval: int = 1000,
    seed: int | None = None,
    *,
    arithmetic_strategy: str = "shuffle",
) -> list[dict[str, Any]]:
    """
    Generate ``num_samples`` samples. Uses ``seed`` for reproducibility when set.

    ``arithmetic_strategy`` applies to REL-A4 / arithmetic (``'shuffle'`` or ``'uniform'``).
    """
    rng = random.Random(seed)
    return [
        generate_sample(task, gridsize, maxval, rng, arithmetic_strategy=arithmetic_strategy)
        for _ in range(num_samples)
    ]


if __name__ == "__main__":
    py = random.Random(42)
    for task_id in ("REL-A1", "REL-A2", "REL-A3", "REL-A4"):
        s = generate_sample(task_id, 3, 1000, py, arithmetic_strategy="shuffle")
        assert len(s["choices"]) == 8
        assert len(s["panels"]) == 3 * 3 - 1
        assert 0 <= s["target"] < 8
        assert all(len(p) == 1 and len(p[0]) == 3 for p in s["panels"])
    data = generate_dataset("REL-A1", 5, 3, 1000, seed=42)
    assert len(data) == 5
    print("generators.py: I-RAVEN-X constant, progression, distribute-three, arithmetic checks passed.")
