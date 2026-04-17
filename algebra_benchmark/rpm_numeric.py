#
# Numerical RPM/RPT: build context and answer_choices text from a sample.
# No visual inputs; panels and choices are numerical matrices/tensors only.
# Compatible with the rpmllm mamba environment (Python 3.10).
#
# Context layout for center_single–style samples (one triple per grid cell) matches
# ``Branch._context`` in raven-large-language-models ``rpm_dataset.py``:
# row-major cells over n×n with bottom-right missing, e.g. for n = nshow = 3:
#   row 1: (…), (…), (…); row 2: (…), (…), (…); row 3: (…), (…),
#

from __future__ import annotations

import math
import numbers
from typing import Any


def _num_str(x: Any) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, numbers.Real) and not isinstance(x, bool) and float(x).is_integer():
        return str(int(x))
    return str(x)


def _is_numeric_scalar(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _is_single_cell_triple(panel: Any) -> bool:
    """True if panel is [[Type, Size, Color]] (one RPM cell)."""
    if not isinstance(panel, list) or len(panel) != 1:
        return False
    row = panel[0]
    if not isinstance(row, list) or len(row) != 3:
        return False
    return all(_is_numeric_scalar(x) for x in row)


def _rpm_cell_display(panel: Any) -> str:
    """
    One RPM grid cell exactly as printed in the row context: ``(Type, Size, Color)``.
    Used for both context tuples and answer-option lines so formatting always matches.
    """
    if _is_single_cell_triple(panel):
        t, s, c = panel[0][0], panel[0][1], panel[0][2]
        return f"({_num_str(t)}, {_num_str(s)}, {_num_str(c)})"
    return _format_panel(panel)


def _triple_grid_prompt_mode(sample: dict[str, Any]) -> bool:
    """True when we use row-major ``row k:`` layout (same as Raven ``Branch._context``)."""
    panels = sample["panels"]
    n = _infer_grid_side(panels)
    return n is not None and all(_is_single_cell_triple(p) for p in panels)


def _format_panel(panel: Any) -> str:
    """Format a single panel (2D or 3D list of numbers) as a string for the prompt."""
    if isinstance(panel, list) and len(panel) > 0:
        if isinstance(panel[0], (int, float)):
            return str(panel)
        if isinstance(panel[0], list):
            return "\n".join(str(row) for row in panel)
        # 3D: list of 2D slices
        return "\n---\n".join(_format_panel(slice_) for slice_ in panel)
    return str(panel)


def _infer_grid_side(panels: list[Any]) -> int | None:
    """
    Return n if ``len(panels) == n*n - 1`` for integer n >= 2; else None.
    """
    m = len(panels) + 1
    if m < 4:
        return None
    n = math.isqrt(m)
    if n * n != m or n < 2:
        return None
    return n


def _branch_style_row_context(cell_strings: list[str], n: int, nshow: int) -> str:
    """
    Same template as ``rpm_dataset.Branch._context`` (raven-large-language-models).
    ``cell_strings`` is row-major length ``n*n - 1`` (missing bottom-right).
    """
    if len(cell_strings) != n * n - 1:
        raise ValueError(f"Expected {n * n - 1} cells, got {len(cell_strings)}")
    arr = cell_strings
    tpl = ""
    for row in range(nshow):
        tpl = tpl + "row " + str(row + 1) + ": {}"
        if row < nshow - 1:
            for _ in range(1, n):
                tpl += ", {}"
            tpl += "; "
        else:
            for _ in range(1, n - 1):
                tpl += ", {}"
            tpl += ", "
    return tpl.format(*arr[((n - nshow) * n) : (n**2 - 1)])


def sample_to_context(sample: dict[str, Any]) -> str:
    """
    Build the context string for the prompt.

    For I-RAVEN-X–style samples (each panel is one ``[[Type, Size, Color]]`` cell and
    ``len(panels) + 1`` is a perfect square), uses the Raven row layout
    (``row 1: …; row 2: …;`` with the last row showing only the first ``n - 1`` cells
    and a trailing comma, matching ``Branch._context`` for ``nshow == n``).

    Otherwise falls back to ``Panel i:`` blocks (e.g. dense matrices / RPT).
    """
    panels = sample["panels"]
    n = _infer_grid_side(panels)
    if _triple_grid_prompt_mode(sample):
        cells = [_rpm_cell_display(p) for p in panels]
        return _branch_style_row_context(cells, n=n, nshow=n)
    parts = []
    for i, p in enumerate(panels):
        parts.append(f"Panel {i}:\n{_format_panel(p)}")
    return "\n\n".join(parts)


def sample_to_answer_choices(sample: dict[str, Any]) -> str:
    """
    Build the answer-choices string for the prompt: the 8 candidate panels.
    In triple-grid mode, each option uses ``_rpm_cell_display`` — the same formatter as
    context cells — so ``Answer k:`` lines match ``(Type, Size, Color)`` exactly.
    """
    choices = sample["choices"]
    triple = _triple_grid_prompt_mode(sample)
    parts = []
    for i, c in enumerate(choices):
        disp = _rpm_cell_display(c) if triple else _format_panel(c)
        parts.append(f"Answer {i}: {disp}")
    return "\n".join(parts)


def build_query(
    sample: dict[str, Any],
    *,
    prefix: str = (
        "Complete the Raven's progressive matrix. Only return the missing panel index (0-7)!\n"
    ),
    incontext: str = "",
) -> str:
    """
    Build the full query string: prefix + optional in-context examples + context + answer set.
    The model is expected to return a single index 0-7 (0-based choice index). The answer set
    uses ``Answer 0`` … ``Answer 7`` (labels match indices).
    """
    context = sample_to_context(sample)
    answer_choices = sample_to_answer_choices(sample)
    out = prefix
    if incontext:
        out += incontext
    out += context
    out += "\n\nAnswer set:\n" + answer_choices
    return out


def get_choices(sample: dict[str, Any]) -> list[Any]:
    """Return the list of 8 choice panels (for scoring: choice_array[pred_idx])."""
    return sample["choices"]


if __name__ == "__main__":
    s = {
        "panels": [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]], [[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0]], [[4.0, 4.0, 4.0]], [[5.0, 5.0, 5.0]]],
        "choices": [[[9.0, 9.0, 9.0]]] * 8,
        "target": 0,
    }
    ctx = sample_to_context(s)
    assert ctx.startswith("row 1:")
    assert "row 2:" in ctx and "row 3:" in ctx
    assert ctx.rstrip().endswith(",")
    first_cell = _rpm_cell_display(s["panels"][0])
    assert first_cell in ctx
    ans = sample_to_answer_choices(s)
    assert ans.splitlines()[0] == f"Answer 0: {_rpm_cell_display(s['choices'][0])}"
    print("rpm_numeric.py: row-major context OK; answer lines match cell formatter")
    print(ctx)
    print(ans.splitlines()[0])
