"""A small, dependency-light composition-based feature vector (CBFV).

This reproduces the idea behind the CBFV package
(https://github.com/kaaiian/CBFV): every element gets a row of tabulated
properties, and a compound is described by statistics of those properties
weighted by the element fractions in its formula.

It is intentionally short so you can read all of it in a few minutes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROPERTY_DIR = Path(__file__).parent / "element_properties"
ELEMENT_TABLES = ("oliynyk", "magpie", "mat2vec", "onehot", "random_200")
STATS = ("avg", "dev", "range", "max", "min", "mode")

_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|\[|\]|\d*\.?\d+)")


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a chemical formula into {element: amount}.

    Handles decimals and nested parentheses, e.g. 'Ba0.5Sr0.5TiO3',
    'Ca10(PO4)6(OH)2', 'N1Nd1'.
    """
    tokens = _TOKEN.findall(formula.replace(" ", ""))
    if "".join(tokens) != formula.replace(" ", ""):
        raise ValueError(f"Could not parse formula: {formula!r}")

    stack: list[dict[str, float]] = [defaultdict(float)]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        is_number = nxt is not None and re.fullmatch(r"\d*\.?\d+", nxt)
        if tok in "([":
            stack.append(defaultdict(float))
        elif tok in ")]":
            group = stack.pop()
            mult = float(nxt) if is_number else 1.0
            for el, n in group.items():
                stack[-1][el] += n * mult
            i += 1 if is_number else 0
        elif tok[0].isupper():
            stack[-1][tok] += float(nxt) if is_number else 1.0
            i += 1 if is_number else 0
        else:
            raise ValueError(f"Unexpected token {tok!r} in {formula!r}")
        i += 1
    if len(stack) != 1:
        raise ValueError(f"Unbalanced parentheses in {formula!r}")
    return {el: n for el, n in stack[0].items() if n > 0}


def element_fractions(formula: str) -> dict[str, float]:
    """Return {element: mole fraction} for a formula."""
    counts = parse_formula(formula)
    total = sum(counts.values())
    return {el: n / total for el, n in counts.items()}


@lru_cache(maxsize=None)
def load_element_properties(elem_prop: str = "oliynyk") -> pd.DataFrame:
    """Load an element-property table (rows = elements, columns = properties)."""
    if elem_prop not in ELEMENT_TABLES:
        raise ValueError(f"elem_prop must be one of {ELEMENT_TABLES}")
    table = pd.read_csv(PROPERTY_DIR / f"{elem_prop}.csv", index_col="element")
    return table.astype(float)


@lru_cache(maxsize=None)
def _property_arrays(elem_prop: str):
    table = load_element_properties(elem_prop)
    return list(table.columns), {el: i for i, el in enumerate(table.index)}, table.to_numpy()


def _feature_values(formula: str, elem_prop: str, stats) -> np.ndarray | None:
    props, row_of, arr = _property_arrays(elem_prop)
    fracs = element_fractions(formula)
    if any(el not in row_of for el in fracs):
        return None
    w = np.array(list(fracs.values()))
    P = arr[[row_of[el] for el in fracs]]           # (n_elements, n_props)
    avg = w @ P
    out = {
        "avg": avg,                                 # composition-weighted mean
        "dev": w @ np.abs(P - avg),                 # weighted mean absolute deviation
        "range": P.max(axis=0) - P.min(axis=0),
        "max": P.max(axis=0),
        "min": P.min(axis=0),
        "mode": P[np.argmax(w)],                    # property of the majority element
    }
    return np.concatenate([out[s] for s in stats])


def featurize_formula(formula: str, elem_prop: str = "oliynyk",
                      stats=STATS) -> pd.Series:
    """Feature vector for a single formula (all NaN if an element is missing)."""
    props, _, _ = _property_arrays(elem_prop)
    values = _feature_values(formula, elem_prop, tuple(stats))
    index = _column_names(props, stats)
    return pd.Series(np.nan if values is None else values, index=index)


def _column_names(props, stats):
    return [f"{s}_{p}" for s in stats for p in props]


def generate_features(df: pd.DataFrame, elem_prop: str = "oliynyk",
                      formula_col: str = "formula", target_col: str | None = "target",
                      stats=STATS, drop_duplicates: bool = True, verbose: bool = True):
    """Featurize a DataFrame of formulas, mirroring CBFV.generate_features.

    Returns X (features), y (targets or None), formulae, skipped (list of
    formulas containing elements absent from the property table).
    Missing feature values are filled with the column median.
    """
    df = df.copy()
    if drop_duplicates:
        n0 = len(df)
        df = df.drop_duplicates(formula_col)
        if verbose and len(df) < n0:
            print(f"Dropped {n0 - len(df)} duplicate formulas")

    props, _, _ = _property_arrays(elem_prop)
    n_cols = len(props) * len(stats)
    rows = [_feature_values(f, elem_prop, tuple(stats)) for f in df[formula_col]]
    rows = [np.full(n_cols, np.nan) if r is None else r for r in rows]
    X = pd.DataFrame(np.vstack(rows), columns=_column_names(props, stats))
    formulae = df[formula_col].reset_index(drop=True)
    y = df[target_col].reset_index(drop=True) if target_col else None

    bad = X.isna().all(axis=1)
    skipped = formulae[bad].tolist()
    X, formulae = X[~bad].reset_index(drop=True), formulae[~bad].reset_index(drop=True)
    if y is not None:
        y = y[~bad].reset_index(drop=True)
    X = X.fillna(X.median())
    if verbose:
        print(f"Featurized {len(X)} formulas into {X.shape[1]} '{elem_prop}' features"
              + (f" (skipped {len(skipped)})" if skipped else ""))
    return X, y, formulae, skipped
