"""Supply risk of a composition: the Herfindahl-Hirschman Index (HHI) of its elements.

HHI measures how concentrated an element's production (or reserves) is among countries:
low HHI (< 1500) = widely produced, high HHI (> 2500) = concentrated supply. A compound's HHI is the
mass-fraction-weighted average of its elements' HHI, following Gaultois et al., "Data-driven review of
thermoelectric materials: performance and resource considerations", Chem. Mater. 25, 2911 (2013).
Element values from pymatgen (MIT license), pymatgen/analysis/hhi_data.csv.
"""

from pathlib import Path

import pandas as pd

from .featurize import element_fractions

_DIR = Path(__file__).resolve().parent / "element_properties"


def _tables():
    hhi = pd.read_csv(_DIR / "hhi.csv").set_index("element")
    mass = pd.read_csv(_DIR / "magpie.csv").set_index("element")["AtomicWeight"]
    return hhi, mass


def compound_hhi(formula, kind="production"):
    """Mass-weighted HHI of a formula (kind = 'production' or 'reserve'); NaN if an element is missing."""
    hhi, mass = _tables()
    col = "HHI_production" if kind == "production" else "HHI_reserve"
    frac = element_fractions(formula)
    try:
        w = {el: x * mass[el] for el, x in frac.items()}
        total = sum(w.values())
        return float(sum(w[el] / total * hhi.loc[el, col] for el in w))
    except KeyError:
        return float("nan")
