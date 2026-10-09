"""Load the workshop datasets from the repo's data/ folder."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "bulk_modulus": "bulk_modulus.csv",
    "shear_modulus": "shear_modulus.csv",
    "log10_thermal_conductivity": "log10_thermal_conductivity.csv",
    "bandgap": "bandgap.csv",
    "heat_capacity_raw": "heat_capacity_raw.csv",
    "heat_capacity_clean": "heat_capacity_clean.csv",
    "agnp": "agnp_flow_synthesis.csv",
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load one of the workshop datasets by short name (see data/README.md)."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Choose from: {sorted(DATASETS)}")
    return pd.read_csv(DATA_DIR / DATASETS[name])
