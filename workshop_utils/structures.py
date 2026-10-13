"""Crystal structures as numbers: neighbor lists, simple structure features and crystal graphs.

Only numpy is needed. A structure is a dict with a 3x3 `lattice` (rows are the
lattice vectors, in angstrom), a list of `species` and fractional coordinates
`frac`, the same information as a CIF file. pymatgen does all of this and more
(`Structure.get_all_neighbors`); the code here is short so you can read it.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import DATA_DIR
from .featurize import load_element_properties

STRUCTURE_FILE = DATA_DIR / "mp_bulk_modulus_structures.json.gz"


def load_structures() -> pd.DataFrame:
    """Materials Project structures with DFT bulk moduli (Matbench log10 K_VRH subset, <= 16 atoms/cell)."""
    with gzip.open(STRUCTURE_FILE, "rt", encoding="utf-8") as fh:
        df = pd.DataFrame(json.load(fh))
    df["K_GPa"] = 10 ** df["log10_K"]
    df["n_sites"] = df["species"].map(len)
    return df


@lru_cache(maxsize=None)
def _atomic_numbers() -> dict[str, int]:
    olk = load_element_properties("oliynyk")
    return {el: int(z) for el, z in olk["Atomic_Number"].items()}


@lru_cache(maxsize=None)
def _element_lookup() -> dict[str, tuple[float, float]]:
    olk = load_element_properties("oliynyk")
    return {el: (float(m), float(r)) for el, m, r in zip(olk.index, olk["Atomic_Weight"], olk["Covalent_Radius"])}


def atomic_number(symbol: str) -> int:
    return _atomic_numbers()[symbol]


def neighbor_list(lattice, frac, cutoff: float = 8.0, max_neighbors: int = 12):
    """Neighbors of every atom in a periodic crystal.

    Returns arrays (center, neighbor, distance). Each atom keeps its
    `max_neighbors` nearest neighbors within `cutoff` angstrom, counting
    periodic images (an atom can neighbor its own image in the next cell).
    """
    L = np.asarray(lattice, float)
    cart = np.asarray(frac, float) @ L
    vol = abs(np.linalg.det(L))
    # how many cells to repeat along each axis so that no neighbor within `cutoff` is missed
    reps = [int(np.ceil(cutoff * np.linalg.norm(np.cross(L[(i + 1) % 3], L[(i + 2) % 3])) / vol))
            for i in range(3)]
    grid = np.stack(np.meshgrid(*[np.arange(-r, r + 1) for r in reps], indexing="ij"), -1).reshape(-1, 3)
    images = (cart[None, :, :] + (grid @ L)[:, None, :]).reshape(-1, 3)        # (n_images * n_atoms, 3)
    image_atom = np.tile(np.arange(len(cart)), len(grid))
    centers, neighbors, dists = [], [], []
    for i, r in enumerate(cart):
        d = np.linalg.norm(images - r, axis=1)
        ok = (d > 1e-4) & (d <= cutoff)
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(d[idx])][:max_neighbors]
        centers.append(np.full(len(idx), i))
        neighbors.append(image_atom[idx])
        dists.append(d[idx])
    return np.concatenate(centers), np.concatenate(neighbors), np.concatenate(dists)


def structure_features(row, tol: float = 1.15) -> dict[str, float]:
    """A handful of physically meaningful, hand-built structure descriptors."""
    L = np.asarray(row["lattice"], float)
    vol = abs(np.linalg.det(L))
    species = row["species"]
    props = _element_lookup()
    mass = sum(props[s][0] for s in species)
    r_cov = np.array([props[s][1] for s in species])
    c, n, d = neighbor_list(L, row["frac"], cutoff=6.0, max_neighbors=24)
    nn = np.array([d[c == i].min() for i in range(len(species))])
    # coordination number: neighbors within 15% of each atom's nearest-neighbor distance
    cn = np.array([(d[c == i] <= tol * nn[i]).sum() for i in range(len(species))])
    return {
        "volume_per_atom": vol / len(species),
        "density_g_cm3": mass / vol * 1.66054,
        "packing_fraction": float((4 / 3) * np.pi * (r_cov ** 3).sum() / vol),
        "mean_nn_distance": nn.mean(),
        "min_nn_distance": nn.min(),
        "mean_coordination": cn.mean(),
        "max_coordination": cn.max(),
        "n_sites": len(species),
    }


def crystal_graph(row, cutoff: float = 8.0, max_neighbors: int = 12) -> dict:
    """Nodes = atoms (atomic numbers), edges = neighbor pairs with their distances."""
    c, n, d = neighbor_list(row["lattice"], row["frac"], cutoff, max_neighbors)
    return {"z": np.array([atomic_number(s) for s in row["species"]]),
            "center": c, "neighbor": n, "dist": d.astype(np.float32)}


def batch_graphs(graphs):
    """Stack several crystal graphs into one big disconnected graph (how GNN libraries batch)."""
    z, cen, nbr, dist, graph_id, offset = [], [], [], [], [], 0
    for k, g in enumerate(graphs):
        z.append(g["z"]); dist.append(g["dist"])
        cen.append(g["center"] + offset); nbr.append(g["neighbor"] + offset)
        graph_id.append(np.full(len(g["z"]), k))
        offset += len(g["z"])
    return {"z": np.concatenate(z), "center": np.concatenate(cen), "neighbor": np.concatenate(nbr),
            "dist": np.concatenate(dist), "graph_id": np.concatenate(graph_id), "n_graphs": len(graphs)}
