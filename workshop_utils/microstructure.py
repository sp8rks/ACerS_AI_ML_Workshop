"""Microstructure images as numbers: 2-point statistics, a synthetic porous ceramic and its conductivity.

2-point statistics follow the PyMKS idea (Kalidindi group): the probability
that two points separated by a vector r both land in the phase of interest.
Computed here with FFTs in numpy so nothing extra needs installing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .data import DATA_DIR

MICROGRAPH_DIR = DATA_DIR / "micrographs"


def load_micrograph(name: str, size: int | None = 256) -> np.ndarray:
    """Load a binarized micrograph from data/micrographs as a 0/1 array (1 = white phase)."""
    from PIL import Image
    path = MICROGRAPH_DIR / (name if name.endswith(".png") else name + ".png")
    im = Image.open(path).convert("L")
    if size:
        s = size / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.NEAREST)
    return (np.asarray(im) > 127).astype(np.uint8)


def two_point_stats(img, periodic: bool = False) -> np.ndarray:
    """Autocorrelation S2(r) of phase 1, centered so r = 0 is in the middle of the map.

    S2(0) is the volume fraction; S2 decays to (volume fraction)^2 at long
    distances, and the shape of the decay encodes feature size and orientation.
    """
    f = np.asarray(img, float)
    if periodic:
        F = np.fft.fft2(f)
        s2 = np.fft.ifft2(F * F.conj()).real / f.size
    else:
        shape = [2 * n for n in f.shape]            # zero-pad, then divide by the number of overlapping pixels
        F = np.fft.fft2(f, shape)
        M = np.fft.fft2(np.ones_like(f), shape)
        s2 = np.fft.ifft2(F * F.conj()).real / np.maximum(np.fft.ifft2(M * M.conj()).real, 1)
    return np.fft.fftshift(s2)


def central_window(s2, half: int):
    """The (2*half+1)^2 block of a centered 2-point map around r = 0."""
    cy, cx = np.array(s2.shape) // 2
    return s2[cy - half: cy + half + 1, cx - half: cx + half + 1]


def make_porous_microstructure(n: int = 64, porosity: float = 0.25, length: float = 3.0,
                               aspect: float = 1.0, angle: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Synthetic porous ceramic: a smoothed random field, thresholded so `porosity` of pixels are pores (1).

    `length` sets pore size (pixels), `aspect` stretches pores along `angle` (degrees from the x axis).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n))
    ky, kx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    t = np.deg2rad(angle)
    k_par = kx * np.cos(t) + ky * np.sin(t)
    k_perp = -kx * np.sin(t) + ky * np.cos(t)
    sig_par, sig_perp = length * np.sqrt(aspect), length / np.sqrt(aspect)
    filt = np.exp(-2 * np.pi ** 2 * ((k_par * sig_par) ** 2 + (k_perp * sig_perp) ** 2))
    field = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    return (field > np.quantile(field, 1 - porosity)).astype(np.uint8)


def effective_conductivity(pores, k_solid: float = 1.0, k_pore: float = 0.02) -> float:
    """Effective thermal conductivity along x (finite volumes, steady state).

    Hot left edge, cold right edge, periodic top/bottom. Face conductances use
    the harmonic mean of the two pixels, so pores block heat flow.
    """
    k = np.where(np.asarray(pores) > 0, k_pore, k_solid).astype(float)
    ny, nx = k.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, vals = [], [], []
    b = np.zeros(nx * ny)
    diag = np.zeros(nx * ny)

    def couple(a, c, g):
        rows.extend([a, c]); cols.extend([c, a]); vals.extend([-g, -g])  # lists of arrays
        np.add.at(diag, a, g); np.add.at(diag, c, g)

    gx = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    couple(idx[:, :-1].ravel(), idx[:, 1:].ravel(), gx.ravel())
    ky_ = np.roll(k, -1, axis=0)
    gy = 2 * k * ky_ / (k + ky_)
    couple(idx.ravel(), np.roll(idx, -1, axis=0).ravel(), gy.ravel())
    # Dirichlet boundaries through a half-cell: T = 1 on the left, T = 0 on the right
    gl, gr = 2 * k[:, 0], 2 * k[:, -1]
    np.add.at(diag, idx[:, 0], gl); b[idx[:, 0]] += gl
    np.add.at(diag, idx[:, -1], gr)
    A = sparse.csr_matrix((np.concatenate(vals + [diag]),
                           (np.concatenate(rows + [np.arange(nx * ny)]),
                            np.concatenate(cols + [np.arange(nx * ny)]))), shape=(nx * ny, nx * ny))
    T = spsolve(A, b).reshape(ny, nx)
    flux = (gl * (1 - T[:, 0])).sum()      # heat entering through the left face
    return float(flux * nx / ny / k_solid)  # normalized: 1.0 for a fully dense part
