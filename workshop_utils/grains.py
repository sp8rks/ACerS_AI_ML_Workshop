"""Grain size and porosity from micrographs: a synthetic sintered ceramic with known ground truth,
plus the classic measurements (ASTM E112 line intercept, equivalent circle diameter, porosity).

The synthetic images mimic a thermally etched SEM micrograph: grains with slightly different
gray levels, dark grooves at grain boundaries, dark pores (mostly on boundaries) with bright rims,
uneven illumination, blur and noise. Because we generated them, we know the true answer.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def _power_voronoi(size, seeds, weights, k=12):
    """Label each pixel with the seed minimizing |x - s|^2 - w (a power diagram: grains of varied size).

    Only the k nearest seeds are checked for each pixel, which is exact for modest weight spreads.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    pts = np.c_[yy.ravel(), xx.ravel()]
    k = min(k, len(seeds))
    d, idx = cKDTree(seeds).query(pts, k=k)
    best = np.argmin(d ** 2 - weights[idx], axis=1)
    return idx[np.arange(len(pts)), best].reshape(size, size) + 1


def synthetic_micrograph(n_grains=150, porosity=0.03, size=512, um_per_px=0.05, noise=0.06,
                         boundary_depth=0.35, size_spread=0.35, seed=0):
    """Make a synthetic SEM-like micrograph of a sintered ceramic.

    Returns a dict with the image (float, 0-1) and the ground truth: ``grains`` (label map, 0 = pore),
    ``pores`` (bool mask), ``boundaries`` (bool mask), ``um_per_px`` and ``truth`` (summary numbers).
    """
    rng = np.random.default_rng(seed)
    # seeds: random, then one Lloyd-like relaxation for more equiaxed grains
    seeds = rng.uniform(0, size, (n_grains, 2))
    lab = _power_voronoi(size, seeds, np.zeros(n_grains))
    seeds = np.array(ndi.center_of_mass(np.ones_like(lab), lab, range(1, n_grains + 1)))
    seeds = np.nan_to_num(seeds, nan=size / 2)
    mean_area = size * size / n_grains
    w = rng.lognormal(0, size_spread, n_grains) * mean_area * 0.25
    grains = _power_voronoi(size, seeds, w)

    # grain boundaries: pixels whose 4-neighbor belongs to another grain
    b = np.zeros_like(grains, bool)
    b[:-1] |= grains[:-1] != grains[1:]
    b[:, :-1] |= grains[:, :-1] != grains[:, 1:]

    # pores: ellipses centered on random boundary pixels until the target porosity is reached
    pores = np.zeros_like(b)
    by, bx = np.nonzero(b)
    r_mean = np.sqrt(mean_area / np.pi) * 0.22
    while pores.mean() < porosity:
        k = rng.integers(len(by))
        cy, cx = by[k], bx[k]
        a, c = r_mean * rng.lognormal(0, 0.35, 2)
        th = rng.uniform(0, np.pi)
        R = int(np.ceil(max(a, c))) + 1
        y0, y1, x0, x1 = max(cy - R, 0), min(cy + R + 1, size), max(cx - R, 0), min(cx + R + 1, size)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dy, dx = yy - cy, xx - cx
        u, v = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        pores[y0:y1, x0:x1] |= (u / a) ** 2 + (v / c) ** 2 <= 1
    grains = np.where(pores, 0, grains)

    # render: grain gray levels, grooves, pores with bright rims, illumination, blur, noise
    shade = rng.normal(0.62, 0.05, n_grains + 1)
    img = shade[grains]
    # etching is uneven: groove depth varies smoothly along the boundaries (some are faint)
    etch = ndi.gaussian_filter(rng.normal(size=(size, size)), size / 25)
    etch = np.clip(0.55 + 0.45 * etch / etch.std(), 0.15, 1.0)
    groove = ndi.gaussian_filter(ndi.binary_dilation(b).astype(float), 1.0)
    img = img - boundary_depth * etch * groove / groove.max()
    rim = ndi.binary_dilation(pores, iterations=2) & ~pores
    img = np.where(rim, img + 0.25, img)
    img = np.where(pores, 0.08, img)
    gy, gx = rng.normal(0, 0.12, 2)
    yy, xx = np.mgrid[0:size, 0:size]
    img = img * (1 + gy * (yy / size - 0.5) + gx * (xx / size - 0.5))
    img = ndi.gaussian_filter(img, 0.8) + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 1)

    truth = grain_stats(grains, um_per_px)
    truth["porosity"] = float(pores.mean())
    truth["mean_intercept_um"] = line_intercept(b | pores, um_per_px)["mean_intercept_um"]
    return {"image": img, "grains": grains, "pores": pores, "boundaries": b, "um_per_px": um_per_px,
            "truth": truth}


def grain_stats(labels, um_per_px, exclude_edges=True, min_px=20):
    """Equivalent circle diameters (ECD) of labeled grains. 0 = background (boundary or pore)."""
    lab = np.asarray(labels)
    ids, area = np.unique(lab[lab > 0], return_counts=True)
    keep = area >= min_px
    if exclude_edges:
        edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        keep &= ~np.isin(ids, edge)
    area_um2 = area[keep] * um_per_px ** 2
    ecd = np.sqrt(4 * area_um2 / np.pi)
    return {"n_grains": int(keep.sum()), "ecd_um": ecd, "mean_ecd_um": float(ecd.mean()) if len(ecd) else np.nan,
            "median_ecd_um": float(np.median(ecd)) if len(ecd) else np.nan}


def line_intercept(boundary_mask, um_per_px, n_lines=12):
    """ASTM E112 (Heyn) mean lineal intercept: lay test lines across the image, count boundary crossings.

    Returns the mean intercept length (µm), the ASTM grain size number G, and the number of intercepts.
    """
    m = np.asarray(boundary_mask, bool)
    h, w = m.shape
    rows = np.linspace(0, h - 1, n_lines + 2)[1:-1].astype(int)
    cols = np.linspace(0, w - 1, n_lines + 2)[1:-1].astype(int)
    crossings = 0
    for line in [m[r] for r in rows] + [m[:, c] for c in cols]:
        crossings += int(np.sum(np.diff(line.astype(int)) == 1))     # entering a boundary = one intercept
    length_um = (len(rows) * w + len(cols) * h) * um_per_px
    L = length_um / max(crossings, 1)
    G = -3.2877 - 6.6439 * np.log10(L / 1000)                        # ASTM E112, L in mm
    return {"mean_intercept_um": float(L), "astm_G": float(G), "intercepts": crossings}


def micrograph_series(n=40, seed=0, size=384, noise=0.06):
    """A set of synthetic micrographs with different grain sizes and porosities, plus a 'measured' hardness.

    Hardness follows a Hall-Petch term and an exponential porosity penalty (illustrative constants):
    H = (H0 + k / sqrt(d)) * exp(-b * P) + noise, with d the true mean ECD in µm.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ng = int(rng.integers(40, 260))
        p = float(rng.uniform(0.0, 0.10))
        m = synthetic_micrograph(n_grains=ng, porosity=p, size=size, noise=noise, seed=seed * 1000 + i)
        d = m["truth"]["mean_ecd_um"]
        H = (14.0 + 6.0 / np.sqrt(d)) * np.exp(-6.0 * m["truth"]["porosity"]) + rng.normal(0, 0.3)
        m["hardness_GPa"] = float(H)
        out.append(m)
    return out


def watershed_grains(elevation, interior, min_distance=7, smooth=1.0):
    """Split an image into grains: one seed per blob of 'interior' pixels, then flood the elevation map.

    elevation: high on grain boundaries (a ridge filter, or a classifier's boundary probability).
    interior:  bool mask of pixels confidently inside a grain.
    Returns a label image with 0 on the one-pixel watershed lines.
    """
    from skimage import feature, filters, measure, segmentation
    dist = filters.gaussian(ndi.distance_transform_edt(interior), smooth)
    peaks = feature.peak_local_max(dist, min_distance=min_distance, labels=measure.label(interior),
                                   exclude_border=False)
    markers = np.zeros(elevation.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(elevation, markers, watershed_line=True)


def matched_fraction(true_labels, pred_labels, iou=0.5):
    """Fraction of true grains (not touching the edge) recovered by a predicted grain with IoU > iou."""
    gt, pred = np.asarray(true_labels), np.asarray(pred_labels)
    edge = set(np.unique(np.r_[gt[0], gt[-1], gt[:, 0], gt[:, -1]]).tolist())
    hit = total = 0
    for i in np.unique(gt):
        if i == 0 or i in edge:
            continue
        mask = gt == i
        total += 1
        lab, cnt = np.unique(pred[mask], return_counts=True)
        cnt, lab = cnt[lab > 0], lab[lab > 0]
        if len(lab):
            j, inter = lab[cnt.argmax()], cnt.max()
            hit += inter / (mask.sum() + (pred == j).sum() - inter) > iou
    return hit / max(total, 1)


def drop_small(mask, min_px):
    """Remove connected blobs smaller than min_px pixels (works across scikit-image versions)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return np.asarray(mask, bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def show_grains(ax, labels, title=None, seed=0):
    """Draw a label image with a random color per grain (neighbors rarely match) and black for 0."""
    lab = np.asarray(labels)
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 1.0, (int(lab.max()) + 1, 3))
    colors[0] = 0
    ax.imshow(colors[lab])
    ax.axis("off")
    if title:
        ax.set_title(title)
