"""Simulated pigmented coatings (e.g. roofing-granule or glaze coatings) for the formulation-featurization demo.

Physics: two-flux Kubelka-Munk. Each pigment has an absorption spectrum K(λ) and a scattering spectrum S(λ)
per unit volume fraction. In a mixture they add up by concentration (Duncan's rule),
    K_mix = Σ c_i K_i + K_binder,   S_mix = Σ c_i S_i  (reduced by pigment crowding),
and a film of thickness X on a substrate of reflectance R_g reflects
    R = [1 − R_g (a − b coth(bSX))] / [a − R_g + b coth(bSX)],   a = 1 + K/S,  b = √(a² − 1).
Solar reflectance weights R(λ) by an approximate solar spectrum (a 5778 K blackbody, 300-2500 nm).

The pigment spectra are stylized (band edges in roughly the right places, magnitudes chosen to give realistic
reflectances). They are illustrative, NOT measured data for any commercial pigment.
"""

import numpy as np
import pandas as pd

WAVELENGTHS = np.arange(300, 2501, 10.0)  # nm
BANDS = {"UV": (300, 400), "VIS": (400, 700), "NIR": (700, 2500)}


def _edge(lam, at, width, high, low=0.0):
    """Absorption that is `high` below wavelength `at` and falls to `low` above it."""
    return low + (high - low) / (1 + np.exp((lam - at) / width))


def _bump(lam, at, width, height):
    return height * np.exp(-0.5 * ((lam - at) / width) ** 2)


def _pigment_spectra(lam=WAVELENGTHS):
    """K and S per unit pigment volume fraction, in 1/µm."""
    def scat(s550, power):
        return s550 * (550 / lam) ** power

    return {
        # name: (K, S)
        "TiO2_rutile": (_edge(lam, 405, 8, 12.0, 0.002), scat(3.0, 1.0)),
        "ZnO": (_edge(lam, 380, 6, 10.0, 0.002), scat(1.3, 1.0)),
        "Fe2O3_hematite": (_edge(lam, 575, 15, 7.0, 0.06) + _bump(lam, 870, 80, 0.4), scat(1.2, 1.3)),
        "Cr2O3_green": (_bump(lam, 450, 40, 2.5) + _bump(lam, 610, 55, 2.0) + _edge(lam, 360, 10, 6.0, 0.02),
                        scat(1.4, 1.2)),
        "carbon_black": (25.0 * (550 / lam) ** 0.6, scat(0.05, 1.0)),
        "IR_black_CrFe": (_edge(lam, 780, 60, 9.0, 0.08), scat(0.9, 1.0)),  # "cool" black: dark but NIR-reflective
        "kaolin_filler": (_edge(lam, 320, 10, 0.5, 0.004), scat(0.25, 1.0)),
    }


# Things a formulator could look up on a datasheet (or measure once per raw material)
_DATASHEET = pd.DataFrame(
    {"refractive_index": [2.70, 2.00, 2.90, 2.50, 1.90, 2.40, 1.56],
     "density_g_cm3": [4.2, 5.6, 5.2, 5.2, 1.8, 5.1, 2.6],
     "BET_m2_g": [12, 10, 9, 4, 120, 6, 18],
     "cost_usd_kg": [3.5, 2.8, 1.2, 4.0, 1.5, 9.0, 0.3]},
    index=list(_pigment_spectra()))


def solar_weights(lam=WAVELENGTHS):
    """Approximate solar spectral irradiance (5778 K blackbody), normalized to sum to 1."""
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    l_m = lam * 1e-9
    w = 1 / (l_m ** 5 * (np.exp(h * c / (l_m * k * 5778)) - 1))
    return w / w.sum()


def pigment_table():
    """One row per pigment: datasheet properties plus band-averaged absorption (K) and scattering (S), 1/µm."""
    rows = {}
    for name, (K, S) in _pigment_spectra().items():
        r = dict(_DATASHEET.loc[name])
        for band, (lo, hi) in BANDS.items():
            m = (WAVELENGTHS >= lo) & (WAVELENGTHS < hi)
            r[f"K_{band}"] = K[m].mean()
            r[f"S_{band}"] = S[m].mean()
        rows[name] = r
    return pd.DataFrame(rows).T.rename_axis("pigment")


def km_reflectance(K, S, thickness_um, substrate_R=0.08):
    """Kubelka-Munk reflectance of a film of given thickness on a substrate."""
    S = np.maximum(S, 1e-6)
    a = 1 + K / S
    b = np.sqrt(a ** 2 - 1)
    coth = 1 / np.tanh(np.minimum(b * S * thickness_um, 50))
    return (1 - substrate_R * (a - b * coth)) / (a - substrate_R + b * coth)


def coating_spectrum(fractions, pvc, thickness_um, firing_C, substrate_R=0.08):
    """Reflectance spectrum of one coating.

    fractions: dict pigment -> fraction of the pigment blend (sums to 1)
    pvc: pigment volume concentration (pigment volume / dry film volume)
    firing_C: cure/firing temperature; hotter firing densifies the binder and removes a little void scattering
    """
    spectra = _pigment_spectra()
    lam = WAVELENGTHS
    K = 0.004 + _edge(lam, 330, 12, 0.3)                    # silicate binder: weak UV absorption
    S = np.zeros_like(lam)
    for name, f in fractions.items():
        Ki, Si = spectra[name]
        K = K + pvc * f * Ki
        S = S + pvc * f * Si
    crowding = 1 - 0.9 * (pvc / 0.55) ** 2                  # scattering per pigment drops as particles crowd
    voids = 0.25 * np.exp(-(firing_C - 400) / 120) * (550 / lam)
    return km_reflectance(K, S * crowding + voids, thickness_um, substrate_R)


def simulate_coatings(n=400, seed=0, noise=0.006, exclude=()):
    """Random 2-4 pigment formulations with processing variables; returns solar and visible reflectance."""
    rng = np.random.default_rng(seed)
    names = [p for p in _pigment_spectra() if p not in exclude]
    w = solar_weights()
    vis = (WAVELENGTHS >= 400) & (WAVELENGTHS < 700)
    rows = []
    for _ in range(n):
        k = rng.integers(2, 5)
        chosen = rng.choice(names, k, replace=False)
        frac = rng.dirichlet(np.ones(k) * 0.8)
        fr = dict(zip(chosen, frac))
        pvc, th, T = rng.uniform(0.08, 0.40), rng.uniform(15, 120), rng.uniform(450, 750)
        R = coating_spectrum(fr, pvc, th, T)
        row = {f"{p}_frac": round(fr.get(p, 0.0), 4) for p in _pigment_spectra()}
        row.update(PVC=round(pvc, 3), thickness_um=round(th, 1), firing_C=round(T, 0),
                   solar_reflectance=float(np.clip((R * w).sum() + rng.normal(0, noise), 0, 1)),
                   visible_reflectance=float(np.clip(R[vis].mean() + rng.normal(0, noise), 0, 1)))
        rows.append(row)
    return pd.DataFrame(rows)


def formulation_vector(df, table=None, processing=("PVC", "thickness_um", "firing_C")):
    """A composition-based feature vector for formulations: the fraction-weighted average of each pigment
    property (like CBFV does with element properties), plus the processing variables."""
    table = pigment_table() if table is None else table
    frac = df[[f"{p}_frac" for p in table.index]].to_numpy()
    feats = pd.DataFrame(frac @ table.to_numpy(), columns=[f"avg_{c}" for c in table.columns], index=df.index)
    return pd.concat([feats, df[list(processing)]], axis=1)
