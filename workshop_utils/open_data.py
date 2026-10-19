"""Download and load open materials datasets (images and tables) that are too big to ship with the workshop.

Everything is fetched over plain HTTPS (no accounts, no API keys) and cached on disk, so a second call is
instant. Big zip archives are read *remotely*: only the files you ask for are downloaded.

    from workshop_utils.open_data import catalog, load_uo2_grain_boundaries
    catalog()                                  # what's available: material, labels, size, license, citation
    pairs = load_uo2_grain_boundaries()        # list of {"name", "image", "boundaries"}

Respect each dataset's license and cite the source (`catalog()` has both).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("WORKSHOP_DATA_CACHE", Path.home() / ".cache" / "acers_workshop"))
_UA = {"User-Agent": "acers-workshop/1.0 (teaching)"}

_ZENODO = "https://zenodo.org/api/records/{}/files/{}/content"
SOURCES = {
    "uo2_grain_boundaries": _ZENODO.format(17986530, "Data.zip"),
    "ebc_coating": _ZENODO.format(22039777, "EBC.zip"),
    "batio3_sintering": _ZENODO.format(18233071, "Dataset%20for%20BaTiO3%20rapid%20sintering.zip"),
    "mgo_ebsd": _ZENODO.format(4073384, "DataFolder.zip"),
    "glass_bead_ct": "https://web.corral.tacc.utexas.edu/digitalporousmedia/archive/DRP-51/DRP-51_archive.zip",
    "magnetic_tiles": "https://codeload.github.com/abin24/Magnetic-tile-defect-datasets./zip/refs/heads/master",
    "sciglass": "https://raw.githubusercontent.com/drcassar/SciGlass/master/sciglass/select/SciGK.csv.zip",
    "matminer": "https://raw.githubusercontent.com/hackingmaterials/matminer/master/matminer/datasets/dataset_metadata.json",
}

_CATALOG = [
    # name, material, modality, what you get, labels, download, license, citation, loader
    ("uo2_grain_boundaries", "UO₂ (sintered nuclear fuel)", "SEM, backscattered electrons",
     "4 image + hand-traced grain-boundary mask pairs (3058×2030 px); more in the archive", "pixel masks",
     "≈16 MB of a 128 MB zip", "CC BY 4.0",
     "Zhao et al., Comput. Mater. Sci. (2026), doi:10.1016/j.commatsci.2026.114622; data doi:10.5281/zenodo.17986530",
     "load_uo2_grain_boundaries()"),
    ("ebc_coating", "Environmental barrier coating (Si/oxide on SiC)", "SEM cross-sections",
     "24 image + mask pairs, split train/val/test", "pixel masks: 0 background, 1 oxide, 2 crack",
     "22 MB", "CC BY 4.0 (repackaged from NASA, MIT)",
     "Stuckner, Harder & Smith, npj Comput. Mater. 8, 200 (2022), doi:10.1038/s41524-022-00878-5; data doi:10.5281/zenodo.22039777",
     "load_ebc_coating()"),
    ("glass_bead_ct", "Sintered soda-lime glass beads (725 °C, 25 min)", "X-ray micro-CT",
     "1,540 slices (2028×2048 px, 10 µm voxels); stated porosity 42%", "porosity (one number)",
     "≈0.7 MB per slice, from a 1 GB zip", "ODC-BY 1.0",
     "Digital Porous Media Portal DRP-51, 'Particulate straining in simple porous media', doi:10.17612/P7K01C",
     "load_glass_bead_ct()"),
    ("batio3_sintering", "BaTiO₃, 4 sintering routes × 5 temperatures", "SEM + tables",
     "grain-size distributions (≈4,000 grains per sample), relative densities; 122 SEM TIFFs on request",
     "measured grain sizes", "tables ≈0.2 MB; ≈1.2 MB per TIFF, from a 188 MB zip", "CC BY 4.0",
     "Mařák et al., J. Eur. Ceram. Soc. (2026), doi:10.1016/j.jeurceramsoc.2026.118629; data doi:10.5281/zenodo.18233071",
     "load_batio3_grain_sizes(), load_batio3_sem()"),
    ("mgo_ebsd", "Polycrystalline MgO (deformed at high P, T)", "EBSD maps (.ctf)",
     "2 orientation maps (413×254 and 267×290 points, 0.3 µm step)", "orientations (Euler angles), band contrast",
     "≈4 MB of a 153 MB zip", "CC BY 4.0",
     "Ledoux et al., Front. Earth Sci. (2022), doi:10.3389/feart.2022.849777; data doi:10.5281/zenodo.4073384",
     "load_mgo_ebsd()"),
    ("magnetic_tiles", "Ferrite magnetic tiles", "optical photos of surfaces",
     "1,344 images: 952 defect-free + 5 defect types (blowhole, break, crack, fray, uneven)",
     "class + pixel masks", "≈53 MB", "no license stated: fetched from the authors' GitHub, not re-hosted",
     "Huang et al., The Visual Computer 36, 85 (2020), doi:10.1007/s00371-018-1588-5",
     "load_magnetic_tiles()"),
    ("sciglass", "Oxide (and other) glasses", "table",
     "≈423,000 glass compositions (mol% oxides) with Tg, density, refractive index, thermal expansion, ...",
     "measured properties", "42 MB", "ODbL / MIT (see repository)",
     "SciGlass database (EPAM), plain-text conversion by D. R. Cassar, doi:10.5281/zenodo.8287159",
     "load_sciglass()"),
    ("matminer", "Inorganic compounds (39 datasets)", "tables",
     "e.g. dielectric_constant (1,056), piezoelectric_tensor (941), expt_gap (6,354), brgoch_superhard_training (2,574)",
     "DFT or measured properties", "kB to a few MB each", "per dataset (mostly CC BY, check the reference)",
     "Ward et al., Comput. Mater. Sci. 152, 60 (2018), doi:10.1016/j.commatsci.2018.05.018 (+ each dataset's own paper)",
     "load_matminer(name), matminer_datasets()"),
]

# Good open sets that don't fit a simple download (very large, or need a login); listed for reference.
_ELSEWHERE = [
    ("Ceramic tiles, normal vs defective (1,600 photos)", "CC BY 4.0, 5.9 GB, file-by-file",
     "https://data.mendeley.com/datasets/47x6jdbr5j/1"),
    ("NFFA SEM nanoscience images (21,169 in 10 classes)", "CC BY 4.0, tar per class (0.1-3 GB)",
     "https://b2share.eudat.eu/records/zja8y-53j14"),
    ("Sm-doped BiFeO₃ atomic-resolution HAADF-STEM + unit-cell parameters", "CC BY 4.0, 7-170 MB per file (HDF5)",
     "https://doi.org/10.5281/zenodo.4555979"),
    ("Si and Ni EBSD patterns for kikuchipy", "pip install kikuchipy; kikuchipy.data", "https://kikuchipy.org"),
    ("MgAl₂O₄ fracture surfaces, intergranular vs transgranular (710 images)", "CC BY, needs a free Globus login",
     "https://doi.org/10.18126/vu60-4htj"),
    ("Pretrained microscopy encoders (MicroNet) for segmentation", "MIT",
     "https://github.com/nasa/pretrained-microscopy-models"),
]


def catalog(elsewhere=False) -> pd.DataFrame:
    """Table of the datasets this module can load (or, with elsewhere=True, good ones it can't)."""
    if elsewhere:
        return pd.DataFrame(_ELSEWHERE, columns=["dataset", "access", "link"])
    cols = ["name", "material", "modality", "contents", "labels", "download", "license", "citation", "loader"]
    return pd.DataFrame(_CATALOG, columns=cols).set_index("name")


# ----------------------------------------------------------------------------------------------- downloading
def _download(url: str, dest: Path) -> Path:
    """Download url to dest (atomically) unless it is already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"downloading {url.split('?')[0]} ...")
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    return dest


class _HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP Range requests, so zipfile can read single members of a remote zip."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={**_UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={**_UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


_REMOTE = {}


def _remote_zip(name) -> zipfile.ZipFile:
    if name not in _REMOTE:
        _REMOTE[name] = zipfile.ZipFile(io.BufferedReader(_HTTPRangeFile(SOURCES[name]), buffer_size=1 << 20))
    return _REMOTE[name]


def list_files(name, pattern="") -> list[str]:
    """Names of the files inside a remote archive (optionally only those matching a regex)."""
    z = _remote_zip(name)
    return [m for m in z.namelist() if not m.endswith("/") and "__MACOSX" not in m
            and not m.endswith(".DS_Store") and re.search(pattern, m)]


def fetch(name, members) -> list[Path]:
    """Download selected members of a remote zip archive into the cache; returns local paths."""
    root = (CACHE / name).resolve()
    out, todo = [], []
    for m in members:
        p = (root / m).resolve()
        if root not in p.parents:
            raise ValueError(f"unsafe path in archive: {m}")
        out.append(p)
        if not p.exists():
            todo.append((m, p))
    if todo:
        z = _remote_zip(name)
        print(f"fetching {len(todo)} file(s) from the {name} archive ...")
        for m, p in todo:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            tmp.write_bytes(z.read(m))
            tmp.replace(p)
    return out


def _whole_zip(name) -> Path:
    """Download a (small) zip once and extract it into the cache."""
    root = CACHE / name
    if (root / ".extracted").exists():
        return root
    zpath = _download(SOURCES[name], CACHE / f"{name}.zip")
    resolved = root.resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if resolved not in (root / m).resolve().parents and (root / m).resolve() != resolved:
                raise ValueError(f"unsafe path in archive: {m}")
        z.extractall(root)
    (root / ".extracted").touch()
    zpath.unlink()
    return root


def read_image(path, as_gray=True) -> np.ndarray:
    """Read an image file as a float array scaled to [0, 1]."""
    from PIL import Image

    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im, dtype=np.float32)
        return (a - a.min()) / max(a.max() - a.min(), 1)
    im = im.convert("L") if as_gray else im.convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------------------------- image loaders
def load_uo2_grain_boundaries(names=("T2_11", "T3_11", "T4_05", "T5_08")):
    """SEM-BSE images of sintered UO₂ with hand-traced grain-boundary masks (True on boundaries).

    The other training images (D1-D6) are 6132×4077 px tiles, 17 MB each: pass e.g. names=("D1",) to get them.
    """
    files = list_files("uo2_grain_boundaries", r"TrainValidData/(Images|GrainBoundary)/")
    out = []
    for n in names:
        img = next(f for f in files if f.endswith(f"/Images/{n}.png"))
        gt = next(f for f in files if f.endswith(f"/GrainBoundary/{n.split('_')[0]}_GT.png"))
        p_img, p_gt = fetch("uo2_grain_boundaries", [img, gt])
        out.append({"name": n, "image": read_image(p_img), "boundaries": read_image(p_gt) > 0.5})
    return out


def load_ebc_coating(split="train"):
    """SEM cross-sections of an environmental barrier coating with masks (0 background, 1 oxide, 2 crack)."""
    root = _whole_zip("ebc_coating") / "EBC" / split
    out = []
    for p in sorted((root / "images").glob("*.tif")):
        m = root / "masks" / p.name
        if not m.exists():
            m = next((root / "masks").glob(p.stem + ".*"))
        out.append({"name": p.stem, "image": read_image(p), "mask": np.asarray(_open(m)).astype(np.uint8)})
    return out


def _open(path):
    from PIL import Image

    return Image.open(path)


def load_magnetic_tiles(classes=None, max_per_class=None, seed=0):
    """Photos of ferrite magnetic-tile surfaces: label (defect type or 'Free'), image, defect mask."""
    root = _whole_zip("magnetic_tiles")
    out = []
    rng = np.random.default_rng(seed)
    for d in sorted(root.glob("*/MT_*")):
        label = d.name[3:]
        if classes and label not in classes:
            continue
        imgs = sorted((d / "Imgs").glob("*.jpg"))
        if max_per_class and len(imgs) > max_per_class:
            imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), max_per_class, replace=False))]
        for p in imgs:
            m = p.with_suffix(".png")
            out.append({"label": label, "name": p.stem, "image": read_image(p),
                        "mask": read_image(m) > 0.5 if m.exists() else None})
    return out


_CT_DIR = "DRP-51/Sintered glass bead - MicroCT/1mm_25min@725Cpeak_42%porosity/"


def load_glass_bead_ct(slices=None, n=20):
    """Micro-CT slices of sintered soda-lime glass beads (10 µm voxels; the authors state 42% porosity).

    slices: slice numbers between 1 and 1540. By default, n slices evenly spread through the whole stack
    (20 slices ≈ 10 MB). Returns {"volume": array (n, 2028, 2048) in [0, 1], "slices", "voxel_um",
    "stated_porosity"}.
    """
    if slices is None:
        slices = np.linspace(1, 1540, n).round().astype(int)
    slices = [int(s) for s in slices]
    if min(slices) < 1 or max(slices) > 1540:
        raise ValueError("slice numbers run from 1 to 1540")
    members = [f"{_CT_DIR}coreA{s:04d}.jpg" for s in slices]
    paths = fetch("glass_bead_ct", members)
    return {"volume": np.stack([read_image(p) for p in paths]), "slices": list(slices),
            "voxel_um": 10.0, "stated_porosity": 0.42}


_BTO_ROUTES = {"CS": "conventional", "RPLS": "rapid pressureless", "SPS": "spark plasma",
               "p-SPS": "pressureless spark plasma"}


def load_batio3_grain_sizes():
    """Every measured BaTiO₃ grain, tidy: sample, route, sintering T (°C), population, grain size (µm)
    and the sample's relative density (%). Grain sizes are 2D diameters × 1.27 (the authors' stereology)."""
    gs, rd = fetch("batio3_sintering", ["4_grain size measurements/4_grain size measurements.csv",
                                        "2_Relative density/2_relative_density.csv"])
    wide = pd.read_csv(gs, sep=";", encoding="utf-8-sig")
    long = wide.melt(var_name="column", value_name="grain_size_um").dropna()
    long["population"] = np.where(long.column.str.endswith("_A"), "abnormal/coarse", "fine")
    long["sample"] = long.column.str.replace("_A", "", regex=False)
    long["route"] = long["sample"].str.extract(r"^([A-Za-z-]+)")[0].map(_BTO_ROUTES)
    long["T_C"] = long["sample"].str.extract(r"(\d+)$")[0].astype(int)
    dens = pd.read_csv(rd, sep=";", encoding="utf-8-sig")
    dens["sample"] = dens["sample"].ffill()
    dens = dens.dropna(subset=["average_relative_density"]).set_index("sample")["average_relative_density"]
    long["relative_density_pct"] = long["sample"].map(dens)
    return long[["sample", "route", "T_C", "population", "grain_size_um", "relative_density_pct"]].reset_index(drop=True)


def load_batio3_sem(sample="CS1225", n=1):
    """The first n SEM TIFFs of a BaTiO₃ sample (e.g. 'CS1225', 'SPS1250', 'RPLS1275', 'p-SPS1350').
    Note: the images still carry the microscope's data bar at the bottom; crop it before analysis."""
    route = re.match(r"^([A-Za-z-]+)", sample).group(1)
    files = [f for f in list_files("batio3_sintering", r"^3_SEM micrographs/")
             if f.lower().endswith((".tif", ".tiff")) and Path(f).name.startswith(sample + "_")
             and f"/{route}/" in f]
    return [{"name": Path(p).stem, "image": read_image(p)} for p in fetch("batio3_sintering", sorted(files)[:n])]


def read_ctf(path):
    """Read an Oxford/HKL .ctf EBSD map: returns (DataFrame of points, header dict)."""
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Phase\tX\tY"))
    header = {}
    for l in lines[:start]:
        parts = l.split("\t")
        if len(parts) >= 2 and parts[0] in ("XCells", "YCells", "XStep", "YStep", "Phases"):
            header[parts[0]] = float(parts[1])
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep="\t")
    return df, header


def load_mgo_ebsd(which=0):
    """One of two EBSD maps of polycrystalline MgO. Returns {"points": DataFrame, "header", "grid": dict of 2D
    arrays (phase, euler1-3 in degrees, band contrast BC, MAD), "step_um"}. Phase 0 = not indexed."""
    files = sorted(list_files("mgo_ebsd", r"Fig6Maps/.*\.ctf$"))
    df, h = read_ctf(fetch("mgo_ebsd", [files[which]])[0])
    nx, ny = int(h["XCells"]), int(h["YCells"])
    grid = {k: df[c].to_numpy().reshape(ny, nx) for k, c in
            [("phase", "Phase"), ("euler1", "Euler1"), ("euler2", "Euler2"), ("euler3", "Euler3"),
             ("BC", "BC"), ("MAD", "MAD")]}
    return {"points": df, "header": h, "grid": grid, "step_um": h["XStep"], "name": Path(files[which]).stem}


# ----------------------------------------------------------------------------------------------- tables
_SCIGLASS_OXIDES = {"SIO2": "SiO2", "AL2O3": "Al2O3", "B2O3": "B2O3", "CAO": "CaO", "K2O": "K2O", "NA2O": "Na2O",
                    "PBO": "PbO", "Li2O": "Li2O", "MgO": "MgO", "SRO": "SrO", "BAO": "BaO", "ZNO": "ZnO",
                    "P2O5": "P2O5", "GEO2": "GeO2", "ZRO2": "ZrO2", "TIO2": "TiO2", "TEO2": "TeO2"}
_SCIGLASS_PROPS = {"TG": "Tg_C", "DENSITY": "density_g_cm3", "ND300": "refractive_index_nd",
                   "ANY_TEC": "thermal_expansion_1e-7_K", "TLiq": "liquidus_C"}


def load_sciglass(prop="TG", only_listed_oxides=True):
    """Glass compositions (mol% of 17 common oxides) with one measured property from SciGlass.

    prop: 'TG' (glass transition, °C), 'DENSITY' (g/cm³), 'ND300' (refractive index n_d),
          'ANY_TEC' (thermal expansion, 1e-7/K), 'TLiq' (liquidus, °C).
    only_listed_oxides: keep only glasses whose 17 listed oxides add up to ~100 mol% (no 'other' components).
    """
    path = _download(SOURCES["sciglass"], CACHE / "SciGK.csv.zip")
    cache = CACHE / "sciglass_main.parquet"
    try:
        df = pd.read_parquet(cache)
    except Exception:
        with zipfile.ZipFile(path) as z:
            df = pd.read_csv(z.open("SciGK.csv"), sep="\t", low_memory=False,
                             usecols=["KOD", "GLASNO", "Year"] + list(_SCIGLASS_OXIDES) + list(_SCIGLASS_PROPS))
        try:
            df.to_parquet(cache)
        except Exception:
            pass
    df = df.dropna(subset=[prop])
    ox = df[list(_SCIGLASS_OXIDES)].fillna(0).rename(columns=_SCIGLASS_OXIDES)
    if only_listed_oxides:
        keep = (ox.sum(axis=1) - 100).abs() < 0.5
        df, ox = df[keep], ox[keep]
    out = pd.concat([df[["KOD", "GLASNO", "Year"]].rename(columns={"KOD": "reference_id", "GLASNO": "glass_no"}),
                     ox, df[[prop]].rename(columns=_SCIGLASS_PROPS)], axis=1)
    return out.reset_index(drop=True)


_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def oxides_to_formula(row, digits=4):
    """Turn oxide mol% (e.g. {'SiO2': 70, 'Na2O': 30}) into an element formula string such as
    'Si0.7O1.7Na0.6', so the Session 2 CBFV featurizer (generate_features) works on glasses."""
    counts = {}
    for ox in _SCIGLASS_OXIDES.values():
        x = float(row.get(ox, 0) or 0) / 100
        if x <= 0:
            continue
        for el, n in _ELEMENT_RE.findall(ox):
            counts[el] = counts.get(el, 0) + x * (int(n) if n else 1)
    return "".join(el + ("" if round(n, digits) == 1 else f"{round(n, digits):g}") for el, n in counts.items())


def matminer_datasets() -> pd.DataFrame:
    """The datasets matminer can load (name, rows, reference), read from matminer's own metadata."""
    meta = json.loads(_download(SOURCES["matminer"], CACHE / "matminer_metadata.json").read_text(encoding="utf-8"))
    return pd.DataFrame([{"name": k, "rows": v.get("num_entries"), "description": v.get("description", "")[:120]}
                         for k, v in meta.items() if v.get("url")]).set_index("name")


def load_matminer(name) -> pd.DataFrame:
    """Load a matminer dataset without installing matminer (structure columns stay as plain dicts)."""
    meta = json.loads(_download(SOURCES["matminer"], CACHE / "matminer_metadata.json").read_text(encoding="utf-8"))
    if name not in meta or not meta[name].get("url"):
        raise KeyError(f"Unknown matminer dataset {name!r}; see matminer_datasets()")
    path = _download(meta[name]["url"], CACHE / "matminer" / f"{name}.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return pd.DataFrame(d["data"], columns=d["columns"], index=d.get("index"))
