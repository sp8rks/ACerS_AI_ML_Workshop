"""A minimal crystal graph convolutional neural network (CGCNN) in plain PyTorch.

After Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018). Atoms are nodes,
neighbor pairs are edges labeled by their distance, and each convolution
lets every atom update its features from its neighbors. Mean pooling over
atoms gives one vector per crystal, and a small network maps it to the property.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DATA_DIR
from .structures import batch_graphs

PRETRAINED = DATA_DIR / "cgcnn_log10K_pretrained.pt"


class GaussianDistance(nn.Module):
    """Expand a bond length into a smooth 'soft one-hot' vector over 0 to dmax angstrom."""

    def __init__(self, n: int = 40, dmax: float = 8.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, dmax, n))
        self.gamma = (n / dmax) ** 2

    def forward(self, d):
        return torch.exp(-self.gamma * (d[:, None] - self.centers) ** 2)


class CGConv(nn.Module):
    """One graph convolution: each atom collects gated messages from its neighbors.

    message(i <- j) = sigmoid(gate) * softplus(core), where [gate, core] is a
    linear function of [x_i, x_j, bond_ij]. (The linear map is split into
    per-atom and per-bond parts, which is the same math but much faster.)
    """

    def __init__(self, h: int, e: int):
        super().__init__()
        self.w_center = nn.Linear(h, 2 * h)
        self.w_neighbor = nn.Linear(h, 2 * h, bias=False)
        self.w_bond = nn.Linear(e, 2 * h, bias=False)
        self.bn_msg, self.bn_out = nn.BatchNorm1d(2 * h), nn.BatchNorm1d(h)

    def forward(self, x, center, neighbor, bond):
        z = self.bn_msg(self.w_center(x)[center] + self.w_neighbor(x)[neighbor] + self.w_bond(bond))
        gate, core = z.chunk(2, dim=1)
        msg = torch.sigmoid(gate) * F.softplus(core)
        summed = torch.zeros_like(x).index_add_(0, center, msg)      # sum messages arriving at each atom
        return F.softplus(x + self.bn_out(summed))


class CGCNN(nn.Module):
    def __init__(self, h: int = 64, n_conv: int = 3, n_gauss: int = 40):
        super().__init__()
        self.embed = nn.Embedding(100, h)          # a learned vector per element (like mat2vec, but learned here)
        self.expand = GaussianDistance(n_gauss)
        self.convs = nn.ModuleList([CGConv(h, n_gauss) for _ in range(n_conv)])
        self.head = nn.Sequential(nn.Linear(h, h), nn.Softplus(), nn.Linear(h, 1))

    def crystal_vectors(self, b):
        """One learned feature vector per crystal (mean over its atoms)."""
        x = self.embed(b["z"])
        bond = self.expand(b["dist"])
        for conv in self.convs:
            x = conv(x, b["center"], b["neighbor"], bond)
        n_atoms = torch.zeros(b["n_graphs"], device=x.device).index_add_(0, b["graph_id"], torch.ones(len(x), device=x.device))
        return torch.zeros(b["n_graphs"], x.shape[1], device=x.device).index_add_(0, b["graph_id"], x) / n_atoms[:, None]

    def forward(self, b):
        return self.head(self.crystal_vectors(b)).squeeze(1)


def to_torch(batch, device="cpu"):
    out = {k: torch.as_tensor(v, device=device) for k, v in batch.items() if k != "n_graphs"}
    out["n_graphs"] = batch["n_graphs"]
    return out


def predict(model, graphs, mean=0.0, std=1.0, batch_size=256, device="cpu"):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(graphs), batch_size):
            b = to_torch(batch_graphs(graphs[k:k + batch_size]), device)
            out.append(model(b).cpu().numpy() * std + mean)
    return np.concatenate(out)


def train(model, graphs, y, epochs=10, batch_size=128, lr=3e-3, device="cpu", val=None, seed=0, verbose=True):
    """Train with an L1 loss on standardized targets. Returns (mean, std, history)."""
    torch.manual_seed(seed)
    y = np.asarray(y, np.float32)
    mean, std = float(y.mean()), float(y.std())
    yt = torch.as_tensor((y - mean) / std, device=device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(graphs) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(epochs):
        model.train()
        perm, tot = rng.permutation(len(graphs)), 0.0
        for k in range(0, len(perm), batch_size):
            ii = perm[k:k + batch_size]
            b = to_torch(batch_graphs([graphs[i] for i in ii]), device)
            loss = F.l1_loss(model(b), yt[ii])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(ii)
        rec = {"epoch": ep + 1, "train_mae": tot / len(graphs) * std}
        if val is not None:
            rec["val_mae"] = float(np.abs(predict(model, val[0], mean, std, device=device) - np.asarray(val[1])).mean())
        history.append(rec)
        if verbose:
            print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
    return mean, std, history
