# Setup

You have two options. **Option A needs no installation** and is recommended if you just want to follow along.

## Option A: Google Colab (recommended)

1. Open the main [README](../README.md) and click the **Open in Colab** badge for a session.
2. Sign in with a Google account.
3. Click **Runtime → Run all**. The first code cell clones this repository into the Colab session automatically.

Colab already includes every package the course needs except `anthropic`, which the Session 6 notebook installs itself.

## Option B: Run locally

You need **Python 3.10 or newer** and `git`.

New to this? These short videos walk through it: [installing miniconda, virtual environments and VS Code](https://www.youtube.com/watch?v=jDy19p6wrA4&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · [cloning a GitHub repo to follow along](https://www.youtube.com/watch?v=zb4M9mYz5TE&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)

```bash
git clone https://github.com/sp8rks/ACerS_AI_ML_Workshop.git
cd ACerS_AI_ML_Workshop
```

Then create an environment with **one** of the following.

**uv** (fast, recommended):
```bash
uv venv
uv pip install -r setup/requirements.txt
uv run jupyter lab
```

**conda:**
```bash
conda create -n acers-ml python=3.11 -y
conda activate acers-ml
pip install -r setup/requirements.txt
jupyter lab
```

**plain pip / venv:**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r setup/requirements.txt
jupyter lab
```

Then open [`setup/00_check_environment.ipynb`](00_check_environment.ipynb) and run it. If every line says ✅, you are ready.

VS Code works too: open the folder, install the Python and Jupyter extensions, and select your environment as the kernel.

## LLM API key (Session 6)

Session 6 calls Claude through the Anthropic API. Create a key at <https://console.anthropic.com/> (API usage is billed separately from a Claude.ai subscription; running the whole notebook typically costs well under US$1).

- **Colab:** click the 🔑 **Secrets** icon in the left sidebar → *Add new secret* → name `ANTHROPIC_API_KEY`, paste the key, and toggle *Notebook access* on.
- **Local:** set an environment variable before starting Jupyter:
  - macOS/Linux: `export ANTHROPIC_API_KEY="sk-ant-..."`
  - Windows (PowerShell): `setx ANTHROPIC_API_KEY "sk-ant-..."`, then open a new terminal

**Never paste a key into a notebook cell or commit it to git.**

No key? The notebook still runs in **offline mode**: the regex baseline, retrieval, the ML tools and the capstone ranking all work, and the LLM cells explain what they would do.

## A note on the CBFV package

The course uses a small built-in featurizer (`workshop_utils/featurize.py`) that reproduces the [CBFV](https://github.com/kaaiian/CBFV) approach with the same element-property tables. We include it because the PyPI `CBFV` package currently fails with recent `pandas` (3.x) and `setuptools` (81+) releases, and because ~100 readable lines are a better teaching tool than a black box.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: workshop_utils` | Run the first (setup) cell of the notebook. Locally, make sure you opened the notebook from inside the cloned repository. |
| Colab: "Warning: this notebook was not authored by Google" | Click **Run anyway**. |
| `boxplot() got an unexpected keyword argument 'orientation'` | Upgrade matplotlib: `pip install -U matplotlib` |
| Session 6 says OFFLINE although you set a key | Restart Jupyter (or the Colab runtime) after setting the key, and check the secret name is exactly `ANTHROPIC_API_KEY`. |
