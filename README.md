# Practical AI and Machine Learning for Materials Science

**ACerS Learning Center short course · October 2026 · Instructor: Taylor D. Sparks (University of Utah)**

This 12-hour short course introduces practical AI and machine learning tools for materials scientists and engineers. It is aimed at researchers, engineers and advanced students in ceramics and related fields. It focuses on hands-on Python workflows for real materials problems: property prediction, materials discovery, process optimization and large language models (LLMs).

By the end of the course you will be able to prototype your own AI-driven materials workflow and judge **when and how** these approaches add value.

---

## 🚀 Quick start (no installation needed)

Every notebook runs in **Google Colab**: click a badge below, then **Runtime → Run all**. The first cell downloads this repository automatically.

To run locally instead, see [setup/SETUP.md](setup/SETUP.md).

## 📅 Schedule

Official course outline: [ACerS_CourseOutline_MLAI.pdf](ACerS_CourseOutline_MLAI.pdf)

| # | Date | Topics | Notebook | Key lesson |
|---|---|---|---|---|
| 1 | **Tue 10/13** | What materials informatics is (and isn't) · high-value ceramics use cases · build → measure → learn · data types & problem framing | [01_materials_informatics_intro](sessions/01_materials_informatics_intro/01_materials_informatics_intro.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sp8rks/ACerS_AI_ML_Workshop/blob/main/sessions/01_materials_informatics_intro/01_materials_informatics_intro.ipynb) | Most ML projects fail from poor problem formulation, not algorithms |
| 2 | **Wed 10/14** | Feature engineering · composition-based feature vectors (CBFV) · domain knowledge vs. automated features | [02_feature_engineering_cbfv](sessions/02_feature_engineering_cbfv/02_feature_engineering_cbfv.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sp8rks/ACerS_AI_ML_Workshop/blob/main/sessions/02_feature_engineering_cbfv/02_feature_engineering_cbfv.ipynb) | Representation matters more than model choice |
| 3 | **Thu 10/15** | Supervised learning (regression + classification) · linear, random forest, boosting · overfitting, validation, metrics | [03_supervised_learning](sessions/03_supervised_learning/03_supervised_learning.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sp8rks/ACerS_AI_ML_Workshop/blob/main/sessions/03_supervised_learning/03_supervised_learning.ipynb) | Validation > accuracy: avoid fooling yourself |
| 4 | **Tue 10/20** | Model interpretation and trust · feature importance and physical insight · uncertainty and failure modes | [04_interpretation_and_trust](sessions/04_interpretation_and_trust/04_interpretation_and_trust.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sp8rks/ACerS_AI_ML_Workshop/blob/main/sessions/04_interpretation_and_trust/04_interpretation_and_trust.ipynb) | Garbage in → garbage out |
| 5 | **Wed 10/21** | Optimization and discovery · Bayesian optimization and active learning · efficient experimental design | [05_bayesian_optimization](sessions/05_bayesian_optimization/05_bayesian_optimization.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sp8rks/ACerS_AI_ML_Workshop/blob/main/sessions/05_bayesian_optimization/05_bayesian_optimization.ipynb) | Exploration vs. exploitation in real experiments |
| 6 | **Thu 10/22** | LLMs for materials science (data extraction, RAG) · agents and tool use · capstone: data → model → decision | [06_llms_and_agents](sessions/06_llms_and_agents/06_llms_and_agents.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sp8rks/ACerS_AI_ML_Workshop/blob/main/sessions/06_llms_and_agents/06_llms_and_agents.ipynb) | LLMs are the glue around validated models, not a replacement for them |

## 📺 Optional pre-watch videos

Each session has a short list of recorded lectures from the semester-long course. Watching them beforehand is **optional**: the sessions are self-contained, but you'll get more out of the hands-on time if the concepts are familiar. Full [YouTube playlist](https://www.youtube.com/playlist?list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) (72 videos).

| Session | Watch before class | Want more? |
|---|---|---|
| 1 | [2. What is Materials Informatics?](https://www.youtube.com/watch?v=65vCdOibwhQ&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[3. How are materials discovered?](https://www.youtube.com/watch?v=RRNcqJSJ6vc&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[4. Machine Learning vs Materials Informatics](https://www.youtube.com/watch?v=nDGDMKB0C6w&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[7. Machine Learning Tasks and Types](https://www.youtube.com/watch?v=N_wFjy3ZWjg&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) | [5. Materials Data Repositories](https://www.youtube.com/watch?v=cdSENQPsAiI&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[6. Materials Data Access (Materials Project API)](https://www.youtube.com/watch?v=Vuu7bNzmL8g&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) |
| 2 | [8. Featurization in Machine Learning](https://www.youtube.com/watch?v=u9ghh22BfaY&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[9. Composition-based feature vector](https://www.youtube.com/watch?v=JctWNNdI9Jc&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) | [10. Structure-based feature vector](https://www.youtube.com/watch?v=VTvfUFph-B4&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[11. Crystal Structure Graphs](https://www.youtube.com/watch?v=3OnXKQwczJM&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) |
| 3 | [14. Linear vs Nonlinear models](https://www.youtube.com/watch?v=YgaBv6CXfvo&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[15. Metrics and evaluation of ML models](https://www.youtube.com/watch?v=u1-m_hsF7DE&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[16. Splitting data in train/val/test splits](https://www.youtube.com/watch?v=_kp19PIVuXc&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) | [18. Ensemble techniques](https://www.youtube.com/watch?v=X6BXE3Bln5M&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) |
| 4 | [17. Best Practices in Materials Informatics](https://www.youtube.com/watch?v=5fMr4mYuCXI&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[20. Can we extrapolate to extraordinary materials?](https://www.youtube.com/watch?v=siphXMgHJ6A&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) | [Accuracy, Uncertainty, Inspectability: composition-restricted attention networks](https://www.youtube.com/watch?v=KtUstqOL33w&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) |
| 5 | [31. Gaussian Processes](https://www.youtube.com/watch?v=zquAOOjG2iI&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[32. Bayesian Optimization](https://www.youtube.com/watch?v=qVEBO1Viv7k&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) | [A gentle introduction to Bayesian optimization](https://www.youtube.com/watch?v=IVaWl2tL06c&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[Traditional sampling (grid vs random vs Sobol vs LHS)](https://www.youtube.com/watch?v=Evua529dAgc&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[Comparing Bayesian optimization with traditional sampling](https://www.youtube.com/watch?v=41fQs4JxRQA&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[Making complex BO simple with Honegumi](https://www.youtube.com/watch?v=1d1rlCyk85g&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[Human-in-the-loop Bayesian Optimization](https://www.youtube.com/watch?v=4tnaL9ts6CQ&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) |
| 6 | [33. Large Language Models in Materials Science](https://www.youtube.com/watch?v=sUl8ovLgK4Q&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)<br>[Your GitHub Repo Is Now a Research Robot (Claude Code + GitHub Actions)](https://www.youtube.com/watch?v=U5sB19DLnOk&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) | [Materials Project Seminar: Materials informatics, moving beyond screening](https://www.youtube.com/watch?v=aDVmb3n4jQs&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) |

**Setting up Python locally?** [Installing miniconda, virtual environments and VS Code](https://www.youtube.com/watch?v=jDy19p6wrA4&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · [How to clone a GitHub repo to follow along](https://www.youtube.com/watch?v=zb4M9mYz5TE&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)

### What happens in each session

<details>
<summary><b>Session 1: What materials informatics is (and isn't)</b></summary>

- **Optional pre-watch:** 📺 [2. What is Materials Informatics?](https://www.youtube.com/watch?v=65vCdOibwhQ&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [3. How are materials discovered?](https://www.youtube.com/watch?v=RRNcqJSJ6vc&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [4. Machine Learning vs Materials Informatics](https://www.youtube.com/watch?v=nDGDMKB0C6w&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [7. Machine Learning Tasks and Types](https://www.youtube.com/watch?v=N_wFjy3ZWjg&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)
- **Demo:** `pandas` exploration of ~4,900 DFT bulk moduli: missing values, distributions, chemistry families, element coverage bias
- **Concept:** what counts as "one sample"? (the same compound at many temperatures → data leakage)
- **Hands-on (optional):** frame an ML-ready problem from your own work with the [problem-framing worksheet](sessions/01_materials_informatics_intro/problem_framing_worksheet.md)
</details>

<details>
<summary><b>Session 2: Feature engineering and CBFV</b></summary>

- **Optional pre-watch:** 📺 [8. Featurization in Machine Learning](https://www.youtube.com/watch?v=u9ghh22BfaY&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [9. Composition-based feature vector](https://www.youtube.com/watch?v=JctWNNdI9Jc&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)
- **Hands-on:** compute a CBFV for Al₂O₃ by hand, then featurize thousands of compounds
- **Experiment:** same model, five representations (one-hot, random, Magpie, Oliynyk, mat2vec), with large data and with 200 samples
- **Discussion:** domain-knowledge features; limits of composition-only models (polymorphs, processing, microstructure)
</details>

<details>
<summary><b>Session 3: Supervised learning</b></summary>

- **Optional pre-watch:** 📺 [14. Linear vs Nonlinear models](https://www.youtube.com/watch?v=YgaBv6CXfvo&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [15. Metrics and evaluation of ML models](https://www.youtube.com/watch?v=u1-m_hsF7DE&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [16. Splitting data in train/val/test splits](https://www.youtube.com/watch?v=_kp19PIVuXc&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)
- **Demo:** full `scikit-learn` workflow: split → dummy baseline → ridge / random forest / gradient boosting → parity plots
- **Hands-on:** predict bulk modulus (regression) and metal vs. non-metal from band gaps (classification); validation curves, cross-validation, learning curves
- **Key experiment:** random vs. grouped cross-validation on heat-capacity data. Leakage makes the error look 10× smaller than it really is.
</details>

<details>
<summary><b>Session 4: Interpretation and trust</b></summary>

- **Optional pre-watch:** 📺 [17. Best Practices in Materials Informatics](https://www.youtube.com/watch?v=5fMr4mYuCXI&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [20. Can we extrapolate to extraordinary materials?](https://www.youtube.com/watch?v=siphXMgHJ6A&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)
- **Hands-on:** clean a raw dataset (negative temperatures, negative heat capacities, missing values) and measure the effect
- **Demo:** impurity vs. permutation importance, correlated features, partial dependence. The model rediscovers that stiffness tracks cohesive energy.
- **Uncertainty:** ensemble σ, calibration, and domain of applicability (nearest-neighbor distance)
- **Discussion:** when NOT to trust a model: new chemistries (hold out all oxides), extrapolation beyond the training range
</details>

<details>
<summary><b>Session 5: Bayesian optimization and active learning</b></summary>

- **Optional pre-watch:** 📺 [31. Gaussian Processes](https://www.youtube.com/watch?v=zquAOOjG2iI&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [32. Bayesian Optimization](https://www.youtube.com/watch?v=qVEBO1Viv7k&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)
- **Demo:** Gaussian process + Expected Improvement / UCB written from scratch; a build → suggest → update loop on a simulated sintering campaign
- **Key concept:** greedy search gets trapped on a local optimum; a small exploration bonus finds the global one
- **Hands-on:** benchmark BO vs. random search on **real** silver-nanoparticle flow-synthesis experiments, plus active learning for model building
</details>

<details>
<summary><b>Session 6: LLMs, agents and the capstone</b></summary>

- **Optional pre-watch:** 📺 [33. Large Language Models in Materials Science](https://www.youtube.com/watch?v=sUl8ovLgK4Q&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) · 📺 [Your GitHub Repo Is Now a Research Robot (Claude Code + GitHub Actions)](https://www.youtube.com/watch?v=U5sB19DLnOk&list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)
- **Demo:** schema-constrained LLM data extraction from ceramics abstracts, scored against an answer key and a regex baseline
- **Demo:** retrieval-augmented generation with citations, and why retrieval quality limits the answer
- **Demo:** an agent loop where the LLM calls *your* ML model, a database lookup and a literature search
- **Capstone:** screen 135 carbides, nitrides and diborides → predict with uncertainty → choose the next 3 experiments → LLM-drafted decision memo
- **Needs** an Anthropic API key for the LLM cells (see [setup/SETUP.md](setup/SETUP.md#llm-api-key-session-6)). Without one the notebook runs in offline mode.
</details>

## 📁 Repository layout

```
├── sessions/          one folder per session, each with a self-contained notebook
├── data/              all datasets used in the course (see data/README.md)
├── workshop_utils/    small helper package: formula parsing, CBFV featurizer, plotting
├── setup/             installation guide, requirements, environment check notebook
└── slides/            lecture slides, posted after each session
```

## 📚 Additional resources

- **Lecture videos:** [Materials Informatics YouTube playlist](https://youtube.com/playlist?list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0) (see the [pre-watch table](#-optional-pre-watch-videos) above)
- **Full semester course** (slides, 30+ worked examples, homework): [sp8rks/MaterialsInformatics](https://github.com/sp8rks/MaterialsInformatics)
- **Best practices paper:** Wang *et al.*, ["Machine Learning for Materials Scientists: An Introductory Guide toward Best Practices"](https://doi.org/10.1021/acs.chemmater.0c01907), *Chem. Mater.* 2020
- **Free textbook:** James *et al.*, [*An Introduction to Statistical Learning*](https://www.statlearning.com/) (Python edition)
- **Software:** [scikit-learn](https://scikit-learn.org/) · [pymatgen](https://pymatgen.org/) · [matminer](https://hackingmaterials.lbl.gov/matminer/) · [CBFV](https://github.com/kaaiian/CBFV) · [BoTorch](https://botorch.org/) / [Ax](https://ax.dev/) · [Honegumi](https://honegumi.readthedocs.io/)
- **Data sources:** [Materials Project](https://next-gen.materialsproject.org/) · [AFLOW](https://aflowlib.org/) · [NOMAD](https://nomad-lab.eu/) · [Citrine](https://citrine.io/) · [Foundry-ML](https://foundry-ml.org/)

## 🙏 Acknowledgments and license

Datasets and element-property tables are adapted from the [Best Practices](https://github.com/anthony-wang/BestPractices) repository (Wang *et al.* 2020, MIT license) and the [CBFV](https://github.com/kaaiian/CBFV) project. The silver-nanoparticle data are from Mekki-Berrada *et al.* (2021) via the Liang *et al.* (2021) Bayesian optimization benchmark. See [data/README.md](data/README.md) for details.

Code and notebooks are released under the [MIT License](LICENSE).
