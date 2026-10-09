# AI/ML for Ceramics and Glass — Curated Examples for Class 1

**Purpose:** Source material for the first 2-hour session of the ACerS short course *Practical AI and Machine Learning for Materials Science*. The goal of Class 1 is to motivate what materials informatics **is** (and **isn't**) using concrete, high-quality examples drawn specifically from ceramics and glass.

**Provenance:** Compiled 2026-10-09 from five Asta (Allen Institute for AI) literature searches — one each for regression, classification, image segmentation, generative models, and AI agents/autonomous labs — filtered and annotated for teaching value.

---

## Framing: what materials informatics is (and isn't)

Talking points to open the class:

- **It is:** learning composition/processing/structure → property relationships from data; accelerating screening and design; extracting quantitative information from images and spectra; closing the loop between prediction and experiment.
- **It isn't:** a replacement for domain knowledge or experiments; magic that works without good data; only deep learning (classical ML like random forests and Gaussian processes dominates most ceramics/glass work); a guarantee of extrapolation outside the training distribution.
- Ceramics and glass are a *great* informatics testbed: a century of accumulated composition-property data (e.g., SciGlass with 300,000+ glass compositions), huge compositional design spaces (oxide glasses span dozens of components), and expensive trial-and-error synthesis that data-driven methods can shortcut.
- A useful narrative arc for the 2 hours: **regression → classification → images → generative design → agents/autonomous labs**, i.e., from "predict a number" to "the lab runs itself." Each step raises the level of autonomy and lowers the human-in-the-loop burden.

---

## 1. Regression — predicting continuous properties

The workhorse of materials informatics. Great for introducing features/descriptors, train/test splits, and model interpretability.

### Headline examples

- **Predicting glass transition temperatures using neural networks** — Cassar, de Carvalho & Zanotto, *Acta Materialia* (2018). [DOI](https://doi.org/10.1016/J.ACTAMAT.2018.08.022)
  A neural network trained on **55,000+ multicomponent oxide glass compositions** (SciGlass) to predict Tg. Arguably *the* canonical glass informatics paper: simple input (composition), huge legacy dataset, useful output. Excellent for showing what a century of accumulated glass data makes possible.

- **Explainable machine learning algorithms for predicting glass transition temperatures** — Alcobaça et al. (incl. Cassar, Zanotto), *Acta Materialia* (2020). [DOI](https://doi.org/10.1016/j.actamat.2020.01.047)
  Benchmarks six ML algorithms (random forest beat the neural net here) on Tg and emphasizes *explainability* — a natural hook for the "informatics isn't a black box if you don't let it be" message.

- **Machine learning approaches for permittivity prediction and rational design of microwave dielectric ceramics** — *Journal of Materiomics* (2021). [DOI](https://doi.org/10.1016/J.JMAT.2021.02.012)
  Predicts permittivity of microwave dielectric ceramics (5G/IoT motivation) from a dataset of only **254 single-phase ceramics** — a nice contrast with the 55,000-sample glass example for discussing small-data realities in ceramics.

- **Predicting the Young's modulus of silicate glasses using high-throughput molecular dynamics and machine learning** — Yang, Bauchy et al., *Scientific Reports* (2019). [DOI](https://doi.org/10.1038/s41598-019-45344-3)
  When experimental data are scarce or inconsistent, simulations can generate the training set. Good for introducing the simulation-plus-ML hybrid strategy.

- **Glass hardness: predicting composition and load effects via symbolic reasoning-informed machine learning** — *Acta Materialia* (2023). [Link](https://api.semanticscholar.org/CorpusId:255999917)
  Around 3,000 glasses; captures the nonlinear indentation size effect. Shows physics-informed ML (symbolic reasoning constraints) beating naive fits.

- **Scalable Gaussian processes for predicting optical, physical, thermal, and mechanical properties of inorganic glasses** — *Materials Advances* (2021). [PDF](https://pdfs.semanticscholar.org/c64e/4e044ca7bac7b59a21d3c37af8d45aa16eb1.pdf)
  Gaussian processes give **uncertainty estimates** with each prediction — the natural bridge to active learning later in the course.

- **Machine learning unveils composition-property relationships in chalcogenide glasses** — *Acta Materialia* (2021). [arXiv](https://arxiv.org/pdf/2106.07749.pdf)
  Extends the paradigm beyond oxides to chalcogenides (photonics applications).

### Teaching angles
- Compare data regimes: 55,000 glasses vs. 254 dielectric ceramics — what changes about model choice and validation?
- Interpretability: which oxides raise Tg? SHAP/feature-importance plots make great slides.
- Failure mode to show: extrapolation beyond the composition space in the training data.

---

## 2. Classification — predicting categories

Introduces decision boundaries, class imbalance, and screening funnels.

### Headline examples

- **Discovery of high-entropy ceramics via machine learning** — Kaufmann, Vecchio et al., *npj Computational Materials* (2020). [DOI](https://doi.org/10.1038/s41524-020-0317-6)
  Classifies whether a candidate high-entropy carbide will form single-phase, using the entropy-forming-ability descriptor — and the predictions were **experimentally validated** by synthesizing new high-entropy carbides. Classification leading directly to the discovery of new ceramics.

- **Machine learning approach for prediction and understanding of glass-forming ability** — Sun et al., *Journal of Physical Chemistry Letters* (2017). [Link](https://www.ncbi.nlm.nih.gov/pubmed/28697303)
  The classic "will it glass?" problem: classifying good vs. poor glass formers from composition. Glass-forming ability is the quintessential ceramics/glass classification task with decades of empirical rules to compare against.

- **A deep-learning technique for phase identification in multiphase inorganic compounds using synthetic XRD powder patterns** — Lee et al., *Nature Communications* (2020). [DOI](https://doi.org/10.1038/s41467-019-13749-3)
  A CNN trained on **1.8 million synthetic XRD patterns** identifies phases in multiphase Sr-Li-Al-O ceramics (LED phosphor space). Beautiful example of classification applied to characterization data rather than composition — and of using *simulated* training data when labels are scarce. Follow-up with phase-fraction regression: [*Inorganic Chemistry Frontiers* (2021)](https://doi.org/10.1039/D0QI01513J).

- **Soliquidy: a descriptor for atomic geometrical confusion** — *npj Computational Materials* (2025). [Link](https://api.semanticscholar.org/CorpusId:275994053)
  A modern descriptor-engineering story applied to glass/no-glass classification of metallic thin films. Useful to make the point that *feature design is where the physics enters*.

### Teaching angles
- Regression vs. classification is often just a reframing: Tg (number) vs. glass former yes/no (label).
- Synthetic training data (simulated XRD) as a strategy when experiments can't supply enough labels.
- Screening-funnel framing: cheap classifier first, expensive experiments only on survivors.

---

## 3. Image segmentation — quantitative microscopy

Computer vision applied to the micrographs every ceramist already collects. Very visual — ideal mid-class energy boost.

### Headline examples

- **Deep learning for three-dimensional segmentation of electron microscopy images of complex ceramic materials** — *npj Computational Materials* (2024). [DOI](https://doi.org/10.1038/s41524-024-01226-5)
  Semantic segmentation of FIB-SEM data of polycrystalline ceramics (grains, pores, secondary phases) that human experts previously traced by hand. The headline ceramics-specific segmentation paper.

- **Deep-learning-based pyramid-transformer for localized porosity analysis of hot-press sintered ceramic paste** — *PLoS ONE* (2024). [Link](https://api.semanticscholar.org/CorpusId:272398552)
  Transformer-based segmentation (PSTNet) of grains and pores in SEM images of sintered ceramics — shows the field moving beyond CNNs/U-Nets.

- **Optimized and autonomous ML framework for characterizing pores, particles, grains and grain boundaries in microstructural images** — *Computational Materials Science* (2021). [Link](https://api.semanticscholar.org/CorpusId:231632142)
  End-to-end automated microstructure quantification; a good "replace weeks of manual point counting" story.

- **Boundary learning by using weighted propagation in convolution network (WPU-Net)** — (2019). [Link](https://api.semanticscholar.org/CorpusId:280070223)
  U-Net variant customized for grain-boundary detection in polycrystalline micrographs; useful for explaining why off-the-shelf vision models need domain adaptation (thin, faint boundaries; class imbalance).

- **Toward quantitative fractography using convolutional neural networks** — *Engineering Fracture Mechanics* (2019). [Link](https://api.semanticscholar.org/CorpusId:199452731)
  CNNs making centuries-old qualitative fractography quantitative — resonates with the brittle-fracture focus of ceramics.

- **Ceramic cracks segmentation with deep learning** — *Applied Sciences* (2021). [Link](https://api.semanticscholar.org/CorpusId:237825894)
  Applied/industrial flavor: automated crack and defect detection for ceramic manufacturing QC (related industrial examples exist for ceramic insulators and zirconia dental crowns).

- Baseline for contrast: **Image segmentation variants for semi-automated quantitative microstructural analysis with ImageJ** — *Praktische Metallographie* (2020). [Link](https://api.semanticscholar.org/CorpusId:228845597)
  The classical thresholding workflow students may already use — frames *why* learned segmentation is a step change.

### Teaching angles
- Before/after slide: manual ImageJ thresholding vs. deep-learning segmentation on the same micrograph.
- Segmentation outputs feed straight into properties students care about: grain size distributions, porosity, crack networks.
- Labeling cost is the bottleneck — a nice echo of the data-scarcity theme from sections 1-2.

---

## 4. Generative models — inverse design

Flips the problem: instead of predicting properties from a composition, generate compositions/structures with target properties.

### Headline examples

- **A generative model for inorganic materials design (MatterGen)** — Zaini et al. (Microsoft Research), *Nature* (2025). [Link](https://api.semanticscholar.org/CorpusId:275591809)
  The current state of the art: a diffusion model that generates stable inorganic crystals conditioned on chemistry, symmetry, and target properties. The "DALL-E for crystals" slide.

- **Inverse design of solid-state materials via a continuous representation (iMatGen)** — Noh et al., *Matter* (2019). [DOI](https://doi.org/10.1016/j.matt.2019.08.017)
  The early VAE landmark: encodes crystals into a continuous latent space and discovers new vanadium oxide polymorphs. Great for explaining *what a latent space is*.

- **An invertible crystallographic representation for general inverse design of inorganic crystals with targeted properties (FTCP)** — Ren et al., *Matter* (2022, arXiv 2020). [arXiv](https://arxiv.org/pdf/2005.07609.pdf)
  General-purpose inverse design not limited to one chemistry — a good step between iMatGen and MatterGen.

- **Generative adversarial networks for crystal structure prediction** — Kim et al., *ACS Central Science* (2020). [Link](https://api.semanticscholar.org/CorpusId:214795159)
  The GAN flavor of the same idea; useful if you want to show the GAN/VAE/diffusion lineage explicitly.

- **Inverse design of porous materials using artificial neural networks** — Kim et al., *Science Advances* (2020). [DOI](https://doi.org/10.1126/sciadv.aax9324)
  A GAN generating zeolites (porous silicates — honorary ceramics) with user-specified methane adsorption properties. Visually striking generated structures.

- **Designing optical glasses by machine learning coupled with a genetic algorithm** — Cassar et al., *Ceramics International* (2021, arXiv 2020). [arXiv](https://arxiv.org/pdf/2008.09187.pdf)
  Inverse design *for glass specifically*: neural-network property models + genetic algorithm search over composition space to propose optical glasses with target refractive index, Abbe number, and Tg. Bridges section 1's regression models to generative design — the same Tg model becomes a design engine.

- **Deep learning aided rational design of oxide glasses** — Ravinder et al., *Materials Horizons* (2020). [Link](https://doi.org/10.1039/c9mh01420a)
  Property maps over oxide glass composition space ("glass selection charts") used for design — an accessible, glass-native stepping stone before full generative models.

### Teaching angles
- Distinguish **inverse design via optimization** (ML model + genetic algorithm — the optical glass paper) from **true generative models** (VAE/GAN/diffusion sampling new structures). Both count; the former is much easier to adopt.
- Honest caveats: synthesizability of generated crystals is the open problem (sets up section 5's A-Lab).
- For context beyond generation: Google DeepMind's GNoME (Merchant et al., *Nature* 2023) scaled graph-network screening to millions of predicted-stable inorganic crystals, many of them oxides/ceramics — pairs naturally with MatterGen on a "scale of modern discovery" slide.

---

## 5. Agents and autonomous laboratories — closing the loop

The frontier: AI that plans, executes, and iterates on experiments.

### Headline examples

- **An autonomous laboratory for the accelerated synthesis of inorganic materials (A-Lab)** — Szymanski et al. (Berkeley Lab/Ceder group), *Nature* (2023). [DOI](https://doi.org/10.1038/s41586-023-06734-w)
  **The** showpiece: robots + ML planners + active learning performed solid-state (ceramic!) synthesis and realized **41 of 58 novel target oxides/phosphates in 17 days**. Directly ceramic-relevant — the targets are inorganic powders made by solid-state routes every ceramist knows. (Worth noting in class: follow-up commentary debated how many products were truly novel single phases — a great scientific-skepticism moment.)

- **Autonomous intelligent agents for accelerated materials discovery (CAMEO)** — *Chemical Science* (2020). [Link](https://api.semanticscholar.org/CorpusId:221931102)
  Agent-based sequential experiment selection (closed-loop at a synchrotron beamline); introduces the "agent decides the next experiment" concept with physics-informed priors.

- **Autonomous inorganic materials discovery via multi-agent physics-aware scientific reasoning (SPARKS)** — (2025). [Link](https://api.semanticscholar.org/CorpusId:280527061)
  LLM-based multi-agent system executing the full discovery cycle — ideation, planning, experiment, iteration — for inorganic materials.

- **Towards scientific intelligence: a survey of LLM-based scientific agents** — (2025). [arXiv](https://api.semanticscholar.org/CorpusId:277467270)
  Reference survey for the lecture: taxonomy of hypothesis-generation, experiment-design, and analysis agents.

- **Autonomous multi-robot synthesis and optimization of metal halide perovskite nanocrystals** — *Nature Communications* (2025). [Link](https://api.semanticscholar.org/CorpusId:280769523)
  Multi-robot self-driving lab in action (perovskite nanocrystals); vivid photos/video assets typically available for slides.

- **Evaluating LLM-based AI agents integrated with materials synthesis tools: the case of atomic layer deposition** — (2026). [Link](https://api.semanticscholar.org/CorpusId:291531204)
  A sober evaluation paper — useful counterweight showing agents are promising but far from solved.

### Teaching angles
- Ladder of autonomy: human-designed experiments → active learning suggests next experiment (CAMEO) → robot executes synthesis (A-Lab) → LLM agents plan campaigns (SPARKS).
- The A-Lab controversy is a feature, not a bug, for teaching: autonomous characterization (automated Rietveld) can overclaim; human experts still matter.
- Connect back to Class 1's opening: agents are only as good as the predictive models (sections 1-2) and characterization tools (section 3) underneath them.

---

## Datasets and tools worth name-dropping in Class 1

- **SciGlass / Interglad** — the legacy glass composition-property databases (300,000+ entries) behind most glass ML papers above; SciGlass is now freely available.
- **GlassPy / GlassNet** (Cassar) — open-source Python package and pretrained models for glass property prediction; students can run the section-1 examples themselves.
- **Materials Project / OQMD / AFLOW** — DFT databases underpinning the classification and generative examples (AFLOW's entropy-forming-ability powered the high-entropy ceramics paper).
- **Matbench / Matminer / pymatgen** — benchmarking and featurization infrastructure used throughout the course.

## Suggested 2-hour flow (strawman)

1. (0:00-0:20) What materials informatics is/isn't; why ceramics and glass are ideal (century of data, huge composition spaces, costly trial-and-error).
2. (0:20-0:45) Regression: Tg from 55,000 glasses; interpretability; small-data dielectric ceramics contrast.
3. (0:45-1:05) Classification: glass-forming ability; high-entropy ceramics discovery; CNN phase ID from XRD.
4. (1:05-1:25) Image segmentation: micrograph before/after; 3D FIB-SEM segmentation; fractography and crack detection.
5. (1:25-1:45) Generative models: optical glass design by ML + genetic algorithm; iMatGen → MatterGen lineage; synthesizability caveat.
6. (1:45-2:00) Agents and autonomous labs: CAMEO → A-Lab → LLM agents; the autonomy ladder; healthy skepticism; course roadmap.

---

*Generated with Asta literature search (Allen Institute for AI) + Claude. Verify author lists and details against the linked records when building slides; a few entries indexed with early online dates may display later years on the publisher page.*
