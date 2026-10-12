# AI/ML for Ceramics and Glass — Curated Examples for Class 1

**Purpose:** Source material for the first 2-hour session of the ACerS short course *Practical AI and Machine Learning for Materials Science*. The goal of Class 1 is to motivate what materials informatics **is** (and **isn't**) using concrete, high-quality examples drawn specifically from ceramics and glass.

**Provenance:** Compiled 2026-10-09 from five Asta (Allen Institute for AI) literature searches — one each for regression, classification, image segmentation, generative models, and AI agents/autonomous labs — filtered and annotated for teaching value. The "More breadth" subsections were added 2026-10-12 to support additional slides; Asta was unavailable that morning (expired token), so those entries were verified against the publisher/index records via web search instead. Later on 2026-10-12, with the Asta token restored, two more Asta searches were run: a deep dive supporting the superhard-materials opening example (every number in that section was verified against the full text of the JACS paper) and a sweep for recent (2024-2026) work, which contributed the "New from Asta" entries.

---

## Framing: what materials informatics is (and isn't)

Talking points to open the class:

- **It is:** learning composition/processing/structure → property relationships from data; accelerating screening and design; extracting quantitative information from images and spectra; closing the loop between prediction and experiment.
- **It isn't:** a replacement for domain knowledge or experiments; magic that works without good data; only deep learning (classical ML like random forests and Gaussian processes dominates most ceramics/glass work); a guarantee of extrapolation outside the training distribution.
- Ceramics and glass are a *great* informatics testbed: a century of accumulated composition-property data (e.g., SciGlass with 300,000+ glass compositions), huge compositional design spaces (oxide glasses span dozens of components), and expensive trial-and-error synthesis that data-driven methods can shortcut.
- A useful narrative arc for the 2 hours: **regression → classification → images → generative design → agents/autonomous labs**, i.e., from "predict a number" to "the lab runs itself." Each step raises the level of autonomy and lowers the human-in-the-loop burden.

---

## Opening deep dive (8-10 slides): the ML-directed superhard materials story

*Mansouri Tehrani, Oliynyk, Parry, Rizvi, Couper, Lin, Miyagi, **Sparks**, Brgoch, "Machine Learning Directed Search for Ultraincompressible, Superhard Materials," J. Am. Chem. Soc. 140, 9844-9853 (2018). [DOI](https://doi.org/10.1021/jacs.8b02717) • [Open-access PDF (NSF PAR)](https://par.nsf.gov/servlets/purl/10094086) • 276 citations.*

Why lead with this one: it is a complete, end-to-end regression story (data → model → screening → synthesis → validation) told in a single paper, it is firsthand (you're a co-author — Utah did the synthesis, SEM, DAC, and hardness work), and regression is the most common ML task students will actually do. All numbers below were checked against the paper's full text on 2026-10-12, so they are safe to put on slides.

**Slide 1 — The target: superhard materials, and why discovery was stuck.** Superhard means Vickers hardness above 40 GPa. Two traditional design-rule families: (1) light-element covalent compounds (diamond, c-BN, B6O, c-BC2N) — cheap elements, but synthesis needs extreme pressure and temperature; (2) transition metals plus light elements with high valence-electron density (ReB2, WB4) — ambient-pressure metallurgy, but scarce, expensive metals. Decades of trial-and-error and simple heuristics; DFT cannot compute hardness directly because indentation plasticity spans scales from bonds to microstructure.

**Slide 2 — The key idea: predict a proxy, not the property.** Bulk modulus (B) and shear modulus (G) both correlate with hardness and *are* reliably computable by DFT — but elastic-tensor calculations are expensive, so in 2018 the Materials Project had elastic constants for only about 6,000 ordered compounds, under 10% of known inorganic solids. The plan: train ML on the DFT moduli that exist, then predict B and G for *everything else*, and use (B, G) as the beacon for hardness. Generalizable teaching point: when your property is unlearnable or unlabeled, find a proxy that scales with it.

**Slide 3 — Training data: 2,572 compounds, and the unglamorous work of cleaning.** 3,248 bulk/shear moduli pulled from Materials Project (DFPT + Voigt-Reuss-Hill; the dataset of de Jong et al., *Scientific Data* 2015, [DOI](https://doi.org/10.1038/sdata.2015.9)), then filtered: cross-referenced against Pearson's Crystal Database so only ambient-pressure-accessible phases remain; manual removal of unphysical values (negative moduli); restricted to binaries/ternaries/quaternaries; no noble gases, H, Tc, or Z > 83 (except U, Th); no thin films/foils. Final training set: **2,572 moduli**. Polymorphs distinguished as "composition, space-group number" (PtV,51 vs PtV,123). Teaching point: a fifth of the raw data was discarded before any learning happened — data curation *is* the method.

**Slide 4 — The model: SVR on 150 descriptors, trained on a desktop.** Support vector machine regression, radial basis function kernel, 10-fold cross-validation; separate models for B and G. Features: 34 elemental/compositional variables (periodic-table position, electronic structure, physical properties) each expanded via difference/average/max/min across the composition, plus 14 structural descriptors (crystal system, space group, cell volume, electron densities) — **150 total**. Genetic-algorithm feature selection kept only 52% of the variables for B and 44% for G. The memorable detail: all of it ran on a single Intel Core i5 desktop PC — the expensive part (DFT) was already done and shared publicly by someone else. That's the materials-informatics economic argument on one slide.

**Slide 5 — How good is the model (and where does it fail)?** Cross-validated parity plots: bulk modulus r² = 0.94, RMSE = 17.2 GPa; shear modulus r² = 0.84, RMSE = 16.5 GPa (comparable to DFT-vs-experiment scatter). More than 86% of compounds land within ±20% for B. The failures teach as much as the successes: G is systematically underpredicted above 200 GPa because the training set has almost nothing up there (two boron nitride points above 300 GPa) — data imbalance in action; the worst B misses (HgO off by 451%) are ionic oxides and halides, while metallic/covalent compounds behave. Which descriptors mattered: cohesive energy, density, covalent/crystal radii, volume per atom, Gilman electron density — ML quantitatively rediscovering classical bonding intuition.

**Slide 6 — The screen: 118,287 compounds in one shot.** The trained models predicted B and G for every binary (15,770), ternary (56,266), and quaternary (46,251) compound in Pearson's Crystal Database. Crucially, 23% contain rare-earth elements and 16% have site disorder or mixing — materials plain DFT *cannot handle* but the descriptor-based model can. Show the paper's Figure 3 (G vs B for all 118k): thresholds at B ≥ 300 GPa cut 118,287 candidates to 359; adding G ≥ 150 GPa leaves **258**. Why G matters: osmium has B near 400 GPa but hardness around 4 GPa — incompressible yet soft, because hardness lives in shear/dislocation resistance.

**Slide 7 — Sanity check, then selection.** The known superhard binaries all fall in the screened corner — WC (B_SVR = 384, G_SVR = 212), WB4 (315/227), TaC (323/210), ReB2 (330/191) — the model rediscovers the answers we already know, which is what licenses trusting it on the unknowns. Among unexplored candidates, the team picked the top ternary, Re0.5W0.5C (5th overall), and the top quaternary, Mo0.9W1.1BC (19th overall). Both had only ever been studied as superconductors (Tc of 3.8 K and 4.2 K), and — decisive practical constraint — both can be arc-melted at ambient pressure. Teaching point: the shortlist was filtered by *synthesizability and economics*, not just the predicted number.

**Slide 8 — Experiment I: making them, and a wrinkle.** Arc melting of elemental powders (about 0.4 g pellets, 80 A, argon). Stoichiometric ReWC2 left unreacted graphite; pure phase required carbon deficiency, ReWCx with x between 0.8 and 1.0 — carbon nonstoichiometry is endemic in refractory carbides, and the model didn't warn them. ReWC0.8 adopts a NaCl-type structure (Re/W mixed on the metal site, deficient carbon site); Mo0.9W1.1BC is a Mo2BC-type orthorhombic stacking with infinite boron zigzag chains. Real synthesis pushed back on the prediction — a point every experimentalist in the room will appreciate.

**Slide 9 — Experiment II: the predictions hold.** Diamond anvil cell compression with synchrotron XRD (ALS beamline 12.2.2, Pt internal standard, Ne medium, up to about 50 GPa) against Birch-Murnaghan fits: ReWC0.8 measured B0 = 380(8) GPa vs ML-predicted 398 GPa (under 5% error); Mo0.9W1.1BC measured B0 = 373(4) GPa vs predicted 370 GPa (under 0.25%). Vickers microindentation (ASTM E384, 10+ indents per load): at 0.49 N load, ReWC0.8 reaches Hv = 40(3) GPa and Mo0.9W1.1BC 42(2) GPa — both cross the superhard line; at 4.9 N they relax to 22.5(7) and 24.9(7) GPa (the indentation size effect, a roughly 40% drop also seen in ReB2 and WB4). Mo0.9W1.1BC beats its parent phase Mo2BC (about 17.8 GPa) and substitutes cheaper Mo for W — the economics angle again.

**Slide 10 — Honest caveats and what happened next.** Caveats worth saying out loud: "superhard at low load" is genuinely contested (the ReB2 controversy; asymptotic high-load hardness is the conservative standard, and the paper argues for reporting the full load dependence); the model predicts intrinsic moduli only — no grain-boundary or microstructural hardening; and hardness was never predicted directly, only reached via proxy. The lineage since: the same Brgoch group built a *direct*, load-dependent hardness model from indentation data ("Finding the Next Superhard Material through Ensemble Learning," *Advanced Materials* 2021, [DOI](https://doi.org/10.1002/adma.202005112)); atomic substitution tuned the borocarbide's hardness/ductility/cost balance (*Chemistry of Materials* 2019, [DOI](https://doi.org/10.1021/ACS.CHEMMATER.9B02596)); ML-informed evolutionary searches found superhard phases beyond known databases (*npj Computational Materials* 2019, [DOI](https://doi.org/10.1038/s41524-019-0226-8)); and physics-based "atomic stiffness" screening carried the ultraincompressibility hunt forward (*Nature Communications* 2023, [DOI](https://doi.org/10.1038/s41467-023-39826-2)). Segue to the rest of the lecture: this was *one* regression with *one* proxy — now the quick tour of everything else ML does for ceramics and glass.

*(If you need to trim to 8 slides: fold Slide 3 into 4, and Slide 8 into 9.)*

**Assets to pull from the paper for these slides:** Figure 1/2 (parity plots with error decomposition), Figure 3 (the 118k-compound G-vs-B screening map — the money figure), Figure 4-5 (XRD refinements and crystal structures), Figure 6 (DAC compression curves), Figure 7 (load-dependent hardness with indent micrographs).

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

### More breadth (added 2026-10-12)

- **Machine learning directed search for ultraincompressible, superhard materials** — Mansouri Tehrani, Brgoch et al. (co-author: T. Sparks), *Journal of the American Chemical Society* 140, 9844-9853 (2018). [Open PDF](https://par.nsf.gov/servlets/purl/10094086)
  **Now the Class 1 opening example — see the dedicated "Opening deep dive" section above** for the full 8-10 slide treatment with verified numbers. Once covered there, this section's remaining examples need only 1-2 slides total.

- **Predicting the dissolution kinetics of silicate glasses using machine learning** — Anoop Krishnan, Smedskjaer, Bauchy et al., *Journal of Non-Crystalline Solids* (2018). [arXiv](https://arxiv.org/abs/1712.06018)
  Neural networks predict aluminosilicate dissolution rates across pH where linear models fail. Chemical durability is *the* property for nuclear waste glasses, bioglasses, and display glass — regression on a property students may not expect ML to touch. (Topology-informed follow-up: *npj Materials Degradation* 2019.)

- **GlassNet: a multitask deep neural network for predicting many glass properties** — Cassar, *Ceramics International* (2023). [arXiv](https://arxiv.org/abs/2303.15538)
  One network, trained on 218,000+ SciGlass compositions, predicting **85 properties** simultaneously (optical, thermal, mechanical, viscosity...). Open-source in GlassPy — students can run it in class. Closes the arc that began with the single-property Tg paper above: from one property in 2018 to 85 in 2023. (Found via Asta 2026-10-12: an independent evaluation of GlassNet for glass stability and glass-forming ability, *Journal of the American Ceramic Society* 2024, [arXiv](https://arxiv.org/abs/2403.10682) — good if you want a critical-assessment beat.)

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

### More breadth (added 2026-10-12)

- **Holistic computational structure screening of more than 12,000 candidates for solid lithium-ion conductor materials** — Sendek et al. (Stanford), *Energy & Environmental Science* (2017). [DOI](https://doi.org/10.1039/C6EE02697D)
  A logistic-regression classifier for "superionic or not" — trained on only a few dozen literature measurements — funnels **12,831 Li-containing solids down to 21 promising ceramic electrolytes**, roughly a million times faster than DFT per candidate. The cleanest screening-funnel slide you can draw, and it's battery-relevant.

- **New tolerance factor to predict the stability of perovskite oxides and halides** — Bartel et al., *Science Advances* 5, eaav0693 (2019). [DOI](https://doi.org/10.1126/sciadv.aav0693)
  ML (the SISSO symbolic-regression method) rediscovered and *beat* a 100-year-old ceramics heuristic: the learned one-line descriptor τ classifies perovskite vs. non-perovskite at 92% accuracy where Goldschmidt's 1926 tolerance factor manages 74%. Perfect "ML can sharpen the rules you already teach" moment — the output is a formula, not a black box.

- **Machine learning modeling of superconducting critical temperature** — Stanev, Takeuchi et al., *npj Computational Materials* 4, 29 (2018). [DOI](https://doi.org/10.1038/s41524-018-0085-8)
  Trained on 12,000+ SuperCon entries, classifies materials as above/below Tc = 10 K at about 92% accuracy — the cuprate superconductors at the heart of it are ceramics — then screens a crystal database to propose 30+ new oxide candidates. Classification and regression chained into one discovery pipeline.

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

### More breadth (added 2026-10-12)

- **Semantic segmentation-assisted deep learning for quantitative multi-phase microstructural analysis** — Hwang et al. (KIST), *Journal of Power Sources* 471, 228458 (2020). [DOI](https://doi.org/10.1016/j.jpowsour.2020.228458)
  Segments the three-phase LSC/GDC/pore microstructure of solid oxide fuel cell cathodes from FIB-SEM slices and automatically extracts the quantities an electrochemist actually wants: phase fractions, particle sizes, and **triple-phase-boundary density**. Segmentation feeding directly into device performance metrics.

- **Deep learning-assisted microstructural analysis of Ni/YSZ anode composites for solid oxide fuel cells** — KIST group, *Materials Characterization* (2021). [DOI](https://doi.org/10.1016/j.matchar.2021.110906)
  Companion study on the classic Ni/YSZ cermet anode; notable for handling *ugly real data* — charging artifacts, curtaining, faint phase boundaries in FIB-SEM — exactly the images students' own instruments produce.

- **Machine-learning-revealed statistics of the particle-carbon/binder detachment in lithium-ion battery cathodes** — Jiang et al., *Nature Communications* 11, 2310 (2020). [DOI](https://doi.org/10.1038/s41467-020-16233-5)
  Mask R-CNN segments **650+ individual NMC oxide particles** in synchrotron X-ray nano-tomography of a battery cathode, turning images into degradation statistics (detachment correlates with charging rate). Shows segmentation scaled to population-level statistics no human would trace by hand — and connects ceramics to batteries for your electrochemistry-inclined students.

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

### More breadth (added 2026-10-12)

- **Microstructure generation via generative adversarial network for heterogeneous, topologically complex 3D materials** — Hsu, Holm, Salvador et al. (CMU/NETL), *JOM* (2021, arXiv 2020). [arXiv](https://arxiv.org/abs/2006.13886)
  A GAN generates synthetic **3D solid oxide fuel cell electrode microstructures** that match the real ones statistically *and* electrochemically (finite-element performance distributions agree; the conventional DREAM.3D generator's don't). Generative models for *microstructure*, not just crystal structures — unlimited realistic digital twins for simulation.

- **Crystal Diffusion Variational Autoencoder (CDVAE)** — Xie et al. (MIT), *ICLR* (2022). [arXiv](https://arxiv.org/abs/2110.06197)
  The missing link in the lineage slide: the paper that brought **diffusion** (the DALL-E/Stable Diffusion idea) to crystal generation, between the VAE/GAN era (iMatGen, FTCP) and MatterGen.

- **Crystal structure generation with autoregressive large language modeling (CrystaLLM)** — Antunes, Butler & Grau-Crespo, *Nature Communications* 15, 10570 (2024). [DOI](https://doi.org/10.1038/s41467-024-54639-7)
  A GPT-style language model trained on millions of CIF files generates plausible inorganic crystal structures *as text* — "ChatGPT, but it speaks crystallography." Instantly intuitive in the LLM era, and there's a free website where students can generate structures live during class.

### New from Asta (2026-10-12)

- **GlazyBench: a benchmark for ceramic glaze property prediction and image generation** — arXiv (2026). [DOI](https://doi.org/10.48550/arXiv.2605.06641)
  **23,148 real ceramic glaze formulations** powering two AI tasks: predict post-firing surface properties (color, transparency) from raw materials, and *generate images of what the fired glaze will look like*. The most ceramics-native example in this whole file — motivated by sparing studio potters the cost of trial-and-error firings. Guaranteed crowd-pleaser: every ceramist in the room has lost a kiln load to a glaze surprise, and it bridges regression and generative AI in one dataset.

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

### More breadth (added 2026-10-12)

- **Accelerated search for BaTiO3-based piezoelectrics with vertical morphotropic phase boundary using Bayesian learning** — Xue, Balachandran, Lookman et al., *PNAS* 113, 13301-13306 (2016). [DOI](https://doi.org/10.1073/pnas.1607412113)
  The ceramics-native active-learning classic that predates the robots: Bayesian learning plus Landau theory guided the synthesis of a (Ba,Ca)TiO3-Ba(Ti,Zr)O3 piezoelectric with better temperature stability than anything in the training set. Human hands did the sintering — the *algorithm* chose what to make next. The first rung on the autonomy ladder.

- **A mobile robotic chemist** — Burger et al. (Cooper group, Liverpool), *Nature* 583, 237-241 (2020). [DOI](https://doi.org/10.1038/s41586-020-2442-2)
  A free-roaming robot ran **688 photocatalysis experiments in 8 days** across a ten-variable space, guided by batched Bayesian optimization, finding mixtures 6x more active than the start. The memorable detail: it uses the *same benches and vials as humans* — autonomy retrofitted into an ordinary lab rather than a purpose-built one.

- **Autonomous chemical research with large language models (Coscientist)** — Boiko, MacKnight, Kline & Gomes (CMU), *Nature* 624, 570-578 (2023). [DOI](https://doi.org/10.1038/s41586-023-06792-0)
  A GPT-4-driven agent that reads documentation, writes code, and plans and executes real experiments (including successful optimization of Pd-catalyzed cross-couplings). The clearest published example of "LLM as scientist" — and a good prompt for discussing what transfers (and doesn't) to solid-state ceramics, where synthesis is slower and feedback is scarcer.

### New from Asta (2026-10-12)

- **Large language models (LLMs) for materials design** — *Advanced Functional Materials* (2025). [DOI](https://doi.org/10.1002/adfm.202525897)
  An up-to-date review mapping what LLMs actually do in materials work today: domain Q&A, automated data extraction from papers, semantic-driven design, synthesis planning, and property prediction. Useful as the "where the field is right now" reference for the final slides — and for answering the inevitable "can I just ask ChatGPT?" question with nuance.

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

## Suggested 2-hour flow and slide budget (revised 2026-10-12)

Structure: one deep worked example first (the superhard story, told firsthand), then a fast breadth tour at 1-2 slides per example. Roughly 30-35 slides total.

1. **(0:00-0:15) Framing — 3-4 slides.** What materials informatics is/isn't; why ceramics and glass are the ideal testbed (century of data, huge composition spaces, costly trial-and-error).
2. **(0:15-0:45) Regression deep dive: the superhard story — 8-10 slides.** Use the "Opening deep dive" section above verbatim; it carries the entire regression pedagogy (proxies, data cleaning, descriptors, validation, failure modes, experimental ground truth).
3. **(0:45-0:55) Regression breadth — 2-3 slides.** Slide A: Cassar's Tg network (55,000 glasses) vs the 254-sample microwave dielectrics study — big-data/small-data contrast on one slide. Slide B: GlassNet — one network, 85 properties, students can run it in GlassPy. Optional slide C: glass dissolution kinetics (waste glass/bioglass durability).
4. **(0:55-1:10) Classification — 3-4 slides (1-2 per example, pick 2-3).** Best picks: Sendek's 12,831-to-21 electrolyte screening funnel (1 slide); Bartel's learned tolerance factor beating Goldschmidt 92% to 74% (1 slide); CNN phase ID from 1.8M synthetic XRD patterns (1 slide); high-entropy ceramics discovery if time allows.
5. **(1:10-1:25) Image segmentation — 3-4 slides.** Before/after: ImageJ thresholding vs deep learning on the same micrograph (1 slide); 3D FIB-SEM segmentation of polycrystalline ceramics + SOFC triple-phase-boundary extraction (1-2 slides); Mask R-CNN particle statistics in battery cathodes (1 slide).
6. **(1:25-1:40) Generative models — 3-4 slides.** Lineage slide: iMatGen → CDVAE → MatterGen (1 slide); Cassar's optical glass design by ML + genetic algorithm — glass-native inverse design (1 slide); GlazyBench glaze prediction/generation as the fun closer (1 slide); CrystaLLM live demo if network allows.
7. **(1:40-1:55) Agents and autonomous labs — 3-4 slides.** Autonomy ladder: BaTiO3 Bayesian design → CAMEO → mobile robotic chemist → A-Lab (1-2 slides); A-Lab achievements *and* the skepticism that followed (1 slide); Coscientist/LLM agents + the sober evaluation counterweight (1 slide).
8. **(1:55-2:00) Wrap — 1-2 slides.** The five tasks on one slide; course roadmap.

---

*Generated with Asta literature search (Allen Institute for AI) + Claude. Verify author lists and details against the linked records when building slides; a few entries indexed with early online dates may display later years on the publisher page.*
