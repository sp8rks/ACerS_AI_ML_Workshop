# Datasets

All datasets are small CSV/JSON files that load in seconds. In the notebooks, use `workshop_utils.load_dataset("<name>")`.

| Name | File | Rows | Columns | Description | Used in |
|---|---|---|---|---|---|
| `bulk_modulus` | `bulk_modulus.csv` | 4,905 | `formula`, `target` | DFT (AFLOW AEL-AGL) Voigt-Reuss-Hill bulk modulus, GPa | 1, 2, 3, 4, 6 |
| `shear_modulus` | `shear_modulus.csv` | 4,905 | `formula`, `target` | DFT (AFLOW AEL-AGL) VRH shear modulus, GPa | exercises |
| `log10_thermal_conductivity` | `log10_thermal_conductivity.csv` | 4,896 | `formula`, `target` | DFT (AFLOW AGL) log₁₀ thermal conductivity at 300 K, W/(m·K) | exercises |
| `bandgap` | `bandgap.csv` | 27,767 | `formula`, `target` | Band gap, eV (0 = metal) | 3 |
| `heat_capacity_raw` | `heat_capacity_raw.csv` | 4,583 | formula, T (K), Cp (J/mol·K) | Heat capacity vs. temperature **as compiled, with errors** (missing values, negative T and Cp) | 4 |
| `heat_capacity_clean` | `heat_capacity_clean.csv` | 4,547 | `formula`, `T`, `Cp` | Cleaned version of the above | 1, 3 |
| `agnp` | `agnp_flow_synthesis.csv` | 3,295 | 5 flow-rate inputs, `loss` | Microfluidic Ag nanoparticle synthesis; `loss` = spectral mismatch to target (lower is better) | 5 |
| — | `ceramics_text_snippets.json` | 10 snippets | text + answer key | **Synthetic** abstract-style paragraphs written for this course, for LLM extraction/RAG exercises | 6 |

## Sources and credits

- **Bulk/shear modulus, thermal conductivity, band gap, heat capacity:** from the companion repository of Wang, A. Y.-T. *et al.*, "Machine Learning for Materials Scientists: An Introductory Guide toward Best Practices", *Chem. Mater.* **32**, 4954 (2020), [doi:10.1021/acs.chemmater.0c01907](https://doi.org/10.1021/acs.chemmater.0c01907) ([GitHub](https://github.com/anthony-wang/BestPractices), MIT license). The original train/val/test splits were concatenated here; the notebooks make their own splits. The elastic and thermal-conductivity values are AFLOW AEL-AGL calculations ([aflowlib.org](https://aflowlib.org/)); see the paper and repository for the provenance of the band-gap and heat-capacity data.
- **Silver nanoparticle synthesis:** Mekki-Berrada, F. *et al.*, "Two-step machine learning enables optimized nanoparticle synthesis", *npj Comput. Mater.* **7** (2021); as packaged in Liang, Q. *et al.*, "Benchmarking the performance of Bayesian optimization across multiple experimental materials science domains", *npj Comput. Mater.* **7** (2021).
- **Element property tables** (`workshop_utils/element_properties/`): from the [CBFV](https://github.com/kaaiian/CBFV) project: Oliynyk, Magpie (Ward *et al.* 2016), mat2vec (Tshitoyan *et al.* 2019), one-hot and random-200.
- **Text snippets:** written for this workshop. Values are typical of the literature but **are not results from any specific paper**. Do not cite them.
