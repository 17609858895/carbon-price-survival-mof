# Carbon-price survival learning for MOF carbon capture

Core analysis and figure code for **Carbon-Price Survival Learning Maps Viability of Metal–Organic Frameworks for Carbon Capture**. This repository contains the code and the conceptual illustration required by the figure renderer. Source datasets, trained models, derived data, numerical-result archives and manuscript files are obtained or generated separately.

## Data

Use version 3 of the PrISMa data: https://doi.org/10.5281/zenodo.12793408 . The original data are distributed under CC BY 4.0. Cite Charalambous et al., A holistic platform for accelerating sorbent-based carbon capture, Nature 632, 89–94 (2024), https://doi.org/10.1038/s41586-024-07683-8 .

Extract the source archives into work/data_raw, retaining the archive directory names (Material_Properties, KPIs_Simulated, KPIs_Experimental, Flags and Cifs). Original source archives and the large material–scenario table are not duplicated here. The data-building script creates the latter from the source data. The data-building and metal-node scripts regenerate the derived descriptors from those archives.

## Environment

Python 3.12.3 was used. Install the versions in requirements.txt in a dedicated environment. The requirements describe the checked research environment; the source data remain a separate download.

## Analysis

From the work directory, set the full analysis configuration in PowerShell:

```powershell
$env:PRISMA_PANEL = "0"
$env:PRISMA_FULL = "1"
$env:PRISMA_THREADS = "4"
$env:PRISMA_RIV_MEAN = "linear+phys"
python scripts/01_build_dataset.py
python scripts/14_metal_node_descriptors.py
python scripts/08_data_landscape.py
$env:PRISMA_SEEDS = "0,1,2"
python scripts/02_target_parameterisation.py
python scripts/04_price_survival.py
python scripts/11_input_tiers.py
python scripts/15_tail_extrapolation.py
python scripts/07_prospective_atlas.py
python scripts/12_atlas_purity.py
python scripts/18_entry_price.py
$env:PRISMA_RIV_MEAN = "lgb"
python scripts/16_conformal_selection.py
$env:PRISMA_RIV_MEAN = "linear+phys"
python scripts/16_conformal_selection.py
python scripts/05_hierarchical_transfer.py
python scripts/09_config_transfer.py
python scripts/06_causal_ground_truth.py
python scripts/10_shap_design_rules.py
python scripts/13_nuisance_capacity.py
python scripts/03_benchmark20_stacking.py
$env:PRISMA_PANEL = "1"
python scripts/02b_decision_focused.py
New-Item -ItemType Directory -Path results/panel54_prelim -Force | Out-Null
Copy-Item -LiteralPath results/exp1b_decision_focused.csv -Destination results/panel54_prelim/exp1b_decision_focused.csv
$env:PRISMA_PANEL = "0"
python figures/make_figures.py
```

The script name containing "prospective" is retained for compatibility; its feedback-round results are a retrospective evaluation on previously published held-out materials. The final response is reciprocal nCAC, in t per euro; plot values in t per thousand euros are scaled by 1,000.

The target comparison repeats RIV and hurdle fits across three material partitions. The twenty-regressor/stacking comparison is exploratory and uses a single level of cross-fitted base predictions. It is not a nested estimate of stacking performance. The distribution intervals use cross-fitted residual scale calibration. Conformal screening uses separate material folds for training, calibration and testing; the model specification was selected using this library, so the reported workflow false-discovery rates are empirical. The AFT implementation supplies original-time censoring bounds and evaluates raw log-time predictions.

Figure source exports combined PNG/PDF and independently rendered PNG/PDF panels. The conceptual illustration is included as an asset; quantitative panels are computed from the analysis results. The numerical summary files retain the analysis column names for reproducibility. Submission-facing variable definitions are in the Supporting Information and supplementary workbook.

The final paper contains ten main figures and 44 independently rendered panels. Former supplementary panels were regrouped by target representation, distribution calibration, frontier ablation, shortlist errors, humidity penalties and effect estimation. The 20-regressor and stacking comparison remains in the numerical supplement. The decision-focused comparison uses a separate 54-scenario, 500-tree setting; ranking relevance grades are computed from complete-scenario responses before fold fitting. Its results describe that exploratory implementation. Re-evaluation with training-only relevance grades is separate from the matched RIV/hurdle target comparison.

## Quick verification

From the repository root, run `python tests/smoke_math.py`. This uses no downloaded data and trains no models. It checks the actual shortlist functions for the Benjamini–Hochberg step, invariance to a common positive score rescaling, permutation equivariance and the plug-in rule's posterior error budget. It also checks the reciprocal-response sign and carbon-price threshold across net-avoiding and net-emitting cases, and parses every supplied Python file.

The full workflow above retrains the study's models and can require substantial computation. It was not rerun merely to publish this repository. Existing numerical results are provided with the manuscript's supplementary tables. Full data, model and source archives are intentionally excluded from Git tracking.

The original PrISMa dataset retains its CC BY 4.0 license and its own authorship. No new license grant is made for third-party material. This repository does not contain the PrISMa simulator itself.
