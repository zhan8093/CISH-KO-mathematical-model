# CISHKO

A mechanistic ODE model of CISH-knockout tumour-infiltrating lymphocyte (TIL) therapy, calibrated to two murine studies and to a first-in-human phase 1 trial in metastatic gastrointestinal cancer.

The model describes interactions between tumour cells, endogenous TILs, and infused CISH-KO and wild-type T-cells. This repository contains the model definitions, the calibration and identifiability analyses, and the scripts that generate the figures in the accompanying manuscript.

## Requirements

- **Julia 1.12.x.** `Manifest.toml` was generated with Julia 1.12.6 and pins all 379 direct and transitive dependencies.
- **Python 3** with `pandas`, `numpy` and `matplotlib`, for Figures 4 and 5 only.

## Setup

```bash
git clone https://github.com/zhan8093/CISH-KO-mathematical-model.git
cd CISH-KO-mathematical-model
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running the analyses

Each script reads and writes relative filenames and expects to be run from its own directory, with `--project` pointing at the repository root:

```bash
cd scripts/figures/figure2_3
julia --project=../../.. good_ness_plot.jl
```

## Repository layout

```
src/       Model and objective function definitions
data/      Input datasets
scripts/
  calibration/   Sequential patient fitting and parameter uncertainty
  figures/       Manuscript figure generation
  supplement/    Profile-likelihood identifiability analysis and per-patient fits
```

Model calibration follows a sequential parameter-reduction procedure guided by pairwise correlation and profile-likelihood analysis. The final model fits seven patient-level parameters (`dL`, `kL`, `pL`, `muL`, `pN`, `f_dose`, `kappa`) together with a lesion-specific tumour growth rate.

The murine directories under `scripts/figures/figure2_3/` also contain an earlier MATLAB implementation of the same model (`.m` files) alongside the Julia version used for the manuscript figures. `scripts/calibration/test_objective.jl` re-evaluates the objective function at the stored optimal parameters as a consistency check, and `scripts/calibration/exp*/CISH_KO.slurm` are the job scripts used to run the fits on an HPC cluster.

## Reproducing the figures

Run each script from its own directory under `scripts/figures/`.

| Figure | Script | Output |
|---|---|---|
| 2a | `figure2_3/2015_mouse/cish_optim_run.jl` | `mouse_fit_2015_julia.png` |
| 2b | `figure2_3/2022_mouse/cish_optim_run.jl` | `mouse_fit_2022_julia.png` |
| 2c, 2d | `figure2_3/goodness_for_mouse.jl` | `mouse_tumor_overall_goodness_fit_{2015,2022}.png` |
| 3a, 3b | `figure2_3/good_ness_plot.jl` | `tumor_volume_prediction_vs_observed.png`, `cish_ratio_log_prediction_vs_observed.png` |
| 4a, 4b | `figure4/plot_mouse_params_v2.py` | `mouse_panel_a.*`, `mouse_panel_b.*` |
| 5a | `figure4/plot_mouse_params_v2.py` | `mouse_clinical_comparison.*` |
| 6a–d | `figure5/2D_parameter_versus_quantity.jl` | `scatter_combined_figure5.*` |
| 6e–h | `km_figure6/KM_curves.jl` | `km_combined_figure5.*` |
| 7a, 7b | `km_figure6/combination_therapy.jl`, then `plot_therapy_results.jl` | `therapy_comparison_L_max.*`, `therapy_comparison_T_AUC_normalized.*` |
| 8a, 8b | `figure8_schedule_sweep/schedule_metrics.jl`, then `schedule_heatmap.jl` | `schedule_sweep/heatmap_ratio.*`, `heatmap_auc.*` |

Supplementary material (S1 Text): `supplement/profile_likelihood.jl` produces the profile-likelihood curves, `supplement/correlation_analysis.jl` the parameter correlation heatmaps, and `supplement/replot_results.jl` the individual patient fits. Patient-level fitted parameters are produced by `calibration/main_optimized.jl`.

## Data availability

The murine tumour growth datasets, digitised from the published figures of the 2015 and 2022 studies, are included in this repository, as are the patient-level fitted model parameters.

Individual patient-level clinical data from the phase 1 trial (ClinicalTrials.gov NCT04426669) are not distributed here, as they contain potentially identifying clinical information. Figures 2 and 4–8 can be regenerated from the files provided; Figure 3 plots observed clinical measurements directly and requires access to the trial data.

## License

Released under the MIT License. See [LICENSE](LICENSE).
