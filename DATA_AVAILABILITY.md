# Data availability statement — draft for submission

Working draft of the PLOS Computational Biology data availability statement, plus
notes on what is true about the repository today.

---

## Statement (paste into the submission form)

> All model code used for simulation, calibration, identifiability analysis and figure
> generation, together with the murine datasets and the patient-level fitted model
> parameters, is available on GitHub at https://github.com/[ORG]/[REPO] and archived on
> Zenodo (DOI: 10.5281/zenodo.[NNNNNNN]).
>
> Individual patient-level clinical data from the phase 1 trial (ClinicalTrials.gov
> NCT04426669) cannot be shared publicly because they contain potentially identifying
> clinical information.

---

## Placeholders to fill

| Placeholder | Notes |
|---|---|
| `[ORG]/[REPO]` | The GitHub repository does not exist yet. |
| `[NNNNNNN]` | Zenodo DOI. Link the GitHub repo to Zenodo and cut a release; the DOI is minted automatically. Your manuscript note is right that PLOS will not accept "to be created soon" — the DOI must resolve at submission. |

Two notes, neither blocking:

- **Licence.** PLOS Computational Biology requires an OSI-approved licence only for
  *Software* articles. For a research article, authors are "free to choose whatever
  license they wish"; a licence is recommended, not mandated. See
  https://journals.plos.org/ploscompbiol/s/code-availability
- **Contact for restricted data.** The code-sharing policy quoted above asks for a
  third-party contact when *code* is restricted, which is not our case. Its final
  bullet defers restricted *data* to PLOS's separate data availability guidelines,
  which do ask for a non-author point of contact. Left out of the statement for
  brevity; add it if the editorial office asks at submission.

---

## Code completeness: satisfied

Every numbered figure in the manuscript now has a script behind it, so the first
sentence is true as written:

| Figure | Script |
|---|---|
| 2 | `figure2_3/*/cish_optim_run.jl`, `goodness_for_mouse.jl` |
| 3 | `figure2_3/good_ness_plot.jl` |
| 4, 5 | `figure4/plot_mouse_params_v2.py` |
| 6 | `figure5/2D_parameter_versus_quantity.jl`, `km_figure6/KM_curves.jl` |
| 7 | `km_figure6/combination_therapy.jl` → `plot_therapy_results.jl` |
| 8 | `figure8_schedule_sweep/schedule_metrics.jl` → `schedule_heatmap.jl` |

`schedule_heatmap.jl` was run against the committed environment and regenerates Fig 8a
and Fig 8b, so that path is verified rather than assumed.

The schedule-optimisation study that arrived in the same folder is in `_archive/` and
is not published: it reports no manuscript figure, so it falls outside the
code-availability requirement. If it becomes part of a revision, move it back into
`scripts/` before resubmitting — the statement promises the code behind every reported
result.

---

## What a clean clone reproduces

The model-derived parameter and outcome tables are now included, so a reviewer who
clones the repository can regenerate almost every figure with no clinical data at all:

| Figure | Runs from a clean clone? | Verified |
|---|---|---|
| 2 | **yes** — murine data included | |
| 4, 5 | **yes** — `mouse_parameters.csv` + `patient_tumor_params.csv` | inputs checked; `pandas` not installed locally so not executed |
| 6a–d | **yes** | ran `2D_parameter_versus_quantity.jl` |
| 6e–h | **yes** | ran `KM_curves.jl` |
| 7 | **yes** | ran `plot_therapy_results.jl` |
| 8a, 8b | **yes** | ran `schedule_heatmap.jl` |
| 3 | no | `good_ness_plot.jl` plots observed clinical measurements directly — irreducibly needs the restricted data |

Re-running the underlying *simulations* rather than the plots still needs the clinical
inputs: `combination_therapy.jl` (Fig 7) and `schedule_metrics.jl` (Fig 8) read tumour
sizes and infused doses to set initial conditions. The committed outcome tables are what
make the figures themselves reproducible.

### What was changed to get here

`Patient_<id>_paras.json`, `patient_tumor_params.csv`,
`patient_tumor_params_with_ratio.csv`, `combination_therapy_results.csv` and
`therapy_comparison_5_results.csv` were un-ignored. Two columns —
`initial_tumor_size` and `cish_dose` — were removed from the tracked copies of the
latter three, because those are direct clinical measurements and the statement above
promises they are not shared. Nothing reads them: the therapy scripts *write* those
columns and recreate them if absent, and the figure scripts never reference them.
Unmodified originals are preserved in `_archive/original_parameter_tables/`.

This also fixed a defect: `plot_mouse_params_v2.py` produces Fig 4 (mouse-only) and
Fig 5 in one run and reads `patient_tumor_params.csv`, so while that file was excluded
the script failed on a clean clone and **Figure 4 was lost along with Figure 5**, even
though Fig 4 contains no patient data.

## Remaining item: `T0` in `metrics.csv`

`metrics.csv` (Fig 8) still carries columns the manuscript does not publish per patient:

| Column | In Fig 8? |
|---|---|
| `patient`, `schedule` | yes — row and column labels |
| `ratio` | yes — Fig 8a |
| `AUC_T` | yes — Fig 8b |
| `T0`, `T_end`, `T_min`, `t_min` | **no** |

`T0` is the patient's initial tumour size — a clinical measurement, and the one column
inconsistent with the stripping described above. `T_end` is a simulated value, but
since `ratio = T_end/T0`, publishing both recovers `T0` anyway.

`schedule_heatmap.jl` only reads `patient`, `schedule`, `ratio` and `AUC_T`, so
dropping the other four would cost nothing. Say the word and it is a one-line change.
