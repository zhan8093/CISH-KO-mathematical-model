# Patient data audit

Inventory of every file in this repository that contains individual patient data
or subject identifiers. **Nothing here has been deleted or moved out of the tree.**
Everything listed is excluded from git by `.gitignore`, except where explicitly
flagged below as still tracked.

Subjects appearing in the analysis: `UMN_002, UMN_003, UMN_006, UMN_009, UMN_012,
UMN_014, UMN_015, UMN_017, UMN_018, UMN_019, UMN_020, UMN_022` (12 patients).
`cish_pt_doses.xlsx` additionally contains rows for **every enrolled subject
UMN_001–UMN_022**, including 10 who do not appear in any analysis.

Of 988 files in the working tree, **629 are excluded** and **359 are staged**.

---

## 1. Read this first — the previous `.gitignore` would have published the raw data

The `.gitignore` that was in place before this pass did not match the two most
sensitive files in the repository. Verified with `git check-ignore`:

| File | Under previous `.gitignore` |
|---|---|
| `CISH_pt_data.xlsx` | **would have been committed** |
| `cish_pt_doses.xlsx` | **would have been committed** |
| `CISHKO_Pct_result.csv` | **would have been committed** |
| `Patient_UMN_*_fit.png` (48 files) | **would have been committed** |
| `supplement_figures.tex` | **would have been committed** |
| `julia_output_*.txt` (54 files) | **would have been committed** |

Its patterns only covered `*.csv`/`*.json` containing the literal strings
`patient`, `clinical` or `UMN`. No `.xlsx` pattern existed at all, and
`CISHKO_Pct_result.csv` contains subject IDs in its *contents* while matching
none of those patterns by name.

One further trap: `Uncertainty_AllPatients.csv` appeared to be ignored on this
machine only because macOS sets `core.ignorecase=true`; the pattern was lowercase
`*patient*` and the filename has a capital `P`. On Linux or CI that file would
have been committed. This repository is now initialised with
`core.ignorecase=false` so local behaviour matches Linux.

---

## 2. Raw clinical trial data — highest sensitivity

| File | Contents | Copies |
|---|---|---|
| `cish_pt_doses.xlsx` | `patient_id`, `Cell_dose_infused`, `IL2_doses_received`, `CISH.KO.by.Western`, `CISH.KO.by.TIDE` for **UMN_001–UMN_022** | 27 (15 in `_archive/`) |
| `CISH_pt_data.xlsx` | Per-patient, per-tumour longitudinal tumour size, with visit day codes (`000_Baseline`, `028_Day_28`, `084_Week_12`, `180_Month_6`, …) | 27 (15 in `_archive/`) |
| `CISHKO_Pct_result.csv` | `Sample_ID` (UMN IDs), `Day`, `CISHKO_over_CISHWT` — 70 longitudinal measurements | 26 (15 in `_archive/`) |

These three are the primary source data. Everything else in this audit is derived from them.

## 3. Derived per-patient results

> **Update.** The model-derived parameter and outcome tables listed below are now
> **deliberately tracked**, so that Figs 2 and 4–8 reproduce from a clean clone. They
> contain fitted parameters and simulated outcomes, which S1 Text Table D publishes
> anyway — not clinical measurements. The two clinical columns that were mixed in,
> `initial_tumor_size` and `cish_dose`, have been stripped from the tracked copies;
> unmodified originals are in `_archive/original_parameter_tables/`. Specifically
> tracked: `Patient_<id>_paras.json`, `patient_tumor_params.csv`,
> `patient_tumor_params_with_ratio.csv`, `combination_therapy_results.csv`,
> `therapy_comparison_5_results.csv`. Everything else in this section stays excluded.

| Pattern | Contents | Copies |
|---|---|---|
| `Patient_UMN_*_paras.json` | Per-patient fitted parameter vectors and objective values | 132 (60 in `_archive/`) |
| `patient_tumor_params_with_ratio.csv` | Per-tumour rows keyed `patient_002_tumor_1` …, with fitted parameters and derived metrics | 24 (14 in `_archive/`) |
| `patient_tumor_params.csv` | As above, without the ratio column | 22 (13 in `_archive/`) |
| `Uncertainty_AllPatients.csv` | `patient_id`, `tumor_id`, per-parameter estimate, SE, confidence interval | 1 |
| `optimal_parameters_by_patient.csv` | Per-patient optimal parameters | 3 (2 in `_archive/`) |
| `real_patient_predictions.csv` | Per-patient model predictions | 2 (both in `_archive/`) |
| `Patient_UMN_*_fit.png` / `.pdf` | Per-patient fit figures, subject ID in filename and title | 60 (24 in `_archive/`) |
| `Aggregate_Profile_*`, per-patient profile plots | Profile-likelihood output | 70 (46 in `_archive/`) |

Note the pseudonymised IDs (`patient_002_tumor_1`) are trivially re-linkable to
`UMN_002` — the numeric part is identical. They are not de-identified.

## 4. Files with no identifier in the name — the easiest to miss

These carry patient data but match no name-based pattern. Two of them were caught
only by scanning file *contents* for per-subject row keys.

| File | Why it matters | Copies |
|---|---|---|
| `combination_therapy_results.csv` | Every row is one patient's tumour (`patient_002_tumor_1`) with the fitted parameter vector and simulated outcomes across six therapy arms. **As found, it also held that patient's initial tumour size and infused cell dose** | 1 |
| `therapy_comparison_5_results.csv` | Same structure, five therapy arms, 36 tumour rows | 1 |
| `julia_output_*.txt` | SLURM stdout: interleaves subject IDs with fitted values (`Results saved to: Patient_UMN_002_paras.json`) | 54 |
| `julia_error_*.txt` | SLURM stderr from the same jobs | 54 |
| `metrics.csv` (Fig 8 sweep) | One row per patient per dosing schedule, keyed by UMN code. **Tracked on purpose** — see below | 1 |

The lowercase `patient_002_tumor_1` row keys are why a name-based rule is not
enough: the filename says "therapy", not "patient".

**Current handling of the two therapy tables:** they are the direct inputs to
`plot_therapy_results.jl`, which draws Fig 7, so excluding them would have made that
figure unreproducible. Instead the two clinical columns (`initial_tumor_size`,
`cish_dose`) were stripped — nothing reads them — and the tables are now tracked.
What remains is fitted parameters and simulated outcomes. Unmodified originals are in
`_archive/original_parameter_tables/`.

### The Fig 8 schedule sweep and final fitting folder

`scripts/figures/figure8_schedule_sweep/` was added after the first audit. Handling:

| File | Status | Why |
|---|---|---|
| `schedule_sweep/metrics.csv` | **tracked** | Its `ratio` and `AUC_T` columns are exactly what Fig 8a/8b publish for all 96 cells, so tracking adds no disclosure and makes the figure checkable. It also carries `T0`, `T_end`, `T_min`, `t_min`, which the manuscript does **not** publish per patient — `T0` is an initial tumour size. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) if you would rather ship a reduced table. |
| `schedule_sweep/heatmap_ratio.*`, `heatmap_auc.*` | **tracked** | These are Fig 8a and Fig 8b themselves. |
| `schedule_*.jl` | **tracked** | Each hardcodes `const PATIENTS = ["UMN_002", …]`. This is the same 12-subject roster Fig 8 prints as its row labels, and the scripts need it to run. |
| `Patient_<id>_schedule.pdf/png` (24) | ignored | Per-patient trajectory figures, not in the manuscript. |
| `Patient_<id>_paras.json` (12) | **tracked** | Fitted parameters only, published as Table D in S1 Text — see the update in section 3. |
| `CISH_pt_data.xlsx`, `cish_pt_doses.xlsx`, `CISHKO_Pct_result.csv` | ignored | Raw clinical inputs, as everywhere else. |
| `*.log` | ignored | Run logs. |

The schedule-optimisation study that shipped in the same folder has been moved to
`_archive/schedule_optimization/` and is therefore excluded. That was the only
remaining set of per-patient numbers (`per_patient_best.csv`, `per_patient_results.csv`,
`cohort_patient_endpoints.csv`) belonging to an analysis the manuscript does not report,
so nothing tracked in this repository now contains unpublished patient-level results.

## 5. Context: the subject codes are not secret

The manuscript itself prints these codes. Figure 8 labels its rows `UMN_002` …
`UMN_022`, and Figure 5 carries the legend entry "Patient 22 (UMN022)". The trial
is registered as NCT04426669 and reported in Lancet Oncol 2025;26(5):559–70.

So the codes are pseudonyms that are published by design, and scrubbing them from
source comments is cosmetic rather than protective. **The substantive protection in
this repository is section 2 — keeping the raw per-patient clinical measurements,
cell doses, IL-2 doses and KO assay results out of version control.** That remains
correct and aligns with the manuscript's own data availability statement, which says
patient-level trial data cannot be shared publicly.

The two source-comment edits below were made anyway, since they cost nothing.

## 6. Source files that mention a code (edited)

These are **source files**, tracked because excluding them would break reproduction.

| File | Line | Status |
|---|---|---|
| `scripts/calibration/run_uncertainty.jl` | 22 | **Edited.** A commented-out `t_end_overrides` example listing three subjects and their treatment end times is now a `<PATIENT_ID>` placeholder. This was the only one carrying clinical detail rather than a bare code. |
| `scripts/calibration/CISHUncertainty.jl` | 453 | **Edited.** Docstring example now reads `Patient_<PATIENT_ID>_paras.json -> <PATIENT_ID>`. |
| `scripts/figures/figure4/plot_mouse_params_v2.py` | 201 | **Left as is.** The legend label `'Patient 22 (UMN022)'` is rendered into **Figure 5** of the manuscript, so the code must keep producing it for the figure to match what was submitted. |

`scripts/supplement/supplement_figures.tex` is currently **excluded** by `.gitignore`
because its captions enumerate the full subject roster. Given section 5, that
exclusion is conservative rather than necessary — the same roster appears in
Figure 8. Un-ignore it if you want it published with the code.

## 6. Consequence for reproducibility

Because the clinical inputs are excluded, a fresh clone reproduces the code but
not the clinical figures. The mouse analyses (`Palmer2015Data.xlsx`,
`CISH KO antiPD1.xlsx`, `CISH KO isotype control.xlsx`) are animal data, are
included, and do run from a clean clone.

If the journal requires the clinical data to be available, the usual routes are a
controlled-access repository (dbGaP, EGA) with an accession number cited in the
data-availability statement, or a de-identified derived dataset — for example
`patient_tumor_params_with_ratio.csv` with the ID column replaced by a random
relabelling, which would be enough to reproduce Figures 5 and 6.

## 7. How to re-run this audit

Scan contents, not just filenames — the therapy CSVs in section 4 pass any
name-based check:

```bash
# any staged file carrying a subject identifier or per-subject row key
git diff --cached --name-only -z | while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  grep -qIE '[Pp]atient[_ -]?[0-9]{2,3}|UMN[_ -]?[0-9]{3}|UMN[0-9]{3}' "$f" \
    2>/dev/null && echo "$f"
done
```

At the time of writing this returns exactly five files: `.gitignore` and
`PATIENT_DATA_AUDIT.md` (which describe the identifiers rather than containing
patient data), plus the three source files listed in section 5.
