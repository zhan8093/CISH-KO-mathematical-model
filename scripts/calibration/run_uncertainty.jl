# run_uncertainty_all.jl
# Run:
#   julia run_uncertainty_all.jl

include("cish_model_fun_optimized.jl")   # defines CISHParams + cish_model_fun_optimized
include("CISHUncertainty.jl")
using .CISHUncertainty
using DataFrames

# --------- settings ----------
dir = "."                       # folder containing Patient_*_paras.json
pattern = "Patient_*_paras.json"

mode   = :patient               # :patient, :tumor, or :both
method = :forward               # :forward (cheaper) or :central (more accurate)
relstep = 1e-6
weight_cish = 0.2

csv_out = "Uncertainty_AllPatients.csv"

# If some patients had special t_end values during fitting, set them here:
# Example:
# t_end_overrides = Dict("<PATIENT_ID>"=>45.0, "<PATIENT_ID>"=>60.0)
t_end_overrides = Dict{String,Float64}()  # leave empty if not needed / all default
# ----------------------------

df = uncertainty_all_patients_to_csv(
    dir=dir,
    pattern=pattern,
    mode=mode,
    method=method,
    relstep=relstep,
    weight_cish=weight_cish,
    t_end_overrides=t_end_overrides,
    csv_out=csv_out
)

println("Saved: $csv_out")
println("Rows: ", nrow(df))
println(first(df, 10))
