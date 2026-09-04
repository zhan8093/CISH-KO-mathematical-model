using CSV, DataFrames, Statistics
df = CSV.read("../patient_tumor_params_with_ratio.csv", DataFrame)
params = [:dL, :kL, :pL, :muL, :pN, :f_dose, :kappa, :gamma]
println("Parameter statistics from fitted patient data:")
println("-"^70)
for col in params
    vals = df[!, col]
    println("$col: min=$(round(minimum(vals), sigdigits=4)), max=$(round(maximum(vals), sigdigits=4)), median=$(round(median(vals), sigdigits=4)), mean=$(round(mean(vals), sigdigits=4))")
end
println("\nMedian patient full parameter set (for use as init_vals):")
for col in params
    println("  $col => $(round(median(df[!, col]), sigdigits=4))")
end
