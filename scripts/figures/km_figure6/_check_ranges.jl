using CSV, DataFrames, Statistics
df = CSV.read("../patient_tumor_params_with_ratio.csv", DataFrame)
for col in [:dL, :gamma, :kL, :pN]
    vals = df[!, col]
    println("$col: min=$(minimum(vals)), max=$(maximum(vals)), median=$(median(vals)), mean=$(round(mean(vals), sigdigits=4))")
end
