# KM_curves.jl - Kaplan-Meier survival curve analysis for CISH model
using Random
using QuasiMonteCarlo
using Plots
using Statistics
# using DifferentialEquations # Uncomment if you are using specific solver options

# -- Style setup (consistent with Figure 4 / plot_mouse_params_v2.py) --
gr(fontfamily = "Computer Modern")
default(
    framestyle  = :axes,
    grid        = true,
    gridalpha   = 0.3,
    gridstyle   = :dash,
    guidefontsize  = 12,
    tickfontsize   = 10,
    titlefontsize  = 11,
    legendfontsize = 9,
    background_color = :white,
    foreground_color = :black,
    dpi  = 300,
    size = (520, 420),
)

# Color palette for KM curves (6 distinguishable colors)
const KM_COLORS = [
    RGB(0.180, 0.525, 0.671),  # #2E86AB steel blue
    RGB(0.910, 0.333, 0.227),  # #E8553A warm red
    RGB(0.498, 0.690, 0.412),  # #7FB069 muted green
    RGB(0.961, 0.651, 0.137),  # #F5A623 amber
    RGB(0.557, 0.267, 0.678),  # #8E44AD purple
    RGB(0.204, 0.286, 0.369),  # #344959 dark slate
]

# Include the optimized CISH model
# Ensure this path is correct relative to where you run the script
include("../cish_model_fun_optimized.jl") 

# --- PARAMETERS (Removed 'const' to allow re-running) ---
vars_names = ["t_infusion","t_end",
              "gamma","lambda","s","K",
              "kL","kN","muL","muN",
              "dL","dN","pL","pN","g",
              "delta","h","l","n","f",
              "f_dose","kappa"]

# === Baseline parameter values from median of fitted patient data ===
# Key changes from previous init_vals:
#   muL:  4e-4 -> 2.5e-8   (was 4 orders of magnitude too high, overshadowing dL)
#   dL:   0.49 -> 0.035    (was at max of patient range, now at median)
#   kL:   0.0126 -> 0.0045 (median patient value)
#   gamma: 0.0056 -> 0.022 (median patient value)
#   pL:   0.0038 -> 0.001  (median patient value)
#   pN:   0.05 -> 0.345    (median patient value)
#   f_dose: 0.1 -> 0.876   (median patient value)
#   kappa: 0.112 -> 0.092  (median patient value)
init_vals = [0.001, 120.0,
             0.022, 0.58, 4.24, 1.00, 
             0.1, 0.00025, 2.5e-8, 4e-4,
             0.035, 0.030, 0.001, 0.005, 1.0e3,
             1.7e-8, 1e3, 578, 200, 1.0, 
             0.876, 0.092]

# QMC bounds updated to reflect actual patient data ranges
# Key: muL upper bound tightened from 0.9 to 1e-4 (median patient muL ~ 2.5e-8)
#      dL bounds match patient data range [0.001, 0.5]
ub_vals = [0.001, 85.0, 
            0.1, 0.90, 5.00, 10.0,
            0.05, 0.0005, 1e-4, 2e-2,
            0.5, 0.500, 0.01, 1, 2.0e7,
            1e-1, 1e3, 2.5e4, 2.5e4, 1.0,
            1.0, 0.5]

lb_vals = [0.001, 85.0,
            0.01, 0.46, 0.10, 1.00,
            0.001, 0.0001, 1.0e-9, 1.0e-9,
            0.001, 0.001, 0.001, 0.06, 0.9e3,
            1e-9, 1e3, 1.0, 1.0, 1.0,
            0.01, 0.05]

vary_names = ["gamma","dL","kL","pL","muL","pN","f_dose","kappa"]

# Map parameter names to indices
name2idx = Dict(n => findfirst(==(n), vars_names) for n in vars_names)

# Extract bounds for varying parameters
lb = [lb_vals[name2idx[n]] for n in vary_names]
ub = [ub_vals[name2idx[n]] for n in vary_names]

# Initial conditions
u0 = [100.0, 0.0, 0.1, 1] 

# === FIX 2: Set Threshold > Initial Condition ===
# If u0=100 and Thresh=100, noise causes immediate death at t=0.
KM_THRESH = 120.0 

# Number of virtual patients
N = 1_000

# Generate QMC samples
Random.seed!(42)
sampler = SobolSample()
A, _ = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, sampler)

# === FIX 3: Ignore index 1 to avoid t=0 detection ===
@inline function first_passage_time(t::AbstractVector, Tseries::AbstractVector, thresh::Real)
    # Start checking from 2nd point to avoid initial condition trigger
    idx = findfirst(>(thresh), @view Tseries[2:end])
    
    if idx === nothing
        return (t[end], false)   # censored
    else
        return (t[idx+1], true)  # event (idx is relative to view)
    end
end

function km_curve_step(times::Vector{<:Real}, events::Vector{Bool})
    @assert length(times) == length(events)
    ev_times = sort(unique(times[events .== true]))
    S_prev = 1.0
    t_plot = Float64[0.0]
    s_plot = Float64[1.0]
    t_max = maximum(times)

    for ti in ev_times
        n_i = count(>=(ti), times)
        d_i = count(x -> x == ti, times[events .== true])
        
        push!(t_plot, ti); push!(s_plot, S_prev)
        S_new = S_prev * (1 - d_i / n_i)
        push!(t_plot, ti); push!(s_plot, S_new)
        S_prev = S_new
    end

    if t_plot[end] < t_max
        push!(t_plot, t_max)
        push!(s_plot, S_prev)
    end

    return t_plot, s_plot
end

@inline function patient_vec_from_A(i::Int)
    v = copy(init_vals)
    for (row, pname) in enumerate(vary_names)
        v[name2idx[pname]] = A[row, i]
    end
    return v
end

function vec_to_params(v::AbstractVector)
    # Using name2idx from global scope (safe in non-const script)
    CISHParams(v[name2idx["t_infusion"]],
               v[name2idx["t_end"]],
               v[name2idx["gamma"]],
               v[name2idx["lambda"]],
               v[name2idx["s"]],
               v[name2idx["K"]],
               v[name2idx["kL"]],
               v[name2idx["kN"]],
               v[name2idx["muL"]],
               v[name2idx["muN"]],
               v[name2idx["dL"]],
               v[name2idx["dN"]],
               v[name2idx["pL"]],
               v[name2idx["pN"]],
               v[name2idx["g"]],
               v[name2idx["delta"]],
               v[name2idx["h"]],
               v[name2idx["l"]],
               v[name2idx["n"]],
               v[name2idx["f"]],
               v[name2idx["f_dose"]],
               v[name2idx["kappa"]])
end

function run_patient_return_event(v::Vector{Float64}, u0_custom::Vector{Float64} = u0)
    params = vec_to_params(v)
    out = cish_model_fun_optimized(params, u0_custom)
    
    # === FIX 4: Handle DifferentialEquations.jl output correctly ===
    # .t is time, .u is state
    if hasproperty(out, :t) && hasproperty(out, :u)
        t = out.t
        # Extract Tumor size (Index 1) from solution
        Tseries = [u[1] for u in out.u]
        return first_passage_time(t, Tseries, KM_THRESH)
    else
        # Fallback if your model returns a custom struct with .x and .y
        return first_passage_time(out.x, view(out.y, 1, :), KM_THRESH)
    end
end

function km_for_level(level_label::AbstractString, level_apply!::Function)
    times  = Vector{Float64}(undef, N)
    events = Vector{Bool}(undef, N)
    
    # Using threads for speed
    Base.Threads.@threads for i in 1:N
        v = patient_vec_from_A(i)
        result = level_apply!(v)
        
        if length(result) == 4
            ti, ei = run_patient_return_event(v, result)
        else
            ti, ei = run_patient_return_event(result)
        end
        
        times[i] = ti
        events[i] = ei
    end
    
    println("For level $level_label: event rate = $(round(count(events)/N * 100, digits=2))%")
    return km_curve_step(times, events)
end

# ===== PLOTTING =====

# 1. KM: Initial Dose(0)
initial_Dose0_grid = [0.0, 1e9, 5e9, 1e10, 5e10, 1e11, 5e11, 1e12]
plt_cell = plot(xlabel="Days", ylabel="Survival probability",
                title="Initial dose",
                xlims=(0, 120), ylims=(0, 1))
for (i, Dose0) in enumerate(initial_Dose0_grid)
    step_t, step_S = km_for_level("Dose(0)=$(Dose0)", v -> begin
        u0_modified = [100.0, 0.0, 0.1, Dose0/1e10]
        return u0_modified
    end)
    plot!(plt_cell, step_t, step_S, label="Dose(0)=$(Dose0)",
          linewidth=2)
end
display(plt_cell)
savefig(plt_cell, "km_initial_Dose0.png")

# 2. KM: Anti-PD1
anti_PD1_dose_grid = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
plt_anti_pd1 = plot(xlabel="Days", ylabel="Survival probability",
                    title="Anti-PD1 dose",
                    xlims=(0, 120), ylims=(0, 1))
for (i, anti_pd1_dose) in enumerate(anti_PD1_dose_grid)
    step_t, step_S = km_for_level("anti-PD1=$(anti_pd1_dose)", v -> begin
        if anti_pd1_dose > 0.0
            v[name2idx["kL"]] = v[name2idx["kL"]] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
            v[name2idx["kN"]] = v[name2idx["kN"]] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
        end
        return v
    end)
    plot!(plt_anti_pd1, step_t, step_S, label="anti-PD1=$(anti_pd1_dose)",
          color=KM_COLORS[min(i, length(KM_COLORS))], linewidth=2)
end
display(plt_anti_pd1)
savefig(plt_anti_pd1, "km_anti_pd1_dose.png")

# 3. KM: Gamma
gamma_grid = collect(range(0.01, 0.1; length=6))
plt_gamma = plot(xlabel="Days", ylabel="Survival probability",
                 title="Tumor growth rate (γ)",
                 xlims=(0, 120), ylims=(0, 1))
for (i, γ) in enumerate(gamma_grid)
    step_t, step_S = km_for_level("γ=$(round(γ, sigdigits=3))", v -> begin
        v[name2idx["gamma"]] = γ
        return v
    end)
    plot!(plt_gamma, step_t, step_S, label="γ=$(round(γ, sigdigits=3))",
          color=KM_COLORS[i], linewidth=2)
end
display(plt_gamma)
savefig(plt_gamma, "km_gamma.png")

# 4. KM: dL (widened to [0.01, 0.5] to cover actual patient range)
dL_grid = collect(range(0.01, 0.5; length=6))
plt_dL = plot(xlabel="Days", ylabel="Survival probability",
              title="Death rate (dL)",
              xlims=(0, 120), ylims=(0, 1))
for (i, dL) in enumerate(dL_grid)
    step_t, step_S = km_for_level("dL=$(round(dL, sigdigits=3))", v -> begin
        v[name2idx["dL"]] = dL
        return v
    end)
    plot!(plt_dL, step_t, step_S, label="dL=$(round(dL, sigdigits=3))",
          color=KM_COLORS[i], linewidth=2)
end
display(plt_dL)
savefig(plt_dL, "km_dL.png")

# 5. KM: pN (endogenous proliferation) — replaces f_dose to match scatter panel (c)
pN_lb, pN_ub = lb_vals[name2idx["pN"]], ub_vals[name2idx["pN"]]
pN_grid = collect(range(0.1, 1; length=6))
plt_pN = plot(xlabel="Days", ylabel="Survival probability",
              title="Endogenous proliferation (pN)",
              xlims=(0, 120), ylims=(0, 1))
for (i, pN) in enumerate(pN_grid)
    step_t, step_S = km_for_level("pN=$(round(pN, sigdigits=3))", v -> begin
        v[name2idx["pN"]] = pN
        return v
    end)
    plot!(plt_pN, step_t, step_S, label="pN=$(round(pN, sigdigits=3))",
          color=KM_COLORS[i], linewidth=2)
end
display(plt_pN)
savefig(plt_pN, "km_pN.png")

# 6. KM: kappa (therapy efficacy parameter)
kappa_lb, kappa_ub = lb_vals[name2idx["kappa"]], ub_vals[name2idx["kappa"]]
kappa_grid = collect(range(kappa_lb, kappa_ub; length=6))
plt_kappa = plot(xlabel="Days", ylabel="Survival probability",
                 title="Dose transfer rate (κ)",
                 xlims=(0, 120), ylims=(0, 1))
for (i, kappa) in enumerate(kappa_grid)
    step_t, step_S = km_for_level("kappa=$(round(kappa, sigdigits=3))", v -> begin
        v[name2idx["kappa"]] = kappa
        return v
    end)
    plot!(plt_kappa, step_t, step_S, label="κ=$(round(kappa, sigdigits=3))",
          color=KM_COLORS[i], linewidth=2)
end
display(plt_kappa)
savefig(plt_kappa, "km_kappa.png")

# 7. KM: kL (immune cell kill rate)
kL_grid = collect(range(0.01, 0.1; length=6))
plt_kL = plot(xlabel="Days", ylabel="Survival probability",
              title="Killing rate (kL)",
              xlims=(0, 120), ylims=(0, 1))
for (i, kL) in enumerate(kL_grid)
    step_t, step_S = km_for_level("kL=$(round(kL, sigdigits=3))", v -> begin
        v[name2idx["kL"]] = kL
        return v
    end)
    plot!(plt_kL, step_t, step_S, label="kL=$(round(kL, sigdigits=3))",
          color=KM_COLORS[i], linewidth=2)
end
display(plt_kL)
savefig(plt_kL, "km_kL.png")

# ===== Combined 1×4 KM figure for publication (panels e–h) =====
combined_km = plot(plt_gamma, plt_dL, plt_pN, plt_kL,
    layout = (1, 4),
    size = (1850, 500),
    margin = 5Plots.mm,
    bottom_margin = 10Plots.mm,
    left_margin   = 5Plots.mm,
    top_margin    = 8Plots.mm,
)
savefig(combined_km, "km_combined_figure5.png")
savefig(combined_km, "km_combined_figure5.pdf")
println("Saved: km_combined_figure5.png / .pdf")