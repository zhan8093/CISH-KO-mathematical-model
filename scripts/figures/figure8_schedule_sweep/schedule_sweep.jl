"""
schedule_sweep.jl

For each of the 12 patients, simulate the CISHKO ODE under several dosing
schedules that share the same total dose but differ in the number of
fractions and the spacing between them. Save one figure per patient
into the `schedule_sweep/` subfolder.

The model + parameters come straight from the fitted JSON for that
patient. Each patient is simulated with its first tumor's T0 (from
CISH_pt_data.xlsx) and the fitted gamma for that tumor.
"""

using CSV, DataFrames, JSON3, XLSX, Plots, Printf
using DiffEqCallbacks  # PresetTimeCallback (not reexported by DifferentialEquations on recent versions)

include("cish_model_fun_optimized.jl")  # brings in DifferentialEquations, StaticArrays, CISHParams, ode_rhs_optimized
include("schedule_plot_style.jl")
apply_cish_plot_style!(size = (900, 620))

const OUT_DIR    = "schedule_sweep"
const N0_DEFAULT = 0.1
const T_END_SIM  = 180.0
const DOSE_SCALE = 1e10  # matches main_optimized.jl: Dose0 = cell_dose / 1e10

# (label, dose times in days). Total dose is split equally across the times.
# Extended to a 56-day fractionation window, scaled from the original 28-day set
# by ~2x so the comparison is structurally analogous.
const SCHEDULES = [
    ("1x @ 0",                    [0.0]),
    ("2x @ 0, 14",                [0.0, 14.0]),
    ("2x @ 0, 28",                [0.0, 28.0]),
    ("2x @ 0, 56",                [0.0, 56.0]),
    ("3x @ 0, 14, 28",            [0.0, 14.0, 28.0]),
    ("3x @ 0, 28, 56",            [0.0, 28.0, 56.0]),
    ("4x @ 0, 14, 28, 42",        [0.0, 14.0, 28.0, 42.0]),
    ("4x @ 0, 14, 28, 56",        [0.0, 14.0, 28.0, 56.0]),
]

const PATIENTS = ["UMN_002","UMN_003","UMN_006","UMN_009","UMN_012","UMN_014",
                  "UMN_015","UMN_017","UMN_018","UMN_019","UMN_020","UMN_022"]

function build_params(raw, gamma::Float64, t_end::Float64)
    vars_names = [String(s) for s in raw["vars_names"]]
    init_vals  = [Float64(v) for v in raw["init_vals"]]
    d = Dict{Symbol, Float64}()
    for (n, v) in zip(vars_names, init_vals)
        d[Symbol(n)] = v
    end
    for (n, v) in zip(raw["vars_fit_name_patient"], raw["paras_fit_patient"])
        d[Symbol(String(n))] = Float64(v)
    end
    d[:gamma] = gamma
    d[:t_end] = t_end
    return CISHParams(
        d[:t_infusion], d[:t_end], d[:gamma], d[:lambda],
        d[:s], d[:K], d[:kL], d[:kN], d[:muL], d[:muN],
        d[:dL], d[:dN], d[:pL], d[:pN], d[:g], d[:delta],
        d[:h], d[:l], d[:n], d[:f], d[:f_dose], d[:kappa]
    )
end

function simulate_schedule(params::CISHParams, T0::Float64, N0::Float64,
                           total_dose::Float64, dose_times::Vector{Float64})
    n_frac    = length(dose_times)
    dose_per  = total_dose / n_frac
    init_dose = dose_times[1] == 0.0 ? dose_per : 0.0
    later     = dose_times[1] == 0.0 ? Vector{Float64}(dose_times[2:end]) : copy(dose_times)

    y0   = SVector{4, Float64}(max(T0, 1e-10), 0.0, max(N0, 1e-10), init_dose)
    prob = ODEProblem{false}(ode_rhs_optimized, y0, (0.0, params.t_end), params)

    if !isempty(later)
        affect! = (integrator) -> begin
            u = integrator.u
            integrator.u = SVector{4, Float64}(u[1], u[2], u[3], u[4] + dose_per)
            return nothing
        end
        cb = PresetTimeCallback(later, affect!)
        return solve(prob, Tsit5();
                     callback = cb,
                     abstol   = 1e-10, reltol = 1e-3,
                     saveat   = 0.5, maxiters = 1_000_000,
                     tstops   = later)
    else
        return solve(prob, Tsit5();
                     abstol = 1e-10, reltol = 1e-3,
                     saveat = 0.5, maxiters = 1_000_000)
    end
end

function first_tumor_obs(patient_id::String, tumor_table::DataFrame)
    pids  = string.(tumor_table[:, 1])
    tids  = string.(tumor_table[:, 4])
    days  = Float64.(tumor_table[:, 3])
    sizes = Float64.(tumor_table[:, 5])
    inds  = findall(pids .== patient_id)
    isempty(inds) && return Float64[], Float64[]
    first_t = tids[inds[1]]
    inds2   = findall((pids .== patient_id) .& (tids .== first_t))
    ord     = sortperm(days[inds2])
    return days[inds2][ord], sizes[inds2][ord]
end

function patient_dose(patient_id::String, dose_table::DataFrame)
    pids = string.(dose_table[:, 1])
    inds = findall(pids .== patient_id)
    isempty(inds) && return 0.0
    return Float64(dose_table[inds[1], 2])
end

function plot_patient(patient_id::String, tumor_table::DataFrame, dose_table::DataFrame)
    json_path = "Patient_$(patient_id)_paras.json"
    if !isfile(json_path)
        @warn "no JSON for $patient_id, skipping"
        return
    end
    raw = JSON3.read(read(json_path, String))

    tumor_gammas = [Float64(t[1]) for t in raw["best_paras_tumors"]]
    gamma1 = tumor_gammas[1]

    obs_days, obs_sizes = first_tumor_obs(patient_id, tumor_table)
    T0 = isempty(obs_sizes) ? 1.0 : obs_sizes[1]

    cell_dose  = patient_dose(patient_id, dose_table)
    total_dose = cell_dose / DOSE_SCALE

    params = build_params(raw, gamma1, T_END_SIM)

    plt = plot(
        title = @sprintf("%s schedule sweep", patient_id),
        xlabel = "Days",
        ylabel = "Tumor size",
        legend = :topright,
        legendfontsize = 8,
        left_margin = 8Plots.mm,
        bottom_margin = 7Plots.mm,
        top_margin = 6Plots.mm,
    )

    ymax = isempty(obs_sizes) ? 0.0 : maximum(obs_sizes)
    for (i, (label, dose_times)) in enumerate(SCHEDULES)
        sol     = simulate_schedule(params, T0, N0_DEFAULT, total_dose, dose_times)
        t_grid  = sol.t
        T_curve = [u[1] for u in sol.u]
        ymax = max(ymax, maximum(T_curve))
        plot!(plt, t_grid, T_curve,
              label = label,
              lw = 2.4,
              color = CISH_FIGURE_COLORS[((i - 1) % length(CISH_FIGURE_COLORS)) + 1])
    end

    if !isempty(obs_days)
        scatter!(plt, obs_days, obs_sizes, label = "Data (tumor 1)",
                 markersize = 4.5,
                 markercolor = :black,
                 markerstrokecolor = :white,
                 markerstrokewidth = 0.7)
    end

    annotate!(plt, 0.03 * T_END_SIM, 0.95 * ymax,
              text(@sprintf("T0 = %.3g\nDose = %.2e cells\nγ = %.3g", T0, cell_dose, gamma1),
                   8, :left, :top))

    isdir(OUT_DIR) || mkpath(OUT_DIR)
    out_path = joinpath(OUT_DIR, "Patient_$(patient_id)_schedule.png")
    save_png_pdf(plt, out_path)
    println("  saved: $out_path")
    return nothing
end

function main()
    isdir(OUT_DIR) || mkpath(OUT_DIR)

    println("Loading data files...")
    tumor_table = DataFrame(XLSX.readtable("CISH_pt_data.xlsx", "Sheet1"))
    dose_table  = DataFrame(XLSX.readtable("cish_pt_doses.xlsx", "Sheet1"))

    println("Running schedule sweep for $(length(PATIENTS)) patients...")
    for pid in PATIENTS
        println("Patient $pid:")
        try
            plot_patient(pid, tumor_table, dose_table)
        catch e
            @warn "error for $pid: $e"
        end
    end
    println("\nDone. Figures in ./$(OUT_DIR)/")
end

main()
