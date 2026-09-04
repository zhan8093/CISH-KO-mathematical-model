"""
schedule_metrics.jl

Re-runs the same schedule sweep as schedule_sweep.jl but extracts numerical
metrics per (patient, schedule) and writes them to schedule_sweep/metrics.csv.

Metrics:
  T0           : initial tumor size (first tumor's first observation)
  T_end        : tumor size at t = T_END_SIM (= 90 days)
  ratio        : T_end / T0  (smaller = better long-term control)
  AUC_T        : trapezoidal integral of T(t) over [0, T_END_SIM]
  T_min        : minimum tumor size reached
  t_min        : time at which T_min was reached
"""

using CSV, DataFrames, JSON3, XLSX, Printf
using DiffEqCallbacks

include("cish_model_fun_optimized.jl")

const OUT_DIR    = "schedule_sweep"
const N0_DEFAULT = 0.1
const T_END_SIM  = 180.0
const DOSE_SCALE = 1e10

const SCHEDULES = [
    ("1x@0",                  [0.0]),
    ("2x@0,14",               [0.0, 14.0]),
    ("2x@0,28",               [0.0, 28.0]),
    ("2x@0,56",               [0.0, 56.0]),
    ("3x@0,14,28",            [0.0, 14.0, 28.0]),
    ("3x@0,28,56",            [0.0, 28.0, 56.0]),
    ("4x@0,14,28,42",         [0.0, 14.0, 28.0, 42.0]),
    ("4x@0,14,28,56",         [0.0, 14.0, 28.0, 56.0]),
]

const PATIENTS = ["UMN_002","UMN_003","UMN_006","UMN_009","UMN_012","UMN_014",
                  "UMN_015","UMN_017","UMN_018","UMN_019","UMN_020","UMN_022"]

function build_params(raw, gamma::Float64, t_end::Float64)
    vars_names = [String(s) for s in raw["vars_names"]]
    init_vals  = [Float64(v) for v in raw["init_vals"]]
    d = Dict{Symbol, Float64}()
    for (n, v) in zip(vars_names, init_vals); d[Symbol(n)] = v; end
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
        return solve(prob, Tsit5(); callback=cb,
                     abstol=1e-10, reltol=1e-3, saveat=0.5,
                     maxiters=1_000_000, tstops=later)
    else
        return solve(prob, Tsit5();
                     abstol=1e-10, reltol=1e-3, saveat=0.5,
                     maxiters=1_000_000)
    end
end

function first_tumor_T0(patient_id::String, tumor_table::DataFrame)
    pids  = string.(tumor_table[:, 1])
    tids  = string.(tumor_table[:, 4])
    days  = Float64.(tumor_table[:, 3])
    sizes = Float64.(tumor_table[:, 5])
    inds  = findall(pids .== patient_id)
    isempty(inds) && return 1.0
    first_t = tids[inds[1]]
    inds2   = findall((pids .== patient_id) .& (tids .== first_t))
    ord     = sortperm(days[inds2])
    return sizes[inds2][ord[1]]
end

function patient_dose(patient_id::String, dose_table::DataFrame)
    pids = string.(dose_table[:, 1])
    inds = findall(pids .== patient_id)
    isempty(inds) && return 0.0
    return Float64(dose_table[inds[1], 2])
end

function trapz(t::AbstractVector, y::AbstractVector)
    s = 0.0
    @inbounds for i in 2:length(t)
        s += 0.5 * (y[i] + y[i-1]) * (t[i] - t[i-1])
    end
    return s
end

function metrics_for_patient(pid::String, tumor_table::DataFrame, dose_table::DataFrame)
    json_path = "Patient_$(pid)_paras.json"
    isfile(json_path) || return DataFrame()
    raw = JSON3.read(read(json_path, String))
    gamma1 = Float64(raw["best_paras_tumors"][1][1])
    T0 = first_tumor_T0(pid, tumor_table)
    total_dose = patient_dose(pid, dose_table) / DOSE_SCALE
    params = build_params(raw, gamma1, T_END_SIM)

    rows = NamedTuple[]
    for (label, dose_times) in SCHEDULES
        sol = simulate_schedule(params, T0, N0_DEFAULT, total_dose, dose_times)
        t = sol.t
        T_curve = [u[1] for u in sol.u]
        # Last point at or before T_END_SIM
        T_end = T_curve[end]
        T_min, idx_min = findmin(T_curve)
        push!(rows, (
            patient = pid,
            schedule = label,
            T0 = T0,
            T_end = T_end,
            ratio = T_end / T0,
            AUC_T = trapz(t, T_curve),
            T_min = T_min,
            t_min = t[idx_min],
        ))
    end
    return DataFrame(rows)
end

function main()
    isdir(OUT_DIR) || mkpath(OUT_DIR)
    tumor_table = DataFrame(XLSX.readtable("CISH_pt_data.xlsx", "Sheet1"))
    dose_table  = DataFrame(XLSX.readtable("cish_pt_doses.xlsx", "Sheet1"))

    all_dfs = DataFrame[]
    for pid in PATIENTS
        try
            df = metrics_for_patient(pid, tumor_table, dose_table)
            push!(all_dfs, df)
        catch e
            @warn "error for $pid: $e"
        end
    end
    df_all = vcat(all_dfs...)

    out_csv = joinpath(OUT_DIR, "metrics.csv")
    CSV.write(out_csv, df_all)
    println("Wrote $(out_csv)")

    # Pretty-print: per-patient ratio and AUC tables
    println("\n=== End/Start tumor ratio (lower = better) ===")
    println(@sprintf("%-9s | %s", "patient", join([s[1] for s in SCHEDULES], " | ")))
    println("-"^90)
    for pid in PATIENTS
        sub = df_all[df_all.patient .== pid, :]
        isempty(sub) && continue
        ratios = [sub[sub.schedule .== s[1], :ratio][1] for s in SCHEDULES]
        # Mark winner with *
        winner = argmin(ratios)
        cells = [@sprintf("%s%6.3f", i == winner ? "*" : " ", ratios[i]) for i in 1:length(ratios)]
        println(@sprintf("%-9s | %s", pid, join(cells, " | ")))
    end

    println("\n=== Tumor AUC over [0,$(T_END_SIM)] (lower = better) ===")
    println(@sprintf("%-9s | %s", "patient", join([s[1] for s in SCHEDULES], " | ")))
    println("-"^90)
    for pid in PATIENTS
        sub = df_all[df_all.patient .== pid, :]
        isempty(sub) && continue
        aucs = [sub[sub.schedule .== s[1], :AUC_T][1] for s in SCHEDULES]
        winner = argmin(aucs)
        cells = [@sprintf("%s%9.1f", i == winner ? "*" : " ", aucs[i]) for i in 1:length(aucs)]
        println(@sprintf("%-9s | %s", pid, join(cells, " | ")))
    end

    # Aggregate: how often is each schedule the winner?
    println("\n=== Schedule wins (across 12 patients) ===")
    println("schedule        | wins by ratio | wins by AUC")
    println("-"^60)
    for (i, (label, _)) in enumerate(SCHEDULES)
        wins_ratio = 0; wins_auc = 0
        for pid in PATIENTS
            sub = df_all[df_all.patient .== pid, :]
            isempty(sub) && continue
            ratios = [sub[sub.schedule .== s[1], :ratio][1] for s in SCHEDULES]
            aucs   = [sub[sub.schedule .== s[1], :AUC_T][1] for s in SCHEDULES]
            argmin(ratios) == i && (wins_ratio += 1)
            argmin(aucs)   == i && (wins_auc   += 1)
        end
        println(@sprintf("%-15s |%14d |%12d", label, wins_ratio, wins_auc))
    end
end

main()
