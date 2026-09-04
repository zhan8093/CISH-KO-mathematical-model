# test_objective.jl
# Two diagnostic tests for a single patient:
#   1. Evaluate the cost function at the saved optimal parameters (no optimization)
#      and compare with the saved best_objective in the JSON file.
#   2. Run the optimizer starting from the saved optimal parameters and check
#      whether it converges to the same objective value.

# Ensure Julia's working directory matches the script's location
cd(@__DIR__)

using CSV, DataFrames, Random, LinearAlgebra, Optim, JSON3, XLSX
using StaticArrays
using Printf
using Statistics

include("cish_model_fun_optimized.jl")
include("cish_obj_fun_tumor_optimized.jl")
include("cish_obj_fun_patient_optimized.jl")

# ── Helper functions (same as main_optimized.jl) ──

function CISHParams(vars_dict::Dict{Symbol, T}) where T<:Real
    CISHParams(
        vars_dict[:t_infusion], vars_dict[:t_end],
        vars_dict[:gamma], vars_dict[:lambda], vars_dict[:s], vars_dict[:K],
        vars_dict[:kL], vars_dict[:kN], vars_dict[:muL], vars_dict[:muN],
        vars_dict[:dL], vars_dict[:dN], vars_dict[:pL], vars_dict[:pN],
        vars_dict[:g], vars_dict[:delta], vars_dict[:h], vars_dict[:l],
        vars_dict[:n], vars_dict[:f], vars_dict[:f_dose], vars_dict[:kappa]
    )
end

function params_to_dict(p::CISHParams)
    Dict{Symbol, Float64}(
        :t_infusion => p.t_infusion, :t_end => p.t_end,
        :gamma => p.gamma, :lambda => p.lambda, :s => p.s, :K => p.K,
        :kL => p.kL, :kN => p.kN, :muL => p.muL, :muN => p.muN,
        :dL => p.dL, :dN => p.dN, :pL => p.pL, :pN => p.pN,
        :g => p.g, :delta => p.delta, :h => p.h, :l => p.l,
        :n => p.n, :f => p.f, :f_dose => p.f_dose, :kappa => p.kappa
    )
end

function update_params(base::CISHParams, param_names::Vector{String}, param_values::Vector{T}) where T
    d = params_to_dict(base)
    for (name, val) in zip(param_names, param_values)
        d[Symbol(name)] = val
    end
    return CISHParams(d)
end

function get_params_by_name(params::CISHParams, param_names::Vector{String})
    return [getfield(params, Symbol(name)) for name in param_names]
end

# ──────────────────────────────────────────────────────────────
# Main test function
# ──────────────────────────────────────────────────────────────

function run_test(; patient_index::Int = 1)
    # ── Load data ──
    tumor_data  = XLSX.readtable("CISH_pt_data.xlsx", "Sheet1")
    tumor_table = DataFrame(tumor_data)
    cish_ratio_table = CSV.read("CISHKO_Pct_result.csv", DataFrame)
    dose_data   = DataFrame(XLSX.readtable("cish_pt_doses.xlsx", "Sheet1"))

    patient_nums_tumor = string.(tumor_table[:, 1])
    tumors_nums        = string.(tumor_table[:, 4])
    days_nums          = tumor_table[:, 3]
    tumor_size_nums    = tumor_table[:, 5]

    patient_nums_cish = string.(cish_ratio_table[:, 1])
    days_nums_cish    = cish_ratio_table[:, 2]
    cish_nums_ratio   = cish_ratio_table[:, 3]

    patients_num_dose = string.(dose_data[:, 1])
    cell_dose         = dose_data[:, 2]

    id_patient = unique(patient_nums_tumor)

    # ── Parameter setup (identical to main_optimized.jl) ──
    vars_names = ["t_infusion","t_end",
                  "gamma","lambda","s","K",
                  "kL","kN","muL","muN",
                  "dL","dN","pL","pN","g",
                  "delta","h","l","n","f","f_dose","kappa"]

    init_vals = [0.001, 85.0,
                 0.01, 0.58, 4.24, 1.00,
                 0.03, 0.0005, 4e-4, 4e-4,
                 0.05, 0.030, 0.05, 0.05, 1.0e3,
                 0.1, 1e3, 578, 200, 1.0, 0.2, 0.25]

    ub_vals = [0.001, 85.0,
               0.1, 0.90, 5.00, 10.0,
               1.0, 0.001, 0.9, 2e-2,
               0.500, 0.500, 1, 1, 2.0e7,
               1e-1, 1e3, 2.5e4, 2.5e4, 1.0, 1.0, 0.5]

    lb_vals = [0.001, 85.0,
               0.001, 0.46, 0.10, 1.00,
               0.001, 0.0001, 1.0e-9, 1.0e-9,
               0.001, 0.001, 0.001, 0.001, 0.9e3,
               1e-9, 1e3, 1.0, 1.0, 1.0, 0.01, 0.05]

    vars_all    = Dict(Symbol(n) => v for (n, v) in zip(vars_names, init_vals))
    vars_all_ub = Dict(Symbol(n) => v for (n, v) in zip(vars_names, ub_vals))
    vars_all_lb = Dict(Symbol(n) => v for (n, v) in zip(vars_names, lb_vals))

    N0_default = 0.1

    i = patient_index

    # ── Load saved JSON ──
    json_path = "Patient_$(id_patient[i])_paras.json"
    if !isfile(json_path)
        error("File $json_path not found!")
    end
    fitted_data = JSON3.read(read(json_path, String))

    paras_fit_patient      = Float64.(fitted_data["paras_fit_patient"])
    best_obj_saved         = Float64(fitted_data["best_objective"])
    vars_fit_name_patient  = String.(fitted_data["vars_fit_name_patient"])
    vars_fit_name_tumor    = String.(fitted_data["vars_fit_name_tumor"])

    paras_tumor_lb = [vars_all_lb[Symbol(v)] for v in vars_fit_name_tumor]
    paras_tumor_ub = [vars_all_ub[Symbol(v)] for v in vars_fit_name_tumor]

    println("="^70)
    println("Testing Patient $i: $(id_patient[i])")
    println("="^70)
    println("\nSaved JSON file: $json_path")
    @printf("Saved best_objective: %.12e\n", best_obj_saved)
    println("\nSaved parameters:")
    for (name, val) in zip(vars_fit_name_patient, paras_fit_patient)
        @printf("  %-10s = %.10e\n", name, val)
    end

    # ── Patient-specific setup ──
    vars_all_current = copy(vars_all)
    if i == 12; vars_all_current[:t_end] = 180.0; end
    if i == 4;  vars_all_current[:t_end] = 45.0;  end
    if i == 5;  vars_all_current[:t_end] = 60.0;  end
    if i == 6;  vars_all_current[:t_end] = 30.0;  end
    if i == 8;  vars_all_current[:t_end] = 40.0;  end
    if i == 9;  vars_all_current[:t_end] = 30.0;  end
    if i == 10; vars_all_current[:t_end] = 60.0;  end
    if i == 11; vars_all_current[:t_end] = 30.0;  end

    params_current = CISHParams(vars_all_current)

    # Get patient data
    inds_tumor = findall(patient_nums_tumor .== id_patient[i])
    id_tumor   = unique(tumors_nums[inds_tumor])

    inds_dose = findall(patients_num_dose .== id_patient[i])
    dose = [cell_dose[inds_dose][1]]

    # Prepare tumor data matrix
    tumors_size = zeros(length(inds_tumor), length(id_tumor) * 2)
    for (j, tumor_id) in enumerate(id_tumor)
        inds = findall((patient_nums_tumor .== id_patient[i]) .& (tumors_nums .== tumor_id))
        tumors_size[1:length(inds), 2*j-1:2*j] .= hcat(days_nums[inds], tumor_size_nums[inds])
    end

    # Prepare CISH ratio data
    inds_cish = findall(patient_nums_cish .== id_patient[i])
    cish_days  = days_nums_cish[inds_cish]
    ratio_data = cish_nums_ratio[inds_cish]
    cish_ratio = hcat(cish_days, ratio_data)
    cish_ratio = cish_ratio[cish_ratio[:, 1] .>= 0, :]

    # Initial conditions (same as main_optimized.jl)
    N0    = N0_default
    L0    = 0.0
    Dose0 = dose[1] / 1e10
    u0_static = SVector{4, Float64}(0.0, L0, N0, Dose0)

    # ── Compute orders from INIT values (same as main_optimized.jl) ──
    paras_patient_init = [vars_all[Symbol(v)] for v in vars_fit_name_patient]
    orders = log10.(paras_patient_init)

    # ── Base-space bounds (same as main_optimized.jl) ──
    paras_patient_lb = [vars_all_lb[Symbol(v)] for v in vars_fit_name_patient]
    paras_patient_ub = [vars_all_ub[Symbol(v)] for v in vars_fit_name_patient]
    base_lb = paras_patient_lb .^ (1 ./ orders)
    base_ub = paras_patient_ub .^ (1 ./ orders)
    base_lb_r = min.(base_lb, base_ub)
    base_ub_r = max.(base_lb, base_ub)

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Direct evaluation (NO optimization)
    # ══════════════════════════════════════════════════════════════
    println("\n" * "="^70)
    println("TEST 1: Direct cost function evaluation (no optimization)")
    println("="^70)

    # Convert fitted params → base space
    base_init = paras_fit_patient .^ (1 ./ orders)

    # Call the SAME objective function used during optimization
    obj_direct = cish_obj_fun_patient_optimized(
        base_init, orders, vars_fit_name_patient,
        params_current, u0_static,
        tumors_size, cish_ratio,
        vars_fit_name_tumor, paras_tumor_lb, paras_tumor_ub
    )

    @printf("\n  Saved best_objective:      %.12e\n", best_obj_saved)
    @printf("  Direct evaluation result:  %.12e\n", obj_direct)
    @printf("  Absolute difference:       %.12e\n", abs(obj_direct - best_obj_saved))
    if best_obj_saved != 0
        @printf("  Relative difference:       %.6e\n", abs(obj_direct - best_obj_saved) / abs(best_obj_saved))
    end

    if abs(obj_direct - best_obj_saved) < 1e-6
        println("  ✓ PASS — values match (within 1e-6)")
    else
        println("  ✗ FAIL — values differ!")
        println("  → This means the saved best_objective in JSON is NOT the patient-level")
        println("    objective; it was likely overwritten by the tumor-level fitting loop.")
    end

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Optimization starting from optimal parameters
    # ══════════════════════════════════════════════════════════════
    println("\n" * "="^70)
    println("TEST 2: Optimization from optimal initial point")
    println("="^70)

    base_init_clamped = clamp.(base_init, base_lb_r, base_ub_r)

    obj_fun = bp -> begin
        if !all(isfinite.(bp))
            return 1e12
        end
        cish_obj_fun_patient_robust(
            bp, orders, vars_fit_name_patient, params_current,
            u0_static, tumors_size, cish_ratio,
            vars_fit_name_tumor, paras_tumor_lb, paras_tumor_ub
        )
    end

    result = optimize(
        obj_fun,
        base_lb_r, base_ub_r, base_init_clamped,
        Fminbox(LBFGS()),
        autodiff = :finite,
        Optim.Options(
            f_reltol  = 1e-6,
            outer_iterations = 8,
            iterations = 100,
            show_trace = true
        )
    )

    obj_optimized     = Optim.minimum(result)
    params_optimized  = Optim.minimizer(result) .^ orders

    @printf("\n  Saved best_objective:            %.12e\n", best_obj_saved)
    @printf("  Direct evaluation (test 1):      %.12e\n", obj_direct)
    @printf("  After re-optimization (test 2):  %.12e\n", obj_optimized)
    @printf("  |test2 - test1|:                 %.12e\n", abs(obj_optimized - obj_direct))
    if obj_direct != 0
        @printf("  Relative |test2 - test1|:        %.6e\n", abs(obj_optimized - obj_direct) / abs(obj_direct))
    end

    println("\n  Parameter comparison (fitted → re-optimized):")
    @printf("  %-10s  %16s  %16s  %12s\n", "Name", "Fitted", "Re-optimized", "Rel.Change")
    for (name, v_fit, v_opt) in zip(vars_fit_name_patient, paras_fit_patient, params_optimized)
        rel_change = v_fit != 0 ? abs(v_opt - v_fit) / abs(v_fit) : abs(v_opt - v_fit)
        @printf("  %-10s  %16.8e  %16.8e  %12.4e\n", name, v_fit, v_opt, rel_change)
    end

    if abs(obj_optimized - obj_direct) < 1e-6
        println("\n  ✓ PASS — optimizer converges to same objective (within 1e-6)")
    elseif obj_optimized < obj_direct
        @printf("\n  ⚠ Optimizer found BETTER objective by %.6e\n", obj_direct - obj_optimized)
        println("    → The saved parameters may not be a true local minimum.")
    else
        @printf("\n  ⚠ Optimizer found WORSE objective by %.6e\n", obj_optimized - obj_direct)
        println("    → Optimizer drifted away. Possible cause: finite-difference noise.")
    end

    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    @printf("  Saved JSON best_objective:       %.12e\n", best_obj_saved)
    @printf("  Test 1 (direct evaluation):      %.12e\n", obj_direct)
    @printf("  Test 2 (re-optimization):        %.12e\n", obj_optimized)
    println("="^70)

    return (; best_obj_saved, obj_direct, obj_optimized,
              paras_fit_patient, params_optimized)
end

# ── Entry point ──
run_test(patient_index = 7)
