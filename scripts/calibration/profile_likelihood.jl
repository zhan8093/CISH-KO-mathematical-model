# profile_likelihood.jl
# Profile likelihood analysis for CISH model patient parameters
# For each parameter in ["dL","kL","kN","pL","delta","f_dose","kappa"],
# fix it at 7 values (0.7x, 0.8x, ..., 1.3x of fitted value),
# re-optimize the remaining parameters, and plot the profile.

# Use non-interactive GR backend (required for headless cluster nodes)
ENV["GKSwstype"] = "100"

using CSV, DataFrames, Random, LinearAlgebra, Optim, Plots, JSON3, XLSX
using StaticArrays
using Base.Threads
using Printf
using Statistics

# Include the optimized model functions
include("cish_model_fun_optimized.jl")
include("cish_obj_fun_tumor_optimized.jl")
include("cish_obj_fun_patient_optimized.jl")

# ──────────────────────────────────────────────────────────────
# Structs / helpers (same as main_optimized.jl)
# ──────────────────────────────────────────────────────────────

# CISHParams struct is defined in cish_model_fun_optimized.jl (included above).
# We need the Dict constructor, params_to_dict, update_params, etc.

# Constructor from dictionary
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
# Profile likelihood optimization for a single fixed-parameter value
# ──────────────────────────────────────────────────────────────

"""
Optimize remaining parameters while fixing one parameter.

Arguments:
- fixed_param_name: name of the parameter to fix
- fixed_param_value: value to fix it at
- other_param_names: names of the remaining parameters to optimize
- other_param_fitted: fitted values of the remaining parameters (used as starting point)
- params_current: base CISHParams struct
- u0_static: initial conditions
- tumors_size, cish_ratio: data
- vars_fit_name_tumor, paras_tumor_lb, paras_tumor_ub: tumor-level fitting info
- vars_all_lb, vars_all_ub: full parameter bounds dictionaries
- nRestarts: number of random restarts for optimization
"""
function profile_optimize(fixed_param_name::String,
                          fixed_param_value::Float64,
                          other_param_names::Vector{String},
                          other_param_fitted::Vector{Float64},
                          params_current::CISHParams,
                          u0_static::SVector{4,Float64},
                          tumors_size::Matrix{Float64},
                          cish_ratio::Matrix{Float64},
                          vars_fit_name_tumor::Vector{String},
                          paras_tumor_lb::Vector{Float64},
                          paras_tumor_ub::Vector{Float64},
                          vars_all_lb::Dict{Symbol,Float64},
                          vars_all_ub::Dict{Symbol,Float64},
                          vars_all::Dict{Symbol,Float64})
    
    # First, fix the parameter in params_current
    params_fixed = update_params(params_current, [fixed_param_name], [fixed_param_value])
    
    # Build bounds for the remaining parameters
    other_lb = Float64[vars_all_lb[Symbol(n)] for n in other_param_names]
    other_ub = Float64[vars_all_ub[Symbol(n)] for n in other_param_names]
    
    # Compute orders from INIT values (consistent with main_optimized.jl)
    # Using fitted values here would distort the base-space landscape
    other_param_init = Float64[vars_all[Symbol(n)] for n in other_param_names]
    orders = log10.(other_param_init)
    
    # Base-space bounds
    base_lb = other_lb .^ (1 ./ orders)
    base_ub = other_ub .^ (1 ./ orders)
    base_lb_r = min.(base_lb, base_ub)
    base_ub_r = max.(base_lb, base_ub)
    
    # Single starting point from previously fitted values (base space)
    base_init = other_param_fitted .^ (1 ./ orders)
    base_init = clamp.(base_init, base_lb_r, base_ub_r)
    
    if !all(isfinite.(base_init))
        return 1e12, copy(other_param_fitted)
    end
    
    obj_fun = bp -> begin
        if !all(isfinite.(bp))
            return 1e12
        end
        cish_obj_fun_patient_robust(
            bp, orders, other_param_names, params_fixed,
            u0_static, tumors_size, cish_ratio,
            vars_fit_name_tumor, paras_tumor_lb, paras_tumor_ub
        )
    end
    
    try
        result = optimize(
            obj_fun,
            base_lb_r, base_ub_r, base_init,
            Fminbox(LBFGS()),
            autodiff = :finite,
            Optim.Options(
                f_reltol = 1e-6,
                outer_iterations = 8,
                iterations = 100,
                show_trace = false
            )
        )
        
        fval = Optim.minimum(result)
        params_opt = Optim.minimizer(result) .^ orders
        
        return fval, params_opt
    catch e
        return 1e12, copy(other_param_fitted)
    end
end

# ──────────────────────────────────────────────────────────────
# Main profile likelihood analysis
# ──────────────────────────────────────────────────────────────

function run_profile_likelihood(; patient_indices=nothing, verbose=true)
    # ── Load data (same as main_optimized.jl) ──
    tumor_data = XLSX.readtable("CISH_pt_data.xlsx", "Sheet1")
    tumor_table = DataFrame(tumor_data)
    cish_ratio_table = CSV.read("CISHKO_Pct_result.csv", DataFrame)
    dose_data_raw = XLSX.readtable("cish_pt_doses.xlsx", "Sheet1")
    dose_data = DataFrame(dose_data_raw)

    patient_nums_tumor = string.(tumor_table[:, 1])
    tumors_nums = string.(tumor_table[:, 4])
    days_nums = tumor_table[:, 3]
    tumor_size_nums = tumor_table[:, 5]

    patient_nums_cish = string.(cish_ratio_table[:, 1])
    days_nums_cish = cish_ratio_table[:, 2]
    cish_nums_ratio = cish_ratio_table[:, 3]

    patients_num_dose = string.(dose_data[:, 1])
    cell_dose = dose_data[:, 2]

    id_patient = unique(patient_nums_tumor)

    # ── Parameter setup ──
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

    vars_all = Dict(Symbol(n) => v for (n, v) in zip(vars_names, init_vals))
    vars_all_ub = Dict(Symbol(n) => v for (n, v) in zip(vars_names, ub_vals))
    vars_all_lb = Dict(Symbol(n) => v for (n, v) in zip(vars_names, lb_vals))

    N0_default = 0.1

    # Multipliers for profile likelihood
    multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    #multipliers = [0.9, 1.0, 1.1]
    idx_1x = findfirst(multipliers .== 1.0)  # index of the 1.0 multiplier

    # Default to all patients if not specified
    if patient_indices === nothing
        patient_indices = 1:length(id_patient)
    end

    # Create top-level output folder
    output_root = "profile_results"
    mkpath(output_root)

    # Storage for aggregated normalized profiles across patients.
    # Keys = parameter name, Values = vector of normalized objective vectors (one per patient).
    aggregated_profiles = Dict{String, Vector{Vector{Float64}}}()

    # ── Loop over patients ──
    for i in patient_indices
        if verbose
            println("\n" * "="^70)
            println("Profile Likelihood for Patient $i: $(id_patient[i])")
            println("="^70)
        end

        # Load fitted parameters from JSON
        json_path = "Patient_$(id_patient[i])_paras.json"
        if !isfile(json_path)
            @warn "File $json_path not found, skipping patient $i"
            continue
        end
        json_str = read(json_path, String)
        fitted_data = JSON3.read(json_str)

        paras_fit_patient     = Float64.(fitted_data["paras_fit_patient"])
        best_obj_original     = Float64(fitted_data["best_objective"])
        vars_fit_name_patient = String.(fitted_data["vars_fit_name_patient"])
        vars_fit_name_tumor   = String.(fitted_data["vars_fit_name_tumor"])

        paras_tumor_lb = [vars_all_lb[Symbol(v)] for v in vars_fit_name_tumor]
        paras_tumor_ub = [vars_all_ub[Symbol(v)] for v in vars_fit_name_tumor]

        # Create patient-specific output subfolder
        patient_dir = joinpath(output_root, id_patient[i])
        mkpath(patient_dir)

        if verbose
            println("Loaded fitted parameters from: $json_path")
            println("Original best objective: $best_obj_original")
            for (name, val) in zip(vars_fit_name_patient, paras_fit_patient)
                @printf("  %-10s = %.6e\n", name, val)
            end
        end

        # ── Patient-specific setup ──
        vars_all_current = copy(vars_all)
        if i == 12;  vars_all_current[:t_end] = 180.0; end
        if i == 4;   vars_all_current[:t_end] = 45.0;  end
        if i == 5;   vars_all_current[:t_end] = 60.0;  end
        if i == 6;   vars_all_current[:t_end] = 30.0;  end
        if i == 8;   vars_all_current[:t_end] = 40.0;  end
        if i == 9;   vars_all_current[:t_end] = 30.0;  end
        if i == 10;  vars_all_current[:t_end] = 60.0;  end
        if i == 11;  vars_all_current[:t_end] = 30.0;  end

        params_current = CISHParams(vars_all_current)

        # Get patient data
        inds_tumor = findall(patient_nums_tumor .== id_patient[i])
        id_tumor = unique(tumors_nums[inds_tumor])

        inds_dose = findall(patients_num_dose .== id_patient[i])
        dose = [cell_dose[inds_dose][1]]

        # Prepare tumor data
        tumors_size = zeros(length(inds_tumor), length(id_tumor) * 2)
        for (j, tumor_id) in enumerate(id_tumor)
            inds = findall((patient_nums_tumor .== id_patient[i]) .& (tumors_nums .== tumor_id))
            tumors_size[1:length(inds), 2*j-1:2*j] .= hcat(days_nums[inds], tumor_size_nums[inds])
        end

        # Prepare CISH ratio data
        inds_cish = findall(patient_nums_cish .== id_patient[i])
        cish_days = days_nums_cish[inds_cish]
        ratio_data = cish_nums_ratio[inds_cish]
        cish_ratio = hcat(cish_days, ratio_data)
        cish_ratio = cish_ratio[cish_ratio[:, 1] .>= 0, :]

        # Initial conditions
        N0 = N0_default
        L0 = 0.0
        Dose0 = dose[1] / 1e10
        u0_static = SVector{4,Float64}(0.0, L0, N0, Dose0)

        # ── Profile likelihood for each parameter (parallelized) ──
        n_params = length(vars_fit_name_patient)
        n_mult = length(multipliers)
        total_jobs = n_params * n_mult  # 7 × 7 = 49 jobs

        # Storage: rows = multipliers, columns = parameters
        profile_objectives = zeros(n_mult, n_params)
        profile_fixed_values = zeros(n_mult, n_params)

        # Pre-compute fixed values and other-param info for each job
        # so threads don't need to recompute
        job_fixed_vals = zeros(n_params, n_mult)
        job_other_names = Vector{Vector{String}}(undef, n_params)
        job_other_fitted = Vector{Vector{Float64}}(undef, n_params)

        for p_idx in 1:n_params
            param_name = vars_fit_name_patient[p_idx]
            fitted_value = paras_fit_patient[p_idx]
            other_indices = setdiff(1:n_params, p_idx)
            job_other_names[p_idx] = vars_fit_name_patient[other_indices]
            job_other_fitted[p_idx] = paras_fit_patient[other_indices]

            for (m_idx, mult) in enumerate(multipliers)
                fixed_val = fitted_value * mult
                lb_val = vars_all_lb[Symbol(param_name)]
                ub_val = vars_all_ub[Symbol(param_name)]
                fixed_val = clamp(fixed_val, lb_val, ub_val)
                job_fixed_vals[p_idx, m_idx] = fixed_val
                profile_fixed_values[m_idx, p_idx] = fixed_val
            end
        end

        # Run all jobs in parallel using threads
        progress = Atomic{Int}(0)
        start_time = time()

        if verbose
            println("\n  Running $total_jobs profile jobs in parallel ($(nthreads()) threads)...")
        end

        @sync begin
            for p_idx in 1:n_params
                for m_idx in 1:n_mult
                    @spawn begin
                        param_name = vars_fit_name_patient[p_idx]
                        fixed_val = job_fixed_vals[p_idx, m_idx]

                        best_fval, _ = profile_optimize(
                            param_name, fixed_val,
                            job_other_names[p_idx], job_other_fitted[p_idx],
                            params_current, u0_static,
                            tumors_size, cish_ratio,
                            vars_fit_name_tumor,
                            paras_tumor_lb, paras_tumor_ub,
                            vars_all_lb, vars_all_ub,
                            vars_all
                        )

                        profile_objectives[m_idx, p_idx] = best_fval

                        done = atomic_add!(progress, 1) + 1
                        if verbose
                            elapsed = time() - start_time
                            @printf("  [%2d/%2d] %s × %.1f → obj = %.6e  (%.1fs)\n",
                                    done, total_jobs, param_name,
                                    multipliers[m_idx], best_fval, elapsed)
                            flush(stdout)
                        end
                    end
                end
            end
        end

        if verbose
            @printf("\n  All %d jobs completed in %.1f seconds.\n", total_jobs, time() - start_time)
        end

        # ── Save results to CSV ──
        results_df = DataFrame()
        results_df.multiplier = multipliers
        for p_idx in 1:n_params
            pname = vars_fit_name_patient[p_idx]
            results_df[!, "$(pname)_fixed_value"] = profile_fixed_values[:, p_idx]
            results_df[!, "$(pname)_objective"] = profile_objectives[:, p_idx]
        end
        csv_path = joinpath(patient_dir, "Profile_Likelihood_$(id_patient[i]).csv")
        CSV.write(csv_path, results_df)
        if verbose
            println("\nResults saved to: $csv_path")
        end

        # ── Normalize: x = multipliers, y = objective / objective_at_1x ──
        x_norm = multipliers  # already [0.7, 0.8, ..., 1.3]

        # ── Plot figures (one per parameter) with normalized axes ──
        for p_idx in 1:n_params
            param_name = vars_fit_name_patient[p_idx]
            fitted_value = paras_fit_patient[p_idx]

            y_raw = profile_objectives[:, p_idx]
            y_at_1x = y_raw[idx_1x]
            y_norm = y_at_1x != 0.0 ? y_raw ./ y_at_1x : y_raw

            # Accumulate normalized profile for aggregation
            if !haskey(aggregated_profiles, param_name)
                aggregated_profiles[param_name] = Vector{Float64}[]
            end
            push!(aggregated_profiles[param_name], y_norm)

            fig = plot(
                x_norm, y_norm,
                xlabel = "Multiplier of $param_name",
                ylabel = "Normalized Objective",
                title = "Profile Likelihood: $param_name\n(Patient $(id_patient[i]))",
                lw = 2,
                marker = :circle,
                markersize = 5,
                legend = false,
                size = (600, 400),
                linecolor = :blue,
                markercolor = :blue,
                grid = true,
                framestyle = :box
            )

            # Mark the 1.0 multiplier with a vertical dashed line
            vline!([1.0], linestyle = :dash, color = :red, lw = 1.5, label = "Fitted (1.0×)")

            # Mark normalized baseline
            hline!([1.0], linestyle = :dot, color = :gray, lw = 1, label = "Baseline")

            # Star at the fitted point
            scatter!([1.0], [y_norm[idx_1x]],
                     marker = :star5, markersize = 10, color = :red, label = "Fitted")

            plot!(legend = :topright)

            fig_path = joinpath(patient_dir, "Profile_$(param_name)_$(id_patient[i]).png")
            savefig(fig, fig_path)
            if verbose
                println("  Plot saved: $fig_path")
            end
        end

        # ── Combined figure with all profiles (normalized) ──
        n_rows = ceil(Int, n_params / 3)
        combined_fig = plot(layout = (n_rows, 3), size = (1400, 350 * n_rows))
        for p_idx in 1:n_params
            param_name = vars_fit_name_patient[p_idx]

            y_raw = profile_objectives[:, p_idx]
            y_at_1x = y_raw[idx_1x]
            y_norm = y_at_1x != 0.0 ? y_raw ./ y_at_1x : y_raw

            plot!(combined_fig[p_idx],
                  x_norm, y_norm,
                  xlabel = "Multiplier",
                  ylabel = "Norm. Obj.",
                  title = param_name,
                  lw = 2, marker = :circle, markersize = 4,
                  legend = false,
                  linecolor = :blue, markercolor = :blue,
                  grid = true, framestyle = :box)

            vline!(combined_fig[p_idx], [1.0],
                   linestyle = :dash, color = :red, lw = 1.5)
        end

        suptitle = "Profile Likelihood — Patient $(id_patient[i])"
        plot!(combined_fig, plot_title = suptitle)

        combined_path = joinpath(patient_dir, "Profile_Likelihood_Combined_$(id_patient[i]).png")
        savefig(combined_fig, combined_path)
        if verbose
            println("  Combined plot saved: $combined_path")
        end
    end

    # ══════════════════════════════════════════════════════════════
    # Aggregate profile likelihood across all patients
    # ══════════════════════════════════════════════════════════════
    if verbose
        println("\n" * "="^70)
        println("Generating aggregate profile likelihood figures")
        println("="^70)
    end

    aggregate_dir = joinpath(output_root, "aggregate")
    mkpath(aggregate_dir)

    all_param_names = sort(collect(keys(aggregated_profiles)))

    for param_name in all_param_names
        profiles = aggregated_profiles[param_name]  # Vector of Vector{Float64}
        n_patients_with_param = length(profiles)

        if n_patients_with_param == 0
            continue
        end

        # Sum normalized objectives across patients
        summed_profile = reduce(.+, profiles)

        # Save aggregate CSV
        agg_df = DataFrame(
            multiplier = multipliers,
            summed_normalized_objective = summed_profile
        )
        agg_csv = joinpath(aggregate_dir, "Aggregate_Profile_$(param_name).csv")
        CSV.write(agg_csv, agg_df)

        # Plot
        fig = plot(
            multipliers, summed_profile,
            xlabel = "Multiplier of $param_name",
            ylabel = "Sum of Normalized Objectives",
            title = "Aggregate Profile Likelihood: $param_name\n($n_patients_with_param patients)",
            lw = 2,
            marker = :circle,
            markersize = 5,
            legend = false,
            size = (600, 400),
            linecolor = :darkgreen,
            markercolor = :darkgreen,
            grid = true,
            framestyle = :box
        )

        vline!([1.0], linestyle = :dash, color = :red, lw = 1.5, label = "Fitted (1.0×)")
        hline!([Float64(n_patients_with_param)], linestyle = :dot, color = :gray, lw = 1,
               label = "Baseline (n=$n_patients_with_param)")
        scatter!([1.0], [summed_profile[idx_1x]],
                 marker = :star5, markersize = 10, color = :red, label = "Fitted")
        plot!(legend = :topright)

        fig_path = joinpath(aggregate_dir, "Aggregate_Profile_$(param_name).png")
        savefig(fig, fig_path)
        if verbose
            println("  Aggregate plot saved: $fig_path")
        end
    end

    # ── Combined aggregate figure ──
    n_agg = length(all_param_names)
    if n_agg > 0
        n_rows_agg = ceil(Int, n_agg / 3)
        combined_agg = plot(layout = (n_rows_agg, 3), size = (1400, 350 * n_rows_agg))
        for (idx, param_name) in enumerate(all_param_names)
            profiles = aggregated_profiles[param_name]
            summed_profile = reduce(.+, profiles)
            n_patients_with_param = length(profiles)

            plot!(combined_agg[idx],
                  multipliers, summed_profile,
                  xlabel = "Multiplier",
                  ylabel = "Sum Norm. Obj.",
                  title = param_name,
                  lw = 2, marker = :circle, markersize = 4,
                  legend = false,
                  linecolor = :darkgreen, markercolor = :darkgreen,
                  grid = true, framestyle = :box)

            vline!(combined_agg[idx], [1.0],
                   linestyle = :dash, color = :red, lw = 1.5)
        end

        plot!(combined_agg, plot_title = "Aggregate Profile Likelihood — All Parameters")
        combined_agg_path = joinpath(aggregate_dir, "Aggregate_Profile_Combined.png")
        savefig(combined_agg, combined_agg_path)
        if verbose
            println("  Combined aggregate plot saved: $combined_agg_path")
        end
    end

    if verbose
        println("\n" * "="^70)
        println("Profile Likelihood Analysis Complete!")
        println("="^70)
    end
end

# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

# Run for all patients that have JSON result files, or specify indices
# Example: run_profile_likelihood(patient_indices=[1], nRestarts=20)
run_profile_likelihood(verbose=true)

