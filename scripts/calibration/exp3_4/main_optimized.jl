# main_optimized_parallel.jl
using CSV, DataFrames, Random, LinearAlgebra, Optim, Plots, JSON3, XLSX
using StaticArrays
using BenchmarkTools  # For performance testing
using Base.Threads
using Printf  # For formatted output

# Include the optimized model functions
include("cish_model_fun_optimized.jl")
include("cish_obj_fun_tumor_optimized.jl")
include("cish_obj_fun_patient_optimized.jl")

const TUMOR_COLORS = (:blue, :red, :green, :orange, :purple, :brown, :black)
# Define parameter structure for type stability
struct CISHParams{T<:Real}
    t_infusion::T
    t_end::T
    gamma::T
    lambda::T
    s::T
    K::T
    kL::T
    kN::T
    muL::T
    muN::T
    dL::T
    dN::T
    pL::T
    pN::T
    g::T
    delta::T
    h::T
    l::T
    n::T
    f::T
    f_dose::T
    kappa::T
end

# Constructor from dictionary (for compatibility with original code)
function CISHParams(vars_dict::Dict{Symbol, T}) where T<:Real
    CISHParams(
        vars_dict[:t_infusion],
        vars_dict[:t_end],
        vars_dict[:gamma],
        vars_dict[:lambda],
        vars_dict[:s],
        vars_dict[:K],
        vars_dict[:kL],
        vars_dict[:kN],
        vars_dict[:muL],
        vars_dict[:muN],
        vars_dict[:dL],
        vars_dict[:dN],
        vars_dict[:pL],
        vars_dict[:pN],
        vars_dict[:g],
        vars_dict[:delta],
        vars_dict[:h],
        vars_dict[:l],
        vars_dict[:n],
        vars_dict[:f],
        vars_dict[:f_dose],
        vars_dict[:kappa]
    )
end

# Convert struct back to dictionary for compatibility
function params_to_dict(p::CISHParams)
    Dict{Symbol, Float64}(
        :t_infusion => p.t_infusion,
        :t_end => p.t_end,
        :gamma => p.gamma,
        :lambda => p.lambda,
        :s => p.s,
        :K => p.K,
        :kL => p.kL,
        :kN => p.kN,
        :muL => p.muL,
        :muN => p.muN,
        :dL => p.dL,
        :dN => p.dN,
        :pL => p.pL,
        :pN => p.pN,
        :g => p.g,
        :delta => p.delta,
        :h => p.h,
        :l => p.l,
        :n => p.n,
        :f => p.f,
        :f_dose => p.f_dose,
        :kappa => p.kappa
    )
end

# Update specific parameters in struct
function update_params(base::CISHParams, param_names::Vector{String}, param_values::Vector{T}) where T
    # Convert to dict for easy updating
    d = params_to_dict(base)
    
    # Update specified parameters
    for (name, val) in zip(param_names, param_values)
        d[Symbol(name)] = val
    end
    
    # Return new struct
    return CISHParams(d)
end

# Extract parameters by name from struct
function get_params_by_name(params::CISHParams, param_names::Vector{String})
    return [getfield(params, Symbol(name)) for name in param_names]
end

# ──────────────────────────────────────────────────────────────
# Latin Hypercube Sampling for better coverage of parameter space
# ──────────────────────────────────────────────────────────────
function latin_hypercube_sample(n_samples::Int, n_dims::Int, lb::Vector{Float64}, ub::Vector{Float64}; rng=Random.default_rng())
    # Create a Latin Hypercube: each dimension divided into n_samples equal strata
    samples = zeros(n_samples, n_dims)
    for d in 1:n_dims
        # Random permutation of strata
        perm = randperm(rng, n_samples)
        for j in 1:n_samples
            # Random point within the assigned stratum
            lo = (perm[j] - 1) / n_samples
            hi = perm[j] / n_samples
            u = lo + rand(rng) * (hi - lo)
            samples[j, d] = lb[d] + u * (ub[d] - lb[d])
        end
    end
    return samples
end

# ──────────────────────────────────────────────────────────────
# 3-stage parallel optimization with progress monitoring
#   Stage 1: Broad multistart with LHS (LBFGS, moderate tolerance)
#   Stage 2: Local refinement of top-K candidates (LBFGS, tight tolerance)
#   Stage 3: NelderMead polish of the single best (derivative-free)
# ──────────────────────────────────────────────────────────────
function parallel_optimize_patient(params_current, u0_static, 
                                  tumors_size, cish_ratio,
                                  vars_fit_name_patient, vars_fit_name_tumor,
                                  orders, base_paras_patient_lb_rearranged, 
                                  base_paras_patient_ub_rearranged,
                                  paras_tumor_lb, paras_tumor_ub,
                                  nTotal::Int; 
                                  verbose::Bool=true,
                                  nRefine::Int=10)
    
    ndim = length(vars_fit_name_patient)
    lb = base_paras_patient_lb_rearranged
    ub = base_paras_patient_ub_rearranged

    # ──────────────────────────────────────────────────────────
    # Shared objective function builder
    # ──────────────────────────────────────────────────────────
    make_obj = () -> (bp -> begin
        if !all(isfinite.(bp))
            return 1e12
        end
        cish_obj_fun_patient_robust(
            bp, orders, vars_fit_name_patient, params_current, 
            u0_static, tumors_size, cish_ratio, 
            vars_fit_name_tumor, paras_tumor_lb, paras_tumor_ub
        )
    end)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Broad multistart with Latin Hypercube Sampling
    # ══════════════════════════════════════════════════════════
    if verbose
        println("\n  ── Stage 1: Broad multistart ($nTotal initial points, LHS) ──")
    end

    # Generate LHS starting points
    lhs_samples = latin_hypercube_sample(nTotal, ndim, lb, ub)

    all_fvals_patient = fill(Inf, nTotal)
    all_paras_patient = zeros(nTotal, ndim)
    
    best_fval_ref = Ref(Inf)
    best_params_patient = zeros(ndim)
    best_lock = ReentrantLock()
    progress = Atomic{Int}(0)
    start_time = time()
    job_times = fill(NaN, nTotal)
    job_threads = fill(0, nTotal)
    
    jobs = Channel{Int}(nTotal)
    @async begin
        for j in 1:nTotal; put!(jobs, j); end
        close(jobs)
    end
    
    @sync for tid in 1:nthreads()
        @spawn begin
            thread_jobs = 0
            obj_fun = make_obj()

            for j in jobs
                job_start = time()
                try
                    base0 = lhs_samples[j, :]
                    # Nudge away from exact boundaries
                    eps_frac = 1e-8
                    base0 = clamp.(base0, lb .+ eps_frac .* (ub .- lb),
                                          ub .- eps_frac .* (ub .- lb))
                    
                    if !all(isfinite.(base0))
                        all_fvals_patient[j] = 1e12
                        continue
                    end
                    
                    result = optimize(
                        obj_fun, lb, ub, base0,
                        Fminbox(LBFGS()),
                        autodiff = :finite,
                        Optim.Options(
                            f_reltol  = 1e-6,
                            outer_iterations = 8, 
                            iterations = 200, 
                            show_trace = false
                        )
                    )
                    
                    fval = Optim.minimum(result)
                    params = Optim.minimizer(result) .^ orders
                    
                    all_fvals_patient[j] = fval
                    @views all_paras_patient[j, :] .= params
                    job_threads[j] = tid
                    thread_jobs += 1
                    
                    lock(best_lock) do
                        if fval < best_fval_ref[]
                            best_fval_ref[] = fval
                            best_params_patient .= params
                        end
                    end
                catch e
                    all_fvals_patient[j] = 1e12
                    if verbose
                        @warn "Stage 1 job $j failed on thread $tid: $e"
                    end
                end
                
                job_times[j] = time() - job_start
                done = atomic_add!(progress, 1) + 1
                
                if verbose && (done % max(1, nTotal ÷ 10) == 0 || done == nTotal)
                    elapsed = time() - start_time
                    avg_time = elapsed / done
                    eta = (nTotal - done) * avg_time
                    @printf("    Stage 1: %3d/%3d (%.0f%%) | Best: %.6e | %.0fs | ETA: %.0fs\n",
                            done, nTotal, 100*done/nTotal, best_fval_ref[], elapsed, eta)
                    flush(stdout)
                end
            end
        end
    end
    
    if verbose
        valid_times = filter(!isnan, job_times)
        @printf("  Stage 1 done in %.1fs. Best = %.6e. Converged: %d/%d\n",
                time() - start_time, best_fval_ref[],
                count(<(1e11), all_fvals_patient), nTotal)
    end

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Refine top-K candidates with tighter tolerances
    # ══════════════════════════════════════════════════════════
    nRefine = min(nRefine, count(<(1e11), all_fvals_patient))
    if nRefine > 0
        if verbose
            println("\n  ── Stage 2: Local refinement (top $nRefine candidates) ──")
        end

        # Select the top-K distinct solutions
        sorted_idx = sortperm(all_fvals_patient)
        top_indices = sorted_idx[1:nRefine]

        refine_fvals = fill(Inf, nRefine)
        refine_paras = zeros(nRefine, ndim)
        refine_progress = Atomic{Int}(0)
        refine_start = time()

        refine_jobs = Channel{Int}(nRefine)
        @async begin
            for j in 1:nRefine; put!(refine_jobs, j); end
            close(refine_jobs)
        end

        @sync for tid in 1:nthreads()
            @spawn begin
                obj_fun = make_obj()

                for rj in refine_jobs
                    try
                        # Start from Stage 1 result, convert back to base space
                        params_real = all_paras_patient[top_indices[rj], :]
                        base_start = params_real .^ (1 ./ orders)
                        eps_frac = 1e-8
                        base_start = clamp.(base_start,
                                            lb .+ eps_frac .* (ub .- lb),
                                            ub .- eps_frac .* (ub .- lb))

                        # Stage 2: LBFGS with tighter tolerance
                        result = optimize(
                            obj_fun, lb, ub, base_start,
                            Fminbox(LBFGS()),
                            autodiff = :finite,
                            Optim.Options(
                                f_reltol  = 1e-8,
                                outer_iterations = 10,
                                iterations = 300,
                                show_trace = false
                            )
                        )
                        
                        fval_lbfgs = Optim.minimum(result)
                        base_lbfgs = Optim.minimizer(result)

                        # NelderMead polish (derivative-free, escapes gradient traps)
                        result_nm = optimize(
                            obj_fun,
                            base_lbfgs,
                            NelderMead(),
                            Optim.Options(
                                f_reltol  = 1e-8,
                                iterations = 500,
                                show_trace = false
                            )
                        )

                        # Use NelderMead result only if it's feasible and better
                        base_nm = Optim.minimizer(result_nm)
                        fval_nm = Optim.minimum(result_nm)
                        if all(lb .<= base_nm .<= ub) && fval_nm < fval_lbfgs
                            refine_fvals[rj] = fval_nm
                            refine_paras[rj, :] .= base_nm .^ orders
                        else
                            refine_fvals[rj] = fval_lbfgs
                            refine_paras[rj, :] .= base_lbfgs .^ orders
                        end

                        done = atomic_add!(refine_progress, 1) + 1
                        if verbose && (done % max(1, nRefine ÷ 5) == 0 || done == nRefine)
                            best_so_far = minimum(refine_fvals)
                            @printf("    Stage 2: %3d/%3d | Best refined: %.6e (%.0fs)\n",
                                    done, nRefine, best_so_far, time() - refine_start)
                            flush(stdout)
                        end
                    catch e
                        refine_fvals[rj] = all_fvals_patient[top_indices[rj]]
                        refine_paras[rj, :] .= all_paras_patient[top_indices[rj], :]
                        atomic_add!(refine_progress, 1)
                        if verbose
                            @warn "Stage 2 job $rj failed: $e"
                        end
                    end
                end
            end
        end

        # Update global best from refinement
        best_refine_idx = argmin(refine_fvals)
        if refine_fvals[best_refine_idx] < best_fval_ref[]
            best_fval_ref[] = refine_fvals[best_refine_idx]
            best_params_patient .= refine_paras[best_refine_idx, :]
        end

        if verbose
            @printf("  Stage 2 done in %.1fs. Best = %.6e\n",
                    time() - refine_start, best_fval_ref[])
        end
    end

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Quick final polish of best solution (LBFGS → NelderMead)
    # ══════════════════════════════════════════════════════════
    if verbose
        println("\n  ── Stage 3: Final polish of best solution ──")
    end
    polish_start = time()
    obj_fun = make_obj()
    eps_frac = 1e-8

    # 3a: Tight LBFGS
    base_best = best_params_patient .^ (1 ./ orders)
    base_best = clamp.(base_best,
                       lb .+ eps_frac .* (ub .- lb),
                       ub .- eps_frac .* (ub .- lb))
    try
        result = optimize(
            obj_fun, lb, ub, base_best,
            Fminbox(LBFGS()),
            autodiff = :finite,
            Optim.Options(
                f_reltol  = 1e-10,
                g_tol     = 1e-10,
                outer_iterations = 12,
                iterations = 500,
                show_trace = false
            )
        )
        fval = Optim.minimum(result)
        if fval < best_fval_ref[]
            best_fval_ref[] = fval
            best_params_patient .= Optim.minimizer(result) .^ orders
        end
    catch; end

    # 3b: NelderMead from the best point
    base_best = best_params_patient .^ (1 ./ orders)
    base_best = clamp.(base_best,
                       lb .+ eps_frac .* (ub .- lb),
                       ub .- eps_frac .* (ub .- lb))
    try
        result_nm = optimize(
            obj_fun, base_best,
            NelderMead(),
            Optim.Options(
                f_reltol  = 1e-10,
                iterations = 500,
                show_trace = false
            )
        )
        base_nm = Optim.minimizer(result_nm)
        fval_nm = Optim.minimum(result_nm)
        if all(lb .<= base_nm .<= ub) && fval_nm < best_fval_ref[]
            best_fval_ref[] = fval_nm
            best_params_patient .= base_nm .^ orders
        end
    catch; end

    if verbose
        @printf("  Stage 3 done in %.1fs. Final best = %.6e\n",
                time() - polish_start, best_fval_ref[])
    end

    # ══════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════
    if verbose
        total_time = time() - start_time
        
        println("\n" * "="^60)
        println("Optimization Complete!")
        println("="^60)
        @printf("Total time: %.2f seconds\n", total_time)
        @printf("Stage 1 starts:     %d\n", nTotal)
        @printf("Stage 2 refinements: %d\n", nRefine)
        @printf("Final objective:    %.10e\n", best_fval_ref[])
        @printf("Convergence rate:   %d/%d successful (Stage 1)\n", 
                count(<(1e11), all_fvals_patient), nTotal)
        
        println("\nBest parameters found:")
        for (name, val) in zip(vars_fit_name_patient, best_params_patient)
            @printf("  %-10s: %.6e\n", name, val)
        end
        println("="^60)
    end
    
    return best_params_patient, best_fval_ref[], all_fvals_patient, all_paras_patient
end

# Main optimization script
function main(; verbose=true, nTotal=200, nRefine=10)
    # Print system info
    if verbose
        println("\n" * "="^60)
        println("Starting CISH Model Optimization")
        println("="^60)
        println("Julia threads: $(nthreads())")
        println("System: $(Sys.CPU_NAME)")
        println("Memory: $(round(Sys.total_memory()/2^30, digits=1)) GB")
        println("="^60 * "\n")
    end
    
    # Load tumor data
    tumor_data = XLSX.readtable("CISH_pt_data.xlsx", "Sheet1")
    tumor_table = DataFrame(tumor_data)

    # Load CISH ratio data from CSV
    cish_ratio_table = CSV.read("CISHKO_Pct_result.csv", DataFrame)

    # Load dose data
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
    il2_doses = dose_data[:, 3]

    id_patient = unique(patient_nums_tumor)

    # ========== Parameter setup (exactly as in original) ==========
    vars_names = ["t_infusion","t_end",
                "gamma","lambda","s","K",
                "kL","kN","muL","muN",
                "dL","dN","pL","pN","g",
                "delta","h","l","n","f","f_dose","kappa"]

    init_vals = [0.001, 85.0,
                0.01, 0.58, 4.24, 1.00, 
                0.03, 0.0005, 4e-4, 4e-4,
                0.05, 0.030, 0.05, 0.05, 1.0e3,
                0.01, 1e3, 578, 200, 1.0, 0.2, 0.25]

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

    # Create dictionaries for compatibility
    vars_all = Dict(Symbol(n) => v for (n, v) in zip(vars_names, init_vals))
    vars_all_ub = Dict(Symbol(n) => v for (n, v) in zip(vars_names, ub_vals))
    vars_all_lb = Dict(Symbol(n) => v for (n, v) in zip(vars_names, lb_vals))

    # Define which parameters to fit (easily changeable!)
    vars_fit_name_tumor = ["gamma"]
    vars_fit_name_patient = ["dL","kL","pL","muL","pN","f_dose","kappa"]

    # Extract bounds and initial values for fitted parameters
    paras_tumor_init = [vars_all[Symbol(v)] for v in vars_fit_name_tumor]
    paras_tumor_lb = [vars_all_lb[Symbol(v)] for v in vars_fit_name_tumor]
    paras_tumor_ub = [vars_all_ub[Symbol(v)] for v in vars_fit_name_tumor]

    paras_patient_init = [vars_all[Symbol(v)] for v in vars_fit_name_patient]
    orders = log10.(paras_patient_init)

    base_patient_init = fill(10.0, length(paras_patient_init))
    paras_patient_lb = [vars_all_lb[Symbol(v)] for v in vars_fit_name_patient]
    paras_patient_ub = [vars_all_ub[Symbol(v)] for v in vars_fit_name_patient]

    base_paras_patient_lb = paras_patient_lb .^ (1 ./ orders)
    base_paras_patient_ub = paras_patient_ub .^ (1 ./ orders)
    base_paras_patient_lb_rearranged = min.(base_paras_patient_lb, base_paras_patient_ub)
    base_paras_patient_ub_rearranged = max.(base_paras_patient_lb, base_paras_patient_ub)

    # ========== Initial conditions (N0 fixed; L0 set per patient from cish_ratio) ==========
    # Default N0
    N0_default = 0.1
    # We'll set u0_static per patient after extracting patient-specific cish_ratio below
    u0_static = SVector{3, Float64}(0.0, 0.0, N0_default)

    # ========== Create initial parameter struct ==========
    params_init = CISHParams(vars_all)

    # ========== Optimization loop over patients ==========
    for i in 3:4
        if verbose
            println("\n" * "="^60)
            println("Fitting Patient $i / $(length(id_patient))")
            println("="^60)
        end
        
        # Update parameters for specific patients
        vars_all_current = copy(vars_all)
        if i == 12
            vars_all_current[:t_end] = 180.0
        end
        if i == 4
            vars_all_current[:t_end] = 45.0
        end 
        if i == 5
            vars_all_current[:t_end] = 60.0
        end          
        if i == 6
            vars_all_current[:t_end] = 30.0
        end   
        if i == 8
            vars_all_current[:t_end] = 40.0
        end   
        if i == 9
            vars_all_current[:t_end] = 30.0
        end   
        if i == 10
            vars_all_current[:t_end] = 60.0
        end   
        if i == 11
            vars_all_current[:t_end] = 30.0
        end   
        # Create struct from current parameters
        params_current = CISHParams(vars_all_current)

        # Get patient data
        inds_tumor = findall(patient_nums_tumor .== id_patient[i])
        id_tumor = unique(tumors_nums[inds_tumor])

        inds_dose = findall(patients_num_dose .== id_patient[i])
        dose = [cell_dose[inds_dose][1]]
        dose_static = SVector{1, Float64}(dose...)

        # Prepare tumor data
        tumors_size = zeros(length(inds_tumor), length(id_tumor)*2)
        for (j, tumor_id) in enumerate(id_tumor)
            inds = findall((patient_nums_tumor .== id_patient[i]) .& (tumors_nums .== tumor_id))
            tumors_size[1:length(inds), 2*j-1:2*j] .= hcat(days_nums[inds], tumor_size_nums[inds])
        end

        # Prepare CISH ratio data
        inds_cish = findall(patient_nums_cish .== id_patient[i])
        cish_days = days_nums_cish[inds_cish]
        ratio_data = cish_nums_ratio[inds_cish]
        cish_ratio = hcat(cish_days, ratio_data)
        cish_ratio = cish_ratio[cish_ratio[:,1] .>= 0, :]

        # Set initial conditions: [T0, L0, N0, Dose0]
        # L0 = 0 (CISH cells will come from Dose compartment)
        # Dose0 = cell_dose from file
        N0 = N0_default
        L0 = 0.0  # Force L0 = 0
        Dose0 = dose[1]/1e10  # Get initial dose from file, scaled (from dose data to mm^2)
        
        # Create 4D initial state: [T0, L0, N0, Dose0]
        u0 = [0.0, L0, N0, Dose0]
        u0_static = SVector{4, Float64}(u0...)

        # ========== Patient-level optimization with parallel processing ==========
        paras_fit_patient, best_fval, all_fvals, all_paras = parallel_optimize_patient(
            params_current, u0_static,
            tumors_size, cish_ratio,
            vars_fit_name_patient, vars_fit_name_tumor,
            orders, base_paras_patient_lb_rearranged, base_paras_patient_ub_rearranged,
            paras_tumor_lb, paras_tumor_ub,
            nTotal;
            verbose=verbose,
            nRefine=nRefine
        )

        # Update patient-level parameters
        params_patient = update_params(params_current, vars_fit_name_patient, paras_fit_patient)
        
        # Also update dictionary for compatibility
        vars_all_patient = params_to_dict(params_patient)

        # ========== Plot tumor fits ==========
        if verbose
            println("\nFitting individual tumors...")
        end

        fig = plot(layout=(2,2), size=(800,600))
        best_paras_tumors = zeros(length(vars_fit_name_tumor), size(tumors_size,2) ÷ 2)

        # Storage for calculating average ratio - simple sum approach
        sum_L = nothing
        sum_N = nothing
        common_time = nothing
        valid_tumor_count = 0

        # decide if tumor-level is scalar (e.g., only "gamma")
        is_scalar_tumor = (length(vars_fit_name_tumor) == 1)
        for k in 1:(size(tumors_size,2) ÷ 2)
            # extract (time, size) and filter out (0,0) rows
            tumor_size = tumors_size[:, 2k-1:2k]
            mask = .!( (tumor_size[:, 1] .== 0.0) .& (tumor_size[:, 2] .== 0.0) )
            tumor_size = tumor_size[mask, :]

            # skip empty tumors
            if size(tumor_size, 1) == 0
                continue
            end

            # set tumor-specific initial condition (T0 from first observation)
            # Keep L0=0, N0, and Dose0 from u0
            u0_new = copy(u0)
            u0_new[1] = tumor_size[1,2]
            u0_tumor_static = SVector{4, Float64}(u0_new...)

            # vector-form tumor objective (expects a Vector even for 1D case)
            obj_vec = p -> cish_obj_fun_tumor_optimized(
                p, vars_fit_name_tumor, params_patient,
                u0_tumor_static, tumor_size, cish_ratio
            )

            # choose optimizer according to dimensionality
            best_paras_tumor = similar(paras_tumor_lb)
            best_fval_tumor = Inf

            if is_scalar_tumor
                # 1D: Brent bracketing search on [lb, ub]; no initial point needed
                f_scalar = γ -> obj_vec([γ])
                res = optimize(
                    f_scalar,
                    paras_tumor_lb[1], paras_tumor_ub[1],
                    Brent();
                    rel_tol = 1e-6,
                    abs_tol = 1e-9,
                    iterations = 100,
                    show_trace = false
                )
                best_paras_tumor[1] = Optim.minimizer(res)
                best_fval_tumor = Optim.minimum(res)
            else
                # multi-D: Fminbox + LBFGS with a couple of vector initial guesses
                init_list = Any[
                    paras_tumor_lb .+ 1e-3,
                    paras_tumor_lb .+ (paras_tumor_ub - paras_tumor_lb) ./ 2
                ]
                for init in init_list
                    res = optimize(
                        obj_vec,
                        paras_tumor_lb, paras_tumor_ub, init,
                        Fminbox(LBFGS()),
                        autodiff = :finite,
                        Optim.Options(
                            f_reltol = 1e-6,
                            outer_iterations = 5,
                            iterations = 50,
                            show_trace = false
                        )
                    )
                    fval = Optim.minimum(res)
                    if fval < best_fval_tumor
                        best_fval_tumor = fval
                        best_paras_tumor .= Optim.minimizer(res)
                    end
                end
            end

            # store best tumor-level params
            best_paras_tumors[:, k] = best_paras_tumor

            # update params and solve ODE with the best tumor-level params
            params_tumor = update_params(params_patient, vars_fit_name_tumor, best_paras_tumor)
            sol = cish_model_fun_optimized(params_tumor, u0_tumor_static)

            # inside your loop
            col = TUMOR_COLORS[(k-1) % length(TUMOR_COLORS) + 1]
            
            plot!(fig[1], sol.x, sol.y[1, :],
                  label = "Tumor $k", lw = 2, color = col)
            
            scatter!(fig[1], tumor_size[:,1], tumor_size[:,2],
                     label = "Data $k", ms = 3,
                     markercolor = col, markerstrokecolor = col)

            # Panel 2: WildType (N)
            plot!(fig[2], sol.x, sol.y[3, :], label="WildType $k", lw=2)

            # Panel 3: CISHKO (L)
            plot!(fig[3], sol.x, sol.y[2, :], label="CISHKO $k", lw=2)

            # Store solution for averaging (we'll interpolate later)
            if sum_L === nothing
                # Initialize with a common time grid based on first tumor
                common_time = Vector{Float64}(sol.x)
                sum_L = Vector{Float64}(sol.y[2, :])
                sum_N = Vector{Float64}(sol.y[3, :])
            else
                # Interpolate this tumor's solution to the common time grid
                L_interp = [begin
                    idx = argmin(abs.(sol.x .- t))
                    sol.y[2, idx]
                end for t in common_time]
                
                N_interp = [begin
                    idx = argmin(abs.(sol.x .- t))
                    sol.y[3, idx]
                end for t in common_time]
                
                sum_L .+= L_interp
                sum_N .+= N_interp
            end
            valid_tumor_count += 1
        end
        
        # Calculate and plot average ratio after all tumors are processed
        if valid_tumor_count > 0 && sum_L !== nothing && common_time !== nothing
            # Calculate average L and N values
            avg_L = sum_L ./ valid_tumor_count
            avg_N = sum_N ./ valid_tumor_count
            
            # Calculate average ratio
            avg_ratio = avg_L ./ max.(avg_N, 1e-12)
            
            # Plot average ratio
            plot!(fig[4], common_time, avg_ratio, label="Average Ratio", lw=3, color=:black, linestyle=:solid)
        end
        
        # Add ratio data once (if provided)
        if size(cish_ratio,1) > 0
            scatter!(fig[4], cish_ratio[:,1], cish_ratio[:,2], ms=3, label="Ratio data")
            ylims!(fig[4], (0, 2 * maximum(cish_ratio[:,2])))
        end

        # tidy labels/titles
        xlabel!(fig[1], "Days"); ylabel!(fig[1], "Tumor size"); title!(fig[1], "Tumor fit")
        xlabel!(fig[2], "Days"); ylabel!(fig[2], "Cells");      title!(fig[2], "WildType (N)")
        xlabel!(fig[3], "Days"); ylabel!(fig[3], "Cells");      title!(fig[3], "CISHKO (L)")
        xlabel!(fig[4], "Days"); ylabel!(fig[4], "CISH ratio"); title!(fig[4], "L /  N")
        savefig(fig, "Patient_$(id_patient[i])_fit.png")
        
        # Save results
        outdata = Dict(
            "paras_fit_patient" => paras_fit_patient,
            "best_paras_tumors" => eachcol(best_paras_tumors) |> collect,
            "vars_names" => vars_names,
            "init_vals" => init_vals,
            "vars_fit_name_tumor" => vars_fit_name_tumor,
            "vars_fit_name_patient" => vars_fit_name_patient,
            "best_objective" => best_fval,
            "all_objectives" => all_fvals
        )

        json_path = "Patient_$(id_patient[i])_paras.json"
        JSON3.write(json_path, outdata)
        
        if verbose
            println("\nResults saved to: $json_path")
            println("Plot saved to: Patient_$(id_patient[i])_fit.png")
        end
    end
    
    if verbose
        println("\n" * "="^60)
        println("Optimization Complete!")
        println("="^60)
    end
end

# Run the main function
main(verbose=true, nTotal=200, nRefine=10)