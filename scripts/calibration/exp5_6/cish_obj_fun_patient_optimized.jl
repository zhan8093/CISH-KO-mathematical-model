# cish_obj_fun_patient_optimized.jl
"""
Optimized patient-specific objective function with parallel tumor fitting
"""

function cish_obj_fun_patient_optimized(base_paras_patient::Vector{Float64},
                                       orders::Vector{Float64},
                                       vars_fit_name_patient::Vector{String},
                                       params_base::CISHParams,
                                       u0::SVector{4,Float64},
                                       tumors_size::Matrix{Float64},
                                       cish_ratio::Matrix{Float64},
                                       vars_fit_name_tumor::Vector{String},
                                       paras_tumor_lb::Vector{Float64},
                                       paras_tumor_ub::Vector{Float64})

    # 1) Map from base-space to real parameter space and update patient-level params
    paras_patient = base_paras_patient .^ orders
    params_patient = update_params(params_base, vars_fit_name_patient, paras_patient)

    num_tumors = size(tumors_size, 2) ÷ 2
    fval_tot = 0.0

    # Use scalar optimizer when tumor-level has exactly 1 parameter (e.g., gamma)
    is_scalar = (length(vars_fit_name_tumor) == 1) && (length(paras_tumor_lb) == 1)

    # Storage for CISH ratio calculation across all tumors
    all_L_vals = Vector{Float64}()
    all_N_vals = Vector{Float64}()
    
    # 2) Process tumors one by one
    for i in 1:num_tumors
        # Extract (time, size) block and filter out (0, 0) rows
        col_start = 2i - 1
        col_end   = 2i
        tumor_size_raw = tumors_size[:, col_start:col_end]
        valid_rows = Vector{Int}()
        @inbounds for row in 1:size(tumor_size_raw, 1)
            if !(tumor_size_raw[row, 1] == 0.0 && tumor_size_raw[row, 2] == 0.0)
                push!(valid_rows, row)
            end
        end
        if isempty(valid_rows)
            continue
        end

        tumor_size_filtered = Matrix{Float64}(tumor_size_raw[valid_rows, :])

        # Set tumor-specific initial condition (T0 from the first observation)
        # Keep L0=0, N0, and Dose0 from u0
        u0_tumor = SVector{4,Float64}(
            tumor_size_filtered[1, 2],
            u0[2], u0[3], u0[4]
        )

        # Vector-form tumor objective (expects a Vector even for 1D) - without CISH calculation
        obj_vec = p -> cish_obj_fun_tumor_only(
            p, vars_fit_name_tumor, params_patient,
            u0_tumor, tumor_size_filtered
        )

        # 3) Optimize tumor-level parameter(s)
        best_fval_tumor = Inf
        best_paras_tumor = nothing

        if is_scalar
            # Scalar case: Brent bracketing search on [lb, ub]; no initial point needed
            f_scalar = γ -> obj_vec([γ])
            res = optimize(f_scalar, paras_tumor_lb[1], paras_tumor_ub[1], Brent();
                           rel_tol=1e-6, abs_tol=1e-9, iterations=100)
            best_fval_tumor = Optim.minimum(res)
            best_paras_tumor = [Optim.minimizer(res)]
        else
            # Multi-dimensional case: Fminbox + LBFGS with a couple of vector initial guesses
            init_points = Any[
                paras_tumor_lb .+ 1e-3,
                paras_tumor_lb .+ (paras_tumor_ub - paras_tumor_lb) ./ 2
            ]
            for init in init_points
                res = optimize(
                    obj_vec,
                    paras_tumor_lb,
                    paras_tumor_ub,
                    init,
                    Fminbox(LBFGS()),
                    Optim.Options(
                        f_reltol = 1e-6,
                        iterations = 20,
                        show_trace = false,
                        store_trace = false,
                        extended_trace = false
                    )
                )
                fval = Optim.minimum(res)
                if fval < best_fval_tumor
                    best_fval_tumor = fval
                    best_paras_tumor = Optim.minimizer(res)
                end
            end
        end

        # Accumulate tumor contributions
        fval_tot += best_fval_tumor
        
        # Collect L_vals and N_vals from the optimized tumor for CISH calculation
        if best_paras_tumor !== nothing && size(cish_ratio, 1) > 0
            # Update parameters with best fit
            params_tumor = update_params(params_patient, vars_fit_name_tumor, best_paras_tumor)
            
            # Solve ODE with best parameters
            sol = cish_model_fun_optimized(params_tumor, u0_tumor)
            
            if length(sol.x) >= 2
                cish_time = Vector{Float64}(cish_ratio[:, 1])
                tvals_cish = time_idxs_optimized(cish_time, sol.x)
                
                # Collect L_vals and N_vals for this tumor
                L_vals_tumor = sol.y[2, tvals_cish]
                N_vals_tumor = sol.y[3, tvals_cish]
                
                append!(all_L_vals, L_vals_tumor)
                append!(all_N_vals, N_vals_tumor)
            end
        end
    end
    
    # Patient-level CISH ratio calculation using average L_vals and N_vals
    obj_cish = 0.0
    if length(all_L_vals) > 0 && length(all_N_vals) > 0 && size(cish_ratio, 1) > 0
        cish_value = Vector{Float64}(cish_ratio[:, 2])
        
        # Reshape collected values: rows = time points, columns = tumors
        n_time_points = length(cish_value)
        
        if num_tumors > 0 && n_time_points * num_tumors == length(all_L_vals)
            L_matrix = reshape(all_L_vals, n_time_points, num_tumors)
            N_matrix = reshape(all_N_vals, n_time_points, num_tumors)
            
            # Calculate average across tumors for each time point
            avg_L = mean(L_matrix, dims=2)[:]  # Average across columns (tumors)
            avg_N = mean(N_matrix, dims=2)[:]
            
            weight = 0.2
            @inbounds for i in 1:n_time_points
                if isfinite(cish_value[i]) && cish_value[i] > 0 && avg_N[i] > 1e-10
                    ratio_model = avg_L[i] / avg_N[i]
                    diff = 1 - ratio_model/cish_value[i]
                    obj_cish += weight * diff * diff
                end
            end
        end
    end

    return fval_tot + obj_cish
end



# Robust wrapper expected by main_optimized.jl
function cish_obj_fun_patient_robust(base_paras_patient,
                                     orders,
                                     vars_fit_name_patient,
                                     params_base::CISHParams,
                                     u0,
                                     tumors_size,
                                     cish_ratio,
                                     vars_fit_name_tumor,
                                     paras_tumor_lb,
                                     paras_tumor_ub)
    try
        # Ensure Float64 matrices for internal computations
        ts = Matrix{Float64}(tumors_size)
        cr = Matrix{Float64}(cish_ratio)
        bp = Vector{Float64}(base_paras_patient)
        ords = Vector{Float64}(orders)
        vfname_p = Vector{String}(vars_fit_name_patient)
        vfname_t = Vector{String}(vars_fit_name_tumor)
        lb = Vector{Float64}(paras_tumor_lb)
        ub = Vector{Float64}(paras_tumor_ub)
        
        # Convert u0 to SVector{4,Float64} - handle various input types
        if isa(u0, SVector{4,Float64})
            u0_static = u0
        elseif isa(u0, SVector)
            u0_static = SVector{4,Float64}(u0[1], u0[2], u0[3], u0[4])
        else
            # Handle Array, Tuple, or other indexable types
            u0_vec = length(u0) == 4 ? u0 : error("u0 must have length 4, got $(length(u0))")
            u0_static = SVector{4,Float64}(u0_vec[1], u0_vec[2], u0_vec[3], u0_vec[4])
        end

        return cish_obj_fun_patient_optimized(bp, ords, vfname_p,
                                              params_base, u0_static,
                                              ts, cr,
                                              vfname_t, lb, ub)
    catch e
        @warn "cish_obj_fun_patient_robust failed: $e"
        return 1e12
    end
end



