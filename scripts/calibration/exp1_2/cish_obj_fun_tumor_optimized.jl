# cish_obj_fun_tumor_optimized.jl
using LinearAlgebra
using StaticArrays

"""
Optimized function to find time indices efficiently using binary search
"""
function time_idxs_optimized(t_exp::AbstractVector{Float64}, t_sim::AbstractVector{Float64})
    n = length(t_exp)
    tvals = Vector{Int}(undef, n)
    
    @inbounds for i in 1:n
        # Use searchsortedfirst for binary search (O(log n))
        idx = searchsortedfirst(t_sim, t_exp[i])
        
        # Handle edge cases and find closest value
        if idx == 1
            tvals[i] = 1
        elseif idx > length(t_sim)
            tvals[i] = length(t_sim)
        else
            # Check which is closer: idx or idx-1
            if abs(t_sim[idx-1] - t_exp[i]) < abs(t_sim[idx] - t_exp[i])
                tvals[i] = idx - 1
            else
                tvals[i] = idx
            end
        end
    end
    
    return tvals
end

"""
Optimized tumor-specific objective function
"""
function cish_obj_fun_tumor_optimized(paras_tumor::Vector{Float64}, 
                                     vars_fit_name_tumor::Vector{String},
                                     params_base::CISHParams,
                                     u0::SVector{4,Float64},
                                     tumor_size::Matrix{Float64},
                                     cish_ratio::Matrix{Float64})
    try
        # Check for invalid parameters
        if any(!isfinite, paras_tumor) || any(<(0), paras_tumor)
            return 1e12
        end
        
        # Update parameters with fitted values
        params = update_params(params_base, vars_fit_name_tumor, paras_tumor)
        
        # Extract kL and kN for constraint checking
        kL = params.kL
        kN = params.kN
        
        # Solve ODE system (using optimized solver) - now handles 4D state
        sol = cish_model_fun_optimized(params, u0)
        
        # Check if solution is valid
        if length(sol.x) < 2
            return 1e12
        end
        
        # Extract time and value vectors - convert views to vectors if needed
        tumor_time = Vector{Float64}(tumor_size[:, 1])
        tumor_value = Vector{Float64}(tumor_size[:, 2])
        
        # Find time indices efficiently
        tvals_tumor = time_idxs_optimized(tumor_time, sol.x)
        
        # Calculate tumor objective
        tumor_ode = sol.y[1, tvals_tumor]
        
        # Check for invalid ODE solutions
        if any(!isfinite, tumor_ode)
            return 1e12
        end
        
        # Vectorized computation for tumor objective
        obj_tumor = 0.0
        @inbounds for i in 1:length(tumor_value)
            if tumor_value[i] > 1e-10  # Only use valid data points
                diff = 1.0 - tumor_ode[i] / tumor_value[i]
                obj_tumor += diff * diff
            end
        end
        
        # CISH ratio objective if data is available
        obj_cish = 0.0
        if size(cish_ratio, 1) > 0
            cish_time = Vector{Float64}(cish_ratio[:, 1])
            cish_value = Vector{Float64}(cish_ratio[:, 2])
            
            tvals_cish = time_idxs_optimized(cish_time, sol.x)
            # Pre-allocate for CISH calculations
            L_vals = sol.y[2, tvals_cish]
            N_vals = sol.y[3, tvals_cish]
            weight = 0.2
            @inbounds for i in 1:length(cish_value)
                if isfinite(cish_value[i]) && cish_value[i] > 0
                    # Calculate ratio with numerical stability
                    if N_vals[i]> 1e-10
                        ratio_model = L_vals[i] / N_vals[i]
                        # Use difference of ratios instead of logs to avoid numerical issues
                        diff = (cish_value[i] - ratio_model)/cish_value[1]
                 #       diff = (-1< diff < 0.5) ? 0.0 : diff
                        obj_cish += weight * diff * diff
                    end
                end
            end
        end
   #     
    #    # Penalties
     #   kN_kL_penalty = 1000.0 * max(kN - kL, 0.0)
      #  
       # # End-point penalty
        #penal = 2.0
#        end_penalty = if length(tvals_tumor) > 0 && sol.y[1, end] > 1e-10
 #           penal * max(0.0, tumor_ode[end] / sol.y[1, end] - 1.0)
  #      else
   #         0.0
    #    end
     #   
      #  # After computing obj_tumor, obj_cish, kN_kL_penalty, end_penalty:
       # if !(obj_tumor ≥ 0 && obj_cish ≥ 0 && kN_kL_penalty ≥ 0 && end_penalty ≥ 0)
        #    @warn "Negative component" obj_tumor obj_cish kN_kL_penalty end_penalty
        #end

        #total = obj_tumor + obj_cish + kN_kL_penalty + end_penalty
        total = obj_tumor + obj_cish
        if total < 0
            @warn "Negative total objective" total obj_tumor obj_cish kN_kL_penalty end_penalty
            total = 0.0                   # clamp to be safe
        end
        return isfinite(total) ? total : 1e12        
        
    catch e
        @warn "Error in tumor objective function: $e"
        return 1e12  # Return large penalty on error
    end
end

"""
Tumor-specific objective function without CISH calculation
"""
function cish_obj_fun_tumor_only(paras_tumor::Vector{Float64}, 
                                 vars_fit_name_tumor::Vector{String},
                                 params_base::CISHParams,
                                 u0::SVector{4,Float64},
                                 tumor_size::Matrix{Float64})
    try
        # Check for invalid parameters
        if any(!isfinite, paras_tumor) || any(<(0), paras_tumor)
            return 1e12
        end
        
        # Update parameters with fitted values
        params = update_params(params_base, vars_fit_name_tumor, paras_tumor)
        
        # Solve ODE system (using optimized solver) - now handles 4D state
        sol = cish_model_fun_optimized(params, u0)
        
        # Check if solution is valid
        if length(sol.x) < 2
            return 1e12
        end
        
        # Extract time and value vectors - convert views to vectors if needed
        tumor_time = Vector{Float64}(tumor_size[:, 1])
        tumor_value = Vector{Float64}(tumor_size[:, 2])
        
        # Find time indices efficiently
        tvals_tumor = time_idxs_optimized(tumor_time, sol.x)
        
        # Calculate tumor objective
        tumor_ode = sol.y[1, tvals_tumor]
        
        # Check for invalid ODE solutions
        if any(!isfinite, tumor_ode)
            return 1e12
        end
        
        # Vectorized computation for tumor objective
        obj_tumor = 0.0
        @inbounds for i in 1:length(tumor_value)
            if tumor_value[i] > 1e-10  # Only use valid data points
                diff = 1.0 - tumor_ode[i] / tumor_value[i]
                obj_tumor += diff * diff
            end
        end
        
        return isfinite(obj_tumor) ? obj_tumor : 1e12        
        
    catch e
        @warn "Error in tumor objective function: $e"
        return 1e12  # Return large penalty on error
    end
end