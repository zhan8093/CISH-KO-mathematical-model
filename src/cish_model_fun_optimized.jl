# cish_model_fun_optimized.jl
using DifferentialEquations
using StaticArrays

# Include struct definition if not already available
if !@isdefined(CISHParams)
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
end

# Out-of-place ODE RHS function (Dual-friendly) - 4D system: [T, L, N, Dose]
function ode_rhs_optimized(y, p::CISHParams, t)
    T_type = eltype(y)

    # Ensure positivity
    T = max(y[1], T_type(1e-10))
    L = max(y[2], T_type(5e-4))
    N = max(y[3], T_type(1e-7))
    Dose_raw = max(y[4], T_type(0.0))
    
    # Dose cutoff threshold for computational efficiency
    # When Dose is negligible, set to zero to speed up computation
    Dose_threshold = T_type(1e-6)  # ~0.01% of typical initial dose
    Dose = Dose_raw > Dose_threshold ? Dose_raw : T_type(0.0)
    dose_active = Dose_raw > Dose_threshold  # Flag for dose effects

    # Early stop for runaway tumor
    if real(T) > 1e5
        return SVector{4,T_type}(zero(T_type), zero(T_type), zero(T_type), zero(T_type))
    end


    # Safe powers
    λ = p.lambda

    LN_sum = L + N
    K_safe = max(p.K, T_type(1e-10))
    capacity = 1 - LN_sum / K_safe

    g_safe = max(p.g, T_type(1e-10))
    f_safe = max(p.f, T_type(1e-10))
    h_safe = max(p.h, T_type(1e-10))
    s_safe = max(p.s, T_type(1e-10))


    kL_term = p.kL * T * L
    kN_term = p.kN * T * N

    dT = p.gamma * T - (kL_term + kN_term) * T
    # Add kappa*Dose term only if dose is above threshold
    dL = -p.muL * L * T - p.dL * L + capacity * p.pL * L + (dose_active ? p.kappa * Dose : T_type(0.0))
    dN = -p.muN * N * T - p.dN * N + capacity * p.pN * N + capacity * p.delta * T / (h_safe + T)
    # Dose dynamics: set to exactly zero once below threshold to save computation
    dDose = dose_active ? (-p.f_dose * Dose - p.kappa * Dose) : T_type(0.0)

    return SVector{4,T_type}(dT, dL, dN, dDose)
end

# In-place version (if needed)
function ode_rhs_optimized!(dy, y, p::CISHParams, t)
    r = ode_rhs_optimized(y, p, t)
    dy[1] = r[1]; dy[2] = r[2]; dy[3] = r[3]; dy[4] = r[4]
    return nothing
end

# ---- Main: single-phase solve; infusion applied at t=0 via initial conditions ----
function cish_model_fun_optimized(params::CISHParams{T}, y0) where T
    # Promote element type (supports Dual, BigFloat, etc.)
    T_type = promote_type(T, eltype(y0))

    # y0 -> typed SVector (T, L, N, Dose)
    # Ensure L0 = 0 (will be populated from Dose compartment)
    if length(y0) == 4
        y0_vec = isa(y0, SVector) ? SVector{4,T_type}(y0) : SVector{4,T_type}(y0...)
    elseif length(y0) == 3
        # Backward compatibility: assume [T, L, N] and set Dose=0
        y0_vec = SVector{4,T_type}(y0[1], T_type(0), y0[3], T_type(0))
    else
        error("Initial condition y0 must have length 3 or 4")
    end

    # Force L0 = 0 (index 2)
    y0_infused = SVector{4,T_type}(
        max(y0_vec[1], T_type(1e-10)),  # T0
        T_type(0.0),                     # L0 = 0 (forced)
        max(y0_vec[3], T_type(1e-10)),  # N0
        max(y0_vec[4], T_type(0.0))      # Dose0
    )

    # Single-phase problem: [0, t_end]
    t0 = T_type(0)
    tF = T_type(params.t_end)
    prob = ODEProblem{false}(ode_rhs_optimized, y0_infused, (t0, tF), params)
    sol = solve(prob, Tsit5(); 
            abstol=T_type(1e-10), 
            reltol=T_type(1e-3), 
            saveat=T_type(0.5), 
            maxiters=1_000_000)

    # Clamp solution before returning
    y_out = hcat(sol.u...)
    y_out = max.(y_out, T_type(1e-10))
    
    return (; x=sol.t, y=y_out)
    # try
    #     sol = solve(prob, Tsit5();
    #                 abstol=T_type(1e-5), reltol=T_type(1e-3),
    #                 saveat=T_type(0.5), maxiters=1_000_000)
    #     if sol.retcode == :Success
    #         return (; x=sol.t, y=hcat(sol.u...))
    #     end
    # catch e
    #     @debug "Solver failed: $e"
    # end

    # # Fallback (single-phase)
    # return cish_model_fun_fallback(params, y0_infused)
end




