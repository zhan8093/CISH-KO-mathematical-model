using OrdinaryDiffEq
using StaticArrays

@inline function ode_rhs(y::SVector{7,Float64}, p, t)
    T  = y[1] > 1e-6 ? y[1] : 1e-6
    L  = y[2] > 1e-6 ? y[2] : 1e-6
    Lo = y[3] > 1e-6 ? y[3] : 1e-6
    N  = y[4] > 1e-6 ? y[4] : 1e-6
    Z  = y[5] > 1e-6 ? y[5] : 1e-6
    dL_c  = y[6] > 1e-6 ? y[6] : 1e-6
    dLo_c = y[7] > 1e-6 ? y[7] : 1e-6

    gamma = p.gamma; K = p.K
    muL = p.muL; muN = p.muN; muZ = p.muZ
    dL = p.dL; dN = p.dN
    pL = p.pL; pN = p.pN
    delta = p.delta; h = p.h
    a = p.a; b = p.b; bL = p.bL
    kL = p.kL; kLo = p.kLo; kN = p.kN
    kappa = p.kappa

    alpha   = (a + 1) / (a + exp(-b  * Z))
    alpha_L = (a + 1) / (a + exp(-bL * Z))
    capacity = 1 - (L + Lo + N) / K

    dT  = gamma*T - alpha_L*kL*L*T - alpha*kLo*Lo*T - alpha*kN*N*T
    dLd = -muL*L*T/(1+Z) - dL*L + capacity*pL*L*(1+Z) + kappa*dL_c
    dLo = -muN*Lo*T/(1+Z) - dN*Lo + capacity*pN*Lo*(1+Z) + kappa*dLo_c
    dN_ = -muN*N*T/(1+Z) - dN*N + capacity*pN*N*(1+Z) + capacity*delta*T/(h+T)
    dZ  = -muZ*Z
    ddL = -kappa*dL_c
    ddLo= -kappa*dLo_c
    return SVector{7,Float64}(dT, dLd, dLo, dN_, dZ, ddL, ddLo)
end

function solve_model(vars_all, y0, cond; saveat=nothing)
    y0m = SVector{7,Float64}(
        y0[1], y0[2], y0[3], y0[4],
        y0[5] + cond[3],
        cond[1],
        cond[2],
    )
    tspan = (0.0, vars_all.t_end)
    prob = ODEProblem(ode_rhs, y0m, tspan, vars_all)
    if saveat === nothing
        solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, maxiters=10^6)
    else
        solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, maxiters=10^6, saveat=saveat)
    end
end
