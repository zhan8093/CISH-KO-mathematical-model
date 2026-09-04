using Base.Threads
using BlackBoxOptim
using Optim
using Random
using Printf
using DataFrames, CSV
using Plots

include("cish_model.jl")
include("cish_obj.jl")

vars_names = (:t_infusion, :t_end,
              :gamma, :K,
              :kL, :kLo, :kN, :muL, :muN,
              :dL, :dN, :pL, :pN,
              :delta, :h,
              :muZ, :a, :b, :bL, :kappa)

init_vals = [7.5, 42.0,
             0.1748, 1.0,
             1.462, 0.5077, 0.0811, 1.2e-4, 1.57e-4,
             0.0796, 0.03, 0.2, 0.05,
             0.1, 1e3,
             0.7, 0.8891, 9.3033, 2.2837, 0.05]

ub_vals = [7.5, 42.0,
           0.5, 10.0,
           7.2, 7.2, 7.2, 2.5e-3, 2.5e-3,
           0.08, 0.04, 2.971, 2.971,
           0.12, 1e3,
           0.7, 10.0, 10.0, 10.0, 2.0]

lb_vals = [7.5, 42.0,
           0.1, 1.0,
           0.014, 0.005, 0.001, 5e-9, 5e-9,
           0.001, 0.02, 0.124, 0.124,
           0.12, 1e3,
           0.7, 0.001, 0.01, 0.01, 0.01]

vars_all = NamedTuple{vars_names}(Tuple(init_vals))
vars_ub  = NamedTuple{vars_names}(Tuple(ub_vals))
vars_lb  = NamedTuple{vars_names}(Tuple(lb_vals))

fit_names = (:gamma, :kL, :kLo, :kN, :pL, :dL, :a, :b, :bL,:kappa)
nFit = length(fit_names)
paras_lb = [getfield(vars_lb, n) for n in fit_names]
paras_ub = [getfield(vars_ub, n) for n in fit_names]

T0 = 25.0; L0 = 0.0; Lw0 = 0.0; N0 = 0.0; Z0 = 0.0; dL0 = 0.0; dLo0 = 0.0
u0 = [T0, L0, Lw0, N0, Z0, dL0, dLo0]

init_datasets()

# Warm up compiled solver
_ = obj_fun(Float64[(l+u)/2 for (l,u) in zip(paras_lb, paras_ub)], fit_names, vars_all, u0)

f_obj = p -> obj_fun(collect(p), fit_names, vars_all, u0)

# ============================================================
# STAGE 1: Global search with Differential Evolution (strict bounds)
# ============================================================
println("Stage 1: Differential Evolution global search")
search_range = [(paras_lb[i], paras_ub[i]) for i in 1:nFit]

bb_res = bboptimize(f_obj;
    SearchRange = search_range,
    NumDimensions = nFit,
    Method = :adaptive_de_rand_1_bin_radiuslimited,
    MaxSteps = 20000,
    PopulationSize = 60,
    TraceMode = :compact,
    TraceInterval = 5.0,
)

de_best = best_candidate(bb_res)
de_fval = best_fitness(bb_res)
@printf("DE best fval: %.6g\n", de_fval)

# ============================================================
# STAGE 2: Multi-start local polish around DE best
# ============================================================
println("\nStage 2: Local polish with LBFGS (Fminbox)")
nPolish = 8
polish_fvals = fill(Inf, nPolish)
polish_paras = [zeros(nFit) for _ in 1:nPolish]
rngs = [MersenneTwister(4242 + i) for i in 1:nPolish]

eps_box = 1e-6 .* (paras_ub .- paras_lb)
lb_int = paras_lb .+ eps_box
ub_int = paras_ub .- eps_box

@threads for i in 1:nPolish
    if i == 1
        p0 = clamp.(de_best, lb_int, ub_int)
    else
        p0 = clamp.(de_best .+ 0.05 .* (paras_ub .- paras_lb) .* (2 .* rand(rngs[i], nFit) .- 1),
                    lb_int, ub_int)
    end
    f_local = p -> obj_fun(p, fit_names, vars_all, u0)
    try
        res = optimize(f_local, paras_lb, paras_ub, p0, Fminbox(LBFGS()),
                       Optim.Options(iterations=200, outer_iterations=5,
                                     f_reltol=1e-10, show_trace=false);
                       autodiff=:finite)
        polish_fvals[i] = Optim.minimum(res)
        polish_paras[i] = clamp.(Optim.minimizer(res), paras_lb, paras_ub)
    catch
        polish_fvals[i] = Inf
    end
    @printf("  polish %d/%d  fval=%.6g\n", i, nPolish, polish_fvals[i])
end

# Pick best between DE and polish
best_polish_idx = argmin(polish_fvals)
if polish_fvals[best_polish_idx] < de_fval
    best_paras = polish_paras[best_polish_idx]
    best_fval  = polish_fvals[best_polish_idx]
    println("Polish improved DE result.")
else
    best_paras = de_best
    best_fval  = de_fval
    println("DE result was already best.")
end

@printf("\nBest fval: %.6g\n", best_fval)
paras_after_opt = update_vars(vars_all, fit_names, best_paras)
println("Best parameters:")
for n in fit_names
    @printf("  %s = %.6g\n", n, getfield(paras_after_opt, n))
end

# ============================================================
# Plot + CSV export
# ============================================================
conds = [(0.0,0.0,0.0), (0.0,0.0,5.0), (0.0,0.5,0.0),
         (0.0,0.5,5.0), (0.5,0.0,0.0), (0.5,0.0,5.0)]
labels = ["No cells","No cells + aPD1","WT","WT + aPD1","CISH KO","CISH KO + aPD1"]

# Distinct, print-friendly palette (matches MATLAB `lines(6)` ordering feel)
palette6 = [
    RGB(0.000, 0.447, 0.741),  # blue
    RGB(0.850, 0.325, 0.098),  # orange-red
    RGB(0.929, 0.694, 0.125),  # gold
    RGB(0.494, 0.184, 0.556),  # purple
    RGB(0.466, 0.674, 0.188),  # green
    RGB(0.301, 0.745, 0.933),  # cyan
]

default(fontfamily="Helvetica", framestyle=:box, grid=false,
        guidefontsize=12, tickfontsize=11, legendfontsize=9, titlefontsize=14)

plt = plot(size=(640, 480),
           xlabel="Time (days)", ylabel="Tumor area (mm²)",
           title="Tumor Growth — Model Fit",
           ylim=(0, 600), xlim=(0, 42),
           legend=:topleft, foreground_color_legend=nothing,
           background_color_legend=RGBA(1,1,1,0.7),
           left_margin=4Plots.mm, bottom_margin=4Plots.mm)

rows = DataFrame(Time=Float64[], Experimental_Data=Float64[],
                 Model_Simulation=Float64[], Condition=String[])

# Pass 1: model curves (labeled — drive the legend)
for (i, d) in enumerate(DATASETS[])
    sol = solve_model(paras_after_opt, u0, conds[i])
    tgrid = collect(range(0.0, paras_after_opt.t_end, length=300))
    ysim = [sol(t)[1] for t in tgrid]
    plot!(plt, tgrid, ysim;
          label=labels[i], color=palette6[i], lw=2.4)
end

# Pass 2: experimental points (same color, no legend entry)
for (i, d) in enumerate(DATASETS[])
    scatter!(plt, d.t, d.v;
             label="", color=palette6[i],
             markerstrokecolor=:black, markerstrokewidth=0.6,
             markersize=5.5, markershape=:circle)

    sol_at = solve_model(paras_after_opt, u0, conds[i]; saveat=d.t)
    model_at = [sol_at.u[k][1] for k in 1:length(sol_at.u)]
    for k in 1:length(d.t)
        push!(rows, (d.t[k], d.v[k], model_at[k], labels[i]))
    end
end

savefig(plt, "mouse_fit_2022_julia.png")
CSV.write("all_results_combined_julia.csv", rows)
println("Saved plot and CSV.")
