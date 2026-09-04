using Base.Threads
using BlackBoxOptim
using Optim
using Random
using Printf
using DataFrames, CSV
using Plots
using Statistics

include("cish_model.jl")
include("cish_obj.jl")

# ============================================================
# Leave-one-therapy-out cross-validation
# Set val_idx to the index (1..6) of the therapy to hold out.
# ============================================================
val_idx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4  # default: WT + aPD1

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

fit_names = (:gamma, :kL, :kLo, :kN, :pL, :dL, :a, :b, :bL)
nFit = length(fit_names)
paras_lb = [getfield(vars_lb, n) for n in fit_names]
paras_ub = [getfield(vars_ub, n) for n in fit_names]

T0 = 25.0; L0 = 0.0; Lw0 = 0.0; N0 = 0.0; Z0 = 0.0; dL0 = 0.0; dLo0 = 0.0
u0 = [T0, L0, Lw0, N0, Z0, dL0, dLo0]

init_datasets()

conds = [(0.0,0.0,0.0), (0.0,0.0,5.0), (0.0,0.5,0.0),
         (0.0,0.5,5.0), (0.5,0.0,0.0), (0.5,0.0,5.0)]
labels = ["No cells","No cells + aPD1","WT","WT + aPD1","CISH KO","CISH KO + aPD1"]

@assert 1 <= val_idx <= length(labels)
train_idx = [i for i in 1:length(DATASETS[]) if i != val_idx]
println("Validation therapy: ", labels[val_idx], "  (idx=", val_idx, ")")
println("Training therapies: ", join(labels[train_idx], ", "))

# Objective restricted to training subset
function obj_fun_subset(paras, fit_names, vars_all, u0, idxs)
    try
        vars_new = update_vars(vars_all, fit_names, paras)
        total = 0.0
        for i in idxs
            d = DATASETS[][i]
            sol = solve_model(vars_new, u0, d.cond; saveat=d.t)
            if sol.retcode != ReturnCode.Success
                return 1e12
            end
            model_T = [sol.u[k][1] for k in 1:length(sol.u)]
            if length(model_T) != length(d.v)
                return 1e12
            end
            total += d.w * sum((d.v .- model_T).^2)
        end
        return isfinite(total) ? total : 1e12
    catch
        return 1e12
    end
end

# Warm up compiled solver
_ = obj_fun_subset(Float64[(l+u)/2 for (l,u) in zip(paras_lb, paras_ub)],
                   fit_names, vars_all, u0, train_idx)

f_obj = p -> obj_fun_subset(collect(p), fit_names, vars_all, u0, train_idx)

# ============================================================
# STAGE 1: Differential Evolution global search
# ============================================================
println("\nStage 1: Differential Evolution global search")
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
# STAGE 2: Multi-start local polish
# ============================================================
println("\nStage 2: Local polish with LBFGS (Fminbox)")
nPolish = 8
polish_fvals = fill(Inf, nPolish)
polish_paras = [zeros(nFit) for _ in 1:nPolish]
rngs = [MersenneTwister(4242 + i) for i in 1:nPolish]

@threads for i in 1:nPolish
    if i == 1
        p0 = copy(de_best)
    else
        p0 = clamp.(de_best .+ 0.05 .* (paras_ub .- paras_lb) .* (2 .* rand(rngs[i], nFit) .- 1),
                    paras_lb, paras_ub)
    end
    f_local = p -> obj_fun_subset(p, fit_names, vars_all, u0, train_idx)
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

@printf("\nBest training fval: %.6g\n", best_fval)
paras_after_opt = update_vars(vars_all, fit_names, best_paras)
println("Best parameters:")
for n in fit_names
    @printf("  %s = %.6g\n", n, getfield(paras_after_opt, n))
end

# ============================================================
# Validation metrics on held-out therapy
# ============================================================
d_val = DATASETS[][val_idx]
sol_val = solve_model(paras_after_opt, u0, conds[val_idx]; saveat=d_val.t)
pred_val = [sol_val.u[k][1] for k in 1:length(sol_val.u)]
resid = d_val.v .- pred_val
sse = sum(resid.^2)
rmse = sqrt(sse / length(resid))
ss_tot = sum((d_val.v .- mean(d_val.v)).^2)
r2 = 1 - sse / ss_tot
@printf("\nValidation (%s):  RMSE=%.4g  R²=%.4g\n", labels[val_idx], rmse, r2)

# ============================================================
# Plot
# ============================================================
palette6 = [
    RGB(0.000, 0.447, 0.741),
    RGB(0.850, 0.325, 0.098),
    RGB(0.929, 0.694, 0.125),
    RGB(0.494, 0.184, 0.556),
    RGB(0.466, 0.674, 0.188),
    RGB(0.301, 0.745, 0.933),
]

default(fontfamily="Helvetica", framestyle=:box, grid=false,
        guidefontsize=12, tickfontsize=11, legendfontsize=9, titlefontsize=13)

# Plot 1: all therapies — training curves solid, validation dashed
plt1 = plot(size=(720, 520),
            xlabel="Time (days)", ylabel="Tumor area (mm²)",
            title="Cross-validation — hold out: $(labels[val_idx])",
            ylim=(0, 600), xlim=(0, 42),
            legend=:topleft, foreground_color_legend=nothing,
            background_color_legend=RGBA(1,1,1,0.7),
            left_margin=4Plots.mm, bottom_margin=4Plots.mm)

for (i, d) in enumerate(DATASETS[])
    sol = solve_model(paras_after_opt, u0, conds[i])
    tgrid = collect(range(0.0, paras_after_opt.t_end, length=300))
    ysim = [sol(t)[1] for t in tgrid]
    ls = (i == val_idx) ? :dash : :solid
    lab = (i == val_idx) ? labels[i]*" (validation)" : labels[i]*" (train)"
    plot!(plt1, tgrid, ysim; label=lab, color=palette6[i], lw=2.4, linestyle=ls)
end
for (i, d) in enumerate(DATASETS[])
    mshape = (i == val_idx) ? :star5 : :circle
    msize  = (i == val_idx) ? 7.5 : 5.5
    scatter!(plt1, d.t, d.v; label="", color=palette6[i],
             markerstrokecolor=:black, markerstrokewidth=0.6,
             markersize=msize, markershape=mshape)
end

savefig(plt1, "crossval_all_$(val_idx).png")

# Plot 2: zoom on validation therapy — data vs prediction
tgrid = collect(range(0.0, paras_after_opt.t_end, length=300))
sol_full = solve_model(paras_after_opt, u0, conds[val_idx])
ypred_full = [sol_full(t)[1] for t in tgrid]

plt2 = plot(size=(640, 480),
            xlabel="Time (days)", ylabel="Tumor area (mm²)",
            title="Validation: $(labels[val_idx])  RMSE=$(round(rmse,digits=2))  R²=$(round(r2,digits=3))",
            legend=:topleft, foreground_color_legend=nothing,
            background_color_legend=RGBA(1,1,1,0.7),
            left_margin=4Plots.mm, bottom_margin=4Plots.mm)
plot!(plt2, tgrid, ypred_full; label="Model prediction",
      color=palette6[val_idx], lw=2.6)
scatter!(plt2, d_val.t, d_val.v; label="Held-out data",
         color=palette6[val_idx], markerstrokecolor=:black,
         markerstrokewidth=0.6, markersize=6.5, markershape=:star5)

savefig(plt2, "crossval_validation_$(val_idx).png")

# ============================================================
# CSV export
# ============================================================
rows = DataFrame(Time=Float64[], Experimental_Data=Float64[],
                 Model_Simulation=Float64[], Condition=String[], Role=String[])
for (i, d) in enumerate(DATASETS[])
    sol_at = solve_model(paras_after_opt, u0, conds[i]; saveat=d.t)
    model_at = [sol_at.u[k][1] for k in 1:length(sol_at.u)]
    role = (i == val_idx) ? "validation" : "train"
    for k in 1:length(d.t)
        push!(rows, (d.t[k], d.v[k], model_at[k], labels[i], role))
    end
end
CSV.write("crossval_results_$(val_idx).csv", rows)

println("\nSaved:")
println("  crossval_all_$(val_idx).png")
println("  crossval_validation_$(val_idx).png")
println("  crossval_results_$(val_idx).csv")
