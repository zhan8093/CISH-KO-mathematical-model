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
# Leave-one-therapy-out cross-validation for the 3 Palmer 2015
# therapies (NT, WT, CISH KO). For each therapy: fit on the other
# two, predict the held-out one, and compare with real data.
# ============================================================

vars_names = (:t_infusion, :t_end,
              :gamma, :K,
              :kL, :kLo, :kN, :muL, :muN,
              :dL, :dN, :pL, :pN,
              :delta, :h,
              :muZ, :a, :b, :bL, :kappa)

init_vals = [7.5, 42.0,
             0.157606, 1.0,
             1.1048, 0.4834, 1e-4, 1.2e-4, 4e-4,
             0.05, 0.03, 0.417132, 0.05,
             0.01, 1e3,
             0.7, 0.151213, 50, 0.01, 0.1516]

ub_vals = [0.1, 32.0,
           1.0, 10.0,
           7.2, 7.2, 7.2, 2.5e-3, 2.5e-3,
           1.0, 0.5, 2.971, 2.971,
           0.12, 1e3,
           0.7, 10.0, 50.0, 10.0, 5.0]

lb_vals = [0.1, 32.0,
           0.1, 1.0,
           0.014, 0.005, 1e-6, 5e-9, 5e-9,
           0.05, 0.001, 0.01, 0.01,
           0.12, 1e3,
           0.7, 0.001, 0.01, 0.01, 0.01]

vars_all = NamedTuple{vars_names}(Tuple(init_vals))
vars_ub  = NamedTuple{vars_names}(Tuple(ub_vals))
vars_lb  = NamedTuple{vars_names}(Tuple(lb_vals))

fit_names = (:gamma, :pL, :dL, :kL)
nFit = length(fit_names)
paras_lb = [getfield(vars_lb, n) for n in fit_names]
paras_ub = [getfield(vars_ub, n) for n in fit_names]

T0 = 40.0; L0 = 0.0; Lw0 = 0.0; N0 = 0.0; Z0 = 0.0; dL0 = 0.0; dLo0 = 0.0
u0 = [T0, L0, Lw0, N0, Z0, dL0, dLo0]

init_datasets()

labels = [d.name for d in DATASETS[]]
nTher = length(DATASETS[])

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
            total += d.w * sum(((d.v .- model_T) ./ d.scale).^2)
        end
        return isfinite(total) ? total : 1e12
    catch
        return 1e12
    end
end

function fit_fold(val_idx)
    train_idx = [i for i in 1:nTher if i != val_idx]
    println("\n================================================")
    println("Fold $(val_idx): holding out $(labels[val_idx])")
    println("================================================")

    f_obj = p -> obj_fun_subset(collect(p), fit_names, vars_all, u0, train_idx)
    _ = obj_fun_subset(Float64[(l+u)/2 for (l,u) in zip(paras_lb, paras_ub)],
                       fit_names, vars_all, u0, train_idx)

    search_range = [(paras_lb[i], paras_ub[i]) for i in 1:nFit]
    bb_res = bboptimize(f_obj;
        SearchRange = search_range,
        NumDimensions = nFit,
        Method = :adaptive_de_rand_1_bin_radiuslimited,
        MaxSteps = 20000,
        PopulationSize = 60,
        TraceMode = :silent,
    )
    de_best = best_candidate(bb_res)
    de_fval = best_fitness(bb_res)
    @printf("  DE fval: %.6g\n", de_fval)

    nPolish = 8
    polish_fvals = fill(Inf, nPolish)
    polish_paras = [zeros(nFit) for _ in 1:nPolish]
    rngs = [MersenneTwister(4242 + val_idx*100 + i) for i in 1:nPolish]

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
    end

    best_polish_idx = argmin(polish_fvals)
    if polish_fvals[best_polish_idx] < de_fval
        best_paras = polish_paras[best_polish_idx]
        best_fval  = polish_fvals[best_polish_idx]
    else
        best_paras = de_best
        best_fval  = de_fval
    end
    @printf("  Best training fval: %.6g\n", best_fval)
    return update_vars(vars_all, fit_names, best_paras), best_fval
end

# ============================================================
# Run all folds
# ============================================================
palette3 = [
    RGB(0.000, 0.447, 0.741),
    RGB(0.850, 0.325, 0.098),
    RGB(0.466, 0.674, 0.188),
]

default(fontfamily="Helvetica", framestyle=:box, grid=false,
        guidefontsize=11, tickfontsize=10, legendfontsize=9, titlefontsize=11)

fold_rmse = zeros(nTher)
fold_r2   = zeros(nTher)
subplots  = Vector{Any}(undef, nTher)

summary_rows = DataFrame(Fold=Int[], Validation=String[],
                         RMSE=Float64[], R2=Float64[], TrainFval=Float64[])
pred_rows = DataFrame(Time=Float64[], Experimental_Data=Float64[],
                      Model_Prediction=Float64[], Validation=String[], Fold=Int[])

for val_idx in 1:nTher
    paras_opt, train_fval = fit_fold(val_idx)

    d_val = DATASETS[][val_idx]
    sol_val = solve_model(paras_opt, u0, d_val.cond; saveat=d_val.t)
    pred_val = [sol_val.u[k][1] for k in 1:length(sol_val.u)]
    resid = d_val.v .- pred_val
    sse = sum(resid.^2)
    rmse = sqrt(sse / length(resid))
    ss_tot = sum((d_val.v .- mean(d_val.v)).^2)
    r2 = ss_tot > 0 ? 1 - sse / ss_tot : NaN
    fold_rmse[val_idx] = rmse
    fold_r2[val_idx]   = r2
    @printf("  Validation RMSE=%.4g  R²=%.4g\n", rmse, r2)

    push!(summary_rows, (val_idx, labels[val_idx], rmse, r2, train_fval))
    for k in 1:length(d_val.t)
        push!(pred_rows, (d_val.t[k], d_val.v[k], pred_val[k], labels[val_idx], val_idx))
    end

    tgrid = collect(range(0.0, paras_opt.t_end, length=300))
    sol_full = solve_model(paras_opt, u0, d_val.cond)
    ypred_full = [sol_full(t)[1] for t in tgrid]

    sp = plot(xlabel="Time (days)", ylabel="Tumor area (mm²)",
              title="$(labels[val_idx])\nRMSE=$(round(rmse,digits=1))  R²=$(round(r2,digits=3))",
              ylim=(0, 600), xlim=(0, 32), legend=:topleft,
              foreground_color_legend=nothing,
              background_color_legend=RGBA(1,1,1,0.7))
    plot!(sp, tgrid, ypred_full; label="Prediction",
          color=palette3[val_idx], lw=2.4)
    scatter!(sp, d_val.t, d_val.v; label="Held-out data",
             color=palette3[val_idx], markerstrokecolor=:black,
             markerstrokewidth=0.6, markersize=5.5, markershape=:star5)
    subplots[val_idx] = sp

    savefig(sp, "crossval_validation_$(val_idx).png")
end

combined = plot(subplots..., layout=(1,3), size=(1500, 450),
                plot_title="Palmer 2015 — Leave-one-therapy-out cross-validation",
                left_margin=4Plots.mm, bottom_margin=4Plots.mm)
savefig(combined, "crossval_all_folds.png")

CSV.write("crossval_summary.csv", summary_rows)
CSV.write("crossval_predictions.csv", pred_rows)

println("\n================================================")
println("Summary")
println("================================================")
for row in eachrow(summary_rows)
    @printf("  Fold %d  %-10s  RMSE=%8.3f  R²=%7.4f\n",
            row.Fold, row.Validation, row.RMSE, row.R2)
end
@printf("  Mean RMSE=%.3f  Mean R²=%.4f\n", mean(fold_rmse), mean(fold_r2))

println("\nSaved:")
println("  crossval_validation_1..$(nTher).png")
println("  crossval_all_folds.png")
println("  crossval_summary.csv")
println("  crossval_predictions.csv")
