"""
schedule_heatmap.jl

Reads schedule_sweep/metrics.csv and produces two heatmap figures:
  schedule_sweep/heatmap_ratio.png  — end/start tumor ratio
  schedule_sweep/heatmap_auc.png    — tumor AUC over [0,180]

Cells are colored by % difference from each patient's 1x bolus baseline
(blue = schedule outperforms single bolus; red = worse). Cells are
annotated with the absolute metric value. Winner per row is prefixed
with a star.
"""

using CSV, DataFrames, Plots, Printf
using Plots.PlotMeasures

include("schedule_plot_style.jl")
apply_cish_plot_style!(size = (1050, 720))

const CSV_PATH = "schedule_sweep/metrics.csv"
const OUT_DIR  = "schedule_sweep"

const SCHEDULE_ORDER = ["1x@0", "2x@0,14", "2x@0,28", "2x@0,56",
                        "3x@0,14,28", "3x@0,28,56",
                        "4x@0,14,28,42", "4x@0,14,28,56"]

function pivot(df::DataFrame, metric_col::Symbol)
    patients = unique(df.patient)
    M = zeros(length(patients), length(SCHEDULE_ORDER))
    for (i, p) in enumerate(patients), (j, s) in enumerate(SCHEDULE_ORDER)
        sub = df[(df.patient .== p) .& (df.schedule .== s), :]
        if !isempty(sub)
            M[i, j] = sub[1, metric_col]
        end
    end
    return patients, M
end

function rel_to_baseline(M::Matrix{Float64})
    R = similar(M)
    for i in axes(M, 1)
        base = M[i, 1]  # 1x@0 is column 1
        for j in axes(M, 2)
            R[i, j] = base == 0 ? 0.0 : (M[i, j] - base) / base * 100
        end
    end
    return R
end

function make_heatmap(M::Matrix{Float64}, R::Matrix{Float64},
                      patients::Vector, title_text::String, fname::String;
                      fmt_fn::Function = v -> @sprintf("%.3f", v))
    nx = length(SCHEDULE_ORDER)
    ny = length(patients)

    max_abs = max(abs(minimum(R)), abs(maximum(R)))
    clim = max(max_abs, 1.0)

    plt = heatmap(
        1:nx, 1:ny, R,
        c = cgrad(:RdBu, rev=true),
        clim = (-clim, clim),
        xticks = (1:nx, SCHEDULE_ORDER),
        yticks = (1:ny, patients),
        xrotation = 25,
        yflip = true,
        xlabel = "Dosing schedule",
        ylabel = "Patient",
        title  = title_text,
        titlefontsize = 12,
        colorbar_title = "% diff vs 1x",
        framestyle = :box,
        bottom_margin = 14mm,
        left_margin   = 10mm,
        right_margin  = 6mm,
        top_margin    = 6mm,
    )

    # Pre-compute winners per row (smaller is better for both metrics)
    winners = [argmin(M[i, :]) for i in 1:ny]

    for i in 1:ny, j in 1:nx
        is_win = (j == winners[i])
        val_str = fmt_fn(M[i, j])
        label = is_win ? "* " * val_str : val_str
        intensity = abs(R[i, j]) / clim
        txt_color = intensity > 0.55 ? :white : :black
        fontsize = is_win ? 8 : 7
        annotate!(plt, j, i, text(label, fontsize, txt_color))
    end

    isdir(OUT_DIR) || mkpath(OUT_DIR)
    out = joinpath(OUT_DIR, fname)
    save_png_pdf(plt, out)
    println("saved: $out")
end

function main()
    df = CSV.read(CSV_PATH, DataFrame)

    patients, M_ratio = pivot(df, :ratio)
    R_ratio = rel_to_baseline(M_ratio)
    make_heatmap(M_ratio, R_ratio, patients,
                 "End/start tumor ratio  (lower = better)", "heatmap_ratio.png";
                 fmt_fn = v -> @sprintf("%.3f", v))

    patients, M_auc = pivot(df, :AUC_T)
    R_auc = rel_to_baseline(M_auc)
    make_heatmap(M_auc, R_auc, patients,
                 "Tumor AUC over [0,180 d]  (lower = better)", "heatmap_auc.png";
                 fmt_fn = v -> @sprintf("%.0f", v))
end

main()
