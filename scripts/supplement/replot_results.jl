# replot_results.jl
# Publication-quality replotting of CISH model fitting results
# Loads optimal parameters from saved .json files and regenerates figures.

using CSV, DataFrames, JSON3, XLSX, Plots, StaticArrays, Printf, Statistics

# Use GR backend for high-quality output
gr()

# ── Include model code ────────────────────────────────────────────
include("cish_model_fun_optimized.jl")

# ── CISHParams helpers (from main_optimized.jl) ──────────────────
function CISHParams(d::Dict{Symbol, Float64})
    CISHParams(
        d[:t_infusion], d[:t_end], d[:gamma], d[:lambda], d[:s], d[:K],
        d[:kL], d[:kN], d[:muL], d[:muN], d[:dL], d[:dN],
        d[:pL], d[:pN], d[:g], d[:delta], d[:h], d[:l],
        d[:n], d[:f], d[:f_dose], d[:kappa]
    )
end

function params_to_dict(p::CISHParams)
    Dict{Symbol,Float64}(
        :t_infusion=>p.t_infusion, :t_end=>p.t_end, :gamma=>p.gamma,
        :lambda=>p.lambda, :s=>p.s, :K=>p.K, :kL=>p.kL, :kN=>p.kN,
        :muL=>p.muL, :muN=>p.muN, :dL=>p.dL, :dN=>p.dN,
        :pL=>p.pL, :pN=>p.pN, :g=>p.g, :delta=>p.delta,
        :h=>p.h, :l=>p.l, :n=>p.n, :f=>p.f, :f_dose=>p.f_dose, :kappa=>p.kappa
    )
end

function update_params(base::CISHParams, param_names::Vector{String}, param_values)
    d = params_to_dict(base)
    for (name, val) in zip(param_names, param_values)
        d[Symbol(name)] = Float64(val)
    end
    return CISHParams(d)
end

# ──────────────────────────────────────────────────────────────────
#  Global style settings
# ──────────────────────────────────────────────────────────────────

# Colorblind-friendly palette for tumors (Wong palette)
const TUMOR_PALETTE = [
    RGB(0/255, 114/255, 178/255),   # blue
    RGB(213/255, 94/255, 0/255),    # vermillion
    RGB(0/255, 158/255, 115/255),   # bluish green
    RGB(204/255, 121/255, 167/255), # reddish purple
    RGB(230/255, 159/255, 0/255),   # orange
    RGB(86/255, 180/255, 233/255),  # sky blue
    RGB(240/255, 228/255, 66/255),  # yellow
]

const MODEL_LW      = 2.5          # line width for model curves
const DATA_MS       = 6            # marker size for data points
const DATA_MSW      = 1.0          # marker stroke width
const RATIO_LW      = 3.0          # line width for ratio curve
const PANEL_SIZE    = (380, 300)   # individual panel size
const FIG_SIZE      = (900, 750)   # 2×2 figure size
const FONT_SIZE     = 11
const TITLE_SIZE    = 13
const TICK_SIZE     = 9
const LEGEND_SIZE   = 8
const DPI           = 300

# Default plot attributes
default(
    fontfamily   = "Helvetica",
    guidefontsize  = FONT_SIZE,
    titlefontsize  = TITLE_SIZE,
    tickfontsize   = TICK_SIZE,
    legendfontsize = LEGEND_SIZE,
    grid           = true,
    gridstyle      = :dot,
    gridalpha      = 0.3,
    framestyle     = :box,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.85),
    legend_font_halign = :left,
    dpi            = DPI,
)

# ──────────────────────────────────────────────────────────────────
#  Helper: reconstruct CISHParams from JSON data
# ──────────────────────────────────────────────────────────────────
function reconstruct_params(json_data, patient_idx::Int)
    vars_names = json_data[:vars_names]
    init_vals  = json_data[:init_vals]

    # Build base dictionary
    vars_all = Dict(Symbol(n) => Float64(v) for (n, v) in zip(vars_names, init_vals))

    # Patient-specific t_end overrides (from main_optimized.jl)
    t_end_overrides = Dict(
        4  => 45.0,
        5  => 60.0,
        6  => 30.0,
        8  => 40.0,
        9  => 30.0,
        10 => 60.0,
        11 => 30.0,
        12 => 180.0,
    )
    if haskey(t_end_overrides, patient_idx)
        vars_all[:t_end] = t_end_overrides[patient_idx]
    end

    # Apply fitted patient-level parameters
    fit_names  = json_data[:vars_fit_name_patient]
    fit_values = json_data[:paras_fit_patient]
    for (n, v) in zip(fit_names, fit_values)
        vars_all[Symbol(n)] = Float64(v)
    end

    return CISHParams(vars_all), vars_all
end

# ──────────────────────────────────────────────────────────────────
#  Load data
# ──────────────────────────────────────────────────────────────────
function load_all_data()
    # Tumor size data
    tumor_data  = XLSX.readtable("CISH_pt_data.xlsx", "Sheet1")
    tumor_table = DataFrame(tumor_data)

    # CISH ratio data
    cish_ratio_table = CSV.read("CISHKO_Pct_result.csv", DataFrame)

    # Dose data
    dose_data_raw = XLSX.readtable("cish_pt_doses.xlsx", "Sheet1")
    dose_data     = DataFrame(dose_data_raw)

    patient_nums_tumor = string.(tumor_table[:, 1])
    tumors_nums        = string.(tumor_table[:, 4])
    days_nums          = tumor_table[:, 3]
    tumor_size_nums    = tumor_table[:, 5]

    patient_nums_cish  = string.(cish_ratio_table[:, 1])
    days_nums_cish     = cish_ratio_table[:, 2]
    cish_nums_ratio    = cish_ratio_table[:, 3]

    patients_num_dose  = string.(dose_data[:, 1])
    cell_dose          = dose_data[:, 2]
    il2_doses          = dose_data[:, 3]

    id_patient = unique(patient_nums_tumor)

    return (;
        patient_nums_tumor, tumors_nums, days_nums, tumor_size_nums,
        patient_nums_cish, days_nums_cish, cish_nums_ratio,
        patients_num_dose, cell_dose, il2_doses,
        id_patient
    )
end

# ──────────────────────────────────────────────────────────────────
#  Extract patient data (tumor sizes, CISH ratios, doses, u0)
# ──────────────────────────────────────────────────────────────────
function extract_patient_data(data, i::Int)
    id = data.id_patient[i]

    # Tumor data
    inds_tumor = findall(data.patient_nums_tumor .== id)
    id_tumor   = unique(data.tumors_nums[inds_tumor])

    tumors_size = zeros(length(inds_tumor), length(id_tumor) * 2)
    for (j, tid) in enumerate(id_tumor)
        inds = findall((data.patient_nums_tumor .== id) .& (data.tumors_nums .== tid))
        tumors_size[1:length(inds), 2j-1:2j] .= hcat(data.days_nums[inds], data.tumor_size_nums[inds])
    end

    # CISH ratio data
    inds_cish  = findall(data.patient_nums_cish .== id)
    cish_days  = data.days_nums_cish[inds_cish]
    ratio_data = data.cish_nums_ratio[inds_cish]
    cish_ratio = hcat(cish_days, ratio_data)
    cish_ratio = cish_ratio[cish_ratio[:, 1] .>= 0, :]

    # Dose
    inds_dose = findall(data.patients_num_dose .== id)
    dose_val  = data.cell_dose[inds_dose][1]

    N0    = 0.1
    L0    = 0.0
    Dose0 = dose_val / 1e10
    u0    = SVector{4, Float64}(0.0, L0, N0, Dose0)

    return (; tumors_size, cish_ratio, id_tumor, u0, id)
end

# ──────────────────────────────────────────────────────────────────
#  Plot one patient – 4-panel figure
# ──────────────────────────────────────────────────────────────────
function plot_patient(data, i::Int; save_dir::String = "figures")
    mkpath(save_dir)

    # Load JSON
    id = data.id_patient[i]
    json_path = "Patient_$(id)_paras.json"
    if !isfile(json_path)
        @warn "JSON not found: $json_path – skipping patient $i ($id)"
        return nothing
    end
    json_data = JSON3.read(read(json_path, String))

    # Reconstruct parameters
    params_patient, _ = reconstruct_params(json_data, i)

    # Patient data
    pd = extract_patient_data(data, i)

    # Best tumor-level parameters from JSON
    best_paras_tumors = json_data[:best_paras_tumors]   # vector of vectors
    vars_fit_name_tumor = json_data[:vars_fit_name_tumor]

    n_tumors = size(pd.tumors_size, 2) ÷ 2

    # ── Create 2×2 figure ──
    fig = plot(
        layout = (2, 2),
        size   = FIG_SIZE,
        margin = 6Plots.mm,
        left_margin   = 8Plots.mm,
        bottom_margin = 7Plots.mm,
        top_margin    = 5Plots.mm,
    )

    # Storage for average ratio calculation
    sum_L = nothing
    sum_N = nothing
    common_time = nothing
    valid_tumor_count = 0

    for k in 1:n_tumors
        # Extract tumor data, filter out (0,0) padding
        tumor_size = pd.tumors_size[:, 2k-1:2k]
        mask = .!((tumor_size[:, 1] .== 0.0) .& (tumor_size[:, 2] .== 0.0))
        tumor_size = tumor_size[mask, :]
        size(tumor_size, 1) == 0 && continue

        # Tumor-specific initial condition
        u0_tumor = SVector{4, Float64}(tumor_size[1, 2], 0.0, pd.u0[3], pd.u0[4])

        # Get tumor-specific gamma
        if k <= length(best_paras_tumors)
            paras_tumor = Float64.(best_paras_tumors[k])
        else
            paras_tumor = Float64.(best_paras_tumors[end])
        end
        params_tumor = update_params(params_patient, Vector{String}(vars_fit_name_tumor), paras_tumor)

        # Solve ODE
        sol = cish_model_fun_optimized(params_tumor, u0_tumor)

        col = TUMOR_PALETTE[(k - 1) % length(TUMOR_PALETTE) + 1]

        # ── Panel 1: Tumor size ──
        plot!(fig[1], sol.x, sol.y[1, :],
              label = "Tumor $k (model)", lw = MODEL_LW, color = col,
              linestyle = :solid)
        scatter!(fig[1], tumor_size[:, 1], tumor_size[:, 2],
                 label = "Tumor $k (data)", ms = DATA_MS, msw = DATA_MSW,
                 color = col, markershape = :circle,
                 markerstrokecolor = col)

        # ── Panel 2: WildType NK cells (N) ──
        plot!(fig[2], sol.x, sol.y[3, :],
              label = "Tumor $k", lw = MODEL_LW, color = col)

        # ── Panel 3: CISHKO cells (L) ──
        plot!(fig[3], sol.x, sol.y[2, :],
              label = "Tumor $k", lw = MODEL_LW, color = col)

        # Accumulate for average ratio
        if sum_L === nothing
            common_time = Vector{Float64}(sol.x)
            sum_L = Vector{Float64}(sol.y[2, :])
            sum_N = Vector{Float64}(sol.y[3, :])
        else
            L_interp = [sol.y[2, argmin(abs.(sol.x .- t))] for t in common_time]
            N_interp = [sol.y[3, argmin(abs.(sol.x .- t))] for t in common_time]
            sum_L .+= L_interp
            sum_N .+= N_interp
        end
        valid_tumor_count += 1
    end

    # ── Panel 4: L / N ratio ──
    if valid_tumor_count > 0 && sum_L !== nothing
        avg_L = sum_L ./ valid_tumor_count
        avg_N = sum_N ./ valid_tumor_count
        avg_ratio = avg_L ./ max.(avg_N, 1e-12)

        plot!(fig[4], common_time, avg_ratio,
              label = "Model (avg)", lw = RATIO_LW, color = :black,
              linestyle = :solid)
    end

    if size(pd.cish_ratio, 1) > 0
        scatter!(fig[4], pd.cish_ratio[:, 1], pd.cish_ratio[:, 2],
                 label = "Data", ms = DATA_MS + 1, msw = DATA_MSW,
                 color = RGB(0.85, 0.33, 0.1),
                 markershape = :diamond,
                 markerstrokecolor = RGB(0.65, 0.2, 0.05))
        y_max = 2.0 * maximum(pd.cish_ratio[:, 2])
        ylims!(fig[4], (0, y_max))
    end

    # ── Axis labels & titles ──
    panel_labels = ["A", "B", "C", "D"]
    titles       = ["Tumor Size", "Endogenous T-Cells (N)", "CISH-KO T-Cells (L)", "CISH-KO / Endogenous Ratio"]
    ylabels      = ["Tumor size (mm²)", "Cell count", "Cell count", "L / N ratio"]

    for (p, lbl, ttl, ylab) in zip(1:4, panel_labels, titles, ylabels)
        xlabel!(fig[p], "Days post infusion")
        ylabel!(fig[p], ylab)
        title!(fig[p],  "$lbl.  $ttl")
    end

    # Overall title
    obj_val = json_data[:best_objective]
    plot!(fig, plot_title = "Patient $(pd.id)   (obj = $(@sprintf("%.4f", obj_val)))",
          plot_titlefontsize = 14)

    # Save
    fname = joinpath(save_dir, "Patient_$(pd.id)_fit.png")
    savefig(fig, fname)
    println("  Saved: $fname")

    # Also save as PDF for publication
    fname_pdf = joinpath(save_dir, "Patient_$(pd.id)_fit.pdf")
    savefig(fig, fname_pdf)

    return fig
end

# ──────────────────────────────────────────────────────────────────
#  Summary figure: all 12 patients – tumor fits in one grid
# ──────────────────────────────────────────────────────────────────
function plot_summary_tumor(data; save_dir::String = "figures")
    mkpath(save_dir)
    n_patients = length(data.id_patient)
    n_rows = ceil(Int, n_patients / 4)

    fig = plot(
        layout = (n_rows, 4),
        size   = (1400, 320 * n_rows),
        margin = 4Plots.mm,
        left_margin   = 6Plots.mm,
        bottom_margin = 6Plots.mm,
        top_margin    = 4Plots.mm,
    )

    for i in 1:n_patients
        id = data.id_patient[i]
        json_path = "Patient_$(id)_paras.json"
        if !isfile(json_path)
            @warn "JSON not found: $json_path – skipping"
            continue
        end
        json_data = JSON3.read(read(json_path, String))
        params_patient, _ = reconstruct_params(json_data, i)
        pd = extract_patient_data(data, i)
        best_paras_tumors = json_data[:best_paras_tumors]
        vars_fit_name_tumor = json_data[:vars_fit_name_tumor]
        n_tumors = size(pd.tumors_size, 2) ÷ 2

        for k in 1:n_tumors
            tumor_size = pd.tumors_size[:, 2k-1:2k]
            mask = .!((tumor_size[:, 1] .== 0.0) .& (tumor_size[:, 2] .== 0.0))
            tumor_size = tumor_size[mask, :]
            size(tumor_size, 1) == 0 && continue

            u0_tumor = SVector{4, Float64}(tumor_size[1, 2], 0.0, pd.u0[3], pd.u0[4])
            paras_tumor = k <= length(best_paras_tumors) ?
                Float64.(best_paras_tumors[k]) : Float64.(best_paras_tumors[end])
            params_tumor = update_params(params_patient, Vector{String}(vars_fit_name_tumor), paras_tumor)
            sol = cish_model_fun_optimized(params_tumor, u0_tumor)

            col = TUMOR_PALETTE[(k - 1) % length(TUMOR_PALETTE) + 1]
            plot!(fig[i], sol.x, sol.y[1, :],
                  label = (k == 1 ? "" : ""), lw = 2.0, color = col)
            scatter!(fig[i], tumor_size[:, 1], tumor_size[:, 2],
                     label = "", ms = 4, msw = 0.8,
                     color = col, markerstrokecolor = col)
        end

        xlabel!(fig[i], "Days")
        ylabel!(fig[i], "Size (mm²)")
        title!(fig[i], "$id", titlefontsize = 10)
    end

    fname = joinpath(save_dir, "Summary_Tumor_Fits.png")
    savefig(fig, fname)
    println("  Saved: $fname")
    savefig(fig, joinpath(save_dir, "Summary_Tumor_Fits.pdf"))
    return fig
end

# ──────────────────────────────────────────────────────────────────
#  Summary figure: all 12 patients – L/N ratio in one grid
# ──────────────────────────────────────────────────────────────────
function plot_summary_ratio(data; save_dir::String = "figures")
    mkpath(save_dir)
    n_patients = length(data.id_patient)
    n_rows = ceil(Int, n_patients / 4)

    fig = plot(
        layout = (n_rows, 4),
        size   = (1400, 320 * n_rows),
        margin = 4Plots.mm,
        left_margin   = 6Plots.mm,
        bottom_margin = 6Plots.mm,
        top_margin    = 4Plots.mm,
    )

    for i in 1:n_patients
        id = data.id_patient[i]
        json_path = "Patient_$(id)_paras.json"
        if !isfile(json_path)
            continue
        end
        json_data = JSON3.read(read(json_path, String))
        params_patient, _ = reconstruct_params(json_data, i)
        pd = extract_patient_data(data, i)
        best_paras_tumors = json_data[:best_paras_tumors]
        vars_fit_name_tumor = json_data[:vars_fit_name_tumor]
        n_tumors = size(pd.tumors_size, 2) ÷ 2

        sum_L = nothing
        sum_N = nothing
        common_time = nothing
        valid_tumor_count = 0

        for k in 1:n_tumors
            tumor_size = pd.tumors_size[:, 2k-1:2k]
            mask = .!((tumor_size[:, 1] .== 0.0) .& (tumor_size[:, 2] .== 0.0))
            tumor_size = tumor_size[mask, :]
            size(tumor_size, 1) == 0 && continue

            u0_tumor = SVector{4, Float64}(tumor_size[1, 2], 0.0, pd.u0[3], pd.u0[4])
            paras_tumor = k <= length(best_paras_tumors) ?
                Float64.(best_paras_tumors[k]) : Float64.(best_paras_tumors[end])
            params_tumor = update_params(params_patient, Vector{String}(vars_fit_name_tumor), paras_tumor)
            sol = cish_model_fun_optimized(params_tumor, u0_tumor)

            if sum_L === nothing
                common_time = Vector{Float64}(sol.x)
                sum_L = Vector{Float64}(sol.y[2, :])
                sum_N = Vector{Float64}(sol.y[3, :])
            else
                sum_L .+= [sol.y[2, argmin(abs.(sol.x .- t))] for t in common_time]
                sum_N .+= [sol.y[3, argmin(abs.(sol.x .- t))] for t in common_time]
            end
            valid_tumor_count += 1
        end

        # Plot average ratio
        if valid_tumor_count > 0 && sum_L !== nothing
            avg_ratio = (sum_L ./ valid_tumor_count) ./ max.(sum_N ./ valid_tumor_count, 1e-12)
            plot!(fig[i], common_time, avg_ratio,
                  label = "Model", lw = 2.5, color = :black)
        end

        # CISH ratio data
        if size(pd.cish_ratio, 1) > 0
            scatter!(fig[i], pd.cish_ratio[:, 1], pd.cish_ratio[:, 2],
                     label = "Data", ms = 5, msw = 0.8,
                     color = RGB(0.85, 0.33, 0.1),
                     markershape = :diamond,
                     markerstrokecolor = RGB(0.65, 0.2, 0.05))
            ylims!(fig[i], (0, 2.0 * maximum(pd.cish_ratio[:, 2])))
        end

        xlabel!(fig[i], "Days")
        ylabel!(fig[i], "L / N")
        title!(fig[i], "$id", titlefontsize = 10)
    end

    fname = joinpath(save_dir, "Summary_Ratio_Fits.png")
    savefig(fig, fname)
    println("  Saved: $fname")
    savefig(fig, joinpath(save_dir, "Summary_Ratio_Fits.pdf"))
    return fig
end

# ──────────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────────
function replot_all(; save_dir = "figures")
    println("="^60)
    println("  Replotting CISH Model Fitting Results")
    println("="^60)

    data = load_all_data()
    n_patients = length(data.id_patient)
    println("Found $n_patients patients: ", join(data.id_patient, ", "))

    # Individual 4-panel figures
    println("\n── Individual patient figures ──")
    for i in 1:n_patients
        print("Patient $i / $n_patients ($(data.id_patient[i])): ")
        plot_patient(data, i; save_dir = save_dir)
    end

    # Summary grids
    println("\n── Summary figures ──")
    plot_summary_tumor(data; save_dir = save_dir)
    plot_summary_ratio(data; save_dir = save_dir)

    println("\n" * "="^60)
    println("  Done! All figures saved to: $save_dir/")
    println("="^60)
end

# Run
replot_all()
