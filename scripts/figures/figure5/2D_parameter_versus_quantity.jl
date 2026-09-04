using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
using GLM
using Random
using MultivariateStats
using Distributions
using Printf

# ── Style setup (consistent with Figure 4 / plot_mouse_params_v2.py) ──
gr(fontfamily = "Computer Modern")
default(
    framestyle  = :axes,       # only left + bottom spines
    grid        = true,
    gridalpha   = 0.3,
    gridstyle   = :dash,
    guidefontsize  = 12,
    tickfontsize   = 10,
    titlefontsize  = 11,
    legendfontsize = 10,
    background_color = :white,
    foreground_color = :black,
    dpi  = 300,
    size = (520, 420),
)

# Color palette (matching Figure 4)
const C_BLUE  = RGB(0.180, 0.525, 0.671)   # #2E86AB steel blue
const C_RED   = RGB(0.910, 0.333, 0.227)   # #E8553A warm red
const C_GREEN = RGB(0.498, 0.690, 0.412)   # #7FB069 muted green

# Parameters to plot on a log scale
const LOG_PARAMS = Set([:dL, :kL, :pL, :muL, :pN])

function load_data(filepath)
    df = CSV.read(filepath, DataFrame)

    # Required ODE-derived metrics columns
    required_cols = ["tumor_ratio", "L_max", "L_AUC", "T_AUC_normalized"]
    name_strings = string.(names(df))
    
    # Check for required columns and standardize names
    for col in required_cols
        # Try different name variations
        candidates = [col, uppercase(col), lowercase(col), replace(col, "_" => "")]
        idx = findfirst(n -> n in candidates, name_strings)
        
        if idx === nothing
            error("Missing required column $col. Available columns: $(name_strings)")
        end
        
        # Standardize column name
        present_name = names(df)[idx]
        target_name = Symbol(col)
        if present_name != target_name
            rename!(df, present_name => target_name)
        end
        
        # Ensure numeric
        try
            df[!, target_name] = Float64.(df[!, target_name])
        catch
            df[!, target_name] = parse.(Float64, string.(df[!, target_name]))
        end
    end

    return df
end





# Remove 2D plotting function - not needed

# Remove 2D plotting function - not needed

# ---------- Generalized parameter correlation analysis ----------
function create_param_vs_metric_scatters(df, metric_col::Symbol, metric_name::String)
    params_to_plot = [:dL, :kL, :pL, :muL, :pN, :f_dose, :kappa, :gamma]
    labels = Dict(
        :dL     => "Death rate dL",
        :kL     => "Killing rate kL",
        :pL     => "Proliferation rate pL",
        :muL    => "Exhaustion rate μL",
        :pN     => "Endogenous proliferation pN",
        :f_dose => "Dose elimination f_dose",
        :kappa  => "Dose transfer rate κ",
        :gamma  => "Tumor growth rate γ"
    )

    plots = Dict{Symbol, Any}()
    correlations = Dict{Symbol, NamedTuple}()  # Store correlation results

    # robust clims for color mapping
    r = skipmissing(df[!, metric_col]); lo, hi = quantile(collect(r), (0.02, 0.98)); clims = (lo, hi)

    println("\n" * "="^60)
    println("PARAMETER-$(uppercase(metric_name)) CORRELATIONS")
    println("="^60)
    println(@sprintf("%-20s %10s %10s %10s", "Parameter", "Pearson r", "P-value", "N"))
    println("-"^60)

    for param in params_to_plot
        x = df[!, metric_col]
        y = df[:, param]
        # mask missing values
        mask = .!ismissing.(x) .& .!ismissing.(y)
        if param in LOG_PARAMS
            mask .= mask .& (Float64.(y) .> 0)
        end
        kept = count(mask)
        total = length(x)
        if kept == 0
            @warn "No valid data to plot $(param) vs tumor_ratio (all missing or non-positive). Skipping."
            continue
        elseif kept < total
            @warn "Filtered $(total - kept) rows with missing or non-positive values when plotting $(param) vs tumor_ratio."
        end

        # Extract valid data for correlation analysis
        x_valid = Float64.(x[mask])
        y_valid = Float64.(y[mask])
        
        # Calculate correlation with raw parameter values
        if length(x_valid) >= 3  # Need at least 3 points for correlation
            pearson_r = cor(x_valid, y_valid)
            
            # Simple t-test for correlation significance
            n = length(x_valid)
            t_stat = pearson_r * sqrt((n-2)/(1-pearson_r^2))
            # Approximate p-value (two-tailed)
            p_value = 2 * (1 - cdf(TDist(n-2), abs(t_stat)))
            
            # Store correlation results
            correlations[param] = (r=pearson_r, p=p_value, n=n)
            
            # Print correlation results
            sig_marker = p_value < 0.001 ? "***" : (p_value < 0.01 ? "**" : (p_value < 0.05 ? "*" : ""))
            println(@sprintf("%-20s %10.4f %10.4f %10d %s", 
                    string(param), pearson_r, p_value, n, sig_marker))
        else
            correlations[param] = (r=NaN, p=NaN, n=kept)
            println(@sprintf("%-20s %10s %10s %10d %s", 
                    string(param), "N/A", "N/A", kept, "(insufficient data)"))
        end

        # Prepare plot data (apply log transform if needed for visualization)
        yplot = y_valid
        ylabel_final = labels[param]
        if param in LOG_PARAMS
            yplot = log10.(yplot)
            ylabel_final = string(labels[param], " (log10)")
        end

        # Build title with correlation info (matches caption style)
        title_str = if haskey(correlations, param) && !isnan(correlations[param].r)
            @sprintf("r = %.3f, p = %.4f", correlations[param].r, correlations[param].p)
        else
            ""
        end

        p = scatter(
            yplot, x_valid;
            xlabel = ylabel_final,
            ylabel = metric_name,
            title  = title_str,
            markersize = 7,
            markerstrokewidth = 0.5,
            alpha  = 0.85,
            zcolor = x_valid, colorbar = true, clim = clims, c = cgrad(:viridis),
            legend = false,
        )

        # Add trend line if correlation is significant
        if haskey(correlations, param) && !isnan(correlations[param].r) &&
           correlations[param].p < 0.05 && length(x_valid) >= 3

            X_matrix = hcat(ones(length(yplot)), yplot)
            coeffs = X_matrix \ x_valid

            y_trend = range(minimum(yplot), maximum(yplot), length=100)
            x_trend = coeffs[1] .+ coeffs[2] .* y_trend

            plot!(p, y_trend, x_trend,
                  color=C_RED, linewidth=2, linestyle=:dash, label="")
        end
        
        plots[param] = p
    end
    
    println("-"^60)
    println("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    # Summary of strongest correlations
    valid_corrs = [(param, abs(corr.r), corr.r, corr.p) for (param, corr) in correlations 
                   if !isnan(corr.r) && corr.p < 0.05]
    
    if !isempty(valid_corrs)
        sort!(valid_corrs, by=x->x[2], rev=true)  # Sort by absolute correlation
        println("\nStrongest significant correlations:")
        for (param, abs_r, r, p) in valid_corrs[1:min(3, end)]
            direction = r > 0 ? "positive" : "negative"
            println(@sprintf("  %-12s: r=%.3f (%s, p=%.3f)", 
                    string(param), r, direction, p))
        end
    else
        println("\nNo statistically significant correlations found (p < 0.05)")
    end
    
    return plots, correlations
end

# ---------- Metric vs Metric Correlation Analysis ----------
function create_metric_correlations(df)
    # Define the metrics and their display names
    metrics = [
        (:tumor_ratio, "Tumor Ratio"),
        (:L_max, "L Max (Peak CISHKO)"),
        (:L_AUC, "L AUC (CISHKO Integration)"),
        (:T_AUC_normalized, "T AUC Normalized")
    ]
    
    # Define specific pairs to analyze
    metric_pairs = [
        ((:tumor_ratio, "Tumor Ratio"), (:L_max, "L Max (Peak CISHKO)")),
        ((:tumor_ratio, "Tumor Ratio"), (:L_AUC, "L AUC (CISHKO Integration)")),
        ((:T_AUC_normalized, "T AUC Normalized"), (:L_max, "L Max (Peak CISHKO)")),
        ((:T_AUC_normalized, "T AUC Normalized"), (:L_AUC, "L AUC (CISHKO Integration)"))
    ]
    
    plots = Dict{String, Any}()
    correlations = Dict{String, NamedTuple}()
    
    println("\n" * "="^60)
    println("METRIC-METRIC CORRELATIONS")
    println("="^60)
    println(@sprintf("%-35s %10s %10s %10s", "Metric Pair", "Pearson r", "P-value", "N"))
    println("-"^60)
    
    for ((x_col, x_name), (y_col, y_name)) in metric_pairs
        x_data = df[!, x_col]
        y_data = df[!, y_col]
        
        # Filter out missing values
        mask = .!ismissing.(x_data) .& .!ismissing.(y_data)
        x_valid = Float64.(x_data[mask])
        y_valid = Float64.(y_data[mask])
        
        if length(x_valid) >= 3
            # Calculate correlation
            pearson_r = cor(x_valid, y_valid)
            n = length(x_valid)
            t_stat = pearson_r * sqrt((n-2)/(1-pearson_r^2))
            p_value = 2 * (1 - cdf(TDist(n-2), abs(t_stat)))
            
            # Store results
            pair_key = "$(x_col)_vs_$(y_col)"
            correlations[pair_key] = (r=pearson_r, p=p_value, n=n, x_name=x_name, y_name=y_name)
            
            # Print correlation results
            sig_marker = p_value < 0.001 ? "***" : (p_value < 0.01 ? "**" : (p_value < 0.05 ? "*" : ""))
            pair_label = "$(x_name) vs $(y_name)"
            println(@sprintf("%-35s %10.4f %10.4f %10d %s", 
                    pair_label, pearson_r, p_value, n, sig_marker))
            
            # Build title with correlation info
            title_str = @sprintf("r = %.3f, p = %.4f", pearson_r, p_value)

            p = scatter(
                x_valid, y_valid;
                xlabel = x_name,
                ylabel = y_name,
                title  = title_str,
                markersize = 7,
                markerstrokewidth = 0.5,
                markerstrokecolor = :white,
                markercolor = C_BLUE,
                alpha  = 0.85,
                legend = false,
            )

            # Add trend line if significant
            if p_value < 0.05 && length(x_valid) >= 3
                X_matrix = hcat(ones(length(x_valid)), x_valid)
                coeffs = X_matrix \ y_valid

                x_trend = range(minimum(x_valid), maximum(x_valid), length=100)
                y_trend = coeffs[1] .+ coeffs[2] .* x_trend

                plot!(p, x_trend, y_trend,
                      color=C_RED, linewidth=2, linestyle=:dash, label="")
            end
            
            plots[pair_key] = p
        else
            pair_key = "$(x_col)_vs_$(y_col)"
            correlations[pair_key] = (r=NaN, p=NaN, n=length(x_valid), x_name=x_name, y_name=y_name)
            pair_label = "$(x_name) vs $(y_name)"
            println(@sprintf("%-35s %10s %10s %10d %s", 
                    pair_label, "N/A", "N/A", length(x_valid), "(insufficient data)"))
        end
    end
    
    println("-"^60)
    println("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    # Summary of significant correlations
    valid_corrs = [(key, corr.r, corr.p, corr.x_name, corr.y_name) for (key, corr) in correlations 
                   if !isnan(corr.r) && corr.p < 0.05]
    
    if !isempty(valid_corrs)
        sort!(valid_corrs, by=x->abs(x[2]), rev=true)
        println("\nSignificant metric correlations:")
        for (key, r, p, x_name, y_name) in valid_corrs
            direction = r > 0 ? "positive" : "negative"
            println(@sprintf("  %-30s: r=%.3f (%s, p=%.3f)", 
                    "$(x_name) vs $(y_name)", r, direction, p))
        end
    else
        println("\nNo statistically significant metric correlations found (p < 0.05)")
    end
    
    return plots, correlations
end

# ---------- Main ----------
function main_analysis(filepath)
    println("Loading data...")
    df = load_data(filepath)
    println("Data loaded: $(nrow(df)) rows, $(ncol(df)) columns")

    # Define metrics to analyze
    metrics = [
        (:tumor_ratio, "Tumor Ratio"),
        (:L_max, "L Max (Peak CISHKO)"), 
        (:L_AUC, "L AUC (CISHKO Integration)"),
        (:T_AUC_normalized, "T AUC Normalized")
    ]
    
    all_correlations = Dict{String, Dict}()
    
    # Analyze each metric
    for (metric_col, metric_name) in metrics
        println("\nCreating param-vs-$(metric_name) scatter plots with correlation analysis...")
        scatters, correlations = create_param_vs_metric_scatters(df, metric_col, metric_name)
        
        # Save plots
        for (param, plotobj) in scatters
            fn = "scatter_$(param)_vs_$(metric_col).png"
            savefig(plotobj, fn)
            println("   Saved: $fn")
        end
        println("   Total param-vs-$(metric_name) plots saved: $(length(scatters))")
        
        # Store correlations
        all_correlations[string(metric_col)] = correlations
    end
    
    # Create metric-vs-metric correlation plots
    println("\nCreating metric-vs-metric correlation plots...")
    metric_plots, metric_correlations = create_metric_correlations(df)
    
    # Save metric correlation plots
    for (pair_key, plotobj) in metric_plots
        fn = "correlation_$(pair_key).png"
        savefig(plotobj, fn)
        println("   Saved: $fn")
    end
    println("   Total metric correlation plots saved: $(length(metric_plots))")
    
    # Save comprehensive correlation results to CSV
    println("\nSaving comprehensive correlation results...")
    corr_df = DataFrame(
        metric = String[],
        parameter = Symbol[],
        correlation = Float64[],
        p_value = Float64[],
        n_points = Int[],
        significant = Bool[]
    )
    
    for (metric_name, correlations) in all_correlations
        for (param, corr_data) in correlations
            push!(corr_df, (
                metric = metric_name,
                parameter = param,
                correlation = corr_data.r,
                p_value = corr_data.p,
                n_points = corr_data.n,
                significant = !isnan(corr_data.p) && corr_data.p < 0.05
            ))
        end
    end
    
    CSV.write("parameter_ode_metrics_correlations.csv", corr_df)
    println("   Saved correlation results: parameter_ode_metrics_correlations.csv")
    
    # Save metric-vs-metric correlation results
    metric_corr_df = DataFrame(
        x_metric = String[],
        y_metric = String[],
        correlation = Float64[],
        p_value = Float64[],
        n_points = Int[],
        significant = Bool[]
    )
    
    for (pair_key, corr_data) in metric_correlations
        if !isnan(corr_data.r)
            # Parse x and y metric names from pair_key
            x_metric, y_metric = split(pair_key, "_vs_")
            push!(metric_corr_df, (
                x_metric = x_metric,
                y_metric = y_metric,
                correlation = corr_data.r,
                p_value = corr_data.p,
                n_points = corr_data.n,
                significant = corr_data.p < 0.05
            ))
        end
    end
    
    CSV.write("metric_metric_correlations.csv", metric_corr_df)
    println("   Saved metric correlation results: metric_metric_correlations.csv")

    # ── Generate combined 2×2 scatter figure for publication ──
    println("\nGenerating combined scatter figure for publication...")
    create_publication_scatter_figure(df)

    return df
end

# ---------- Combined 2×2 scatter figure (Figure 5 a–d) ----------
function create_publication_scatter_figure(df)
    panel_specs = [
        # (param, metric_col, xlabel, metric_name)
        (:gamma,  :tumor_ratio,       "Tumor growth rate γ",              "Tumor Ratio"),
        (:dL,     :tumor_ratio,       "Death rate dL (log10)",            "Tumor Ratio"),
        (:pN,     :T_AUC_normalized,  "Endogenous proliferation pN (log10)", "Normalized Tumor AUC"),
        (:kL,     :T_AUC_normalized,  "Killing rate kL (log10)",          "Normalized Tumor AUC"),
    ]

    panels = []
    for (param, metric, xlab, metric_name) in panel_specs
        x_raw = Float64.(df[!, param])
        y_raw = Float64.(df[!, metric])

        mask = .!isnan.(x_raw) .& .!isnan.(y_raw)
        if param in LOG_PARAMS
            mask .= mask .& (x_raw .> 0)
        end

        x = x_raw[mask]
        y = y_raw[mask]
        xplot = param in LOG_PARAMS ? log10.(x) : x

        # Correlation
        pearson_r = cor(xplot, y)
        n = length(xplot)
        t_stat = pearson_r * sqrt((n - 2) / (1 - pearson_r^2))
        p_value = 2 * (1 - cdf(TDist(n - 2), abs(t_stat)))
        title_str = @sprintf("%s (r=%.3f, p=%.4f)", metric_name, pearson_r, p_value)

        # robust clims for this panel
        lo_p, hi_p = quantile(y, (0.02, 0.98))

        p = scatter(xplot, y;
            xlabel = xlab,
            title  = title_str,
            markersize = 7,
            markerstrokewidth = 0.5,
            alpha  = 0.85,
            zcolor = y, colorbar = true, clim = (lo_p, hi_p), c = cgrad(:viridis),
            colorbar_tickfontsize = 6,
            right_margin = 0Plots.mm,
            legend = false,
        )

        if p_value < 0.05 && n >= 3
            X_mat = hcat(ones(n), xplot)
            coeffs = X_mat \ y
            xt = range(minimum(xplot), maximum(xplot), length=100)
            yt = coeffs[1] .+ coeffs[2] .* xt
            plot!(p, xt, yt, color=C_RED, linewidth=2, linestyle=:dash, label="")
        end

        push!(panels, p)
    end

    combined = plot(panels..., layout = (1, 4),
        size = (2200, 500),
        margin = 5Plots.mm,
        bottom_margin = 10Plots.mm,
        left_margin   = 5Plots.mm,
        top_margin    = 8Plots.mm,
    )

    savefig(combined, "scatter_combined_figure5.png")
    savefig(combined, "scatter_combined_figure5.pdf")
    println("   Saved: scatter_combined_figure5.png / .pdf")
    return combined
end

# Run the analysis if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        # Use default path if no argument provided
        default_path = "../patient_tumor_params_with_ratio.csv"
        println("No file path provided. Using default: $default_path")
        main_analysis(default_path)
    else
        main_analysis(ARGS[1])
    end
end
