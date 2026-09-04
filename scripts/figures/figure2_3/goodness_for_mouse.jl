# goodness_for_mouse.jl
# Goodness of fit analysis for mouse tumor data
# Data format: [day, observed_size, predicted_size, category]

using CSV, DataFrames, Plots, Statistics, Printf
using LinearAlgebra

# Set a clean default theme for all plots
gr(fontfamily = "Computer Modern")
default(
    background_color = :white,
    foreground_color = :black,
    grid = true,
    gridalpha = 0.3,
    gridstyle = :dash,
    framestyle = :axes,
    guidefontsize = 14,
    tickfontsize = 11,
    legendfontsize = 10,
    titlefontsize = 13,
    margin = 5Plots.mm,
    dpi = 300
)

"""
Calculate goodness of fit statistics
"""
function calculate_goodness_stats(observed, predicted)
    # Remove any NaN or missing values
    valid_idx = .!isnan.(observed) .& .!isnan.(predicted) .& .!ismissing.(observed) .& .!ismissing.(predicted)
    obs_clean = observed[valid_idx]
    pred_clean = predicted[valid_idx]
    
    if length(obs_clean) < 2
        return (n=0, correlation=NaN, r_squared=NaN, rmse=NaN, mae=NaN)
    end
    
    # Pearson correlation
    correlation = cor(obs_clean, pred_clean)
    
    # R-squared
    ss_res = sum((obs_clean .- pred_clean).^2)
    ss_tot = sum((obs_clean .- mean(obs_clean)).^2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # RMSE
    rmse = sqrt(mean((obs_clean .- pred_clean).^2))
    
    # MAE
    mae = mean(abs.(obs_clean .- pred_clean))
    
    return (n=length(obs_clean), correlation=correlation, r_squared=r_squared, rmse=rmse, mae=mae)
end

"""
Create goodness of fit plots (legacy function - kept for compatibility)
"""
function create_goodness_plots(df)
    println("Creating goodness of fit plots...")
    
    # Get unique categories
    categories = unique(df.category)
    n_categories = length(categories)
    println("Found $(n_categories) categories: $(categories)")
    
    # Define a publication-quality color palette (colorblind-friendly)
    colors = [RGB(0.0, 0.45, 0.70), RGB(0.90, 0.62, 0.0), RGB(0.0, 0.62, 0.45),
              RGB(0.80, 0.47, 0.65), RGB(0.34, 0.71, 0.91), RGB(0.94, 0.89, 0.26),
              RGB(0.84, 0.37, 0.0), RGB(0.50, 0.50, 0.50), RGB(0.60, 0.20, 0.80), RGB(0.20, 0.80, 0.40)]
    
    # Overall plot (all categories combined)
    println("\n" * "="^60)
    println("OVERALL GOODNESS-OF-FIT STATISTICS (ALL CATEGORIES)")
    println("="^60)
    
    overall_stats = calculate_goodness_stats(df.observed, df.predicted)
    println("Number of data points: $(overall_stats.n)")
    println("Pearson correlation: $(round(overall_stats.correlation, digits=4))")
    println("R-squared: $(round(overall_stats.r_squared, digits=4))")
    println("RMSE: $(round(overall_stats.rmse, digits=6))")
    println("MAE: $(round(overall_stats.mae, digits=6))")
    println("="^60)
    
    # Create overall scatter plot with different colors for each category
    p_overall = plot(xlabel="Observed Tumor Size", 
                    ylabel="Predicted Tumor Size",
                    title="Overall Model Fit by Category\nR2 = $(round(overall_stats.r_squared, digits=3)),  n = $(overall_stats.n)",
                    legend=:topleft,
                    size=(800, 700),
                    margin=8Plots.mm)
    
    # Plot each category with different colors
    for (i, cat) in enumerate(categories)
        cat_data = filter(row -> row.category == cat, df)
        color_idx = ((i-1) % length(colors)) + 1
        scatter!(p_overall, cat_data.observed, cat_data.predicted,
                alpha=0.75,
                color=colors[color_idx],
                markerstrokewidth=0.5,
                markerstrokecolor=:gray30,
                markersize=6,
                label=string(cat))
    end
    
    # Add perfect fit line (y=x)
    min_val = min(minimum(df.observed), minimum(df.predicted))
    max_val = max(maximum(df.observed), maximum(df.predicted))
    pad = 0.05 * (max_val - min_val)
    plot!(p_overall, [min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 
          line=:dash, color=:black, linewidth=2, label="Perfect Fit (y=x)")
    
    # Add regression line
    X = hcat(ones(length(df.observed)), df.observed)
    β = X \ df.predicted
    x_range = range(min_val, max_val, length=100)
    y_fit = β[1] .+ β[2] .* x_range
    plot!(p_overall, x_range, y_fit, 
          color=RGB(0.2, 0.7, 0.2), linewidth=2.5, label="Linear Fit")
    
    # Save overall plot
    savefig(p_overall, "mouse_tumor_overall_goodness_fit.png")
    println("\nOverall plot saved as: mouse_tumor_overall_goodness_fit.png")
    
    # Category-wise analysis
    println("\n" * "="^60)
    println("CATEGORY-WISE GOODNESS-OF-FIT STATISTICS")
    println("="^60)
    
    # Create subplot for each category
    plots_by_category = []
    
    for (i, cat) in enumerate(categories)
        cat_data = filter(row -> row.category == cat, df)
        cat_stats = calculate_goodness_stats(cat_data.observed, cat_data.predicted)
        
        println("\nCategory $(cat):")
        println("  Data points: $(cat_stats.n)")
        println("  Correlation: $(round(cat_stats.correlation, digits=4))")
        println("  R-squared: $(round(cat_stats.r_squared, digits=4))")
        println("  RMSE: $(round(cat_stats.rmse, digits=6))")
        println("  MAE: $(round(cat_stats.mae, digits=6))")
        
        # Create individual plot for this category
        color_idx = ((i-1) % length(colors)) + 1
        p_cat = scatter(cat_data.observed, cat_data.predicted,
                       alpha=0.7,
                       xlabel="Observed",
                       ylabel="Predicted", 
                       title="$(cat)  (R2=$(round(cat_stats.r_squared, digits=3)), n=$(cat_stats.n))",
                       color=colors[color_idx],
                       markerstrokewidth=0.5,
                       markerstrokecolor=:gray30,
                       legend=false,
                       markersize=5)
        
        # Add perfect fit line
        if cat_stats.n > 0
            cat_min = min(minimum(cat_data.observed), minimum(cat_data.predicted))
            cat_max = max(maximum(cat_data.observed), maximum(cat_data.predicted))
            cat_pad = 0.05 * (cat_max - cat_min)
            plot!(p_cat, [cat_min - cat_pad, cat_max + cat_pad], [cat_min - cat_pad, cat_max + cat_pad], 
                  line=:dash, color=:black, linewidth=1.5)
        end
        
        push!(plots_by_category, p_cat)
    end
    
    # Create combined subplot layout (2x3 or 3x2 depending on number of categories)
    if n_categories <= 4
        layout = (2, 2)
    elseif n_categories <= 6
        layout = (2, 3) 
    else
        layout = (3, 3)
    end
    
    # Pad plots_by_category if needed for clean layout
    while length(plots_by_category) < prod(layout)
        push!(plots_by_category, plot(framestyle=:none, showaxis=false, grid=false))
    end
    
    p_combined = plot(plots_by_category[1:prod(layout)]..., 
                     layout=layout, 
                     size=(1200, 900),
                     plot_title="Mouse Tumor Goodness of Fit by Category",
                     plot_titlefontsize=15,
                     left_margin=6Plots.mm,
                     bottom_margin=6Plots.mm)
    
    savefig(p_combined, "mouse_tumor_category_goodness_fit.png")
    println("\nCategory-wise plot saved as: mouse_tumor_category_goodness_fit.png")
    
    # Time series plots by category
    println("\nCreating time series plots...")
    
    time_plots = []
    for (i, cat) in enumerate(categories)
        cat_data = filter(row -> row.category == cat, df)
        if nrow(cat_data) > 0
            color_idx = ((i-1) % length(colors)) + 1
            p_time = plot(cat_data.day, cat_data.observed, 
                         seriestype=:scatter, 
                         label="Observed",
                         color=colors[color_idx],
                         markerstrokewidth=0.5,
                         markerstrokecolor=:gray30,
                         markersize=5,
                         xlabel="Day", 
                         ylabel="Tumor Size",
                         title="$(cat)")
            
            plot!(p_time, cat_data.day, cat_data.predicted,
                  seriestype=:line,
                  label="Predicted", 
                  color=colors[color_idx],
                  linewidth=2.5,
                  linestyle=:dash)
            
            push!(time_plots, p_time)
        end
    end
    
    # Create time series layout
    while length(time_plots) < prod(layout)
        push!(time_plots, plot(framestyle=:none, showaxis=false, grid=false))
    end
    
    p_time_combined = plot(time_plots[1:prod(layout)]..., 
                          layout=layout,
                          size=(1200, 900),
                          plot_title="Mouse Tumor Time Series by Category",
                          plot_titlefontsize=15,
                          left_margin=6Plots.mm,
                          bottom_margin=6Plots.mm)
    
    savefig(p_time_combined, "mouse_tumor_timeseries_by_category.png")
    println("Time series plot saved as: mouse_tumor_timeseries_by_category.png")
    
    return overall_stats
end

"""
Analyze a single dataset
"""
function analyze_single_dataset(csv_filename, year_suffix)
    println("\n" * "="^80)
    println("ANALYZING $(csv_filename)")
    println("="^80)
    
    # Try to find the CSV file
    base_dir = dirname(pwd())  # Parent directory 
    csv_path = joinpath(base_dir, csv_filename)
    
    if !isfile(csv_path)
        # Try current directory
        csv_path = csv_filename
        if !isfile(csv_path)
            # Try parent directory with relative path
            csv_path = "../$(csv_filename)"
            if !isfile(csv_path)
                println("⚠️  Could not find '$(csv_filename)'. Skipping this dataset.")
                return nothing
            end
        end
    end
    
    println("Loading data from: $(csv_path)")
    
    # Load the CSV data
    try
        df = CSV.read(csv_path, DataFrame)
        println("Successfully loaded $(nrow(df)) rows of data")
        
        # Display column names to verify structure
        println("Columns found: $(names(df))")
        
        # Rename columns to standard names (assuming order: day, observed, predicted, category)
        if ncol(df) >= 4
            rename!(df, [names(df)[1] => :day, 
                        names(df)[2] => :observed, 
                        names(df)[3] => :predicted, 
                        names(df)[4] => :category])
        else
            println("❌ CSV file must have at least 4 columns: [day, observed_size, predicted_size, category]")
            return nothing
        end
        
        # Show first few rows
        println("\nFirst 5 rows of data:")
        println(first(df, 5))
        
        # Show data summary
        println("\nData summary:")
        println("  Days range: $(minimum(df.day)) to $(maximum(df.day))")
        println("  Observed size range: $(minimum(df.observed)) to $(maximum(df.observed))")  
        println("  Predicted size range: $(minimum(df.predicted)) to $(maximum(df.predicted))")
        println("  Categories: $(unique(df.category))")
        
        # Create goodness of fit plots with year suffix
        stats = create_goodness_plots_with_suffix(df, year_suffix)
        
        return stats
        
    catch e
        println("❌ Error reading CSV file: $(e)")
        return nothing
    end
end

"""
Create goodness of fit plots with filename suffix
"""
function create_goodness_plots_with_suffix(df, suffix)
    println("Creating goodness of fit plots for $(suffix) data...")
    
    # Get unique categories
    categories = unique(df.category)
    n_categories = length(categories)
    println("Found $(n_categories) categories: $(categories)")
    
    # Define a publication-quality color palette (colorblind-friendly)
    colors = [RGB(0.0, 0.45, 0.70), RGB(0.90, 0.62, 0.0), RGB(0.0, 0.62, 0.45),
              RGB(0.80, 0.47, 0.65), RGB(0.34, 0.71, 0.91), RGB(0.94, 0.89, 0.26),
              RGB(0.84, 0.37, 0.0), RGB(0.50, 0.50, 0.50), RGB(0.60, 0.20, 0.80), RGB(0.20, 0.80, 0.40)]
    
    # Overall plot (all categories combined)
    println("\n" * "="^60)
    println("OVERALL GOODNESS-OF-FIT STATISTICS (ALL CATEGORIES) - $(suffix)")
    println("="^60)
    
    overall_stats = calculate_goodness_stats(df.observed, df.predicted)
    println("Number of data points: $(overall_stats.n)")
    println("Pearson correlation: $(round(overall_stats.correlation, digits=4))")
    println("R-squared: $(round(overall_stats.r_squared, digits=4))")
    println("RMSE: $(round(overall_stats.rmse, digits=6))")
    println("MAE: $(round(overall_stats.mae, digits=6))")
    println("="^60)
    
    # Create overall scatter plot with different colors for each category
    p_overall = plot(xlabel="Observed Tumor Size", 
                    ylabel="Predicted Tumor Size",
                    title="Overall Model Fit by Category ($(suffix))\nR2 = $(round(overall_stats.r_squared, digits=3)),  n = $(overall_stats.n)",
                    legend=:topleft,
                    size=(800, 700),
                    margin=8Plots.mm)
    
    # Plot each category with different colors
    for (i, cat) in enumerate(categories)
        cat_data = filter(row -> row.category == cat, df)
        color_idx = ((i-1) % length(colors)) + 1
        scatter!(p_overall, cat_data.observed, cat_data.predicted,
                alpha=0.75,
                color=colors[color_idx],
                markerstrokewidth=0.5,
                markerstrokecolor=:gray30,
                markersize=6,
                label=string(cat))
    end
    
    # Add perfect fit line (y=x)
    min_val = min(minimum(df.observed), minimum(df.predicted))
    max_val = max(maximum(df.observed), maximum(df.predicted))
    pad = 0.05 * (max_val - min_val)
    plot!(p_overall, [min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 
          line=:dash, color=:black, linewidth=2, label="Perfect Fit (y=x)")
    
    # Add regression line
    X = hcat(ones(length(df.observed)), df.observed)
    β = X \ df.predicted
    x_range = range(min_val, max_val, length=100)
    y_fit = β[1] .+ β[2] .* x_range
    plot!(p_overall, x_range, y_fit, 
          color=RGB(0.2, 0.7, 0.2), linewidth=2.5, label="Linear Fit")
    
    # Save overall plot with suffix
    savefig(p_overall, "mouse_tumor_overall_goodness_fit_$(suffix).png")
    println("\nOverall plot saved as: mouse_tumor_overall_goodness_fit_$(suffix).png")
    
    # Category-wise analysis
    println("\n" * "="^60)
    println("CATEGORY-WISE GOODNESS-OF-FIT STATISTICS - $(suffix)")
    println("="^60)
    
    # Create subplot for each category
    plots_by_category = []
    
    for (i, cat) in enumerate(categories)
        cat_data = filter(row -> row.category == cat, df)
        cat_stats = calculate_goodness_stats(cat_data.observed, cat_data.predicted)
        
        println("\nCategory $(cat):")
        println("  Data points: $(cat_stats.n)")
        println("  Correlation: $(round(cat_stats.correlation, digits=4))")
        println("  R-squared: $(round(cat_stats.r_squared, digits=4))")
        println("  RMSE: $(round(cat_stats.rmse, digits=6))")
        println("  MAE: $(round(cat_stats.mae, digits=6))")
        
        # Create individual plot for this category
        color_idx = ((i-1) % length(colors)) + 1
        p_cat = scatter(cat_data.observed, cat_data.predicted,
                       alpha=0.7,
                       xlabel="Observed",
                       ylabel="Predicted", 
                       title="$(cat) ($(suffix))  (R2=$(round(cat_stats.r_squared, digits=3)), n=$(cat_stats.n))",
                       color=colors[color_idx],
                       markerstrokewidth=0.5,
                       markerstrokecolor=:gray30,
                       legend=false,
                       markersize=5)
        
        # Add perfect fit line
        if cat_stats.n > 0
            cat_min = min(minimum(cat_data.observed), minimum(cat_data.predicted))
            cat_max = max(maximum(cat_data.observed), maximum(cat_data.predicted))
            cat_pad = 0.05 * (cat_max - cat_min)
            plot!(p_cat, [cat_min - cat_pad, cat_max + cat_pad], [cat_min - cat_pad, cat_max + cat_pad], 
                  line=:dash, color=:black, linewidth=1.5)
        end
        
        push!(plots_by_category, p_cat)
    end
    
    # Create combined subplot layout
    if n_categories <= 4
        layout = (2, 2)
    elseif n_categories <= 6
        layout = (2, 3) 
    else
        layout = (3, 3)
    end
    
    # Pad plots_by_category if needed for clean layout
    while length(plots_by_category) < prod(layout)
        push!(plots_by_category, plot(framestyle=:none, showaxis=false, grid=false))
    end
    
    p_combined = plot(plots_by_category[1:prod(layout)]..., 
                     layout=layout, 
                     size=(1200, 900),
                     plot_title="Mouse Tumor Goodness of Fit by Category ($(suffix))",
                     plot_titlefontsize=15,
                     left_margin=6Plots.mm,
                     bottom_margin=6Plots.mm)
    
    savefig(p_combined, "mouse_tumor_category_goodness_fit_$(suffix).png")
    println("\nCategory-wise plot saved as: mouse_tumor_category_goodness_fit_$(suffix).png")
    
    # Time series plots by category
    println("\nCreating time series plots...")
    
    time_plots = []
    for (i, cat) in enumerate(categories)
        cat_data = filter(row -> row.category == cat, df)
        if nrow(cat_data) > 0
            color_idx = ((i-1) % length(colors)) + 1
            p_time = plot(cat_data.day, cat_data.observed, 
                         seriestype=:scatter, 
                         label="Observed",
                         color=colors[color_idx],
                         markerstrokewidth=0.5,
                         markerstrokecolor=:gray30,
                         markersize=5,
                         xlabel="Day", 
                         ylabel="Tumor Size",
                         title="$(cat) ($(suffix))")
            
            plot!(p_time, cat_data.day, cat_data.predicted,
                  seriestype=:line,
                  label="Predicted", 
                  color=colors[color_idx],
                  linewidth=2.5,
                  linestyle=:dash)
            
            push!(time_plots, p_time)
        end
    end
    
    # Create time series layout
    while length(time_plots) < prod(layout)
        push!(time_plots, plot(framestyle=:none, showaxis=false, grid=false))
    end
    
    p_time_combined = plot(time_plots[1:prod(layout)]..., 
                          layout=layout,
                          size=(1200, 900),
                          plot_title="Mouse Tumor Time Series by Category ($(suffix))",
                          plot_titlefontsize=15,
                          left_margin=6Plots.mm,
                          bottom_margin=6Plots.mm)
    
    savefig(p_time_combined, "mouse_tumor_timeseries_by_category_$(suffix).png")
    println("Time series plot saved as: mouse_tumor_timeseries_by_category_$(suffix).png")
    
    return overall_stats
end

"""
Main analysis function
"""
function analyze_mouse_data()
    println("Starting comprehensive mouse tumor goodness of fit analysis...")
    
    # Analyze both datasets
    datasets = [
        ("all_results_combined_2022.csv", "2022"),
        ("all_results_combined_2015.csv", "2015")
    ]
    
    results = Dict()
    
    for (filename, year) in datasets
        stats = analyze_single_dataset(filename, year)
        if stats !== nothing
            results[year] = stats
        end
    end
    
    # Summary comparison if both datasets were analyzed
    if haskey(results, "2022") && haskey(results, "2015")
        println("\n" * "="^80)
        println("COMPARISON SUMMARY")
        println("="^80)
        println("2022 Dataset: R² = $(round(results["2022"].r_squared, digits=4)), n = $(results["2022"].n)")
        println("2015 Dataset: R² = $(round(results["2015"].r_squared, digits=4)), n = $(results["2015"].n)")
        println("="^80)
    end
    
    println("\nComprehensive analysis complete!")
    return results
end

# Run the analysis
if abspath(PROGRAM_FILE) == @__FILE__
    analyze_mouse_data()
end