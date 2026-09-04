using CSV, DataFrames, Plots, Statistics
using StatsPlots
using HypothesisTests

# -- Style setup (consistent with other figures) --
gr(fontfamily = "Computer Modern")
default(
    framestyle  = :axes,
    grid        = true,
    gridalpha   = 0.3,
    gridstyle   = :dash,
    guidefontsize  = 14,
    tickfontsize   = 11,
    titlefontsize  = 12,
    legendfontsize = 9,
    background_color = :white,
    foreground_color = :black,
    dpi  = 300,
)

# Function to find file in current directory or parent directories
function find_file(filename, max_levels=3)
    current_dir = pwd()
    
    # Check current directory first
    if isfile(joinpath(current_dir, filename))
        return joinpath(current_dir, filename)
    end
    
    # Check parent directories
    for level in 1:max_levels
        parent_dir = dirname(current_dir)
        if parent_dir == current_dir  # Reached root
            break
        end
        
        file_path = joinpath(parent_dir, filename)
        if isfile(file_path)
            return file_path
        end
        
        current_dir = parent_dir
    end
    
    return nothing
end

function create_therapy_scatter_plots()
    println("Loading therapy results...")
    
    # Load the results from combination therapy analysis
    results_file = find_file("combination_therapy_results.csv")
    if results_file === nothing
        println("Error: combination_therapy_results.csv not found. Please run combination_therapy.jl first.")
        return
    end
    
    results_df = CSV.read(results_file, DataFrame)
    println("Loaded results from: $results_file")
    println("Loaded $(nrow(results_df)) parameter combinations")
    
    # Define therapy scenarios and metrics
    therapy_scenarios = ["no_therapy", "cish", "double_cish", "anti_pd1", "cish_anti_pd1", "double_cish_anti_pd1"]
    therapy_labels = ["No Therapy", "CISH", "Double CISH", "anti-PD1", "CISH+anti-PD1", "Double CISH+anti-PD1"]
    metrics = ["L_max", "L_AUC", "T_AUC_normalized", "tumor_ratio"]
    metric_titles = [
        "Maximum Lymphocyte Count (L_max)",
        "Lymphocyte Area Under Curve (L_AUC)", 
        "Normalized Tumor AUC (T_AUC)",
        "Tumor Size Ratio (Final/Initial)"
    ]
    
    # Set up colors for each therapy
    colors = [:gray, :blue, :darkblue, :red, :purple, :darkred]
    
    # Create 4 separate scatter plots
    # Figures that should use larger fonts
    large_font_metrics = ["T_AUC_normalized", "L_max"]

    for (metric_idx, metric) in enumerate(metrics)
        println("Creating scatter plot for $metric...")

        # Use larger fonts for selected figures
        if metric in large_font_metrics
            guidefont  = 20
            tickfont   = 16
            sig_fontsize = 16
        else
            guidefont  = 14
            tickfont   = 11
            sig_fontsize = 11
        end

        # Initialize plot
        p = plot(size=(1000, 700),
                ylabel=metric_titles[metric_idx],
                xticks=(1:6, therapy_labels),
                xrotation=30,
                legend=false,
                guidefontsize=guidefont,
                tickfontsize=tickfont,
                left_margin=5Plots.mm,
                bottom_margin=8Plots.mm,
                top_margin=5Plots.mm,
                right_margin=5Plots.mm)
        
        # Collect and plot data for each therapy scenario
        all_y_values = Float64[]
        
        for (therapy_idx, scenario) in enumerate(therapy_scenarios)
            col_name = Symbol("$(scenario)_$(metric)")
            
            # Get values for this therapy scenario
            if hasproperty(results_df, col_name)
                values = results_df[!, col_name]
                valid_values = values[.!isnan.(values)]
                
                if length(valid_values) > 0
                    # Add boxplot for this therapy
                    boxplot!(p, [therapy_idx], valid_values,
                            color=colors[therapy_idx],
                            alpha=0.3,
                            outliers=false,
                            whisker_width=:match,
                            linewidth=2,
                            fillalpha=0.3)
                    
                    # Add some jitter to x-coordinates to avoid overlap
                    x_jitter = therapy_idx .+ 0.2 .* (rand(length(valid_values)) .- 0.5)
                    
                    # Plot scatter points on top of boxplot
                    scatter!(p, x_jitter, valid_values,
                            color=colors[therapy_idx],
                            alpha=0.7,
                            markersize=3,
                            markerstrokewidth=0.5,
                            markerstrokecolor=:white)
                    
                    # Store all values for y-axis scaling
                    append!(all_y_values, valid_values)
                    
                    # Calculate median for reporting
                    med_val = median(valid_values)
                    println("  $(therapy_labels[therapy_idx]): $(length(valid_values)) points, median = $(round(med_val, digits=4))")
                else
                    println("  $(therapy_labels[therapy_idx]): No valid data")
                end
            else
                println("  Warning: Column $col_name not found in data")
            end
        end
        
        # Set y-axis limits with some padding
        if length(all_y_values) > 0
            # Special handling for normalized tumor AUC
            if metric == "T_AUC_normalized"
                y_min = 30
                y_max = 100
                y_range = y_max - y_min
            else
                y_min = minimum(all_y_values)
                y_max = maximum(all_y_values)
                y_range = y_max - y_min
            end
            
            # First, count how many significant pairs there are to allocate space
            # Only compare cish (index 2) against all others
            sig_count = 0
            i = 2  # cish
            for j in 1:length(therapy_scenarios)
                if j == i
                    continue
                end
                col_i = Symbol("$(therapy_scenarios[i])_$(metric)")
                col_j = Symbol("$(therapy_scenarios[j])_$(metric)")
                
                if hasproperty(results_df, col_i) && hasproperty(results_df, col_j)
                    values_i = results_df[!, col_i]
                    values_j = results_df[!, col_j]
                    valid_i = values_i[.!isnan.(values_i)]
                    valid_j = values_j[.!isnan.(values_j)]
                    
                    if length(valid_i) > 0 && length(valid_j) > 0
                        test_result = MannWhitneyUTest(valid_i, valid_j)
                        p_value = pvalue(test_result)
                        if p_value < 0.05
                            sig_count += 1
                        end
                    end
                end
            end
            
            # Add extra space at top for significance markers
            extra_space = max(0.2*y_range, sig_count * 0.08*y_range)
            y_max_extended = y_max + extra_space
            ylims!(p, (y_min - 0.05*y_range, y_max_extended))
        end
        
        # Set x-axis limits
        xlims!(p, (0.5, 6.5))
        
        # Perform comparisons: cish vs all other therapies
        println("Pairwise comparisons for $metric (CISH vs Others):")
        sig_y_pos = y_max + 0.03*y_range
        sig_y_offset = 0.07*y_range
        
        i = 2  # cish
        for j in 1:length(therapy_scenarios)
            if j == i
                continue
            end
            col_i = Symbol("$(therapy_scenarios[i])_$(metric)")
            col_j = Symbol("$(therapy_scenarios[j])_$(metric)")
            
            if hasproperty(results_df, col_i) && hasproperty(results_df, col_j)
                values_i = results_df[!, col_i]
                values_j = results_df[!, col_j]
                valid_i = values_i[.!isnan.(values_i)]
                valid_j = values_j[.!isnan.(values_j)]
                
                if length(valid_i) > 0 && length(valid_j) > 0
                    # Perform Mann-Whitney U test
                    test_result = MannWhitneyUTest(valid_i, valid_j)
                    p_value = pvalue(test_result)
                    
                    # Determine significance level
                    sig_marker = ""
                    if p_value < 0.001
                        sig_marker = "***"
                    elseif p_value < 0.01
                        sig_marker = "**"
                    elseif p_value < 0.05
                        sig_marker = "*"
                    end
                    
                    if sig_marker != ""
                        println("  $(therapy_labels[i]) vs $(therapy_labels[j]): p=$(round(p_value, digits=4)) $sig_marker")
                        
                        # Add significance line and marker on plot
                        plot!(p, [i, j], [sig_y_pos, sig_y_pos], 
                             color=:black, linewidth=2, label="", legend=false)
                        mid_x = (i + j) / 2
                        annotate!(p, mid_x, sig_y_pos + sig_y_offset*0.2,
                                 text(sig_marker, sig_fontsize, :black, :center, :bold))
                        sig_y_pos += sig_y_offset
                    end
                end
            end
        end
        
        # Save the plot
        output_filename = "scatter_$(metric).png"
        savefig(p, output_filename)
        println("Saved scatter plot to $output_filename")
        
        # Also save as PDF
        pdf_filename = "scatter_$(metric).pdf"
        savefig(p, pdf_filename)
        println("Saved scatter plot to $pdf_filename")
        
        # Print summary statistics
        println("Summary statistics for $metric:")
        for (therapy_idx, scenario) in enumerate(therapy_scenarios)
            col_name = Symbol("$(scenario)_$(metric)")
            if hasproperty(results_df, col_name)
                values = results_df[!, col_name]
                valid_values = values[.!isnan.(values)]
                
                if length(valid_values) > 0
                    println("  $(therapy_labels[therapy_idx]): n=$(length(valid_values)), median=$(round(median(valid_values), digits=4)), mean=$(round(mean(valid_values), digits=4)) ± $(round(std(valid_values), digits=4))")
                end
            end
        end
        
        # Perform pairwise comparisons and add significance markers
        println("Pairwise comparisons for $metric:")
        max_y = maximum(all_y_values)
        sig_y_offset = max_y * 0.05
        sig_y_pos = max_y + sig_y_offset
        
        for i in 1:(length(therapy_scenarios)-1)
            for j in (i+1):length(therapy_scenarios)
                col_i = Symbol("$(therapy_scenarios[i])_$(metric)")
                col_j = Symbol("$(therapy_scenarios[j])_$(metric)")
                
                if hasproperty(results_df, col_i) && hasproperty(results_df, col_j)
                    values_i = results_df[!, col_i]
                    values_j = results_df[!, col_j]
                    valid_i = values_i[.!isnan.(values_i)]
                    valid_j = values_j[.!isnan.(values_j)]
                    
                    if length(valid_i) > 0 && length(valid_j) > 0
                        # Perform Mann-Whitney U test
                        test_result = MannWhitneyUTest(valid_i, valid_j)
                        p_value = pvalue(test_result)
                        
                        # Determine significance level
                        sig_marker = ""
                        if p_value < 0.001
                            sig_marker = "***"
                        elseif p_value < 0.01
                            sig_marker = "**"
                        elseif p_value < 0.05
                            sig_marker = "*"
                        end
                        
                        if sig_marker != ""
                            println("  $(therapy_labels[i]) vs $(therapy_labels[j]): p=$(round(p_value, digits=4)) $sig_marker")
                            
                            # Add significance line and marker on plot
                            mid_x = (i + j) / 2
                            plot!(p, [i, j], [sig_y_pos, sig_y_pos], 
                                 color=:black, linewidth=1.5, label="")
                            annotate!(p, mid_x, sig_y_pos + sig_y_offset*0.5,
                                     text(sig_marker, sig_fontsize, :black, :center))
                            sig_y_pos += sig_y_offset * 1.5
                        end
                    end
                end
            end
        end
        println()
    end
    
    println("All scatter plots created successfully!")
    println("Generated files:")
    for metric in metrics
        println("  - scatter_$(metric).png")
        println("  - scatter_$(metric).pdf")
    end
end

# Run the function
if abspath(PROGRAM_FILE) == @__FILE__
    create_therapy_scatter_plots()
end