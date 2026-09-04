using CSV, DataFrames, Plots, Statistics
using StatsPlots

function plot_therapy_results()
    println("Loading therapy results...")
    
    # Load the results from combination therapy analysis
    if !isfile("combination_therapy_results.csv")
        println("Error: combination_therapy_results.csv not found. Please run combination_therapy.jl first.")
        return
    end
    
    results_df = CSV.read("combination_therapy_results.csv", DataFrame)
    println("Loaded $(nrow(results_df)) parameter combinations")
    
    # Define therapy scenarios and metrics
    therapy_scenarios = ["no_therapy", "cish", "double_cish", "anti_pd1", "cish_anti_pd1", "double_cish_anti_pd1"]
    therapy_labels = ["No Therapy", "CISH", "Double CISH", "Anti-PD1", "CISH + Anti-PD1", "Double CISH + Anti-PD1"]
    metrics = ["L_max", "L_AUC", "T_AUC_normalized", "tumor_ratio"]
    metric_labels = ["L_max", "L_AUC", "T_AUC (normalized)", "Tumor Ratio (Final/Initial)"]
    
    # Set up plot theme
    theme(:default)
    
    # Create plots for each metric
    for (metric_idx, metric) in enumerate(metrics)
        println("Creating plot for $metric...")
        
        # Collect data for this metric across all therapies
        plot_data = []
        therapy_nums = []
        
        for (therapy_idx, scenario) in enumerate(therapy_scenarios)
            col_name = Symbol("$(scenario)_$(metric)")
            values = results_df[!, col_name]
            
            # Remove NaN values
            valid_values = values[.!isnan.(values)]
            
            # Add to plot data
            append!(plot_data, valid_values)
            append!(therapy_nums, fill(therapy_idx, length(valid_values)))
        end
        
        # Create scatter plot
        p = scatter(therapy_nums, plot_data,
                   xlabel="Therapy Scenario",
                   ylabel=metric_labels[metric_idx],
                   title="$(metric_labels[metric_idx]) Across Therapy Scenarios",
                   xticks=(1:6, therapy_labels),
                   xrotation=45,
                   alpha=0.6,
                   markersize=3,
                   legend=false,
                   size=(800, 600),
                   left_margin=5Plots.mm,
                   bottom_margin=8Plots.mm)
        
        # Add box plots overlay to show distributions
        violin!(p, therapy_nums, plot_data, 
               alpha=0.3, 
               linewidth=0,
               side=:both)
        
        # Calculate and display summary statistics
        println("\nSummary statistics for $metric:")
        for (therapy_idx, scenario) in enumerate(therapy_scenarios)
            col_name = Symbol("$(scenario)_$(metric)")
            values = results_df[!, col_name]
            valid_values = values[.!isnan.(values)]
            
            if length(valid_values) > 0
                med_val = median(valid_values)
                mean_val = mean(valid_values)
                std_val = std(valid_values)
                println("  $(therapy_labels[therapy_idx]): n=$(length(valid_values)), median=$(round(med_val, digits=4)), mean=$(round(mean_val, digits=4)) ± $(round(std_val, digits=4))")
                
                # Add median line
                hline!(p, [med_val], 
                      color=:red, 
                      linestyle=:dash, 
                      alpha=0.3,
                      linewidth=1,
                      xlims=(therapy_idx-0.4, therapy_idx+0.4))
            else
                println("  $(therapy_labels[therapy_idx]): No valid data")
            end
        end
        
        # Save the plot
        output_filename = "therapy_comparison_$(metric).png"
        savefig(p, output_filename)
        println("Saved plot to $output_filename")
        
        # Also save as PDF for publications
        pdf_filename = "therapy_comparison_$(metric).pdf"
        savefig(p, pdf_filename)
        println("Saved plot to $pdf_filename")
    end
    
    # Create a combined summary plot
    println("Creating combined summary plot...")
    
    # Normalize all metrics to 0-1 scale for comparison
    normalized_data = DataFrame()
    
    for scenario in therapy_scenarios
        scenario_data = Float64[]
        metric_names = String[]
        
        for metric in metrics
            col_name = Symbol("$(scenario)_$(metric)")
            values = results_df[!, col_name]
            valid_values = values[.!isnan.(values)]
            
            if length(valid_values) > 0
                # Normalize to median value across all therapies for this metric
                all_values = Float64[]
                for s in therapy_scenarios
                    col = Symbol("$(s)_$(metric)")
                    vals = results_df[!, col]
                    append!(all_values, vals[.!isnan.(vals)])
                end
                
                if length(all_values) > 0
                    baseline = median(all_values)
                    if baseline > 0
                        normalized_val = median(valid_values) / baseline
                    else
                        normalized_val = 1.0
                    end
                else
                    normalized_val = 1.0
                end
            else
                normalized_val = NaN
            end
            
            push!(scenario_data, normalized_val)
            push!(metric_names, metric)
        end
        
        normalized_data[!, Symbol(scenario)] = scenario_data
    end
    
    # Create heatmap
    heatmap_matrix = Matrix(normalized_data[:, therapy_scenarios])
    
    p_combined = heatmap(therapy_labels, metric_labels, heatmap_matrix,
                        title="Normalized Treatment Effects\n(Relative to Median Across All Therapies)",
                        xlabel="Therapy Scenario",
                        ylabel="Metric",
                        color=:coolwarm,
                        size=(1000, 600),
                        xrotation=45,
                        left_margin=5Plots.mm,
                        bottom_margin=8Plots.mm)
    
    # Add text annotations with values
    for i in 1:length(metric_labels)
        for j in 1:length(therapy_labels)
            val = heatmap_matrix[i, j]
            if !isnan(val)
                annotate!(p_combined, j, i, text(string(round(val, digits=2)), 10, :white, :center))
            end
        end
    end
    
    savefig(p_combined, "therapy_comparison_heatmap.png")
    savefig(p_combined, "therapy_comparison_heatmap.pdf")
    println("Saved combined heatmap to therapy_comparison_heatmap.png and .pdf")
    
    println("\nPlotting complete! Generated files:")
    for metric in metrics
        println("  - therapy_comparison_$(metric).png")
        println("  - therapy_comparison_$(metric).pdf")
    end
    println("  - therapy_comparison_heatmap.png")
    println("  - therapy_comparison_heatmap.pdf")
end

# Function to plot 5-therapy comparison results
function plot_therapy_5_results()
    println("\n" * "="^50)
    println("Loading 5-therapy comparison results...")
    
    # Load the results from 5-therapy comparison analysis
    if !isfile("therapy_comparison_5_results.csv")
        println("Error: therapy_comparison_5_results.csv not found. Please run therapy_comparison_5.jl first.")
        return
    end
    
    results_df = CSV.read("therapy_comparison_5_results.csv", DataFrame)
    println("Loaded $(nrow(results_df)) parameter combinations")
    
    # Define therapy scenarios and metrics for 5-therapy comparison
    therapy_scenarios = ["control_dose", "dl_reduced", "gamma_reduced", "fdose_reduced", "kappa_increased"]
    therapy_labels = ["Control Dose", "dL ÷ 2", "γ ÷ 2", "f_dose ÷ 2", "κ × 2"]
    metrics = ["L_max", "L_AUC", "T_AUC_normalized", "tumor_ratio"]
    metric_labels = ["L_max", "L_AUC", "T_AUC (normalized)", "Tumor Ratio (Final/Initial)"]
    
    # Set up plot theme
    theme(:default)
    
    # Create plots for each metric
    for (metric_idx, metric) in enumerate(metrics)
        println("Creating plot for $metric...")
        
        # Collect data for this metric across all therapies
        plot_data = []
        therapy_nums = []
        
        for (therapy_idx, scenario) in enumerate(therapy_scenarios)
            col_name = Symbol("$(scenario)_$(metric)")
            values = results_df[!, col_name]
            
            # Remove NaN values
            valid_values = values[.!isnan.(values)]
            
            # Add to plot data
            append!(plot_data, valid_values)
            append!(therapy_nums, fill(therapy_idx, length(valid_values)))
        end
        
        # Create scatter plot
        p = scatter(therapy_nums, plot_data,
                   xlabel="Therapy Scenario",
                   ylabel=metric_labels[metric_idx],
                   title="$(metric_labels[metric_idx]) - 5-Therapy Comparison",
                   xticks=(1:5, therapy_labels),
                   xrotation=45,
                   alpha=0.6,
                   markersize=3,
                   legend=false,
                   size=(800, 600),
                   left_margin=5Plots.mm,
                   bottom_margin=8Plots.mm)
        
        # Add violin plots overlay to show distributions
        violin!(p, therapy_nums, plot_data, 
               alpha=0.3, 
               linewidth=0,
               side=:both)
        
        # Calculate and display summary statistics
        println("\nSummary statistics for $metric (5-therapy):")
        for (therapy_idx, scenario) in enumerate(therapy_scenarios)
            col_name = Symbol("$(scenario)_$(metric)")
            values = results_df[!, col_name]
            valid_values = values[.!isnan.(values)]
            
            if length(valid_values) > 0
                med_val = median(valid_values)
                mean_val = mean(valid_values)
                std_val = std(valid_values)
                println("  $(therapy_labels[therapy_idx]): n=$(length(valid_values)), median=$(round(med_val, digits=4)), mean=$(round(mean_val, digits=4)) ± $(round(std_val, digits=4))")
                
                # Add median line
                hline!(p, [med_val], 
                      color=:red, 
                      linestyle=:dash, 
                      alpha=0.3,
                      linewidth=1,
                      xlims=(therapy_idx-0.4, therapy_idx+0.4))
            else
                println("  $(therapy_labels[therapy_idx]): No valid data")
            end
        end
        
        # Save the plot
        output_filename = "therapy_5_comparison_$(metric).png"
        savefig(p, output_filename)
        println("Saved plot to $output_filename")
        
        # Also save as PDF for publications
        pdf_filename = "therapy_5_comparison_$(metric).pdf"
        savefig(p, pdf_filename)
        println("Saved plot to $pdf_filename")
    end
    
    # Create a combined summary heatmap for 5-therapy comparison
    println("Creating combined summary plot for 5-therapy comparison...")
    
    # Normalize all metrics to 0-1 scale for comparison
    normalized_data = DataFrame()
    
    for scenario in therapy_scenarios
        scenario_data = Float64[]
        metric_names = String[]
        
        for metric in metrics
            col_name = Symbol("$(scenario)_$(metric)")
            values = results_df[!, col_name]
            valid_values = values[.!isnan.(values)]
            
            if length(valid_values) > 0
                # Normalize to median value across all therapies for this metric
                all_values = Float64[]
                for s in therapy_scenarios
                    col = Symbol("$(s)_$(metric)")
                    vals = results_df[!, col]
                    append!(all_values, vals[.!isnan.(vals)])
                end
                
                if length(all_values) > 0
                    baseline = median(all_values)
                    if baseline > 0
                        normalized_val = median(valid_values) / baseline
                    else
                        normalized_val = 1.0
                    end
                else
                    normalized_val = 1.0
                end
            else
                normalized_val = NaN
            end
            
            push!(scenario_data, normalized_val)
            push!(metric_names, metric)
        end
        
        normalized_data[!, Symbol(scenario)] = scenario_data
    end
    
    # Create heatmap
    heatmap_matrix = Matrix(normalized_data[:, therapy_scenarios])
    
    p_combined = heatmap(therapy_labels, metric_labels, heatmap_matrix,
                        title="Normalized Treatment Effects - 5-Therapy Comparison\n(Relative to Median Across All Therapies)",
                        xlabel="Therapy Scenario",
                        ylabel="Metric",
                        color=:coolwarm,
                        size=(900, 600),
                        xrotation=45,
                        left_margin=5Plots.mm,
                        bottom_margin=8Plots.mm)
    
    # Add text annotations with values
    for i in 1:length(metric_labels)
        for j in 1:length(therapy_labels)
            val = heatmap_matrix[i, j]
            if !isnan(val)
                annotate!(p_combined, j, i, text(string(round(val, digits=2)), 10, :white, :center))
            end
        end
    end
    
    savefig(p_combined, "therapy_5_comparison_heatmap.png")
    savefig(p_combined, "therapy_5_comparison_heatmap.pdf")
    println("Saved combined heatmap to therapy_5_comparison_heatmap.png and .pdf")
    
    println("\n5-Therapy Comparison Plotting complete! Generated files:")
    for metric in metrics
        println("  - therapy_5_comparison_$(metric).png")
        println("  - therapy_5_comparison_$(metric).pdf")
    end
    println("  - therapy_5_comparison_heatmap.png")
    println("  - therapy_5_comparison_heatmap.pdf")
    println("="^50 * "\n")
end

# Run the plotting functions
if abspath(PROGRAM_FILE) == @__FILE__
    plot_therapy_results()
    plot_therapy_5_results()
end