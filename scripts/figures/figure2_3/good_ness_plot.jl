# data_analysis.jl
# Analysis of model predictions vs observed tumor volume data with goodness-of-fit diagnostics

using CSV, DataFrames, JSON3, Plots, Statistics, Printf, XLSX
using StaticArrays

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

# Include the model functions
include("../cish_model_fun_optimized.jl")

# Define parameter structure (copy from main file for compatibility)
struct CISHParams{T<:Real}
    t_infusion::T
    t_end::T
    gamma::T
    lambda::T
    s::T
    K::T
    kL::T
    kN::T
    muL::T
    muN::T
    dL::T
    dN::T
    pL::T
    pN::T
    g::T
    delta::T
    h::T
    l::T
    n::T
    f::T
    f_dose::T
    kappa::T
end

# Constructor from dictionary
function CISHParams(vars_dict::Dict{Symbol, T}) where T<:Real
    CISHParams(
        vars_dict[:t_infusion], vars_dict[:t_end], vars_dict[:gamma], vars_dict[:lambda],
        vars_dict[:s], vars_dict[:K], vars_dict[:kL], vars_dict[:kN], vars_dict[:muL],
        vars_dict[:muN], vars_dict[:dL], vars_dict[:dN], vars_dict[:pL], vars_dict[:pN],
        vars_dict[:g], vars_dict[:delta], vars_dict[:h], vars_dict[:l], vars_dict[:n],
        vars_dict[:f], vars_dict[:f_dose], vars_dict[:kappa]
    )
end

# Convert struct back to dictionary for compatibility
function params_to_dict(p::CISHParams)
    Dict{Symbol, Float64}(
        :t_infusion => p.t_infusion, :t_end => p.t_end, :gamma => p.gamma,
        :lambda => p.lambda, :s => p.s, :K => p.K, :kL => p.kL,
        :kN => p.kN, :muL => p.muL, :muN => p.muN, :dL => p.dL,
        :dN => p.dN, :pL => p.pL, :pN => p.pN, :g => p.g,
        :delta => p.delta, :h => p.h, :l => p.l, :n => p.n,
        :f => p.f, :f_dose => p.f_dose, :kappa => p.kappa
    )
end

# Update specific parameters in struct
function update_params(base::CISHParams, param_names::Vector{String}, param_values::Vector{T}) where T
    # Convert to dict for easy updating
    d = params_to_dict(base)
    
    # Update specified parameters
    for (name, val) in zip(param_names, param_values)
        d[Symbol(name)] = val
    end
    
    return CISHParams(d)
end

function analyze_model_predictions()
    println("Starting comprehensive model prediction analysis...")
    
    # Load tumor volume data from Excel file
    base_dir = dirname(pwd())  # Get parent directory
    tumor_data_raw = XLSX.readtable(joinpath(base_dir, "CISH_pt_data.xlsx"), "Sheet1")
    tumor_data = DataFrame(tumor_data_raw)
    
    # Load dose data
    dose_data_raw = XLSX.readtable(joinpath(base_dir, "cish_pt_doses.xlsx"), "Sheet1")
    dose_data = DataFrame(dose_data_raw)
    
    # Load CISH ratio data from CSV
    cish_ratio_data = CSV.read(joinpath(base_dir, "CISHKO_Pct_result.csv"), DataFrame)
    
    # Extract data columns (following main_optimized.jl structure)
    patient_nums_tumor = string.(tumor_data[:, 1])
    tumor_nums = string.(tumor_data[:, 4])
    days_nums = tumor_data[:, 3]
    tumor_size_nums = tumor_data[:, 5]
    
    patients_num_dose = string.(dose_data[:, 1])
    cell_dose = dose_data[:, 2]
    il2_doses = dose_data[:, 3]
    
    # Get unique patient IDs
    id_patient = unique(patient_nums_tumor)
    println("Found $(length(id_patient)) patients: $id_patient")
    
    # Parameter setup (exactly as in main_optimized.jl)
    vars_names = ["t_infusion","t_end",
                "gamma","lambda","s","K",
                "kL","kN","muL","muN",
                "dL","dN","pL","pN","g",
                "delta","h","l","n","f","f_dose","kappa"]

    init_vals = [0.001, 85.0,
                0.01, 0.58, 4.24, 1.00, 
                0.03, 0.0005, 4e-4, 4e-4,
                0.05, 0.030, 0.05, 0.05, 1.0e3,
                0.01, 1e3, 578, 200, 1.0, 0.2, 0.25]
    
    vars_fit_name_patient = ["dL","kL","pL","muL","pN","f_dose","kappa"]
    vars_fit_name_tumor = ["gamma"]
    
    # Initialize storage for tumor volume predictions and observations
    all_observed = Float64[]
    all_predicted = Float64[]
    all_patient_info = String[]
    all_tumor_info = String[]
    all_time_info = Float64[]
    
    # Initialize storage for CISH ratio predictions and observations
    all_observed_ratio = Float64[]
    all_predicted_ratio = Float64[]
    all_patient_info_ratio = String[]
    all_time_info_ratio = Float64[]
    
    # Default initial condition values
    N0_default = 0.1
    
    # Process each patient (only first 6 as in main_optimized.jl)
    for i in 1:min(6, length(id_patient))
        patient_id = id_patient[i]
        println("\nProcessing patient $i: $patient_id")
        
        # Load optimized parameters for this patient
        param_file = "../Patient_$(patient_id)_paras.json"
        if !isfile(param_file)
            println("Warning: Parameter file $param_file not found, skipping...")
            continue
        end
        
        params_data = JSON3.read(read(param_file, String))
        paras_fit_patient = params_data.paras_fit_patient
        best_paras_tumors = params_data.best_paras_tumors
        
        # Create initial parameter dictionary
        vars_all = Dict(Symbol(n) => v for (n, v) in zip(vars_names, init_vals))
        
        # Update with fitted patient parameters
        for (j, param_name) in enumerate(vars_fit_name_patient)
            vars_all[Symbol(param_name)] = paras_fit_patient[j]
        end
        
        # Set patient-specific t_end values (from main_optimized.jl)
        if i == 12
            vars_all[:t_end] = 180.0
        elseif i == 4
            vars_all[:t_end] = 45.0
        elseif i == 5
            vars_all[:t_end] = 60.0          
        elseif i == 6
            vars_all[:t_end] = 30.0
        elseif i == 8
            vars_all[:t_end] = 40.0
        elseif i == 9
            vars_all[:t_end] = 30.0
        elseif i == 10
            vars_all[:t_end] = 60.0
        elseif i == 11
            vars_all[:t_end] = 30.0
        end
        
        # Create parameter struct with patient-level fitted parameters
        params_patient = CISHParams(vars_all)
        
        # Get patient data
        inds_tumor = findall(patient_nums_tumor .== patient_id)
        id_tumor = unique(tumor_nums[inds_tumor])
        
        # Get dose data
        inds_dose = findall(patients_num_dose .== patient_id)
        dose = [cell_dose[inds_dose][1]]
        
        # Get CISH ratio data for this patient (exclude negative days)
        patient_cish_data = filter(row -> row.Sample_ID == patient_id && row.Day >= 0, cish_ratio_data)
        cish_time_points = sort(unique(patient_cish_data.Day))
        println("CISH ratio time points for $patient_id: $cish_time_points")
        
        # Prepare tumor data for each tumor
        tumors_size = zeros(length(inds_tumor), length(id_tumor)*2)
        for (j, tumor_id) in enumerate(id_tumor)
            inds = findall((patient_nums_tumor .== patient_id) .& (tumor_nums .== tumor_id))
            tumors_size[1:length(inds), 2*j-1:2*j] .= hcat(days_nums[inds], tumor_size_nums[inds])
        end
        
        # Set initial conditions following main_optimized.jl structure
        N0 = N0_default
        L0 = 0.0  # Force L0 = 0
        Dose0 = dose[1]/1e10  # Get initial dose from file, scaled
        
        println("Number of tumors: $(length(id_tumor))")
        
        # Process each tumor
        for k in 1:length(id_tumor)
            # Extract tumor-specific data
            tumor_size = tumors_size[:, 2k-1:2k]
            mask = .!( (tumor_size[:, 1] .== 0.0) .& (tumor_size[:, 2] .== 0.0) )
            tumor_size = tumor_size[mask, :]
            
            if size(tumor_size, 1) == 0
                continue
            end
            
            println("  Processing tumor $k with $(size(tumor_size,1)) data points")
            
            # Get tumor-specific parameters
            if k <= length(best_paras_tumors)
                tumor_params_vec = Vector{Float64}(best_paras_tumors[k])
                vars_fit_name_tumor_vec = Vector{String}(vars_fit_name_tumor)
                
                # Update parameters with tumor-specific values
                params_tumor = update_params(params_patient, vars_fit_name_tumor_vec, tumor_params_vec)
                
                # Set tumor-specific initial condition (T0 from first observation)
                u0 = [tumor_size[1,2], L0, N0, Dose0]  # [T0, L0, N0, Dose0]
                u0_tumor_static = SVector{4, Float64}(u0...)
                
                # Solve ODE
                try
                    sol = cish_model_fun_optimized(params_tumor, u0_tumor_static)
                    
                    # Extract predictions at observed time points
                    for row_idx in 1:size(tumor_size, 1)
                        time_point = tumor_size[row_idx, 1]
                        observed_volume = tumor_size[row_idx, 2]
                        
                        # Find closest time point in solution
                        if time_point >= minimum(sol.x) && time_point <= maximum(sol.x)
                            # Interpolate to get prediction at exact time point
                            idx = argmin(abs.(sol.x .- time_point))
                            predicted_volume = sol.y[1, idx]  # T component is tumor volume
                            
                            # Store data
                            push!(all_observed, observed_volume)
                            push!(all_predicted, predicted_volume)
                            push!(all_patient_info, patient_id)
                            push!(all_tumor_info, "T$k")
                            push!(all_time_info, time_point)
                            
                            println("    Time $(time_point): Observed = $(round(observed_volume, digits=4)), Predicted = $(round(predicted_volume, digits=4))")
                        end
                    end
                    
                    # Also extract CISH ratio predictions for this tumor
                    for cish_time in cish_time_points
                        if cish_time >= minimum(sol.x) && cish_time <= maximum(sol.x)
                            # Get observed CISH ratio data
                            observed_cish_data = filter(row -> row.Sample_ID == patient_id && row.Day == cish_time, patient_cish_data)
                            if nrow(observed_cish_data) > 0
                                observed_ratio = observed_cish_data.CISHKO_over_CISHWT[1]
                                
                                # Interpolate to get prediction at exact time point
                                idx = argmin(abs.(sol.x .- cish_time))
                                L_pred = sol.y[2, idx]  # CISHKO cells
                                N_pred = sol.y[3, idx]  # Wildtype cells
                                predicted_ratio = L_pred / max(N_pred, 1e-12)  # Avoid division by zero
                                
                                # Store CISH ratio data (only once per patient-time combination, not per tumor)
                                if k == 1  # Only store once for the first tumor to avoid duplicates
                                    push!(all_observed_ratio, observed_ratio)
                                    push!(all_predicted_ratio, predicted_ratio)
                                    push!(all_patient_info_ratio, patient_id)
                                    push!(all_time_info_ratio, cish_time)
                                    
                                    println("    CISH Time $(cish_time): Observed = $(round(observed_ratio, digits=4)), Predicted = $(round(predicted_ratio, digits=4))")
                                end
                            end
                        end
                    end
                    
                catch e
                    println("    Error solving ODE for tumor $k: $e")
                    continue
                end
            else
                println("    No parameters found for tumor $k")
            end
        end
    end
    
    # Calculate goodness-of-fit statistics for tumor volumes
    if length(all_observed) > 0
        # Pearson correlation
        correlation = cor(all_observed, all_predicted)
        
        # Root mean square error
        rmse = sqrt(mean((all_observed - all_predicted).^2))
        
        # Mean absolute error
        mae = mean(abs.(all_observed - all_predicted))
        
        # R-squared
        ss_res = sum((all_observed - all_predicted).^2)
        ss_tot = sum((all_observed .- mean(all_observed)).^2)
        r_squared = 1 - ss_res/ss_tot
        
        # Print statistics for tumor volumes
        println("\n" * "="^60)
        println("TUMOR VOLUME GOODNESS-OF-FIT STATISTICS")
        println("="^60)
        @printf("Number of data points: %d\n", length(all_observed))
        @printf("Pearson correlation: %.4f\n", correlation)
        @printf("R-squared: %.4f\n", r_squared)
        @printf("RMSE: %.6f\n", rmse)
        @printf("MAE: %.6f\n", mae)
        println("="^60)
        
        # Create scatter plot
        p = scatter(all_observed, all_predicted, 
                   xlabel="Observed Tumor Volume", 
                   ylabel="Fitted Tumor Volume",
                   #title="Model Fitted vs Observed Data\n(Pearson r = $(round(correlation, digits=3)), R² = $(round(r_squared, digits=3)))",
                   legend=false,
                   markersize=7,
                   markershape=:circle,
                   markerstrokewidth=0.8,
                   markerstrokecolor=:gray30,
                   alpha=0.75,
                   color=RGB(0.0, 0.45, 0.70),
                   size=(700, 600),
                   guidefontsize=16,
                   tickfontsize=13,
                   margin=8Plots.mm)
        
        # Add 1:1 line for reference
        min_val = min(minimum(all_observed), minimum(all_predicted))
        max_val = max(maximum(all_observed), maximum(all_predicted))
        pad = 0.05 * (max_val - min_val)
        plot!(p, [min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 
              line=:dash, color=:black, linewidth=2.5, label="Perfect fit (1:1)")
        
        # Add correlation coefficient as text annotation
        annotate!(p, [(0.05 * (max_val - min_val) + min_val, 
                      0.92 * (max_val - min_val) + min_val, 
                      text("r = $(round(correlation, digits=3))\nR2 = $(round(r_squared, digits=3))\nRMSE = $(round(rmse, digits=4))", 
                           :left, 13, "Computer Modern"))])
        
        # Save plot
        savefig(p, "tumor_volume_prediction_vs_observed.png")
        println("\nPlot saved as: tumor_volume_prediction_vs_observed.png")
        

        

        
    else
        println("No valid tumor volume predictions could be generated!")
    end
    
    # Calculate goodness-of-fit statistics for CISH ratios
    if length(all_observed_ratio) > 0
        # Pearson correlation
        correlation_ratio = cor(all_observed_ratio, all_predicted_ratio)
        
        # Root mean square error
        rmse_ratio = sqrt(mean((all_observed_ratio - all_predicted_ratio).^2))
        
        # Mean absolute error
        mae_ratio = mean(abs.(all_observed_ratio - all_predicted_ratio))
        
        # R-squared
        ss_res_ratio = sum((all_observed_ratio - all_predicted_ratio).^2)
        ss_tot_ratio = sum((all_observed_ratio .- mean(all_observed_ratio)).^2)
        r_squared_ratio = 1 - ss_res_ratio/ss_tot_ratio
        
        # Print statistics for CISH ratios
        println("\n" * "="^60)
        println("CISH RATIO GOODNESS-OF-FIT STATISTICS")
        println("="^60)
        @printf("Number of data points: %d\n", length(all_observed_ratio))
        @printf("Pearson correlation: %.4f\n", correlation_ratio)
        @printf("R-squared: %.4f\n", r_squared_ratio)
        @printf("RMSE: %.6f\n", rmse_ratio)
        @printf("MAE: %.6f\n", mae_ratio)
        println("="^60)
        

        

        

        
        # Log-scale analysis for CISH ratios
        # Filter out zero or negative values before taking log
        valid_indices = (all_observed_ratio .> 0) .& (all_predicted_ratio .> 0)
        if sum(valid_indices) > 0
            log_observed_ratio = log10.(all_observed_ratio[valid_indices])
            log_predicted_ratio = log10.(all_predicted_ratio[valid_indices])
            
            # Calculate log-scale statistics
            correlation_log = cor(log_observed_ratio, log_predicted_ratio)
            rmse_log = sqrt(mean((log_observed_ratio - log_predicted_ratio).^2))
            mae_log = mean(abs.(log_observed_ratio - log_predicted_ratio))
            ss_res_log = sum((log_observed_ratio - log_predicted_ratio).^2)
            ss_tot_log = sum((log_observed_ratio .- mean(log_observed_ratio)).^2)
            r_squared_log = 1 - ss_res_log/ss_tot_log
            
            println("\n" * "="^60)
            println("CISH RATIO LOG-SCALE GOODNESS-OF-FIT STATISTICS")
            println("="^60)
            @printf("Number of valid data points (>0): %d\n", sum(valid_indices))
            @printf("Pearson correlation (log): %.4f\n", correlation_log)
            @printf("R-squared (log): %.4f\n", r_squared_log)
            @printf("RMSE (log): %.6f\n", rmse_log)
            @printf("MAE (log): %.6f\n", mae_log)
            println("="^60)
            
            # Create log-scale scatter plot
            p_log = scatter(log_observed_ratio, log_predicted_ratio, 
                           xlabel="log10(Observed CISH Ratio)", 
                           ylabel="log10(Fitted CISH Ratio)",
                           #title="CISH Ratio (Log Scale): Model Fitted vs Observed Data\n(Pearson r = $(round(correlation_log, digits=3)), R² = $(round(r_squared_log, digits=3)))",
                           legend=false,
                           markersize=7,
                           markershape=:circle,
                           markerstrokewidth=0.8,
                           markerstrokecolor=:gray30,
                           alpha=0.75,
                           color=RGB(0.56, 0.27, 0.68),
                           size=(700, 600),
                           guidefontsize=16,
                           tickfontsize=13,
                           margin=8Plots.mm)
            
            # Add 1:1 line for reference
            min_log = min(minimum(log_observed_ratio), minimum(log_predicted_ratio))
            max_log = max(maximum(log_observed_ratio), maximum(log_predicted_ratio))
            log_pad = 0.05 * (max_log - min_log)
            plot!(p_log, [min_log - log_pad, max_log + log_pad], [min_log - log_pad, max_log + log_pad], 
                  line=:dash, color=:black, linewidth=2.5, label="Perfect fit (1:1)")
            
            # Adjust x-axis limits to better fit the data
            xlims!(p_log, (-2.5, max_log + 0.1 * (max_log - min_log)))
            ylims!(p_log, (min_log - 0.1 * (max_log - min_log), max_log + 0.1 * (max_log - min_log)))
            
            # Add correlation coefficient as text annotation
            annotate!(p_log, [(min_log + 0.05 * (max_log - min_log), 
                              max_log - 0.08 * (max_log - min_log), 
                              text("r = $(round(correlation_log, digits=3))\nR2 = $(round(r_squared_log, digits=3))\nRMSE = $(round(rmse_log, digits=4))", 
                                   :left, 13, "Computer Modern"))])
            
            savefig(p_log, "cish_ratio_log_prediction_vs_observed.png")
            println("CISH ratio log-scale plot saved as: cish_ratio_log_prediction_vs_observed.png")
            

            

        else
            println("No valid positive values for log-scale analysis!")
        end
        
    else
        println("No valid CISH ratio predictions could be generated!")
    end
    
    println("\nAnalysis complete!")
end

# Run the analysis
if abspath(PROGRAM_FILE) == @__FILE__
    analyze_model_predictions()
end
