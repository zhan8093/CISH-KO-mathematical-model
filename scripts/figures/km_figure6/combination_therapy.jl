using CSV, DataFrames, JSON3, Printf, XLSX
using StaticArrays

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

# Include the model functions
model_file = find_file("cish_model_fun_optimized.jl")
if model_file !== nothing
    include(model_file)
else
    error("Could not find cish_model_fun_optimized.jl in current or parent directories")
end

function main()
    println("Loading existing parameter data...")
    
    # Load the existing CSV file with parameters
    params_file_with_ratio = find_file("patient_tumor_params_with_ratio.csv")
    params_file = find_file("patient_tumor_params.csv")
    
    if params_file_with_ratio !== nothing
        params_df = CSV.read(params_file_with_ratio, DataFrame)
        println("Loaded parameters from: $params_file_with_ratio")
    elseif params_file !== nothing
        params_df = CSV.read(params_file, DataFrame)
        println("Loaded parameters from: $params_file")
    else
        error("Could not find patient_tumor_params.csv or patient_tumor_params_with_ratio.csv in current or parent directories")
    end
    
    println("Found $(nrow(params_df)) parameter combinations")
    
    # Add new columns if they don't exist
    therapy_scenarios = ["no_therapy", "cish", "double_cish", "anti_pd1", "cish_anti_pd1", "double_cish_anti_pd1"]
    metrics = ["L_max", "L_AUC", "T_AUC_normalized", "tumor_ratio"]
    
    # Add basic columns
    if !("initial_tumor_size" in names(params_df))
        params_df[!, :initial_tumor_size] = Vector{Float64}(undef, nrow(params_df))
    end
    if !("cish_dose" in names(params_df))
        params_df[!, :cish_dose] = Vector{Float64}(undef, nrow(params_df))
    end
    
    # Add columns for each therapy scenario
    for scenario in therapy_scenarios
        for metric in metrics
            col_name = Symbol("$(scenario)_$(metric)")
            if !(string(col_name) in names(params_df))
                params_df[!, col_name] = Vector{Float64}(undef, nrow(params_df))
            end
        end
    end
    
    # Load supporting data files
    println("Loading tumor and dose data...")
    
    # Load tumor volume data
    tumor_data_file = find_file("CISH_pt_data.xlsx")
    if tumor_data_file === nothing
        error("Could not find CISH_pt_data.xlsx in current or parent directories")
    end
    
    tumor_data_raw = XLSX.readtable(tumor_data_file, "Sheet1")
    tumor_data = DataFrame(tumor_data_raw)
    println("Loaded tumor data from: $tumor_data_file")
    patient_nums_tumor = string.(tumor_data[:, 1])
    tumor_nums = string.(tumor_data[:, 4])
    days_nums = tumor_data[:, 3]
    tumor_size_nums = tumor_data[:, 5]
    
    # Load dose data
    dose_data_file = find_file("cish_pt_doses.xlsx")
    if dose_data_file === nothing
        error("Could not find cish_pt_doses.xlsx in current or parent directories")
    end
    
    dose_data_raw = XLSX.readtable(dose_data_file, "Sheet1")
    dose_data = DataFrame(dose_data_raw)
    println("Loaded dose data from: $dose_data_file")
    patients_num_dose = string.(dose_data[:, 1])
    cell_dose = dose_data[:, 2]
    
    # Get unique patient IDs (same as in good_ness_plot.jl)
    id_patient = unique(patient_nums_tumor)
    println("Found patients: $id_patient")
    
    # Parameter setup (exactly as in good_ness_plot.jl)
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
    
    # Function to calculate metrics for a given therapy scenario
    function calculate_therapy_metrics(vars_all, T0, dose_value, therapy_scenario)
        try
            # Create a copy of parameters for this scenario
            vars_scenario = copy(vars_all)
            
            # Set anti-PD1 dose
            anti_pd1_dose = 0.5
            
            # Modify parameters based on therapy scenario
            if therapy_scenario == "no_therapy"
                Dose0 = 0.0
            elseif therapy_scenario == "cish"
                Dose0 = dose_value/1e10
            elseif therapy_scenario == "double_cish"
                Dose0 = 2*dose_value/1e10
            elseif therapy_scenario == "anti_pd1"
                Dose0 = 0.0
                vars_scenario[:kL] = vars_scenario[:kL] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
                vars_scenario[:kN] = vars_scenario[:kN] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
            elseif therapy_scenario == "cish_anti_pd1"
                Dose0 = dose_value/1e10
                vars_scenario[:kL] = vars_scenario[:kL] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
                vars_scenario[:kN] = vars_scenario[:kN] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
            elseif therapy_scenario == "double_cish_anti_pd1"
                Dose0 = 2*dose_value/1e10
                vars_scenario[:kL] = vars_scenario[:kL] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
                vars_scenario[:kN] = vars_scenario[:kN] * 1.9 / (0.9 + exp(-9*anti_pd1_dose))
            # 5-Therapy comparison scenarios
            elseif therapy_scenario == "control_dose"
                Dose0 = dose_value/1e10
            elseif therapy_scenario == "dl_reduced"
                Dose0 = dose_value/1e10
                vars_scenario[:dL] = vars_scenario[:dL] / 2
            elseif therapy_scenario == "gamma_reduced"
                Dose0 = dose_value/1e10
                vars_scenario[:gamma] = vars_scenario[:gamma] / 2
            elseif therapy_scenario == "fdose_reduced"
                Dose0 = dose_value/1e10
                vars_scenario[:f_dose] = vars_scenario[:f_dose] / 2
            elseif therapy_scenario == "kappa_increased"
                Dose0 = dose_value/1e10
                vars_scenario[:kappa] = vars_scenario[:kappa] * 2
            end
            
            # Create parameter struct for this scenario
            params_struct = CISHParams(
                vars_scenario[:t_infusion],
                vars_scenario[:t_end],
                vars_scenario[:gamma],
                vars_scenario[:lambda],
                vars_scenario[:s],
                vars_scenario[:K],
                vars_scenario[:kL],
                vars_scenario[:kN],
                vars_scenario[:muL],
                vars_scenario[:muN],
                vars_scenario[:dL],
                vars_scenario[:dN],
                vars_scenario[:pL],
                vars_scenario[:pN],
                vars_scenario[:g],
                vars_scenario[:delta],
                vars_scenario[:h],
                vars_scenario[:l],
                vars_scenario[:n],
                vars_scenario[:f],
                vars_scenario[:f_dose],
                vars_scenario[:kappa]
            )
            
            # Set initial conditions
            L0 = 0.0
            N0 = 0.1
            u0 = [T0, L0, N0, Dose0]
            u0_static = SVector{4, Float64}(u0...)
            
            # Solve ODE
            sol = cish_model_fun_optimized(params_struct, u0_static)
            
            if sol !== nothing
                # Extract solution in 0-100 day range
                t_points = sol.x
                day_100_mask = t_points .<= 100.0
                t_filtered = t_points[day_100_mask]
                T_vals = sol.y[1, day_100_mask]
                L_vals = sol.y[2, day_100_mask]
                
                if length(t_filtered) > 1
                    # Calculate L_max
                    L_max = maximum(L_vals)
                    
                    # Calculate AUC using trapezoidal rule
                    L_AUC = 0.0
                    T_AUC = 0.0
                    
                    for j in 1:(length(t_filtered)-1)
                        dt = t_filtered[j+1] - t_filtered[j]
                        L_AUC += 0.5 * (L_vals[j] + L_vals[j+1]) * dt
                        T_AUC += 0.5 * (T_vals[j] + T_vals[j+1]) * dt
                    end
                    
                    T_AUC_normalized = T_AUC / T0
                    
                    # Calculate tumor ratio (final / initial)
                    T_final = T_vals[end]
                    tumor_ratio = T_final / T0
                    
                    return (L_max, L_AUC, T_AUC_normalized, tumor_ratio)
                else
                    return (NaN, NaN, NaN, NaN)
                end
            else
                return (NaN, NaN, NaN, NaN)
            end
            
        catch e
            return (NaN, NaN, NaN, NaN)
        end
    end
    
    # Process each row in the parameter dataframe
    for (row_idx, row) in enumerate(eachrow(params_df))
        println("Processing $(row.ID) ($(row_idx)/$(nrow(params_df)))")
        
        # Parse patient and tumor info
        id_parts = split(row.ID, "_")
        patient_id = id_parts[2]  # e.g., "002"
        tumor_num = parse(Int, id_parts[4])  # e.g., 1
        patient_full = "UMN_$(patient_id)"
        
        # Find patient index in the original patient list
        i = findfirst(id_patient .== patient_full)
        if i === nothing
            println("  Patient $patient_full not found in data")
            params_df[row_idx, :initial_tumor_size] = NaN
            params_df[row_idx, :cish_dose] = NaN
            # Set all therapy scenario metrics to NaN
            for scenario in therapy_scenarios
                for metric in metrics
                    params_df[row_idx, Symbol("$(scenario)_$(metric)")] = NaN
                end
            end
            continue
        end
        
        try
            # Create initial parameter dictionary
            vars_all = Dict(Symbol(n) => v for (n, v) in zip(vars_names, init_vals))
            
            # Update with fitted parameters from CSV
            vars_all[:dL] = row.dL
            vars_all[:kL] = row.kL
            vars_all[:muL] = row.muL
            vars_all[:pL] = row.pL
            vars_all[:pN] = row.pN
            vars_all[:f_dose] = row.f_dose
            vars_all[:kappa] = row.kappa
            vars_all[:gamma] = row.gamma
            
            # Set patient-specific t_end values (from good_ness_plot.jl)
            # if i == 12
            #     vars_all[:t_end] = 180.0
            # elseif i == 4
            #     vars_all[:t_end] = 45.0
            # elseif i == 5
            #     vars_all[:t_end] = 60.0          
            # elseif i == 6
            #     vars_all[:t_end] = 30.0
            # elseif i == 8
            #     vars_all[:t_end] = 40.0
            # elseif i == 9
            #     vars_all[:t_end] = 30.0
            # elseif i == 10
            #     vars_all[:t_end] = 60.0
            # elseif i == 11
            #     vars_all[:t_end] = 30.0
            # end
            
            # Create parameter struct (unused in current implementation)
            # params_struct = CISHParams(vars_all)  # This was causing errors
            
            # Get dose for this patient
            inds_dose = findall(patients_num_dose .== patient_full)
            if length(inds_dose) > 0
                dose = [cell_dose[inds_dose][1]]
                Dose0 = dose[1]/1e10
                dose_value = dose[1]
            else
                Dose0 = 0.1/1e10
                dose_value = 0.1
            end
            
            # Get initial tumor size
            tumor_id_str = "T_$(tumor_num)"
            inds_tumor = findall((patient_nums_tumor .== patient_full) .& (tumor_nums .== tumor_id_str))
            
            if length(inds_tumor) > 0
                T0 = tumor_size_nums[inds_tumor[1]]
            else
                T0 = 1.0
            end
            
            # Store initial tumor size and dose
            params_df[row_idx, :initial_tumor_size] = T0
            params_df[row_idx, :cish_dose] = dose_value
            
            println("  T0: $T0, Dose: $dose_value")
            
            # Calculate metrics for all therapy scenarios
            for scenario in therapy_scenarios
                L_max, L_AUC, T_AUC_normalized, tumor_ratio = calculate_therapy_metrics(vars_all, T0, dose_value, scenario)
                
                # Store results with scenario-specific column names
                params_df[row_idx, Symbol("$(scenario)_L_max")] = L_max
                params_df[row_idx, Symbol("$(scenario)_L_AUC")] = L_AUC
                params_df[row_idx, Symbol("$(scenario)_T_AUC_normalized")] = T_AUC_normalized
                params_df[row_idx, Symbol("$(scenario)_tumor_ratio")] = tumor_ratio
                
                @printf("    %s: L_max=%.6f, L_AUC=%.6f, T_AUC_norm=%.6f, tumor_ratio=%.6f\n", 
                       scenario, L_max, L_AUC, T_AUC_normalized, tumor_ratio)
            end
            
        catch e
            println("  Error: ", e)
            params_df[row_idx, :initial_tumor_size] = NaN
            params_df[row_idx, :cish_dose] = NaN
            # Set all therapy scenario metrics to NaN
            for scenario in therapy_scenarios
                for metric in metrics
                    params_df[row_idx, Symbol("$(scenario)_$(metric)")] = NaN
                end
            end
        end
    end
    
    # Save results to CSV file
    output_filename = "combination_therapy_results.csv"
    CSV.write(output_filename, params_df)
    println("Results saved to $(output_filename)")
    println("Total rows processed: $(nrow(params_df))")
    
    # ===== NOW CALCULATE 5-THERAPY COMPARISON RESULTS =====
    println("\n" * "="^60)
    println("Starting 5-therapy comparison analysis...")
    println("="^60)
    
    # Create new dataframe for 5-therapy comparison
    therapy_scenarios_5 = ["control_dose", "dl_reduced", "gamma_reduced", "fdose_reduced", "kappa_increased"]
    results_5_df = copy(params_df[:, [:ID, :initial_tumor_size, :cish_dose]])
    
    # Add columns for each 5-therapy scenario
    for scenario in therapy_scenarios_5
        for metric in metrics
            col_name = Symbol("$(scenario)_$(metric)")
            if !(string(col_name) in names(results_5_df))
                results_5_df[!, col_name] = Vector{Float64}(undef, nrow(results_5_df))
            end
        end
    end
    
    # Re-process each row for 5-therapy scenarios
    for (row_idx, row) in enumerate(eachrow(params_df))
        println("Processing $(row.ID) for 5-therapy ($(row_idx)/$(nrow(params_df)))")
        
        # Parse patient and tumor info
        id_parts = split(row.ID, "_")
        patient_id = id_parts[2]
        tumor_num = parse(Int, id_parts[4])
        patient_full = "UMN_$(patient_id)"
        
        # Find patient index in the original patient list
        i = findfirst(id_patient .== patient_full)
        if i === nothing
            println("  Patient $patient_full not found in data")
            for scenario in therapy_scenarios_5
                for metric in metrics
                    results_5_df[row_idx, Symbol("$(scenario)_$(metric)")] = NaN
                end
            end
            continue
        end
        
        try
            # Create initial parameter dictionary
            vars_all = Dict(Symbol(n) => v for (n, v) in zip(vars_names, init_vals))
            
            # Update with fitted parameters from CSV
            vars_all[:dL] = row.dL
            vars_all[:kL] = row.kL
            vars_all[:muL] = row.muL
            vars_all[:pL] = row.pL
            vars_all[:pN] = row.pN
            vars_all[:f_dose] = row.f_dose
            vars_all[:kappa] = row.kappa
            vars_all[:gamma] = row.gamma
            
            # Get dose for this patient
            inds_dose = findall(patients_num_dose .== patient_full)
            if length(inds_dose) > 0
                dose = [cell_dose[inds_dose][1]]
                dose_value = dose[1]
            else
                dose_value = 0.1
            end
            
            # Get initial tumor size
            tumor_id_str = "T_$(tumor_num)"
            inds_tumor = findall((patient_nums_tumor .== patient_full) .& (tumor_nums .== tumor_id_str))
            
            if length(inds_tumor) > 0
                T0 = tumor_size_nums[inds_tumor[1]]
            else
                T0 = 1.0
            end
            
            println("  T0: $T0, Dose: $dose_value")
            
            # Calculate metrics for all 5-therapy scenarios
            for scenario in therapy_scenarios_5
                L_max, L_AUC, T_AUC_normalized, tumor_ratio = calculate_therapy_metrics(vars_all, T0, dose_value, scenario)
                
                # Store results with scenario-specific column names
                results_5_df[row_idx, Symbol("$(scenario)_L_max")] = L_max
                results_5_df[row_idx, Symbol("$(scenario)_L_AUC")] = L_AUC
                results_5_df[row_idx, Symbol("$(scenario)_T_AUC_normalized")] = T_AUC_normalized
                results_5_df[row_idx, Symbol("$(scenario)_tumor_ratio")] = tumor_ratio
                
                @printf("    %s: L_max=%.6f, L_AUC=%.6f, T_AUC_norm=%.6f, tumor_ratio=%.6f\n", 
                       scenario, L_max, L_AUC, T_AUC_normalized, tumor_ratio)
            end
            
        catch e
            println("  Error: ", e)
            for scenario in therapy_scenarios_5
                for metric in metrics
                    results_5_df[row_idx, Symbol("$(scenario)_$(metric)")] = NaN
                end
            end
        end
    end
    
    # Save 5-therapy results to separate CSV file
    output_filename_5 = "therapy_comparison_5_results.csv"
    CSV.write(output_filename_5, results_5_df)
    println("\n5-Therapy Results saved to $(output_filename_5)")
    println("Total rows processed: $(nrow(results_5_df))")
    println("="^60)
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end