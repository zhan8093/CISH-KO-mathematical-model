using DataFrames, CSV, JSON3, Statistics, Plots, Plots.PlotMeasures, XLSX

"""
    analyze_optimal_parameter_correlations(; verbose=true)

Extracts optimal fitted parameters for each patient and analyzes correlations.
"""
function analyze_optimal_parameter_correlations(; verbose=true)
    
    # Load patient IDs
    tumor_data = XLSX.readtable("CISH_pt_data.xlsx", "Sheet1")
    tumor_table = DataFrame(tumor_data)
    id_patient = unique(string.(tumor_table[:, 1]))
    
    # Extract fitted parameters from each patient's JSON
    param_rows = []
    for i in 1:length(id_patient)-1
        pid = id_patient[i]
        json_path = "Patient_$(pid)_paras.json"
        
        if !isfile(json_path)
            continue
        end
        
        outdata = JSON3.read(read(json_path), Dict)
        vars_fit_name = String.(outdata["vars_fit_name_patient"])
        paras_fit = Float64.(outdata["paras_fit_patient"])
        
        # Create dictionary with proper type (Any to accept both String and Float64)
        param_dict = Dict{String, Any}()
        param_dict["Patient"] = pid
        for (param_name, param_val) in zip(vars_fit_name, paras_fit)
            param_dict[param_name] = param_val
        end
        push!(param_rows, param_dict)
    end
    
    params_df = DataFrame(param_rows)
    CSV.write("optimal_parameters_by_patient.csv", params_df)
    
    # Compute correlations
    param_names = setdiff(names(params_df), ["Patient"])
    param_matrix = Matrix{Float64}(params_df[!, param_names])
    cor_matrix = cor(param_matrix)
    
    cor_df = DataFrame(cor_matrix, Symbol.(param_names))
    cor_df[!, :Parameter] = param_names
    select!(cor_df, :Parameter, Not(:Parameter))
    CSV.write("optimal_parameter_correlations.csv", cor_df)
    
    # Correlation heatmap
    p_heatmap = heatmap(param_names, param_names, cor_matrix,
                       xlabel="Parameter",
                       ylabel="Parameter",
                       title="Optimal Parameter Correlations",
                       color=:RdBu,
                       clims=(-1, 1),
                       aspect_ratio=:equal,
                       xrotation=45,
                       size=(900, 800),
                       bottom_margin=12mm,
                       left_margin=12mm,
                       colorbar_title="Correlation")
    
    # Annotate values
    for i in 1:length(param_names)
        for j in 1:length(param_names)
            val = cor_matrix[i, j]
            txt_color = abs(val) > 0.5 ? :white : :black
            annotate!(p_heatmap, j, i, text(string(round(val, digits=2)), 7, txt_color))
        end
    end
    
    savefig(p_heatmap, "optimal_parameter_correlation_heatmap.png")
    
    if verbose
        println("Results saved:")
        println("  • optimal_parameters_by_patient.csv")
        println("  • optimal_parameter_correlations.csv")
        println("  • optimal_parameter_correlation_heatmap.png")
    end
    
    return params_df, cor_df
end

analyze_optimal_parameter_correlations()