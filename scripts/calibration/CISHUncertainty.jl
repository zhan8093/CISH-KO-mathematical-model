module CISHUncertainty

using JSON3
using XLSX
using CSV
using DataFrames
using StaticArrays
using LinearAlgebra
using Glob
# ------------------------------------------------------------
# You must include your ODE model file from the caller script:
include("cish_model_fun_optimized.jl")
# ------------------------------------------------------------

# -------------------------
# Nearest-time index helper
# -------------------------
"""
Return indices into `sol_times` for each `data_times` using nearest neighbor.
Stable and consistent with your plotting approach.
"""
function time_idxs_nearest(data_times::AbstractVector{<:Real},
                           sol_times::AbstractVector{<:Real})
    idxs = Vector{Int}(undef, length(data_times))
    @inbounds for i in eachindex(data_times)
        t = data_times[i]
        idxs[i] = argmin(abs.(sol_times .- t))
    end
    return idxs
end

# -------------------------
# Load fit JSON
# -------------------------
"""
Load saved patient JSON produced by your fitting code.
Returns a NamedTuple.
"""
function load_fit_json(json_path::String)
    d = JSON3.read(read(json_path, String))
    return (; 
        vars_names = Vector{String}(d["vars_names"]),
        init_vals = Vector{Float64}(d["init_vals"]),
        vars_fit_name_patient = Vector{String}(d["vars_fit_name_patient"]),
        paras_fit_patient = Vector{Float64}(d["paras_fit_patient"]),
        vars_fit_name_tumor = Vector{String}(d["vars_fit_name_tumor"]),
        best_paras_tumors = d["best_paras_tumors"]
    )
end

# -------------------------
# Build/update CISHParams
# -------------------------
"""
Build CISHParams from (vars_names, init_vals) in the exact struct field order.

Requires CISHParams to be defined (from your included ODE file).
"""
function build_params_from_init(vars_names::Vector{String}, init_vals::Vector{<:Real})
    d = Dict{Symbol,Float64}()
    @assert length(vars_names) == length(init_vals)
    for (nm, val) in zip(vars_names, init_vals)
        d[Symbol(nm)] = Float64(val)
    end

    return CISHParams(
        d[:t_infusion], d[:t_end], d[:gamma], d[:lambda], d[:s], d[:K],
        d[:kL], d[:kN], d[:muL], d[:muN], d[:dL], d[:dN],
        d[:pL], d[:pN], d[:g], d[:delta], d[:h], d[:l], d[:n], d[:f],
        d[:f_dose], d[:kappa]
    )
end

"""
Set one named field on immutable CISHParams by reconstructing.
"""
function setparam(p::CISHParams, name::String, value::Float64)
    s = Symbol(name)
    return CISHParams(
        s == :t_infusion ? value : p.t_infusion,
        s == :t_end      ? value : p.t_end,
        s == :gamma      ? value : p.gamma,
        s == :lambda     ? value : p.lambda,
        s == :s          ? value : p.s,
        s == :K          ? value : p.K,
        s == :kL         ? value : p.kL,
        s == :kN         ? value : p.kN,
        s == :muL        ? value : p.muL,
        s == :muN        ? value : p.muN,
        s == :dL         ? value : p.dL,
        s == :dN         ? value : p.dN,
        s == :pL         ? value : p.pL,
        s == :pN         ? value : p.pN,
        s == :g          ? value : p.g,
        s == :delta      ? value : p.delta,
        s == :h          ? value : p.h,
        s == :l          ? value : p.l,
        s == :n          ? value : p.n,
        s == :f          ? value : p.f,
        s == :f_dose     ? value : p.f_dose,
        s == :kappa      ? value : p.kappa
    )
end

"""
Apply a vector of fitted values to CISHParams.
"""
function apply_fit(p::CISHParams, names::Vector{String}, vals::Vector{Float64})
    @assert length(names) == length(vals)
    for (nm, v) in zip(names, vals)
        p = setparam(p, nm, v)
    end
    return p
end

# -------------------------
# Data loading / extraction
# -------------------------
"""
Load the three datasets once.
Returns a NamedTuple of raw columns similar to your main().
"""
function load_all_data(; tumor_xlsx::String="CISH_pt_data.xlsx",
                         cish_csv::String="CISHKO_Pct_result.csv",
                         dose_xlsx::String="cish_pt_doses.xlsx")

    tumor_table = DataFrame(XLSX.readtable(tumor_xlsx, "Sheet1"))
    cish_ratio_table = CSV.read(cish_csv, DataFrame)
    dose_data = DataFrame(XLSX.readtable(dose_xlsx, "Sheet1"))

    patient_nums_tumor = string.(tumor_table[:, 1])
    tumors_nums        = string.(tumor_table[:, 4])
    days_nums          = [ismissing(x) ? NaN : Float64(x) for x in tumor_table[:, 3]]
    tumor_size_nums    = [ismissing(x) ? NaN : Float64(x) for x in tumor_table[:, 5]]

    patient_nums_cish  = string.(cish_ratio_table[:, 1])
    days_nums_cish     = [ismissing(x) ? NaN : Float64(x) for x in cish_ratio_table[:, 2]]
    cish_nums_ratio    = [ismissing(x) ? NaN : Float64(x) for x in cish_ratio_table[:, 3]]

    patients_num_dose  = string.(dose_data[:, 1])
    cell_dose          = [ismissing(x) ? NaN : Float64(x) for x in dose_data[:, 2]]

    return (; patient_nums_tumor, tumors_nums, days_nums, tumor_size_nums,
            patient_nums_cish, days_nums_cish, cish_nums_ratio,
            patients_num_dose, cell_dose)
end

"""
Extract one tumor trajectory + cish ratio + u0 for patient_id and tumor_k.

tumor_k is 1-based index into the patient's unique tumor list.
Returns (tumor_size, cish_ratio, u0, tumor_id)
"""
function extract_one_patient_one_tumor(data, patient_id::String, tumor_k::Int;
                                       N0_default::Float64=0.1)

    # tumor ids for patient
    inds_tumor = findall(data.patient_nums_tumor .== patient_id)
    @assert !isempty(inds_tumor) "No tumor records for patient $patient_id"

    id_tumor = unique(data.tumors_nums[inds_tumor])
    @assert 1 ≤ tumor_k ≤ length(id_tumor) "tumor_k out of range for patient $patient_id"
    tumor_id = id_tumor[tumor_k]

    inds = findall((data.patient_nums_tumor .== patient_id) .& (data.tumors_nums .== tumor_id))
    tumor_size = hcat(data.days_nums[inds], data.tumor_size_nums[inds])
    tumor_size = Matrix{Float64}(tumor_size)

    # filter out (0,0) rows
    mask = .!( (tumor_size[:,1] .== 0.0) .& (tumor_size[:,2] .== 0.0) )
    tumor_size = tumor_size[mask, :]

    @assert size(tumor_size,1) ≥ 1 "Empty tumor trajectory after filtering for patient $patient_id tumor $tumor_id"

    # cish ratio for patient
    inds_cish = findall(data.patient_nums_cish .== patient_id)
    if isempty(inds_cish)
        cish_ratio = zeros(Float64, 0, 2)
    else
        cish_ratio = hcat(data.days_nums_cish[inds_cish], data.cish_nums_ratio[inds_cish])
        cish_ratio = Matrix{Float64}(cish_ratio)
        cish_ratio = cish_ratio[cish_ratio[:,1] .>= 0.0, :]
    end

    # dose
    inds_dose = findall(data.patients_num_dose .== patient_id)
    @assert !isempty(inds_dose) "No dose record for patient $patient_id"
    dose = data.cell_dose[inds_dose][1]
    Dose0 = dose / 1e10

    # u0 for tumor: T0 from first observation, L0=0, N0 fixed, Dose0 from file
    T0 = tumor_size[1,2]
    u0 = SVector{4,Float64}(T0, 0.0, N0_default, Dose0)

    return tumor_size, cish_ratio, u0, tumor_id
end

# -------------------------
# Residual vector (matches your objective)
# -------------------------
"""
Residual vector r(θ). SSE = sum(r.^2) matches your objective (up to skipping rules).
"""
function residual_vec(params::CISHParams,
                      u0::SVector{4,Float64},
                      tumor_size::Matrix{Float64},
                      cish_ratio::Matrix{Float64};
                      weight_cish::Float64 = 0.2)

    sol = cish_model_fun_optimized(params, u0)

    tumor_time  = tumor_size[:, 1]
    tumor_value = tumor_size[:, 2]
    tvals_tumor = time_idxs_nearest(tumor_time, sol.x)
    T_model = sol.y[1, tvals_tumor]

    r = Float64[]
    sizehint!(r, size(tumor_size,1) + size(cish_ratio,1))

    @inbounds for i in eachindex(tumor_value)
        if tumor_value[i] > 1e-10 && isfinite(T_model[i])
            push!(r, 1.0 - T_model[i] / tumor_value[i])
        else
            push!(r, 0.0)
        end
    end

    if size(cish_ratio, 1) > 0
        cish_time  = cish_ratio[:, 1]
        cish_value = cish_ratio[:, 2]
        tvals_cish = time_idxs_nearest(cish_time, sol.x)

        Lm = sol.y[2, tvals_cish]
        Nm = sol.y[3, tvals_cish]

        denom = cish_value[1]
        w_sqrt = sqrt(weight_cish)

        @inbounds for i in eachindex(cish_value)
            if isfinite(cish_value[i]) && cish_value[i] > 0 && Nm[i] > 1e-10
                ratio_model = Lm[i] / Nm[i]
                diff = (cish_value[i] - ratio_model) / denom
                push!(r, w_sqrt * diff)
            else
                push!(r, 0.0)
            end
        end
    end

    if any(x -> !isfinite(x), r)
        fill!(r, 1e6)
    end
    return r
end

# -------------------------
# Finite-difference Jacobian
# -------------------------
function fd_jacobian(θ::Vector{Float64}, make_params::Function,
                     u0::SVector{4,Float64},
                     tumor_size::Matrix{Float64},
                     cish_ratio::Matrix{Float64};
                     method::Symbol = :forward,
                     relstep::Float64 = 1e-6,
                     weight_cish::Float64 = 0.2)

    r0 = residual_vec(make_params(θ), u0, tumor_size, cish_ratio; weight_cish=weight_cish)
    n = length(r0); p = length(θ)
    J = Matrix{Float64}(undef, n, p)

    if method == :forward
        for j in 1:p
            h = relstep * max(abs(θ[j]), 1.0)
            θp = copy(θ); θp[j] += h
            rp = residual_vec(make_params(θp), u0, tumor_size, cish_ratio; weight_cish=weight_cish)
            @inbounds J[:, j] = (rp .- r0) ./ h
        end
    elseif method == :central
        for j in 1:p
            h = relstep * max(abs(θ[j]), 1.0)
            θp = copy(θ); θp[j] += h
            θm = copy(θ); θm[j] -= h
            rp = residual_vec(make_params(θp), u0, tumor_size, cish_ratio; weight_cish=weight_cish)
            rm = residual_vec(make_params(θm), u0, tumor_size, cish_ratio; weight_cish=weight_cish)
            @inbounds J[:, j] = (rp .- rm) ./ (2h)
        end
    else
        error("method must be :forward or :central")
    end

    return J, r0
end

# -------------------------
# Gauss–Newton error bars
# -------------------------
"""
Compute Cov ≈ s² (J'J)^+ using pseudoinverse, SE = sqrt(diag(Cov))
"""
function gn_errorbars(J::Matrix{Float64}, r::Vector{Float64})
    n, p = size(J)
    SSE = sum(abs2, r)
    dof = max(n - p, 1)
    s2 = SSE / dof
    Cov = s2 * pinv(J' * J)  # robust for sloppiness / near-singularity
    SE = sqrt.(diag(Cov))
    Corr = Cov ./ (SE * SE')
    return (; SSE, dof, s2, Cov, SE, Corr)
end

# -------------------------
# High-level API
# -------------------------
"""
Compute Jacobian-based SE/CI for one patient & one tumor using saved fit JSON.

mode:
  :patient -> uncertainty for patient-level parameters only (tumor params fixed)
  :tumor   -> uncertainty for tumor-level parameters only (patient params fixed)
  :both    -> uncertainty for concatenated vector

Returns NamedTuple including names, estimates, SE, CI, etc.
"""
function uncertainty_one_tumor(json_path::String,
                               patient_id::String,
                               tumor_k::Int;
                               data = nothing,
                               mode::Symbol = :patient,
                               method::Symbol = :forward,
                               relstep::Float64 = 1e-6,
                               weight_cish::Float64 = 0.2,
                               t_end_override::Union{Nothing,Float64}=nothing)

    data === nothing && (data = load_all_data())

    tumor_size, cish_ratio, u0, tumor_id =
        extract_one_patient_one_tumor(data, patient_id, tumor_k)

    fit = load_fit_json(json_path)
    params_base = build_params_from_init(fit.vars_names, fit.init_vals)

    # Apply t_end override if needed to match what was used in fitting
    if t_end_override !== nothing
        params_base = setparam(params_base, "t_end", t_end_override)
    end

    # Apply patient optimum
    params_patient = apply_fit(params_base, fit.vars_fit_name_patient, fit.paras_fit_patient)

    # Tumor optimum
    tumor_vec = Float64.(fit.best_paras_tumors[tumor_k])
    params_opt = apply_fit(params_patient, fit.vars_fit_name_tumor, tumor_vec)

    if mode == :patient
        θ0 = copy(fit.paras_fit_patient)
        names = fit.vars_fit_name_patient
        make_params = θ -> begin
            p = apply_fit(params_base, fit.vars_fit_name_patient, θ)
            p = apply_fit(p, fit.vars_fit_name_tumor, tumor_vec) # keep tumor fixed
            p
        end
    elseif mode == :tumor
        θ0 = copy(tumor_vec)
        names = fit.vars_fit_name_tumor
        make_params = θ -> apply_fit(params_patient, fit.vars_fit_name_tumor, θ)
    elseif mode == :both
        θ0 = vcat(fit.paras_fit_patient, tumor_vec)
        names = vcat(fit.vars_fit_name_patient, fit.vars_fit_name_tumor)
        np = length(fit.paras_fit_patient)
        make_params = θ -> begin
            p = apply_fit(params_base, fit.vars_fit_name_patient, collect(θ[1:np]))
            p = apply_fit(p, fit.vars_fit_name_tumor, collect(θ[np+1:end]))
            p
        end
    else
        error("mode must be :patient, :tumor, or :both")
    end

    J, r0 = fd_jacobian(θ0, make_params, u0, tumor_size, cish_ratio;
                        method=method, relstep=relstep, weight_cish=weight_cish)

    stats = gn_errorbars(J, r0)
    z = 1.96
    ci_lo = θ0 .- z .* stats.SE
    ci_hi = θ0 .+ z .* stats.SE

    return (; patient_id, tumor_id, tumor_k, mode,
            names, θ0, SE=stats.SE, ci_lo, ci_hi,
            SSE=stats.SSE, dof=stats.dof, s2=stats.s2,
            J, r0, params_opt)
end

"""
Batch over all tumors for one patient JSON, write a CSV table.

tumor_ks: vector of tumor indices; if nothing, runs 1:n_tumors for that patient.

Returns DataFrame, and writes to csv_out if provided.
"""
function uncertainty_patient_to_table(json_path::String, patient_id::String;
                                      data = nothing,
                                      mode::Symbol = :patient,
                                      method::Symbol = :forward,
                                      relstep::Float64 = 1e-6,
                                      weight_cish::Float64 = 0.2,
                                      t_end_override::Union{Nothing,Float64}=nothing,
                                      csv_out::Union{Nothing,String}=nothing)

    data === nothing && (data = load_all_data())

    # determine n tumors for patient
    inds_tumor = findall(data.patient_nums_tumor .== patient_id)
    id_tumor = unique(data.tumors_nums[inds_tumor])
    n_tumors = length(id_tumor)

    rows = DataFrame(patient_id=String[], tumor_id=String[], tumor_k=Int[],
                     mode=String[], param=String[],
                     estimate=Float64[], SE=Float64[],
                     ci_lo=Float64[], ci_hi=Float64[],
                     SSE=Float64[], dof=Int[])

    for k in 1:n_tumors
        out = uncertainty_one_tumor(json_path, patient_id, k;
                                    data=data, mode=mode,
                                    method=method, relstep=relstep,
                                    weight_cish=weight_cish,
                                    t_end_override=t_end_override)

        for i in eachindex(out.names)
            push!(rows, (
                patient_id,
                out.tumor_id,
                out.tumor_k,
                String(out.mode),
                out.names[i],
                out.θ0[i],
                out.SE[i],
                out.ci_lo[i],
                out.ci_hi[i],
                out.SSE,
                out.dof
            ))
        end
    end

    if csv_out !== nothing
        CSV.write(csv_out, rows)
    end
    return rows
end

"""
Extract patient_id from filename like:
  Patient_<PATIENT_ID>_paras.json -> <PATIENT_ID>
"""
function patient_id_from_json_path(json_path::String)
    base = splitpath(json_path)[end]
    # remove prefix/suffix
    s = replace(base, r"^Patient_" => "")
    s = replace(s, r"_paras\.json$" => "")
    return s
end

"""
Batch: scan directory for Patient_*_paras.json, run uncertainty for all patients,
append into one table, write single CSV.

- mode: :patient, :tumor, or :both
- method: :forward or :central
- t_end_overrides: optional Dict{String,Float64} mapping patient_id => t_end used in fitting
"""
function uncertainty_all_patients_to_csv(; 
        dir::String=".",
        pattern::String="Patient_*_paras.json",
        mode::Symbol = :patient,
        method::Symbol = :forward,
        relstep::Float64 = 1e-6,
        weight_cish::Float64 = 0.2,
        t_end_overrides::Dict{String,Float64} = Dict{String,Float64}(),
        csv_out::String="Uncertainty_AllPatients.csv",
        tumor_xlsx::String="CISH_pt_data.xlsx",
        cish_csv::String="CISHKO_Pct_result.csv",
        dose_xlsx::String="cish_pt_doses.xlsx"
    )

    # load data once
    data = load_all_data(tumor_xlsx=tumor_xlsx, cish_csv=cish_csv, dose_xlsx=dose_xlsx)

    json_files = sort(glob(pattern, dir))
    @assert !isempty(json_files) "No JSON files found with pattern=$pattern in dir=$dir"

    rows = DataFrame(
        patient_id=String[], tumor_id=String[], tumor_k=Int[],
        mode=String[], param=String[],
        estimate=Float64[], SE=Float64[],
        ci_lo=Float64[], ci_hi=Float64[],
        SSE=Float64[], dof=Int[]
    )

    for json_path in json_files
        pid = patient_id_from_json_path(json_path)

        # optional per-patient t_end override
        t_end_override = haskey(t_end_overrides, pid) ? t_end_overrides[pid] : nothing

        # determine number of tumors for this patient from the data (same logic as fitting)
        inds_tumor = findall(data.patient_nums_tumor .== pid)
        if isempty(inds_tumor)
            @warn "Skipping patient with no tumor records in data" pid json_path
            continue
        end
        id_tumor = unique(data.tumors_nums[inds_tumor])
        n_tumors = length(id_tumor)

        @info "Running uncertainty" patient_id=pid json=json_path n_tumors=n_tumors mode=mode method=method

        for k in 1:n_tumors
            out = uncertainty_one_tumor(json_path, pid, k;
                                        data=data,
                                        mode=mode,
                                        method=method,
                                        relstep=relstep,
                                        weight_cish=weight_cish,
                                        t_end_override=t_end_override)

            for i in eachindex(out.names)
                push!(rows, (
                    pid,
                    out.tumor_id,
                    out.tumor_k,
                    String(out.mode),
                    out.names[i],
                    out.θ0[i],
                    out.SE[i],
                    out.ci_lo[i],
                    out.ci_hi[i],
                    out.SSE,
                    out.dof
                ))
            end
        end
    end

    CSV.write(csv_out, rows)
    return rows
end

export load_all_data, uncertainty_one_tumor, uncertainty_patient_to_table,
       uncertainty_all_patients_to_csv


end # module
