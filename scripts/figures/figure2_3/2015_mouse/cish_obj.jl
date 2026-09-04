using XLSX

function load_palmer2015(filename)
    xf = XLSX.readxlsx(filename)
    sh = xf[XLSX.sheetnames(xf)[1]]
    m = sh[:]
    # columns: 1-2 NT, 3-4 WT, 5-6 CISH KO; row 1 may be header
    function extract_pair(col_t, col_v)
        ts = Float64[]
        vs = Float64[]
        for r in 1:size(m, 1)
            tv = m[r, col_t]; vv = m[r, col_v]
            tv isa Number && vv isa Number || continue
            (isnan(tv) || isnan(vv)) && continue
            push!(ts, Float64(tv)); push!(vs, Float64(vv))
        end
        return ts, vs
    end
    nt_t, nt_v       = extract_pair(1, 2)
    wt_t, wt_v       = extract_pair(3, 4)
    cish_t, cish_v   = extract_pair(5, 6)
    return (nt_t, nt_v), (wt_t, wt_v), (cish_t, cish_v)
end

if !@isdefined(DATASETS)
    const DATASETS = Ref{Any}(nothing)
end

function init_datasets()
    nt, wt, cish = load_palmer2015("Palmer2015Data.xlsx")
    entries = [
        (nt,   (0.0, 0.0, 0.0), "NT",      1.0),
        (wt,   (0.0, 0.5, 0.0), "WT",      5.0),
        (cish, (0.5, 0.0, 0.0), "CISH KO", 5.0),
    ]
    ds = []
    for ((t, v), cond, name, w) in entries
        scale = max(maximum(v), 10.0)
        push!(ds, (t=t, v=v, cond=cond, name=name, w=w, scale=scale))
    end
    DATASETS[] = ds
    return ds
end

function update_vars(vars_all, fit_names, paras)
    nt = vars_all
    for (i, n) in enumerate(fit_names)
        nt = merge(nt, NamedTuple{(n,)}((paras[i],)))
    end
    return nt
end

function obj_fun(paras, fit_names, vars_all, u0)
    try
        vars_new = update_vars(vars_all, fit_names, paras)
        total = 0.0
        for d in DATASETS[]
            sol = solve_model(vars_new, u0, d.cond; saveat=d.t)
            if sol.retcode != ReturnCode.Success
                return 1e12
            end
            model_T = [sol.u[k][1] for k in 1:length(sol.u)]
            if length(model_T) != length(d.v)
                return 1e12
            end
            total += d.w * sum(((d.v .- model_T) ./ d.scale).^2)
        end
        return isfinite(total) ? total : 1e12
    catch
        return 1e12
    end
end
