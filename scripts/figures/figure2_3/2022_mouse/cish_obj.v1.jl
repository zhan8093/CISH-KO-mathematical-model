using XLSX

function load_data(filename)
    xf = XLSX.readxlsx(filename)
    sh = xf[XLSX.sheetnames(xf)[1]]
    m = sh[:]
    t = Float64.(collect(m[1, :]))
    v = Float64.(collect(m[2, :]))
    return t, v
end

if !@isdefined(DATASETS)
    const DATASETS = Ref{Any}(nothing)
end

function init_datasets()
    files = [
        ("no cells.xlsx",                  (0.0, 0.0, 0.0), 1.0),
        ("no cells antiPD1.xlsx",          (0.0, 0.0, 5.0), 1.0),
        ("wild type isotype control.xlsx", (0.0, 0.5, 0.0), 1.0),
        ("wild type antiPD1.xlsx",         (0.0, 0.5, 5.0), 10.0),
        ("CISH KO isotype control.xlsx",   (0.5, 0.0, 0.0), 50.0),
        ("CISH KO antiPD1.xlsx",           (0.5, 0.0, 5.0), 200.0),
    ]
    ds = []
    for (f, cond, w) in files
        t, v = load_data(f)
        push!(ds, (t=t, v=v, cond=cond, w=w))
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
            total += d.w * sum((d.v .- model_T).^2)
        end
        return isfinite(total) ? total : 1e12
    catch
        return 1e12
    end
end
