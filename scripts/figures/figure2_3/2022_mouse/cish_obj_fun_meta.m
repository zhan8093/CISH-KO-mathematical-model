function [obj_fun_val] = cish_obj_fun_meta(paras, vars_fit_name, vars_all, u0)
    try
        paras_fit = paras;
        
        % Update vars_all with new values
        for i = 1:length(vars_fit_name)
            vars_all.(vars_fit_name(i)) = paras_fit(i);
        end
        
        norm = 1;
        get_obj = @(filename, cond) ...
            get_squared_error(filename, cond, u0, vars_all, norm);
        
        obj1 = get_obj('no cells.xlsx', [0 0 0]);
        obj2 = get_obj('no cells antiPD1.xlsx', [0 0 5]);
        obj3 = get_obj('wild type isotype control.xlsx', [0 1 0]);
        obj4 = get_obj('wild type antiPD1.xlsx', [0 1 5]);
        obj5 = get_obj('CISH KO isotype control.xlsx', [1 0 0]);
        obj6 = get_obj('CISH KO antiPD1.xlsx', [1 0 5]);
        
        obj_fun_val = obj1 + obj2 + obj3 + 10* obj4 + 50 * obj5 + 200* obj6;
        
        if ~isfinite(obj_fun_val)
            obj_fun_val = 1e12;
        end
    catch
        obj_fun_val = 1e12;
        disp('Objective function failed at:');
        disp(paras);
    end
end

function obj = get_squared_error(filename, cond, u0, vars_all, norm)
    data = readmatrix(filename);
    data_time = data(1, :);
    data_cells = data(2, :);
    sol = cish_model_fun_v2(vars_all, u0, cond);
    tvals = time_idxs(data_time, sol.x);
    model_vals = sol.y(1, tvals);  % tumor size
    obj = sum(((data_cells - model_vals) / norm).^2);
end

function tvals = time_idxs(t_exp, t_sim)
    t_sim = t_sim(:)';  % Force row vector
    tvals = nan(1, length(t_exp));
    for i = 1:length(t_exp)
        [~, tvals(i)] = min(abs(t_sim - t_exp(i)));
    end
end