function [obj_fun_val] = cish_obj_fun_meta(paras, vars_fit_name, vars_all, u0)
    try
        % Update parameters
        for i = 1:length(vars_fit_name)
            vars_all.(vars_fit_name(i)) = paras(i);
        end
        norm = 1;
        
        % Read shared Excel file once
        data = readmatrix('Palmer2015Data.xlsx');
        
        % Extract NT (columns A and B)
        data_nt = data(:, 1:2);
        % WT (columns C and D)
        data_wt = data(:, 3:4);
        % CISH KO (columns E and F)
        data_cish = data(:, 5:6);
        
        % Define therapy conditions
        cond_nt   = [0 0 0];
        cond_wt   = [0 0.5 0];
        cond_cish = [0.5 0 0];
        
        % Compute squared errors
        obj1 = get_squared_error(data_nt,   cond_nt,   u0, vars_all, norm);
        obj2 = get_squared_error(data_wt,   cond_wt,   u0, vars_all, norm);
        obj3 = get_squared_error(data_cish, cond_cish, u0, vars_all, norm);
        
        obj_fun_val = obj1 + 10* obj2 + 15 * obj3;
        
        if ~isfinite(obj_fun_val)
            obj_fun_val = 1e12;
        end
    catch
        obj_fun_val = 1e12;
        disp('Objective function failed at:');
        disp(paras);
    end
end

function obj = get_squared_error(data, cond, u0, vars_all, norm)
    data = data(~any(isnan(data), 2), :);  % remove empty rows
    data_time = data(:, 1)';
    data_cells = data(:, 2)';
    sol = cish_model_fun_v2(vars_all, u0, cond);
    tvals = time_idxs(data_time, sol.x);
    model_vals = sol.y(1, tvals);  % tumor size
    obj = sum(((data_cells - model_vals) / norm).^2);
end

function tvals = time_idxs(t_exp, t_sim)
    tvals = nan(1, length(t_exp));
    for i = 1:length(t_exp)
        [~, tvals(i)] = min(abs(t_sim - t_exp(i)));
    end
end