clear
close all

% --- Variable names and values ---
vars_names = ["t_infusion","t_end",...
              "gamma","K",...
              "kL","kLo","kN","muL","muN",...
              "dL","dN","pL","pN",...
              "delta","h",...
              "muZ","a","b","bL","kappa"];

init_vals = [7.5, 42, ...
             0.1748, 1,...
             1.462, 0.5077, 0.0811, 1.2e-4, 1.57e-4,...
             0.0796, 0.03, 0.2, 0.05,...
             0.1, 1e3,...
             0.7, 0.8891, 9.3033, 2.2837, 0.1];

ub_vals = [7.5, 42,...
           0.5, 10,...
           7.2, 7.2, 7.2, 2.5e-3, 2.5e-3,...
           0.08, 0.04, 2.971, 2.971,...
           0.12, 1e3,...
           0.7, 10, 10, 10, 2.0];


lb_vals = [7.5, 42,...
           0.1, 1,...
           0.014, 0.005, 0.001, 5e-9, 5e-9,...
           0.001, 0.02, 0.124, 0.124,...
           0.12, 1e3,...
           0.7, 0.001, 0.01, 0.01, 0.01];

vars_all = struct();
vars_all_ub = struct();
vars_all_lb = struct();

for i = 1:length(vars_names)
    name = vars_names(i);
    vars_all.(name) = init_vals(i);
    vars_all_ub.(name) = ub_vals(i);
    vars_all_lb.(name) = lb_vals(i);
end

% --- Optimization target parameters ---
vars_fit_name = ["gamma","kL","kLo","kN","dL","a","b","bL"];
nFit = length(vars_fit_name);

paras_init = zeros(1, nFit);
paras_lb   = zeros(1, nFit);
paras_ub   = zeros(1, nFit);
for i = 1:nFit
    paras_init(i) = vars_all.(vars_fit_name(i));
    paras_lb(i)   = vars_all_lb.(vars_fit_name(i));
    paras_ub(i)   = vars_all_ub.(vars_fit_name(i));
end

% Initial conditions (MODIFIED: add dose_L and dose_Lo compartments)
T0 = 25; L0 = 0; Lw0 = 0; N0 = 0; Z0 = 0; dose_L0 = 0; dose_Lo0 = 0;
u0 = [T0 L0 Lw0 N0 Z0 dose_L0 dose_Lo0];

% Constraints
A = []; b = []; Aeq = []; beq = []; nonlcon = [];

% Optimization settings
options = optimset('MaxIter', 100, 'Display', 'off');

nTotal = 10;
% Preallocate results
all_fvals = zeros(nTotal, 1);
all_paras = zeros(nTotal, nFit);

% Explicitly pass needed variables into parfor
vars_all_local = vars_all;
u0_local = u0;
vars_fit_name_local = vars_fit_name;

% --- Progress monitor ---
global progress
progress = 0;
q = parallel.pool.DataQueue;
afterEach(q, @updateProgress);

% --- Main optimization loop ---
parfor i = 1:nTotal
    % Random initialization (force row vector)
    paras_0 = paras_lb + rand(1, nFit) .* (paras_ub - paras_lb);

    [paras_optim, fval] = fmincon(@(paras) ...
        cish_obj_fun_meta(paras, vars_fit_name_local, vars_all_local, u0_local), ...
        paras_0, A, b, Aeq, beq, ...
        paras_lb, paras_ub, ...
        nonlcon, options);

    all_fvals(i) = fval;
    all_paras(i,:) = paras_optim(:)';  % Force row vector

    % Report progress
    send(q, i);
end

% --- Select best result ---
[best_fval, best_idx] = min(all_fvals);
best_paras = all_paras(best_idx, :);
paras_fit = best_paras;

% --- Update parameter struct with optimized values ---
paras_after_opt = vars_all;
for i = 1:nFit
    paras_after_opt.(vars_fit_name(i)) = paras_fit(i);
end

% (Optional) Display best parameters
disp('Best optimized parameters:');
disp(paras_after_opt);

conds = [0 0 0;
         0 0 5;
         0 1 0;
         0 1 5;
         1 0 0;
         1 0 5];

filenames = {
    'no cells.xlsx',
    'no cells antiPD1.xlsx',
    'wild type isotype control.xlsx',
    'wild type antiPD1.xlsx',
    'CISH KO isotype control.xlsx',
    'CISH KO antiPD1.xlsx'
};

labels = {'No cells', 'No cells + aPD1', 'WT', 'WT + aPD1', 'CISH KO', 'CISH KO + aPD1'};

sols = cell(1, 6);
data = cell(1, 6);

for i = 1:6
    sols{i} = cish_model_fun_v2(paras_after_opt, u0, conds(i,:));
    data{i} = readmatrix(filenames{i});
end

% === Plot: 2022 Mouse Tumor Size ===
fig = figure('Position', [100 100 500 400], 'Color', 'w');
hold on; box on;

% --- Style settings ---
colors = lines(6);
markerSize = 50;
lineWidth = 2.2;
axFontSize = 12;
titleFontSize = 14;
legendFontSize = 9;

% --- Simulation curves ---
for i = 1:6
    plot(sols{i}.x, sols{i}.y(1,:), '-', ...
        'Color', colors(i,:), 'LineWidth', lineWidth);
end

% --- Data points ---
for i = 1:6
    scatter(data{i}(1,:), data{i}(2,:), markerSize, colors(i,:), ...
        'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end

xlabel('Time (days)', 'FontSize', axFontSize);
ylabel('Tumor area (mm^2)', 'FontSize', axFontSize);
title('Tumor Cells', 'FontSize', titleFontSize, 'FontWeight', 'bold');
legend(labels, 'Location', 'best', 'FontSize', legendFontSize, 'Box', 'off');
ylim([0 600]);
set(gca, 'FontSize', axFontSize, 'LineWidth', 1);

% --- Save ---
exportgraphics(fig, 'mouse_fit_2022.png', 'Resolution', 300);

% --- Export Data to CSV (Fixed Dimensions) ---
export_tables = {};
disp('Processing data for export...');

for i = 1:6
    % 1. Extract experimental time and values (ensure they are columns)
    time_points = data{i}(1, :)';
    exp_values  = data{i}(2, :)';

    % 2. Clean simulation data to ensure unique time points
    sim_t = sols{i}.x;
    sim_y = sols{i}.y(1,:);

    % Find unique time points to prevent interp1 errors
    [unique_t, unique_idx] = unique(sim_t);
    unique_y = sim_y(unique_idx);

    % 3. Interpolate
    sim_values_interp = interp1(unique_t, unique_y, time_points, 'linear', 'extrap');

    % 4. Create a label column for this condition
    cond_label = repmat(labels(i), length(time_points), 1);

    % 5. Create a temporary table for this iteration
    temp_tbl = table(time_points, exp_values, sim_values_interp, cond_label, ...
        'VariableNames', {'Time', 'Experimental_Data', 'Model_Simulation', 'Condition'});

    % 6. Store in cell array
    export_tables{i} = temp_tbl;
end

% Combine all tables vertically
final_data_table = vertcat(export_tables{:});

% Define filename
output_filename = 'all_results_combined.csv';

% Write to CSV
writetable(final_data_table, output_filename);

fprintf('Data successfully saved to %s\n', output_filename);

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function updateProgress(~)
    global progress
    progress = progress + 1;
    fprintf('Finished %d/10\n', progress);
end