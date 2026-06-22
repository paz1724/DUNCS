%% run_comparison.m
%  MATLAB wrapper for DUNCS vs SparseNet (and any future model) comparison.
%
%  Reads models_config.json, calls Python to train each model,
%  reads the raw .mat output, and creates overlaid comparison plots.
%  Saves every figure as .fig and .png.
%
%  Usage:
%      >> run_comparison          % from MATLAB command window
%
%  Prerequisites:
%      - Python environment with project dependencies available on PATH
%      - models_config.json in the project root

clear; clc; close all;

%% ---- Configuration ----
project_root = fileparts(mfilename('fullpath'));
config_file  = fullfile(project_root, 'models_config.json');

% Read JSON config
config = jsondecode(fileread(config_file));
model_names  = fieldnames(config.models);
samples_size = config.samples_size;
test_ratio   = config.test_ratio;
output_dir   = fullfile(project_root, config.output_dir);
plots_dir    = fullfile(output_dir, 'plots');

if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

% Colors for up to 6 models
colors_solid  = lines(numel(model_names));
colors_dashed = min(colors_solid + 0.35, 1);  % lighter shade for validation

%% ---- Phase 1: Train each model via Python ----
fprintf('\n========================================\n');
fprintf('  Phase 1: Training models via Python\n');
fprintf('========================================\n\n');

for i = 1:numel(model_names)
    name     = model_names{i};
    cfg_path = config.models.(name).config_path;

    result_file = fullfile(output_dir, [name '_results.mat']);

    fprintf('--- Training %s ---\n', name);
    cmd = sprintf( ...
        'python "%s" --model_name %s --config_path "%s" --samples_size %d --test_ratio %.2f --output_dir "%s"', ...
        fullfile(project_root, 'train_single_model.py'), ...
        name, cfg_path, samples_size, test_ratio, output_dir);

    fprintf('Command: %s\n', cmd);
    status = system(cmd);

    if status ~= 0
        error('Python training failed for model %s (exit code %d).', name, status);
    end

    if ~isfile(result_file)
        error('Expected output file not found: %s', result_file);
    end
    fprintf('%s training complete. Results at: %s\n\n', name, result_file);
end

%% ---- Phase 2: Load all results ----
fprintf('\n========================================\n');
fprintf('  Phase 2: Loading results\n');
fprintf('========================================\n\n');

results = struct();
for i = 1:numel(model_names)
    name = model_names{i};
    result_file = fullfile(output_dir, [name '_results.mat']);
    data = load(result_file);
    results.(name) = data;
    fprintf('Loaded %s: %d epochs, test RMSPE = %.6f\n', ...
        name, numel(data.epochs), data.test_rmspe);
end

%% ---- Phase 3: Create comparison plots ----
fprintf('\n========================================\n');
fprintf('  Phase 3: Generating comparison plots\n');
fprintf('========================================\n\n');

% ---- Plot 1: Training Loss ----
fig1 = figure('Name', 'Training Loss Comparison', 'Position', [100 100 900 600]);
hold all; grid on;
legend_entries = {};
for i = 1:numel(model_names)
    name = model_names{i};
    d = results.(name);
    plot(d.epochs, d.loss_train, '-', 'Color', colors_solid(i,:), 'LineWidth', 2);
    legend_entries{end+1} = name; %#ok<SAGROW>
end
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss (RMSPE)', 'FontSize', 12);
title('Training Loss Comparison', 'FontSize', 14);
legend(legend_entries, 'Location', 'northeast', 'FontSize', 11);
hold off;
savefig(fig1, fullfile(plots_dir, 'training_loss_comparison.fig'));
saveas(fig1, fullfile(plots_dir, 'training_loss_comparison.png'));
fprintf('Saved: training_loss_comparison\n');

% ---- Plot 2: Validation Loss ----
fig2 = figure('Name', 'Validation Loss Comparison', 'Position', [150 100 900 600]);
hold all; grid on;
legend_entries = {};
for i = 1:numel(model_names)
    name = model_names{i};
    d = results.(name);
    plot(d.epochs, d.loss_valid, '-', 'Color', colors_solid(i,:), 'LineWidth', 2);
    legend_entries{end+1} = name; %#ok<SAGROW>
end
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss (RMSPE)', 'FontSize', 12);
title('Validation Loss Comparison', 'FontSize', 14);
legend(legend_entries, 'Location', 'northeast', 'FontSize', 11);
hold off;
savefig(fig2, fullfile(plots_dir, 'validation_loss_comparison.fig'));
saveas(fig2, fullfile(plots_dir, 'validation_loss_comparison.png'));
fprintf('Saved: validation_loss_comparison\n');

% ---- Plot 3: Train & Validation Loss (all models, solid=train, dashed=valid) ----
fig3 = figure('Name', 'Train & Validation Loss', 'Position', [200 100 900 600]);
hold all; grid on;
legend_entries = {};
h_lines = [];
for i = 1:numel(model_names)
    name = model_names{i};
    d = results.(name);
    h1 = plot(d.epochs, d.loss_train, '-', 'Color', colors_solid(i,:), 'LineWidth', 2);
    h2 = plot(d.epochs, d.loss_valid, '--', 'Color', colors_dashed(i,:), 'LineWidth', 2);
    h_lines = [h_lines, h1, h2]; %#ok<AGROW>
    legend_entries{end+1} = [name ' Train']; %#ok<SAGROW>
    legend_entries{end+1} = [name ' Valid']; %#ok<SAGROW>
end
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss (RMSPE)', 'FontSize', 12);
title('Training & Validation Loss', 'FontSize', 14);
legend(h_lines, legend_entries, 'Location', 'northeast', 'FontSize', 11);
hold off;
savefig(fig3, fullfile(plots_dir, 'train_valid_loss_comparison.fig'));
saveas(fig3, fullfile(plots_dir, 'train_valid_loss_comparison.png'));
fprintf('Saved: train_valid_loss_comparison\n');

% ---- Plot 4: Source Estimation Accuracy (only models that have it) ----
has_accuracy = false(numel(model_names), 1);
for i = 1:numel(model_names)
    name = model_names{i};
    acc = results.(name).acc_train;
    has_accuracy(i) = ~isempty(acc) && any(acc > 0);
end

if any(has_accuracy)
    fig4 = figure('Name', 'Source Estimation Accuracy', 'Position', [250 100 900 600]);
    hold all; grid on;
    legend_entries = {};
    h_lines = [];
    for i = 1:numel(model_names)
        if ~has_accuracy(i), continue; end
        name = model_names{i};
        d = results.(name);
        h1 = plot(d.epochs, d.acc_train, '-', 'Color', colors_solid(i,:), 'LineWidth', 2);
        h2 = plot(d.epochs, d.acc_valid, '--', 'Color', colors_dashed(i,:), 'LineWidth', 2);
        h_lines = [h_lines, h1, h2]; %#ok<AGROW>
        legend_entries{end+1} = [name ' Train']; %#ok<SAGROW>
        legend_entries{end+1} = [name ' Valid']; %#ok<SAGROW>
    end
    xlabel('Epoch', 'FontSize', 12);
    ylabel('Accuracy (%)', 'FontSize', 12);
    title('Source Estimation Accuracy', 'FontSize', 14);
    legend(h_lines, legend_entries, 'Location', 'southeast', 'FontSize', 11);
    hold off;
    savefig(fig4, fullfile(plots_dir, 'accuracy_comparison.fig'));
    saveas(fig4, fullfile(plots_dir, 'accuracy_comparison.png'));
    fprintf('Saved: accuracy_comparison\n');
end

% ---- Plot 5: Test RMSPE Bar Chart ----
fig5 = figure('Name', 'Test RMSPE Comparison', 'Position', [300 100 900 600]);
bar_names = {};
bar_values = [];
bar_colors_list = [];

for i = 1:numel(model_names)
    name = model_names{i};
    bar_names{end+1} = name; %#ok<SAGROW>
    bar_values(end+1) = results.(name).test_rmspe; %#ok<SAGROW>
    bar_colors_list = [bar_colors_list; colors_solid(i,:)]; %#ok<AGROW>
end

% Add classical ESPRIT from the first model's results
first_model = model_names{1};
bar_names{end+1} = 'ESPRIT (classical)';
bar_values(end+1) = results.(first_model).esprit_rmspe;
bar_colors_list = [bar_colors_list; 0.3 0.7 0.3];

b = bar(bar_values, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.2);
b.CData = bar_colors_list;
set(gca, 'XTickLabel', bar_names, 'FontSize', 11);
ylabel('RMSPE Loss', 'FontSize', 12);
title('Test Set RMSPE Comparison', 'FontSize', 14);
grid on; set(gca, 'GridAlpha', 0.3);

% Add value labels on bars
for k = 1:numel(bar_values)
    text(k, bar_values(k), sprintf('%.4f', bar_values(k)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold', 'FontSize', 10);
end

savefig(fig5, fullfile(plots_dir, 'test_rmspe_comparison.fig'));
saveas(fig5, fullfile(plots_dir, 'test_rmspe_comparison.png'));
fprintf('Saved: test_rmspe_comparison\n');

%% ---- Summary Table ----
fprintf('\n========================================\n');
fprintf('  Final Comparison\n');
fprintf('========================================\n\n');
fprintf('%-25s %12s %12s\n', 'Method', 'Test RMSPE', 'Accuracy');
fprintf('%s\n', repmat('-', 1, 50));
for i = 1:numel(model_names)
    name = model_names{i};
    rmspe = results.(name).test_rmspe;
    acc   = results.(name).test_accuracy;
    if isempty(acc) || acc == 0
        acc_str = 'N/A';
    else
        acc_str = sprintf('%.2f%%', acc * 100);
    end
    fprintf('%-25s %12.6f %12s\n', name, rmspe, acc_str);
end
fprintf('%-25s %12.6f %12s\n', 'ESPRIT (classical)', ...
    results.(first_model).esprit_rmspe, 'N/A');

fprintf('\nPlots saved to: %s\n', plots_dir);
fprintf('Done!\n');
