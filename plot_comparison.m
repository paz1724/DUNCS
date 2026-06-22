%% plot_comparison.m
%  Read pre-computed .mat results and plot DUNCS vs SparseNet comparison.
%  No training — reads existing results only.
%
%  Usage:
%      >> plot_comparison          % from MATLAB command window
%
%  Output:
%      data/simulations/Plots/loss_comparison.fig  (.png)
%      data/simulations/Plots/accuracy_comparison.fig  (.png)

clear; clc; close all;

%% ---- Paths ----
project_root = fileparts(mfilename('fullpath'));
results_dir  = fullfile(project_root, 'data', 'simulations', 'results');
plots_dir    = fullfile(project_root, 'data', 'simulations', 'Plots');

if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

%% ---- Load results ----
duncs   = load(fullfile(results_dir, 'DUNCS_results.mat'));
sparse  = load(fullfile(results_dir, 'SparseNet_results.mat'));

fprintf('Loaded DUNCS:     %d epochs, test RMSPE = %.6f, test accuracy = %.2f%%\n', ...
    numel(duncs.epochs), duncs.test_rmspe, duncs.test_accuracy * 100);
fprintf('Loaded SparseNet: %d epochs, test RMSPE = %.6f, test accuracy = %.2f%%\n', ...
    numel(sparse.epochs), sparse.test_rmspe, sparse.test_accuracy * 100);

%% ---- Colors ----
blue       = [0.0  0.45 0.74];
light_blue = [0.39 0.70 0.92];
red        = [0.85 0.33 0.10];
light_red  = [0.93 0.60 0.45];

%% ---- Figure 1: Loss Comparison ----
fig1 = figure('Name', 'Loss Comparison', 'Position', [100 100 900 600]);
hold on; grid on;

h1 = plot(duncs.epochs,  duncs.loss_train,  '-',  'Color', blue,       'LineWidth', 2);
h2 = plot(duncs.epochs,  duncs.loss_valid,  '--', 'Color', light_blue, 'LineWidth', 2);
h3 = plot(sparse.epochs, sparse.loss_train, '-',  'Color', red,        'LineWidth', 2);
h4 = plot(sparse.epochs, sparse.loss_valid, '--', 'Color', light_red,  'LineWidth', 2);

xlabel('Epoch', 'FontSize', 12);
ylabel('Loss (RMSPE)', 'FontSize', 12);
title('DUNCS vs SparseNet — Training & Validation Loss', 'FontSize', 14);
legend([h1 h2 h3 h4], ...
    {'DUNCS Train', 'DUNCS Valid', 'SparseNet Train', 'SparseNet Valid'}, ...
    'Location', 'northeast', 'FontSize', 11);
hold off;

savefig(fig1, fullfile(plots_dir, 'loss_comparison.fig'));
saveas(fig1, fullfile(plots_dir, 'loss_comparison.png'));
fprintf('Saved: loss_comparison\n');

%% ---- Figure 2: Accuracy Comparison ----
fig2 = figure('Name', 'Accuracy Comparison', 'Position', [150 100 900 600]);
hold on; grid on;

h1 = plot(duncs.epochs,  duncs.acc_train,  '-',  'Color', blue,       'LineWidth', 2);
h2 = plot(duncs.epochs,  duncs.acc_valid,  '--', 'Color', light_blue, 'LineWidth', 2);
h3 = plot(sparse.epochs, sparse.acc_train, '-',  'Color', red,        'LineWidth', 2);
h4 = plot(sparse.epochs, sparse.acc_valid, '--', 'Color', light_red,  'LineWidth', 2);

xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title('DUNCS vs SparseNet — Source Estimation Accuracy', 'FontSize', 14);
legend([h1 h2 h3 h4], ...
    {'DUNCS Train', 'DUNCS Valid', 'SparseNet Train', 'SparseNet Valid'}, ...
    'Location', 'southeast', 'FontSize', 11);
hold off;

savefig(fig2, fullfile(plots_dir, 'accuracy_comparison.fig'));
saveas(fig2, fullfile(plots_dir, 'accuracy_comparison.png'));
fprintf('Saved: accuracy_comparison\n');

%% ---- Summary ----
fprintf('\n========================================\n');
fprintf('  Summary\n');
fprintf('========================================\n');
fprintf('%-15s %12s %12s\n', 'Model', 'Test RMSPE', 'Test Acc');
fprintf('%s\n', repmat('-', 1, 40));
fprintf('%-15s %12.6f %11.2f%%\n', 'DUNCS',     duncs.test_rmspe,  duncs.test_accuracy * 100);
fprintf('%-15s %12.6f %11.2f%%\n', 'SparseNet', sparse.test_rmspe, sparse.test_accuracy * 100);
fprintf('%-15s %12.6f %12s\n',     'ESPRIT',    duncs.esprit_rmspe, 'N/A');
fprintf('\nPlots saved to: %s\n', plots_dir);
fprintf('Done!\n');
