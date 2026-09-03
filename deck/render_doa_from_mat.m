function [] = render_doa_from_mat()
%% Render DOA overlays for ALL table algorithms straight through DoA_Wrapper's Plot_DOA.
%  Loads results_<case>_r<r>.mat (written by the Python bridge), each containing:
%     angles    [1 x K]  angle grid in degrees
%     anglesGT  [1 x M]  ground-truth angles in degrees
%     results   struct   one field per algorithm (sanitized name), each with
%                        .psd_dB [1 x K] (NaN allowed for angle-only methods),
%                        .estimated_DoA [1 x n], .df_error (scalar)
%     order     cellstr  (optional) field order for a stable legend / colors
%  Renders each with the real Plot_DOA and saves a PNG.

set(0, 'DefaultFigureVisible', 'off');
indir = 'C:\Users\Daniel\AppData\Local\Temp\claude\c--GitHub-DUNCS\64e8dd82-9435-48e6-a804-bb899502d269\scratchpad\doa_png4';
cases = {'single', 'reuse15', 'multipath15'};
for ci = 1:numel(cases)
    for r = 1:2
        mat = fullfile(indir, sprintf('results_%s_r%d.mat', cases{ci}, r));
        if ~exist(mat, 'file'); fprintf('missing %s\n', mat); continue; end
        S = load(mat);
        Plot_DOA(double(S.angles), S.results, double(S.anglesGT));
        png = fullfile(indir, sprintf('doa_all_%s_r%d.png', cases{ci}, r));
        exportgraphics(gcf, png, 'Resolution', 150);
        close(gcf);
        fprintf('wrote %s\n', png);
    end
end
end

%% ~~~~~~~~~~~~~~~~~~~~ Plot_DOA (from DoA_Wrapper.m, headless-adapted) ~~~~~~~~~~~~~~~~~~~~ %%
function [] = Plot_DOA(angles, results, anglesGT)
figure('Position', [100 100 1400 900]);
hold all;

modeFields = fields(results);
lineHandles = struct();
for j = 1:numel(modeFields)
    mode = modeFields{j};
    lineHandles.(mode) = plot(angles, results.(mode).psd_dB, 'LineWidth', 1.5);
end

xlabel('Angle (degrees)');
ylabel('Power Spectrum (dB, peak-normalized)');
xline(anglesGT, '--g', 'LineWidth', 2);
legendLabels = cellfun(@(m) sprintf('%s (DF Error: %.2f\xB0)', strrep(m, '_', '-'), results.(m).df_error), modeFields, 'UniformOutput', false);
legendStr = vertcat(legendLabels, {'Ground Truth Angles'});
legend(legendStr, 'Location', 'eastoutside', 'Interpreter', 'none');

% Add xline for each estimated DoA with the color matching the respective plot line
for j = 1:numel(modeFields)
    mode = modeFields{j};
    estimated_DoAs = results.(mode).estimated_DoA;
    color = get(lineHandles.(mode), 'Color');
    for k = 1:numel(estimated_DoAs)
        xline(estimated_DoAs(k), '--', 'Color', color, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
end

title({'DoA Estimation — all algorithms', "GT Angles: " + num2str(anglesGT) + " [deg]"});
grid minor;
end
