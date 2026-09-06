function [] = render_mp_why()
%% Render the 3 multipath failure examples through DoA_Wrapper's Plot_DOA.
set(0, 'DefaultFigureVisible', 'off');
indir = 'C:\Users\Daniel\AppData\Local\Temp\claude\c--GitHub-DUNCS\64e8dd82-9435-48e6-a804-bb899502d269\scratchpad\doa_mp';
for r = 1:3
    mat = fullfile(indir, sprintf('results_mp_r%d.mat', r));
    if ~exist(mat, 'file'); fprintf('missing %s\n', mat); continue; end
    S = load(mat);
    Plot_DOA(double(S.angles), S.results, double(S.anglesGT));
    png = fullfile(indir, sprintf('doa_mp_why_r%d.png', r));
    exportgraphics(gcf, png, 'Resolution', 150);
    close(gcf);
    fprintf('wrote %s\n', png);
end
end

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
for j = 1:numel(modeFields)
    mode = modeFields{j};
    estimated_DoAs = results.(mode).estimated_DoA;
    color = get(lineHandles.(mode), 'Color');
    for k = 1:numel(estimated_DoAs)
        xline(estimated_DoAs(k), '--', 'Color', color, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
end
title({'Why multipath fails — MUSIC(raw cov) vs SubspaceNet(CNN cov) vs MFOCUSS vs IAA', "GT Angles: " + num2str(anglesGT) + " [deg]"});
grid minor;
end
