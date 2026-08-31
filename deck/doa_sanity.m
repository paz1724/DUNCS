function [] = doa_sanity()
%% Sanity DOA power-spectrum plots — straight from DoA_Wrapper's algorithms + Plot_DOA.
%  Overlays MFOCUSS / MUSIC / MVDR / IAA / BF on ONE axis; GT = vertical green dashed line.
%  "Sim" conditions: recorded ULA3 geometry @150 MHz, SNR 15 dB, 8 snapshots.
%  Case 1 = single front source; Case 2 = reuse-15 (two non-coherent front sources, 15 deg apart).
%  3 realizations per case, each saved as its own figure.

addpath(genpath('C:\GitHub\Hof\DOA1\Code'));   % cMath, cStruct, cPlot (used by MUSIC/MVDR/MFOCUSS)
addpath('C:\GitHub\Sandboxes\OFDM');            % cSubSpace (MVDR beamformer helpers)

doaModes = ["MFOCUSS", "MUSIC", "MVDR", "IAA", "BF"];

%% Array + signal parameters — recorded ULA3 @150 MHz (the "Sim" column conditions)
Nelements                 = 5;
element_positions         = [0 0.35 1.58 1.91 2.3];   % recorded ULA3 geometry [m]
freq                      = 150e6;   c = 3e8;   lambda = c / freq;
L                         = 8;        % snapshots (matches DUNCS T=8)
sParams.K                 = 1801;     % 0.1-deg grid over [-90, 90]
sParams.delta             = 1e-6;
sParams.Niters            = 100;
sParams.tol               = 1e-6;
sParams.p                 = 0.8;
sParams.thOMP             = 1e-3;
sParams.stapes_lambda     = 0.1;
sParams.calibrationErrors = diag(1 + 0 * randn(Nelements, 1));
sParams.SNR_dB            = 15;
sParams.N                 = numel(element_positions);
sParams.lambda            = lambda;
sParams.blockSize         = sParams.N;   % no spatial smoothing (bSmooth = 0)
sParams.bSmooth           = 0;
sParams.modeOMP           = "FFT";
sParams.nullSteerVec      = [];

%% Steering matrix over the display grid
angles = linspace(-90, 90, sParams.K);
A = Get_steering_matrix(element_positions, angles, lambda);

%% Output dir for the sanity PNGs
outdir = 'C:\Users\Daniel\AppData\Local\Temp\claude\c--GitHub-DUNCS\64e8dd82-9435-48e6-a804-bb899502d269\scratchpad\doa_png';
if ~exist(outdir, 'dir'); mkdir(outdir); end
set(0, 'DefaultFigureVisible', 'off');

%% Two Sim cases x 3 realizations; each realization = its own figure
seeds     = [1724, 2025, 4242];
gtSingle  = [12, -23, 34];      % single front-cone source [deg]
gtReuse1  = [-20, 5, 28];       % first of the reuse pair; second = first + 15 deg
caseNames = {'single', 'reuse15'};
for ci = 1:2
    for r = 1:3
        rng('default'); rng(seeds(r));
        if ci == 1
            gt0 = gtSingle(r); nAdd = 0; dTheta = 0;
        else
            gt0 = gtReuse1(r); nAdd = 1; dTheta = 15;
        end
        sParams.nRandTargets = nAdd;

        data = Generate_training_data(element_positions, lambda, L, sParams.SNR_dB, ...
                                      gt0, nAdd, dTheta, 0, 0);   % bCoherentSig=0 (reuse), bOnlyNoise=0
        y        = data.input_data(:, :, 1);
        anglesGT = data.output_data(1, :);
        sParams.anglesGT = anglesGT;
        sParams.M = size(data.input_data, 2);

        %% Collect results for all DoA algorithms (eval reaches the local functions)
        results = struct();
        for doaMode = doaModes
            try
                eval("results." + doaMode + " = " + doaMode + "(A, y, angles, sParams);");
            catch ME
                fprintf('  [%s r%d] %s FAILED: %s\n', caseNames{ci}, r, doaMode, ME.message);
            end
        end

        %% DF errors
        modeFields = fields(results);
        for j = 1:numel(modeFields)
            mode = modeFields{j};
            results.(mode).df_error = mean(min(abs(anglesGT.' - results.(mode).estimated_DoA)));
        end

        %% Plot (straight from Plot_DOA) and save
        Plot_DOA(angles, results, anglesGT);
        png = fullfile(outdir, sprintf('doa_%s_r%d.png', caseNames{ci}, r));
        exportgraphics(gcf, png, 'Resolution', 150);
        close(gcf);
        fprintf('wrote %s | GT=%s | modes=%s\n', png, mat2str(anglesGT), strjoin(string(modeFields), ','));
    end
end

end

%% ~~~~~~~~~~~~~~~~~~~~ Remove_projection_of_steering_angle ~~~~~~~~~~~~~~~~~~~~ %%
function [y_new] = Remove_projection_of_steering_angle(y, element_positions, angleToNull, lambda)

if isempty(angleToNull)
    y_new = y;
else
    %% Get null steering vector
    v_null = Get_steering_matrix(element_positions, angleToNull, lambda);

    %% Normalize the null steering vector
    v_null = v_null / norm(v_null);

    %% Calculate the projection matrix P_null
    P_null = (v_null * v_null');

    %% Calculate the orthogonal projection matrix P_perp
    P_perp = eye(size(v_null,1)) - P_null;

    %% Remove the projection of v_null from X
    y_new = P_perp * y;
end
end

%% ~~~~~~~~~~~~~~~~~~~~ Plot_DOA ~~~~~~~~~~~~~~~~~~~~ %%
function [] = Plot_DOA(angles, results, anglesGT)
figure;
hold all;

modeFields = fields(results);
lineHandles = struct();
for j = 1:numel(modeFields)
    mode = modeFields{j};
    lineHandles.(mode) = plot(angles, results.(mode).psd_dB, 'LineWidth', 1.5);
end

xlabel('Angle (degrees)');
ylabel('Power Spectrum (dB)');
xline(anglesGT, '--g', 'LineWidth', 2);
legendLabels = cellfun(@(doaMode) sprintf('%s (DF Error: %.2f°)', doaMode, results.(doaMode).df_error), modeFields, 'UniformOutput', false);
legendStr = vertcat(legendLabels{:}, "Ground Truth Angles");
legend(cellstr(legendStr), 'Location', 'best');   % (was cPlot.Click_Legend — plain legend for headless render)

% Add xline for each estimated DoA with the color matching the respective plot line
for j = 1:numel(modeFields)
    mode = modeFields{j};
    estimated_DoAs = results.(mode).estimated_DoA;
    color = get(lineHandles.(mode), 'Color');
    for k = 1:numel(estimated_DoAs)
        xline(estimated_DoAs(k), '--', 'Color', color, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
end

title({'DoA Estimation Results',"GT Angles: "+num2str(anglesGT)+" [deg]"});
grid minor;
% plotbrowser('on');   % (interactive — disabled for headless render)
end

%% ~~~~~~~~~~~~~~~~~~~~ Generate_training_data ~~~~~~~~~~~~~~~~~~~~ %%
function data = Generate_training_data( ...
    element_positions, ...
    lambda, ...
    L, ...
    SNR_dB, ...
    angles_GT, ...
    nRandTargets, ...
    dTheta, ...
    bCoherentSig, ...
    bOnlyNoise)
% Generate training data function

nAngles = numel(angles_GT);
N = length(element_positions);  % Number of array elements
input_data = zeros(N, L, nAngles);  % Input data matrix
output_data = zeros(nAngles, nRandTargets + 1);  % Output data matrix

for i = 1:nAngles
    true_angle = angles_GT(i);
    % random_angles = randi([-90, 90], 1, nRandTargets);
    random_angles = true_angle+(1:nRandTargets)*dTheta;
    all_angles = [true_angle, random_angles];
    A_true = Get_steering_matrix(element_positions, all_angles, lambda);
    if bCoherentSig
        x_true = (randn(1, L) + 1i * randn(1, L)) / sqrt(2);
        x_true = repmat(x_true,[nRandTargets+1,1]);
    else
        x_true = (randn(nRandTargets+1, L) + 1i * randn(nRandTargets+1, L)) / sqrt(2);
    end

    sigma_n = 10 ^ (-SNR_dB / 20);
    noise = sigma_n * (randn(N, L) + 1i * randn(N, L)) / sqrt(2);
    if bOnlyNoise
        Y = noise;
    else
        Y = A_true * x_true + noise;
        % Y = A_true * x_true;
    end

    input_data(:,:,i) = Y;
    output_data(i, :) = all_angles;
end

data.input_data = input_data;
data.output_data = output_data;
end

%% ~~~~~~~~~~~~~~~~~~~~ Get_steering_matrix ~~~~~~~~~~~~~~~~~~~~ %%
function A = Get_steering_matrix(element_positions, angles, lambda)
% Compute the steering matrix for a ULA
angles_rad = deg2rad(angles);
A = exp(1i * 2 * pi * element_positions'/lambda * sin(angles_rad));
end

%% ~~~~~~~~~~~~~~~~~~~~ Find_PSD_peaks ~~~~~~~~~~~~~~~~~~~~ %%
function [peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets)
psd_dB = 10*log10(psdEst);
% psd_dB = psd_dB - max(psd_dB);

% Find DoA by locating the peaks in the PSD estimate
[peaks, locs] = findpeaks(psd_dB, angles, 'SortStr', 'descend', 'NPeaks', nTargets);
end

%% ~~~~~~~~~~~~~~~~~~~~ Get_subspace_statistics ~~~~~~~~~~~~~~~~~~~~ %%
function sStat = Get_subspace_statistics(y, sParams)
%% Preprocessing Step: Compute the spatially smoothed covariance matrix
% Adjust the size of subblocks for spatial smoothing (L) as needed
R = cMath.Spatial_Smoothing(y, sParams.blockSize, sParams.bSmooth);

%% Eigen Decomposition
[E, D] = cMath.Eigen_Decomposition(R);

%% Estimate_number_of_taps
p  = size(D,1);
N  = size(y,2); % The inner dimension of the auto-correlation

%% MDL: Minimum_Description_Length
% numPaths = cMath.Minimum_Description_Length(D, p, N, sParams);
sParams.numPaths = 2;

%% Noise Whitening
sStat = cMath.Signal_And_Noise_Decomposition(E, D, sParams);

end

%% ~~~~~~~~~~~~~~~~~~~~ BF ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = BF(A, y, angles, sParams)
% Matched filter
s = A' * y;

% Spectrum calculation
P_k = diag(s * s');

% Final PSD estimate
psdEst = P_k;

%% Find_PSD_peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ CSBL ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = CSBL(A, y, angles, sParams)

[M, T] = size(y);
[~, N] = size(A);

% Initialization
alpha = 1e-6 * ones(N, 1);  % Initial variance of the sources (sparsity prior)
sigma2 = 1e-2;  % Noise variance
Phi = A;  % Array manifold matrix

for iter = 1:sParams.Niters
    % E-Step: Compute posterior mean and covariance of sources
    Sigma_x = (Phi' * Phi / sigma2 + diag(1 ./ alpha)) \ eye(N);
    mu_x = Sigma_x * Phi' * y / sigma2;

    % M-Step: Update source variances and noise variance
    gamma_new = mean(abs(mu_x).^2 + diag(Sigma_x), 2);
    sigma2_new = norm(y - Phi * mu_x, 'fro')^2 / (M * T);

    % Convergence check
    if norm(gamma_new - alpha) < sParams.tol
        break;
    end

    % Update parameters
    alpha = gamma_new;
    sigma2 = sigma2_new;
end

psdEst = abs(alpha);

%% Find_PSD_peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ RIMAX ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = RIMAX(A, y, angles, sParams)

% Initialization
maxIter = sParams.Niters;       % Maximum number of iterations
tol = sParams.tol;              % Convergence tolerance
[M, T] = size(y);               % Number of sensors and snapshots
[~, N] = size(A);               % Number of potential DoA grid points
sigma2 = 1e-2;                  % Initial noise variance
noise_cov = eye(M);             % Initial noise covariance matrix
log_likelihood_old = -Inf;      % Initial log-likelihood value

% Placeholder for source powers and estimated DoAs
source_amplitudes = zeros(N, 1);
psdEst = zeros(N, 1);
nTargets = sParams.nRandTargets + 1;

% Iterative Maximum Likelihood Estimation Loop
for iter = 1:maxIter
    % Step 1: Update AoA Estimates - Compute spatial spectrum (MUSIC-like)
    P_music = zeros(1, N);
    for i = 1:N
        a_theta = A(:, i);  % Steering vector for each angle
        P_music(i) = 1 / (a_theta' * (noise_cov \ a_theta));  % MUSIC spectrum estimate
    end

    % Find DoA by locating the peaks in the PSD estimate
    [peaks, locs] = findpeaks(abs(P_music), 1:numel(P_music), 'SortStr', 'descend', 'NPeaks', nTargets);

    % Find peaks in the MUSIC spectrum to estimate current AoAs
    A_est = A(:, locs);  % Corresponding steering vectors for estimated AoAs

    % Step 2: Estimate Source Amplitudes (Least Squares fitting)
    source_amplitudes = (A_est' * (noise_cov \ A_est)) \ (A_est' * (noise_cov \ y));

    % Step 3: Update Noise Covariance
    residual = y - A_est * source_amplitudes;
    noise_cov_new = (residual * residual') / T + sigma2 * eye(M);

    % Step 4: Log-Likelihood Calculation for Convergence Check
    log_likelihood_new = -trace((residual' / noise_cov_new) * residual) - log(det(noise_cov_new));

    % Convergence Check
    dist = abs(log_likelihood_new - log_likelihood_old);
    if dist < tol
        fprintf('Convergence achieved after %d iterations.\n', iter);
        break;
    end

    % Update parameters for the next iteration
    noise_cov = noise_cov_new;
    log_likelihood_old = log_likelihood_new;
end

% Final PSD Estimate
A_est = A(:, locs);  % Extract the steering matrix for the estimated directions

% Compute the contribution to the power from the estimated directions
psdEst = mean(abs(A'*(A_est * source_amplitudes)).^2,2);
% for i = 1:N
%     a_theta = A(:, i);  % Steering vector for angle i
%     % Project onto the estimated source amplitudes and compute power
%     psdEst(i) = abs(a_theta' * (A_est * source_amplitudes)).^2 / T;  % Average power contribution
% end


%% Find_PSD_peaks
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ SDP ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = SDP(A, y, angles, sParams)

%% Get_subspace_statistics
sStat = Get_subspace_statistics(y, sParams);

M = size(y,1); % Number of sensors in the ULA
T = size(y,2); % Number of time snapshots

% Atomic Norm Minimization Problem
cvx_begin sdp quiet
variable Z(M, T) complex
minimize(norm_nuc(Z))  % Nuclear norm minimization (relaxation of atomic norm)
subject to
norm(y - Z, 'fro') <= 1e-6;  % Data fitting constraint
cvx_end

% Estimate the AoA based on the recovered Z
[U, ~, ~] = svd(Z);
Z_est = U(:, 1:sStat.numPaths);  % Extract the estimated array manifold matrix

% Matched filter
s = A' * Z_est;

% Spectrum calculation
P_k = diag(s * s');

% Final PSD estimate
psdEst = P_k;

%% Find_PSD_peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ slim_aoa ~~~~~~~~~~~~~~~~~~~~ %%
function theta_est = slim_aoa(y, A, lambda, max_iter, tol)
% SLIM Algorithm for AoA Estimation
% y: received signal vector (Mx1)
% A: steering matrix (MxN) where M is the number of sensors and N is the number of angles
% lambda: regularization parameter
% max_iter: maximum number of iterations
% tol: convergence tolerance

% Initialization
[M, N] = size(A);    % M is number of sensors, N is number of potential angles
x = zeros(N, 1);     % Sparse solution vector
theta_est = zeros(N, 1); % Initialize angle estimates

% Iterative Minimization Loop
for iter = 1:max_iter
    % Save the previous estimate for convergence check
    x_old = x;

    % Solve the Lasso problem using the regularization parameter lambda
    x = lasso(A, y, 'Lambda', lambda);

    % Check for convergence
    if norm(x - x_old, 2) < tol
        break;
    end

    % Update the regularization parameter if needed (optional)
    % lambda = update_lambda(lambda, iter); % Uncomment if dynamic lambda is used
end

% Extract estimated angles based on the non-zero entries of x
theta_est = find(abs(x) > tol);  % Indices of non-zero elements correspond to estimated angles

% Convert indices to angles (if A matrix columns correspond to specific angles)
% This step assumes that A was generated for a specific grid of angles
% For example, if angles range from -90 to 90 degrees:
% angles_grid = linspace(-90, 90, N);
% theta_est = angles_grid(theta_est);
end

%% ~~~~~~~~~~~~~~~~~~~~ SLIM ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = SLIM(A, Y, angles, sParams)

% Initialization
[Nelements, Nsnapshots] = size(Y);
[M, N] = size(A);
sigma2 = db2pow(-sParams.SNR_dB);
lambda = sParams.delta;
p = sParams.p;
tol = sParams.tol;
x_est = A'*Y;%zeros(N, Nsnapshots);  % Initialize with zeros for all snapshots
W = eye(N);           % Initial weight matrix
R = (A * W * A' + sigma2 * eye(M)) / Nsnapshots;  % Initial covariance matrix
distVec = nan(sParams.Niters,1);

for k = 1:sParams.Niters
    % Update X for all snapshots using the current R and W
    x_old = x_est;
    % x_est = (A' / R * A + lambda * W) \ (A' / R * Y);

    eta_t = (1/Nsnapshots) * norm(Y-A*x_old,'fro');
    P_k = abs(mean(x_old,2)).^(2-p);
    P_k_inv = diag(1./P_k);
    x_est = (A'*A+eta_t*P_k_inv) \ A'*Y;

    % % Update the weight matrix W
    % P_k = mean(abs(x_est).^(p-2), 2);  % Average over snapshots and apply the sparsity rule
    % W = diag(P_k);
    %
    % % Update the covariance matrix R
    % R = (A * W * A' + sigma2 * eye(M)) / Nsnapshots;

    % Convergence check
    dist = norm(x_est - x_old, 'fro');
    distVec(k) = dist;
    if dist < tol
        break;
    end
end

P_k = abs(mean(x_est,2)).^(2-p);

% Find PSD peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(P_k, angles, nTargets);

% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;
end

%% ~~~~~~~~~~~~~~~~~~~~ LWS_SBL ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = LWS_SBL(A, y, angles, sParams)

%% Parameters
M               = 5; % Number of elements in the ULA
N               = 1801; % Number of potential source directions (1 degree resolution)
SNR_dB          = 30; % Signal to noise ratio in dB
xhat_th_dB      = 16.7;
xhat_diff_th_dB = -14.8;
delta_u         = 0.1;
n_tilde         = 100;
ITER            = 5;

%% Derivative Parameters
sigma_n         = 10^(-SNR_dB/20);

%% Create LWS_SBL object
sParams.M                 = M; % Number of elements in the ULA
sParams.N                 = N; % Number of potential source directions
sParams.K                 = N-1; % Number of sources
sParams.lambda            = sigma_n^2; % Initial estimate of noise variance;
sParams.xhat_th_dB        = xhat_th_dB;
sParams.xhat_diff_th_dB   = xhat_diff_th_dB;
sParams.delta_u           = delta_u;
sParams.n_tilde           = n_tilde        ;
sParams.ITER              = ITER           ;
oLWS_SBL                  = cLWS_SBL(sParams);

%% Estimate the directions using LWS-SBL
[x_hat, sEst, last_x_hat] = oLWS_SBL.Apply(A, y);

% Final PSD estimate
psdEst = abs(x_hat);
psdEst = min(psdEst,1e-8);

% Find PSD peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

% Results
sResults.estimated_DoA = locs;
sResults.peaks = peaks;
sResults.psd_dB = psd_dB;

end

% %% ~~~~~~~~~~~~~~~~~~~~ SLIM ~~~~~~~~~~~~~~~~~~~~ %
% function [sResults] = SLIM(A, y, angles, sParams)
% % Matched filter
% s = A' * y;
%
% % Spectrum calculation
% P_k = diag(s * s');
% tol = sParams.tol;
% maxIter = sParams.Niters;
% delta = sParams.delta;
% nTargets = sParams.nRandTargets + 1;
%
% % Iterative minimization
% for iter = 1:maxIter
%     R = A * diag(P_k) * A' + delta * eye(size(A,1));
%     s = diag(diag(A' * (R \ A))) \ (A' * (R \ y));
%     P_k_new = diag(s * s');
%     if norm(P_k_new - P_k) < tol
%         break;
%     end
%     P_k = P_k_new;
% end
%
% % Final PSD estimate
% psdEst = P_k;
%
% % Find PSD peaks
% [peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);
%
% % Results
% sResults.estimated_DoA = locs;
% sResults.peaks  = peaks;
% sResults.psd_dB = psd_dB;
% end

%% ~~~~~~~~~~~~~~~~~~~~ IAA ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = IAA(A, y, angles, sParams)

% Matched filter
s = A' * y;

% Spectrum calculation
P_k = diag(s * s');

% Initialization
P_k_mat = nan(sParams.K, sParams.Niters);
distVec = nan(sParams.Niters, 1);
lambda = 1e-3; % Regularization parameter for diagonal loading

% Iterate for the IAA algorithm
for iter = 1:sParams.Niters

    % Covariance matrix calculation with diagonal loading
    R = A * diag(P_k) * A' + lambda * eye(size(A, 1));

    % Efficient matrix inversion using Cholesky decomposition
    [L, p] = chol(R, 'lower');
    if p == 0
        Linv = inv(L);
        Rinv = Linv * Linv';
    else
        Rinv = pinv(R); % Use pseudoinverse if R is singular
    end

    % Weighted Least Squares estimation
    W = diag(diag(A' * Rinv * A));
    s = W \ (A' * Rinv * y);

    % Update spectrum
    P_k_new = diag(s * s');

    % Matrix for debugging
    P_k_mat(:, iter) = P_k_new;

    % Check convergence
    dist = norm(P_k - P_k_new, 2)^2;
    distVec(iter) = dist;
    if dist < sParams.tol
        break;
    end

    P_k = P_k_new;
end

% Final PSD estimate
psdEst = P_k;

% Find PSD peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

% Results
sResults.estimated_DoA = locs;
sResults.peaks = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ SparseIAA ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = SparseIAA(A, y, angles, sParams)

% Dimensions
[Nsensors, Nsnapshots] = size(y);  % M is the number of sensors, N is the number of snapshots
K = size(A, 2);  % K is the number of angles/hypotheses

% Initialize alpha and P_k
alpha = A' * y;  % Initial estimate of alpha (K x N)
p_k = diag(alpha * alpha');
% p_k = mean(abs(alpha).^2, 2);  % Initial power spectrum estimate (K x 1)
P_k = diag(p_k);

% Iteration Parameters
P_k_mat = nan(K, sParams.Niters);
distVec = nan(sParams.Niters, 1);
lambda = 0;%1e-5;  % Regularization parameter for diagonal loading

% Iterate for the Sparse IAA algorithm with Soft Thresholding
for iter = 1:sParams.Niters

    alpha_prev = alpha;
    %% Step 1: Calculate R
    % lowTh = 1e-6;
    % P_k(isnan(P_k)) = 0;
    % P_k_norm = P_k/norm(P_k);
    R = A * P_k * A' + lambda * eye(Nsensors);
    if any(isnan(R(:)))
        break;
    end
    % Cholesky decomposition for inversion
    [L, p] = chol(R, 'lower');
    if p == 0
        Linv = inv(L);
        Rinv = Linv * Linv';
    else
        Rinv = pinv(R);  % Use pseudoinverse if R is singular
    end

    %% Step 2: Calculate wk using the provided formula for soft thresholding
    wk = zeros(K, Nsnapshots);
    for k = 1:K
        ak = A(:, k);  % Column vector of size (M x 1)
        mu_k = 1 - mean(abs(alpha(k, :)).^2) * (ak' * Rinv * ak);
        % Compute wk using covariance matrix R_y
        wk(k,:) = (1 / (Nsensors * sum(abs(alpha(k, :)).^2))) * ...
            (trace(y' *Rinv * y)/Nsnapshots - (abs(ak' * Rinv * y).^2 / (ak' * Rinv * ak)) * ...
            (1 - sum(abs(alpha(k, :)).^2) * (ak' * Rinv * ak)) / mu_k);
        if any(isnan(wk(k,:)))
            break;
        end
    end

    %% Step 3: Calculate alpha_k(i) using the soft thresholding formula
    for k = 1:K
        ak = A(:, k);  % Column vector of size (M x 1)
        mu_k = 1 - sum(abs(alpha(k, :)).^2) * (ak' * Rinv * ak);
        alpha(k, :) = (ak' * Rinv * y) / (ak' * Rinv * ak + wk(k,:) * mu_k);
        if any(isnan(alpha(k, :)))
            break;
        end
    end

    % Update spectrum
    P_k_new = sum(abs(alpha).^2, 2) / Nsnapshots;  % Power spectrum update (K x 1)

    % Save the power estimates for debugging
    P_k_mat(:, iter) = P_k_new;

    %% Step 4: Check convergence
    dist = sum(abs(mean(alpha - alpha_prev, 2,"omitnan")).^2,"omitnan");
    distVec(iter) = dist;
    if dist < sParams.tol
        break;
    end

    p_k = P_k_new;
end

% Final PSD estimate
psdEst = p_k;

% Find PSD peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

% Results
sResults.estimated_DoA = locs;
sResults.peaks = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ MUSIC ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = MUSIC(A, y, angles, sParams)
%% Get_subspace_statistics
sStat = Get_subspace_statistics(y, sParams);

%% Calculate the MUSIC spectrum in a vectorized manner
EnEnH = sStat.En * sStat.En';  % Compute the product once to avoid repetition
AGal = A(1:sParams.blockSize,:);
% PSpecVec = 1 ./ sum(conj(AGal) .* (EnEnH * AGal), 1).';
PSpecVec = zeros(sParams.K,1);
for i = 1:sParams.K
    PSpecVec(i) = 1./(AGal(:,i)' * (EnEnH* AGal(:,i)));
end
psdEst = abs(PSpecVec);

%% Find_PSD_peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ MVDR ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = MVDR(A, y, angles, sParams)
%% MVDR / Capon spectrum from the sample covariance — the wrapper's own inline formula
%  (lines below were commented in the source; used here because the cSubSpace beamformer
%  helper on this machine has a mismatched signature). Standard Capon: 1/(a' R^{-1} a).
AGal = A(1:sParams.blockSize, :);
yGal = y(1:sParams.blockSize, :);

R    = (yGal * yGal') / size(yGal, 2);                              % sample covariance
R    = R + sParams.delta * (trace(R) / size(R, 1)) * eye(size(R, 1));   % diagonal loading
invR = inv(R);

PSpecVec = zeros(sParams.K, 1);
for i = 1:sParams.K
    PSpecVec(i) = 1 ./ real(AGal(:, i)' * (invR * AGal(:, i)));     % Capon power
end
psdEst = abs(PSpecVec);

%% Find_PSD_peaks (same peak reader as the other spectrum methods)
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ OMP ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = OMP(A, y, angles, sParams)

%% Get_subspace_statistics
sStat = Get_subspace_statistics(y, sParams);

%% Covariance Martix Size
Nfb            = size(sStat.Rss,1);

AGal = A(1:sParams.blockSize,:);

F_OverCompDict = AGal;
W_t            = zeros(size(sStat.Rss,1),sStat.numPaths);

XGal           = y(1:Nfb,:);
r_t            = XGal;
LTotal         = zeros(size(F_OverCompDict,2),sStat.numPaths);
maxLVec        = zeros(1,sStat.numPaths);
J_t            = zeros(1,sStat.numPaths);

%% Generate_MVDR_W: MVDR Beamformer Matrix
switch sParams.modeOMP
    case "MVDR"
        W = cSubSpace.Generate_MVDR_W(F_OverCompDict, sStat.Rss);
    case "FFT"
        W = F_OverCompDict;
end
for t = 1:sStat.numPaths

    %% MVDR_Projection
    PSpec = cSubSpace.MVDR_Projection(W, r_t);

    %% Projection of atoms
    L = abs(PSpec);

    %% argmax over the projection of atoms
    [maxL,j_t] = max(L);

    %% Stop Condition
    if maxL<sParams.thOMP
        LTotal               = LTotal(:,1:t-1) ;
        maxLVec              = maxLVec(:,1:t-1);
        J_t                  = J_t(:,1:t-1)    ;
        sStat.numPaths       = t-1;
        break;
    end

    %% Aggregate atoms
    LTotal(:,t)  = L;
    maxLVec(:,t) = maxL;
    J_t(:,t)     = j_t;
    W_t(:,t)     = W(:,j_t).';

    %% Least Squares for optimal coeffiecients Fn
    Wt       = W_t(:,1:t);
    aLS      = Wt \ XGal;
    XHat_t   = Wt * aLS;

    %% Subtract Residual
    r_t      = XGal-XHat_t;
end

psd_dB = 10*log10(LTotal(:,1));
psd_dB = psd_dB - max(psd_dB);

%% Results
sResults.estimated_DoA = angles(J_t);
sResults.peaks  = maxLVec;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ SBL ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = SBL(A, y, angles, sParams)

%% Create LWS_SBL object
sParamsSBL.M                 = sParams.N; % Number of elements in the ULA
sParamsSBL.N                 = sParams.K; % Number of potential source directions
sParamsSBL.K                 = sParamsSBL.N-1; % Number of sources
sParamsSBL.lambda            = 10^(-sParams.SNR_dB/10); % Initial estimate of noise variance;
sParamsSBL.xhat_th_dB        = 16.7;
sParamsSBL.xhat_diff_th_dB   = -14.8;
sParamsSBL.delta_u           = 0.1;
sParamsSBL.n_tilde           = 100        ;
sParamsSBL.ITER              = 20           ;
oLWS_SBL                  = cLWS_SBL(sParamsSBL);

[x_hat, sEst, last_x_hat] = oLWS_SBL.Apply(A, y);

% Final PSD estimate
psdEst = abs(x_hat);

%% Find_PSD_peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(psdEst, angles, nTargets);

psd_dB = max(psd_dB,-50);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ MFOCUSS ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = MFOCUSS(Phi, Y, angles, sParams)

%% Defaults
sParams = cStruct.Set_default_value(sParams,'lambda',       0.6);
sParams = cStruct.Set_default_value(sParams,'p',            0.75);
sParams = cStruct.Set_default_value(sParams,'PRUNE_GAMMA',  1e-3);
sParams = cStruct.Set_default_value(sParams,'EPSILON',      1e-4);
sParams = cStruct.Set_default_value(sParams,'bPrint',       false);
sParams = cStruct.Set_default_value(sParams,'MAX_ITERS',    100);
sParams = cStruct.Set_default_value(sParams,'bForceGT',     0);

%% Dimension of the Problem
[~, M] = size(Phi);
[~, L] = size(Y);

%% Normalize Input
factorY = 100./sqrt(sum(abs(Y).^2,"All"));
Y = Y.*factorY;

%% Initializations
phasorDiffVec = nan(sParams.MAX_ITERS,1);
angleDiffVec = nan(sParams.MAX_ITERS,1);

sIn.count = 0;                 % record iterations
sIn.gamma = 0.1*ones(M,1);         % initialization of gamma_i
sIn.Phi   = Phi;
sIn.Y     = Y;
sIn.mu    = zeros(M,L);   
% initialization of the solution matrix

%% Force GT
if sParams.bForceGT
    angleIndsGT = find(ismember(angles, sParams.anglesGT));
    sIn.gamma = zeros(M,1);
    sIn.gamma(angleIndsGT) = 20;
end

%% Learning loop
while (1)

    %% MFOCUSS_Core
    sOut = MFOCUSS_Core(sIn, sParams);

    %% Stop Condition
    if sOut.bBreak
        break;
    else
        sIn.mu    = sOut.mu;
        sIn.gamma = sOut.gamma;
        sIn.count = sOut.count;

        phasorDiffVec(sIn.count) = sOut.phasorDiffVal;
        angleDiffVec(sIn.count) = sOut.angleDiffVal;
    end

end
sOut

gamma_ind = find(sOut.index);
gamma_est = zeros(M,1);
gamma_est(sOut.index) = sOut.gamma(sOut.index);
Pafter = abs(gamma_est);
Pafter(Pafter==0) = eps;

%% Find_PSD_peaks
nTargets = sParams.nRandTargets + 1;
[peaks, locs, psd_dB] = Find_PSD_peaks(Pafter, angles, nTargets);

psd_dB = max(psd_dB,-50);

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;

end

%% ~~~~~~~~~~~~~~~~~~~~ MFOCUSS_Core ~~~~~~~~~~~~~~~~~~~~ %%
function [sOut] = MFOCUSS_Core(sIn, sParams)

bPrint      = sParams.bPrint;
p           = sParams.p;
lambda      = sParams.lambda;
PRUNE_GAMMA = sParams.PRUNE_GAMMA;
EPSILON     = sParams.EPSILON;
MAX_ITERS   = sParams.MAX_ITERS;

count = sIn.count;
gamma = sIn.gamma;
Phi   = sIn.Phi;
Y     = sIn.Y;
mu    = sIn.mu;

[N, M] = size(Phi);
[N, L] = size(Y);

index = gamma > PRUNE_GAMMA;
m = sum(index);

if (m == 0)
    sOut.bBreak = 1;
else

    gamma_old = gamma(index);

    % ====== Compute new weights ======
    G = repmat(sqrt(gamma_old)',N,1);
    PhiG = Phi(:,index).*G;
    [U,S,V] = svd(PhiG,'econ');

    [d1,d2] = size(S);
    if (d1 > 1)
        diag_S = diag(S);
    else
        diag_S = S(1);
    end

    %% U Scaled
    UscaleFactor = repmat((diag_S./(diag_S.^2 + sqrt(lambda) + 1e-16))',N,1);
    U_scaled = U(:,1:min(N,m)).*UscaleFactor;

    %% Spectrum Scale Matrix
    Xi = G'.*(V*U_scaled');

    %% mu
    mu_old = mu;
    mu     = Xi*Y;

    %% Update hyperparameters
    mu2_bar = sum(abs(mu).^2,2);
    gamma(index) = (mu2_bar/L).^(1-p/2);

    %% Check stopping conditions, etc.
    count = count + 1;

    %% diffValVec
    Yhat = Phi(:,index)*mu;
    phaseDiff = wrapToPi(angle(Yhat)-angle(Y));
    angleDiffVal = norm(phaseDiff,'fro');%/sqrt(size(phaseDiff,2));
    Ydiff = Yhat-Y;
    phasorDiffVal = abs(sum(Ydiff(:))).^2;

    %% Calculate quality factor
    % Gamma Stability
    gammaStability = max(abs(gamma(index) - gamma_old)) / (1 + max(gamma_old));

    % Mu Stability
    if size(mu) == size(mu_old)
        muStability = max(max(abs(mu_old - mu))) ./ (1 + max(abs(mu_old(:))));
    else
        muStability = Inf;  % Undefined if sizes mismatch (should not happen)
    end

    % Combine into a quality factor (lower is better)
    QF = gammaStability + muStability + phasorDiffVal + angleDiffVal;

    sOut.QF = QF; % Lower values indicate higher solution stability

    %% Update outputs
    sOut.Yhat = Yhat;
    sOut.angleDiffVal = angleDiffVal;
    sOut.phasorDiffVal = phasorDiffVal;

    sOut.mu    = mu;
    sOut.gamma = gamma;
    sOut.count = count;
    sOut.index = index;

    sOut.bBreak = 0;
    
    %% Stop Conditions
    if (bPrint)
        disp(['iters: ',num2str(count),'   num coeffs: ',num2str(m), ...
            '   gamma change: ',num2str(max(abs(gamma - gamma_old))), ...
            '   QF: ', num2str(QF)]);
    end

    if (count >= MAX_ITERS)
        sOut.bBreak = 1;
    end

    if (size(mu) == size(mu_old))
        dmu = max(max(abs(mu_old - mu)));
        if (dmu < EPSILON)
            sOut.bBreak = 1;
        end
    end

end

end


%% ~~~~~~~~~~~~~~~~~~~~ STAPES ~~~~~~~~~~~~~~~~~~~~ %%
function [sResults] = STAPES(A, y, angles, sParams)
% STAPES Algorithm for Direction of Arrival Estimation for Complex Data using Lasso
% Inputs:
%   X: Received complex signal matrix (M x T)
%   A_grid: Complex array manifold matrix for the grid (M x N)
%   lambda: Regularization parameter for sparsity
%   K: Number of sources
%   maxIterations: Maximum number of iterations for refinement
%   tol: Convergence tolerance for the iterative refinement
%   calibrationErrors: Calibration error matrix (M x M)
% Outputs:
%   theta_final: Final DoA estimates
%   source_powers: Power estimates of the sources


stapes_lambda     = sParams.stapes_lambda;
K                 = sParams.K;
tol               = sParams.tol;
maxIterations     = sParams.Niters;
calibrationErrors = sParams.calibrationErrors;

lambda            = sParams.lambda;

% Parameters
[M, T] = size(y);  % M: Number of sensors, T: Number of snapshots
N = size(A, 2);  % N: Number of potential angles in the grid

% Incorporate calibration errors into the array manifold matrix
if ~isempty(calibrationErrors)
    A = calibrationErrors * A;
end

% Convert the problem to a real-valued formulation
X_real = [real(y); imag(y)];  % Convert complex X to real
A_grid_real = [real(A), -imag(A); imag(A), real(A)];  % Convert complex A_grid to real

% Sparse recovery: Step 1 (Initial DoA Estimation)
fprintf('Starting sparse recovery using real-valued lasso...\n');

% Use lasso for sparse recovery (each column in X represents one snapshot)
B_sparse_real = nan(size(A_grid_real,2),size(X_real,2));
for i = 1:size(X_real,2)
    Y = X_real(:,i);
    [B_sparse_real(:,i), FitInfo(i)] = lasso(A_grid_real, Y, 'Lambda', stapes_lambda);
end

% Recombine real and imaginary parts into complex sparse coefficients
B_sparse_complex = B_sparse_real(1:N, :) + 1i * B_sparse_real(N+1:end, :);

% Sum across snapshots to get total power contribution from each grid point
b_sparse = sum(abs(B_sparse_complex), 2);  % Sparse signal strength on each grid point

% Identify the K largest peaks in the sparse spectrum
[~, idx_peaks] = sort(b_sparse, 'descend');
theta_initial = angles(idx_peaks(1:K));  % Initial DoA estimates

% Iterative refinement: Step 2 (EM-like Refinement)
fprintf('Starting iterative refinement...\n');
theta_current = theta_initial;
for iter = 1:maxIterations


    % Step 2.1: E-Step (Estimate source signals)
    A_est = array_manifold(theta_current, M, calibrationErrors, lambda);  % Array manifold for current DoAs
    S_hat = (A_est' * A_est+sParams.delta*eye(size(A_est,2))) \ (A_est' * y);  % Source signal estimates

    % Step 2.2: M-Step (Update DoA estimates by refining theta)
    theta_next = refine_DoA(y, S_hat, M, calibrationErrors, theta_current, lambda, idx_peaks);

    % Check convergence (stop if the angle updates are smaller than tolerance)
    dist = mean(abs(theta_next - theta_current));

    disp("Iteration "+iter+" , Dist: "+dist);
    if dist < tol
        fprintf('Convergence achieved after %d iterations.\n', iter);
        break;
    end

    % Update theta for the next iteration
    theta_current = theta_next;
end

% Final DoA estimates
theta_final = theta_current;

% Estimate source powers
A_final = array_manifold(theta_final, M, calibrationErrors, lambda);
S_final = (A_final' * A_final) \ (A_final' * y);  % Final source signals
source_powers = sum(abs(S_final).^2, 2) / T;  % Average power per source

fprintf('STAPES finished. Estimated DoAs: %s degrees\n', num2str(theta_final));

%% Results
sResults.estimated_DoA = locs;
sResults.peaks  = peaks;
sResults.psd_dB = psd_dB;
end

%% Helper function to generate array manifold for a given set of angles
function A = array_manifold(theta, M, calibrationErrors, lambda)
% Generates the array manifold matrix for given DoAs
% Inputs:
%   theta: Vector of DoAs in degrees
%   M: Number of sensors
%   calibrationErrors: Calibration error matrix (M x M)
% Outputs:
%   A: Array manifold matrix

% lambda = 3e8 / 2.4e9;  % Wavelength (example: fc = 2.4 GHz)
d = lambda / 2;  % Inter-element spacing for ULA (half wavelength)

% Initialize array manifold matrix
A = zeros(M, length(theta));

% Compute steering vectors for each DoA
for k = 1:length(theta)
    steering_vector = exp(1i * 2 * pi * d * (0:M-1).' * sind(theta(k)) / lambda);
    if ~isempty(calibrationErrors)
        A(:, k) = calibrationErrors * steering_vector;
    else
        A(:, k) = steering_vector;
    end
end
end

%% Helper function to refine DoAs (M-Step)
function theta_next = refine_DoA(X, S_hat, M, calibrationErrors, theta_current, lambda, idx_peaks)
% Refines the DoA estimates using maximum likelihood or similar methods
% Inputs:
%   X: Received signal matrix (M x T)
%   S_hat: Estimated source signals (K x T)
%   M: Number of sensors
%   calibrationErrors: Calibration error matrix (M x M)
%   theta_current: Current DoA estimates
% Outputs:
%   theta_next: Refined DoA estimates

% Define a small search grid around each current DoA
delta_theta = 0.1;  % Fine-tuning step size (degrees)
theta_range = [-1, 0, 1] * delta_theta;  % Search in 3 directions (current, left, right)

% Initialize refined DoA estimates
theta_next = zeros(size(theta_current));

for k = 1:length(theta_current)
    % Search for the best DoA around the current estimate
    theta_candidates = theta_current(k) + theta_range;

    % Evaluate likelihood or fitting error for each candidate
    best_error = inf;
    for theta_candidate = theta_candidates
        A_test = array_manifold(theta_candidate, M, calibrationErrors, lambda);
        error = norm(X - A_test * S_hat(idx_peaks(k), :), 'fro')^2;
        if error < best_error
            best_error = error;
            theta_next(k) = theta_candidate;
        end
    end
end
end
