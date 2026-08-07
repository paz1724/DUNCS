"""
DU-MFOCUSS: Deep-Unfolded Multiple-measurement FOCUSS for DoA estimation.

Unrolls the M-FOCUSS sparse-recovery iteration (Cotter et al., IEEE T-SP 2005)
into a fixed number of layers over an overcomplete angle-grid steering
dictionary, and learns the per-layer regularization weight (lambda_k) and the
lp-diversity exponent (p_k) end-to-end with the RMSPE criterion. The steering
dictionary is built from the system model's manifold (the recorded ULA3
manifold when an antenna pattern is loaded, the analytic ULA otherwise), so the
estimation manifold always matches the data-generation manifold.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.metrics.criterions import set_criterions
from src.utils import device


class DUMFOCUSS(ParentModel):
    """Deep-unfolded M-FOCUSS network with learned per-layer lambda and p."""

    def __init__(self, system_model: SystemModel, num_iterations: int = 10,
                 grid_size: int = 121, criterion: str = "rmspe",
                 grid_range_deg: float = None,
                 learn_calibration: bool = False,   # trainable N×N dictionary calibration C (init=I,
                 cal_reg_weight: float = 1e-3,       # so reduce-to-MFOCUSS holds); ||C-I||_F^2 kept small
                 lam_multi_scale: float = 0.02,
                 peak_lim_deg: float = None,
                 fine_cols_per_180deg: int = 901,
                 p_init_floor: float = 0.01,
                 p_init_amp: float = 0.98,
                 p_init_decay: float = 0.27,
                 p_init_clamp_eps: float = 1e-3,
                 lam_init: float = 0.99,
                 lam_eps: float = 1e-5,
                 mcp_raw_init: float = -3.0,
                 temp_raw_init: float = -3.0,
                 temp_eps: float = 1e-3,
                 train_temp_floor: float = 0.3,
                 spectrum_loss_weight: float = 0.5,
                 spectrum_loss_sigma_deg: float = 1.0,
                 conv_kernel_halfwidth_deg: float = 4.0,
                 conv_kernel_min_cols: int = 2,
                 bce_clamp_eps: float = 1e-6,
                 ls_init_reg: float = 1e-6,
                 rms_clamp_eps: float = 1e-12,
                 reweight_floor: float = 1e-12,
                 rn_eps: float = 1e-9,
                 spectrum_norm_eps: float = 1e-9,
                 hyper_norm_eps: float = 1e-30,
                 hyper_log_eps: float = 1e-12,
                 pick_half_frac: float = 0.025,
                 pick_supp_frac: float = 0.03,
                 pick_min_cols: int = 1,
                 extend_p_decay: float = 0.2,
                 extend_p_target: float = 0.01,
                 refine_win_deg: float = 2.0,
                 eval_chunk: int = 256,
                 count_peak_thr: float = 0.5):
        """Initializes the DU-MFOCUSS model.

        Args:
            system_model (SystemModel): Array geometry and signal parameters.
            num_iterations (int): Number of unrolled M-FOCUSS layers (K).
            grid_size (int): Number of angle-grid columns in the steering dictionary.
            criterion (str): Training criterion name (e.g. "rmspe").
            grid_range_deg (float | [lo, hi]): If given, build the steering dictionary
                over this azimuth range instead of the system doa_range. A scalar v means
                the symmetric [-v, +v]; a 2-sequence gives an explicit [lo, hi] (e.g.
                [-180, 180] for full-azimuth recovery, matching classical MFOCUSS). A grid
                WIDER than the data cone keeps front-cone close pairs away from the grid
                boundary (cleaner recovery) and lets endfire / back-hemisphere sources be
                represented. The recorded ULA3 manifold's front/back asymmetry makes the
                full-azimuth dictionary well-posed (an ideal Vandermonde ULA could not).

        The remaining keyword args are hoisted numeric tunables / thresholds /
        epsilons; their defaults reproduce the historical hard-coded values
        exactly (see the constants block below for what each one controls).
        """
        super().__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.num_iter = num_iterations
        self.grid_size = grid_size
        self.M = system_model.params.M

        # ---- Hoisted tunables / schedule constants / epsilons (defaults == historical literals) ----
        self.lam_multi_scale = lam_multi_scale  # scale the learned lambda down for >=2 sources; 0.02 (sharper than
                                                # MFOCUSS's 0.05) selected on the real train split and better on
                                                # synthetic too (synth reuse 10->5%, multipath 16->9%, real 9->2% MD)
        # Peak-search limit (deg): suppress spurious spectral peaks in the grid-edge columns beyond the
        # trained/valid azimuth range. On real data the DataSim-trained (front-cone) unfolding leaks energy
        # into the untrained |theta|>~82 edge columns -> the greedy argmax lands there (44% of samples at
        # |a|>=88). Masking |theta|>peak_lim_deg before peak-picking recovers the true interior peak (44->8% MD).
        self.peak_lim_deg = peak_lim_deg
        self.fine_cols_per_180deg = fine_cols_per_180deg    # fine-grid density (columns per 180 deg span)
        self.p_init_floor = p_init_floor                    # p-schedule init: floor + amp * exp(-decay * k)
        self.p_init_amp = p_init_amp
        self.p_init_decay = p_init_decay
        self.p_init_clamp_eps = p_init_clamp_eps            # clamp p_sched to [eps, 2 - eps]
        self.lam_init = lam_init                            # lambda init (matches Hof MFOCUSS lambda = 0.99)
        self.lam_eps = lam_eps                              # softplus offset: lambda_k = softplus(raw) + eps
        self.mcp_raw_init = mcp_raw_init                    # raw MCP init -> softplus ~ 0 (plain FOCUSS at init)
        self.temp_raw_init = temp_raw_init                  # raw readout temp init -> softplus(-3)+eps ~ 0.05
        self.temp_eps = temp_eps                            # readout temp = softplus(raw) + eps
        self.train_temp_floor = train_temp_floor            # train-time readout temperature floor
        self.spectrum_loss_weight = spectrum_loss_weight    # dense heat-map aux loss weight (see training_step)
        self.spectrum_loss_sigma_deg = spectrum_loss_sigma_deg  # Gaussian heat-map target sigma (deg)
        self.conv_kernel_halfwidth_deg = conv_kernel_halfwidth_deg  # "conv" aux-loss kernel half-width (deg)
        self.conv_kernel_min_cols = conv_kernel_min_cols    # "conv" kernel minimum half-width (columns)
        self.bce_clamp_eps = bce_clamp_eps                  # BCE input clamp to [eps, 1 - eps]
        self.ls_init_reg = ls_init_reg                      # least-squares init regularization (A A^H + reg I)
        self.rms_clamp_eps = rms_clamp_eps                  # RMS normalization clamp floor
        self.reweight_floor = reweight_floor                # FOCUSS reweight row-norm floor (matches MFOCUSS)
        self.rn_eps = rn_eps                                # MCP relative-magnitude normalization epsilon
        self.spectrum_norm_eps = spectrum_norm_eps          # spectrum max-normalization epsilon
        self.hyper_norm_eps = hyper_norm_eps                # hyper-feature norm-ratio denominator epsilon
        self.hyper_log_eps = hyper_log_eps                  # hyper-feature log10 argument epsilon
        self.pick_half_frac = pick_half_frac                # soft-argmax window half-width (fraction of G)
        self.pick_supp_frac = pick_supp_frac                # greedy peak suppression radius (fraction of G)
        self.pick_min_cols = pick_min_cols                  # minimum window/refine half-width (columns)
        self.extend_p_decay = extend_p_decay                # extension-tail Hof p-anneal decay rate
        self.extend_p_target = extend_p_target              # extension-tail p-anneal target
        self.refine_win_deg = refine_win_deg                # fine-grid local refine window (+/- deg)
        self.eval_chunk = eval_chunk                        # fine-grid pass chunk size (memory bound)
        self.count_peak_thr = count_peak_thr                # source-count local-max significance threshold

        self._lam_scale = 1.0

        # Overcomplete steering dictionary on the (recorded or analytic) manifold.
        # Source-adaptive grid: a COARSE grid (grid_size) for >=2 sources -- broader peaks
        # detect both close sources (a fine grid over-sparsifies and drops the weaker one),
        # and a FINE grid (grid_size_fine) for a single source -- sub-grid precision. The
        # learned per-layer lambda/p are grid-independent, so the same schedule drives both.
        if grid_range_deg is None:
            lo, hi = system_model.params.doa_range
        elif np.isscalar(grid_range_deg):
            lo, hi = -float(grid_range_deg), float(grid_range_deg)
        else:
            lo, hi = float(grid_range_deg[0]), float(grid_range_deg[1])
        grid_rad = np.deg2rad(np.linspace(lo, hi, grid_size))
        A = np.asarray(system_model.steering_vec(grid_rad))      # (N, G) coarse
        self.register_buffer("grid", torch.as_tensor(grid_rad, dtype=torch.float64))
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.complex128))
        # Fine (single-source) grid: preserve angular density (~0.2 deg/col at the ±90 default)
        # across whatever range was requested, so full-azimuth grids stay well-resolved.
        self.grid_size_fine = max(int(grid_size), int(round(self.fine_cols_per_180deg * abs(hi - lo) / 180.0)))
        grid_rad_f = np.deg2rad(np.linspace(lo, hi, self.grid_size_fine))
        Af = np.asarray(system_model.steering_vec(grid_rad_f))   # (N, Gf) fine (single-source precision)
        self.register_buffer("grid_fine", torch.as_tensor(grid_rad_f, dtype=torch.float64))
        self.register_buffer("A_fine", torch.as_tensor(Af, dtype=torch.complex128))
        self._A_use = None          # active dictionary/grid (set per-forward by source count)
        self._grid_use = None

        # Learned per-layer parameters (reparameterized to valid ranges):
        #   lambda_k = softplus(raw_lambda_k) + eps  > 0   (regularization)
        #   p_k      = 2 * sigmoid(raw_p_k) in (0, 2)       (lp diversity)
        # Initialized to reproduce classical MFOCUSS (Hof annealing): lambda_k = 0.99
        # (constant) and p_k annealed 0.99 -> ~0.1 across the K layers. With the RMS
        # normalization + least-squares init in get_learned_covariance, untrained
        # DU-MFOCUSS *is* MFOCUSS (a strict special case: DU with a fixed schedule),
        # so end-to-end training can only improve on the classical baseline.
        ks = torch.arange(num_iterations, dtype=torch.float32)
        p_sched = (self.p_init_floor + self.p_init_amp * torch.exp(-self.p_init_decay * ks)
                   ).clamp(self.p_init_clamp_eps, 2.0 - self.p_init_clamp_eps)    # 0.99 -> ~0.1
        raw_p0 = torch.log((p_sched / 2.0) / (1.0 - p_sched / 2.0))               # invert 2*sigmoid
        lam0 = float(np.log(np.exp(self.lam_init) - 1.0))                         # invert softplus -> lambda~=0.99
        self._raw_lambda = nn.Parameter(torch.full((num_iterations,), lam0))
        self._raw_p = nn.Parameter(raw_p0)
        # MCP (minimax concave penalty) per-layer de-bias strength m_k = softplus(raw_mcp_k) >= 0.
        # The MCP term reduces FOCUSS's over-shrinkage of already-strong atoms by boosting their
        # reweight in proportion to the PREVIOUS iterate's relative magnitude (see the loop below).
        # Initialized at raw=-3 -> m_k ~= 0.05 (a <=5% de-bias at init): near-exact reduce-to-MFOCUSS,
        # but with a LIVE gradient (softplus'(-3)~0.05; the earlier -16 init had softplus'~1e-7 -> the
        # MCP was permanently frozen at 1e-7 and training could never wake it).
        self._raw_mcp = nn.Parameter(torch.full((num_iterations,), self.mcp_raw_init))
        # INPUT-ADAPTIVE hyper-parameters: per-layer corrections to (raw_lambda_k, raw_p_k, raw_mcp_k)
        # computed from the PREVIOUS iterate's state — [log residual ratio, row-sparsity ratio].
        # Zero-initialized: at init the corrections are exactly 0, so the reduce-to-MFOCUSS
        # equivalence is preserved; training learns data-dependent schedules (incl. the MCP) on top.
        self._hyper = nn.ModuleList([nn.Linear(2, 3) for _ in range(num_iterations)])
        for lin in self._hyper:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.adaptive_hyper = True
        # Readout temperature for the differentiable local soft-argmax peak picker
        # (init softplus(-3)+1e-3 ~= 0.05, matching the MFOCUSS peak picker).
        self._raw_temp = nn.Parameter(torch.tensor(self.temp_raw_init))
        self.spectrum_loss_mode = "bce"     # "bce" = Gaussian heat-map BCE; "conv" = convolutional
                                            # window matching: smooth BOTH spectrum and GT spikes with a
                                            # triangular kernel and L2-match — an angle offset between
                                            # peaks becomes an INTEGRAL of power difference (kernel
                                            # overlap), giving wide-basin, fully derivable gradients
        self._last_spectrum = None
        # Learnable dictionary calibration: A_cal = C @ A with C = _cal_re + j*_cal_im, init C = I_N.
        # Registered ONLY when enabled, so the default model's state_dict is byte-identical (old
        # weights strict-load). At init C = I -> A unchanged -> reduce-to-MFOCUSS preserved; a trained
        # C corrects the effective dictionary geometry (the trainable analog of DU-MFOCUSS-cal).
        self.learn_calibration = bool(learn_calibration)
        self.cal_reg_weight = cal_reg_weight
        if self.learn_calibration:
            self._cal_re = nn.Parameter(torch.eye(system_model.params.N))
            self._cal_im = nn.Parameter(torch.zeros(system_model.params.N, system_model.params.N))

    def get_model_params(self):
        """Returns a short string summarizing the model's hyper-parameters.

        Returns:
            str: String of the form "K=<num_iter>_grid=<grid_size>".
        """
        return f"K={self.num_iter}_grid={self.grid_size}"

    def get_learned_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the unrolled M-FOCUSS layers and returns the recovered spectrum.

        Args:
            x (torch.Tensor): Input snapshots, shape [B, N, T].

        Returns:
            torch.Tensor: Normalized angular power spectrum, shape [B, G].
        """
        Y = x.to(torch.complex128)                                # [B, N, T]
        A = self._A_use if self._A_use is not None else self.A    # [N, G] source-adaptive
        if getattr(self, "learn_calibration", False):
            C = (self._cal_re + 1j * self._cal_im).to(A.dtype).to(A.device)   # [N,N], init I
            A = C @ A                                                          # calibrated dictionary
        B = Y.shape[0]
        eye = torch.eye(A.shape[0], dtype=torch.complex128, device=Y.device)

        # RMS-normalize the input (match MFOCUSS): brings the signal to RMS=1 so the
        # learned regularization lambda operates at a signal-scale-independent point.
        rms = torch.sqrt(torch.mean(Y.abs() ** 2, dim=(1, 2), keepdim=True)).clamp_min(self.rms_clamp_eps)
        Y = Y / rms.to(torch.complex128)

        # Least-squares (min-norm) initialization: s0 = A^H (A A^H)^-1 Y (match MFOCUSS).
        AAh = A @ A.conj().t()
        s = torch.einsum("gn,bnt->bgt", A.conj().t(),
                         torch.linalg.solve(AAh + self.ls_init_reg * eye, Y))  # [B, G, T]

        # Iteration budget parity with classical MFOCUSS (K=100): after the K learned layers,
        # optionally continue iterating with the LAST layer's learned (lambda, p, mcp), annealing p
        # down the same Hof tail (p -> 0.01). At init this reduces DU to MFOCUSS-(K+extend) — the
        # deployed MFOCUSS budget — so the learned model can only add on top, never lag on budget.
        extend = int(getattr(self, "extend_iters", 0))
        total_iters = self.num_iter + extend
        for k in range(total_iters):
            kk = min(k, self.num_iter - 1)                        # extension reuses the last layer
            row_norm = torch.linalg.norm(s, dim=2)                # [B, G] real

            if getattr(self, "adaptive_hyper", False):
                # Input-adaptive (lambda_k, p_k): correct the learned per-layer schedule by a
                # zero-initialized linear map of the PREVIOUS iterate's state. Features:
                #   f1 = log10(||Y - A s||_F / ||Y||_F)   residual ratio (fit quality)
                #   f2 = (||rn||_1/||rn||_2)/sqrt(G)      row-sparsity ratio in (0, 1]
                resid = Y - torch.einsum("ng,bgt->bnt", A, s)
                f1 = torch.log10(torch.linalg.norm(resid.reshape(B, -1), dim=1) /
                                 (torch.linalg.norm(Y.reshape(B, -1), dim=1) + self.hyper_norm_eps) + self.hyper_log_eps)
                f2 = (row_norm.sum(dim=1) / (torch.linalg.norm(row_norm, dim=1) + self.hyper_norm_eps)) / np.sqrt(A.shape[1])
                feat = torch.stack([f1, f2], dim=1).to(torch.float32)                    # [B, 2]
                delta = self._hyper[kk](feat).to(torch.float64)                           # [B, 3]
                p_k = (2.0 * torch.sigmoid(self._raw_p[kk] + delta[:, 1])).unsqueeze(1)   # [B, 1]
                lam_k = (F.softplus(self._raw_lambda[kk] + delta[:, 0]) + self.lam_eps) * self._lam_scale   # [B]
                m_k = F.softplus(self._raw_mcp[kk] + delta[:, 2]).unsqueeze(1)             # [B, 1] MCP de-bias
            else:
                p_k = 2.0 * torch.sigmoid(self._raw_p[kk])
                lam_k = (F.softplus(self._raw_lambda[kk]) + self.lam_eps) * self._lam_scale
                m_k = F.softplus(self._raw_mcp[kk])

            if k >= self.num_iter:
                # extension tail: continue the Hof p-anneal from the last layer's p toward 0.01
                decay = float(np.exp(-self.extend_p_decay * (k - self.num_iter + 1)))
                p_k = self.extend_p_target + (p_k - self.extend_p_target) * decay

            # FOCUSS reweight + learned MCP de-bias: the (1 + m_k*rn) factor boosts the reweight for
            # already-strong atoms (relative magnitude rn in [0,1] from the PREVIOUS iterate),
            # countering FOCUSS's over-shrinkage of large coefficients (the minimax-concave idea:
            # taper the penalty as a coefficient grows). m_k~0 at init -> factor ~1 -> classical FOCUSS.
            rn = row_norm / (row_norm.amax(dim=1, keepdim=True) + self.rn_eps)   # [B, G] relative magnitude
            w = (row_norm + self.reweight_floor) ** (1.0 - p_k / 2.0) * (1.0 + m_k * rn)   # floor matches MFOCUSS (was 1e-9: kept dying atoms ~1000x more alive)

            AW = A.unsqueeze(0) * w.unsqueeze(1).to(torch.complex128)   # [B, N, G]
            gram = torch.einsum("bng,bmg->bnm", AW, AW.conj())    # [B, N, N]
            lam_b = lam_k if lam_k.dim() else lam_k.expand(B)     # per-sample lambda [B]
            reg = gram + lam_b.to(torch.complex128).view(-1, 1, 1) * eye   # [B, N, N]
            inv_Y = torch.linalg.solve(reg, Y)                    # [B, N, T]
            q = torch.einsum("bng,bnt->bgt", AW.conj(), inv_Y)    # [B, G, T]
            s = w.unsqueeze(2).to(torch.complex128) * q           # [B, G, T]

        spectrum = torch.linalg.norm(s, dim=2)                    # [B, G] real power
        spectrum = spectrum / (spectrum.amax(dim=1, keepdim=True) + self.spectrum_norm_eps)
        self._last_spectrum = spectrum                            # for the training aux loss
        return spectrum

    def _soft_argmax(self, spectrum: torch.Tensor, source_number: int) -> torch.Tensor:
        """Differentiable peak picker: local soft-argmax around the top peaks.

        Args:
            spectrum (torch.Tensor): Normalized power spectrum, shape [B, G].
            source_number (int): Number of peaks (sources) to extract.

        Returns:
            torch.Tensor: Estimated angles in radians, shape [B, source_number].
        """
        B, G = spectrum.shape
        grid = self._grid_use if self._grid_use is not None else self.grid
        half = max(self.pick_min_cols, int(self.pick_half_frac * G))   # local soft-argmax window (~sub-min_gap)
        supp = max(half, int(self.pick_supp_frac * G))                 # suppression radius to separate sources
        temp = F.softplus(self._raw_temp) + self.temp_eps
        if self.training:
            # TRAIN-time temperature floor: at temp~0.05 the window softmax over the normalized
            # spectrum saturates (d(angle)/d(spectrum) ~ 0), so the RMSPE term supplied NO gradient
            # to (lambda_k, p_k, m_k). A 0.3 floor during training keeps the readout informative;
            # EVAL keeps the learned sharp temperature (readout accuracy unchanged).
            temp = torch.clamp(temp, min=self.train_temp_floor)
        grid_idx = torch.arange(G, device=spectrum.device)
        offsets = torch.arange(-half, half + 1, device=spectrum.device)
        # Greedy peak picking on a detached copy: take the argmax, suppress its
        # neighborhood, repeat — so the M peaks are distinct (topk alone could pick
        # adjacent cells of a single peak and miss a second source).
        work = spectrum.detach().clone()
        centers = []
        for _ in range(source_number):
            idx = work.argmax(dim=1)                                            # [B]
            centers.append(idx)
            mask = (grid_idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= supp      # [B, G]
            work = work.masked_fill(mask, float("-inf"))
        out = torch.zeros(B, source_number, dtype=torch.float64, device=spectrum.device)
        for j, idx in enumerate(centers):
            win = (idx.unsqueeze(1) + offsets).clamp(0, G - 1)                   # [B, W]
            vals = torch.gather(spectrum, 1, win)                               # [B, W]
            angles_win = grid[win]                                              # [B, W]
            weights = torch.softmax(vals / temp, dim=1)
            out[:, j] = (weights * angles_win).sum(dim=1)
        return out

    def _estimate_count(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Estimates the source count from the number of significant spectral peaks.

        Args:
            spectrum (torch.Tensor): Normalized power spectrum, shape [B, G].

        Returns:
            torch.Tensor: Estimated source count per sample, shape [B] (>= 1).
        """
        center = spectrum[:, 1:-1]
        left = spectrum[:, :-2]
        right = spectrum[:, 2:]
        peaks = (center > left) & (center >= right) & (center > self.count_peak_thr)
        return peaks.sum(dim=1).clamp(min=1).float()

    def forward(self, x: torch.Tensor, sources_num: int = None, phase: str = "train"):
        """Recovers the sparse spectrum and reads out the DoAs.

        Args:
            x (torch.Tensor): Input snapshots, shape [B, N, T].
            sources_num (int): Number of sources to estimate.
            phase (str): Unused; kept for interface compatibility.

        Returns:
            tuple: (doa_prediction [B, M], source_estimation [B], None).
        """
        src = int(sources_num) if sources_num is not None else self.M
        # Source-adaptive regularization: lower lambda for >=2 sources sharpens the spectrum so close
        # sources resolve (the learned single-source lambda over-smooths and merges them).
        self._lam_scale = 1.0 if src <= 1 else self.lam_multi_scale
        # Source-adaptive grid: FINE dictionary for a single source (sub-grid precision), COARSE for
        # >=2 sources (broader peaks detect both close sources; the fine grid over-sparsifies -> misses one).
        single = src <= 1
        self._A_use = self.A_fine if single else self.A
        self._grid_use = self.grid_fine if single else self.grid
        spectrum = self.get_learned_covariance(x)
        if self.peak_lim_deg is not None:
            edge = (self._grid_use.abs() > np.deg2rad(self.peak_lim_deg)).unsqueeze(0)  # [1, G] grid-edge columns
            spectrum = spectrum.masked_fill(edge, 0.0)                                  # suppress spurious edge peaks
        doa_prediction = self._soft_argmax(spectrum, src)
        source_estimation = self._estimate_count(spectrum)
        if not single and getattr(self, "refine_pairs", True):
            # Coarse-detect -> FINE-refine: multi-source angles were read off the COARSE grid
            # (broad peaks, ~0.5 deg/col) -> quantization + peak-shape bias dominate the pair RMS.
            # Re-run the recovery on the FINE dictionary and take a local soft-argmax within a
            # small window around each coarse peak (detection unchanged; accuracy improves).
            self._A_use = self.A_fine
            self._grid_use = self.grid_fine
            spec_f = torch.cat([self.get_learned_covariance(x[i:i + self.eval_chunk])   # chunked: fine-grid pass
                                for i in range(0, x.shape[0], self.eval_chunk)])        # OOMs on 4 GB at B=3000
            doa_prediction = self._refine_local(spec_f, doa_prediction)
        return doa_prediction, source_estimation, None

    def _refine_local(self, spec_f: torch.Tensor, doa: torch.Tensor, win_deg: float = None) -> torch.Tensor:
        """Local soft-argmax on the FINE grid within +/-win_deg of each detected angle.

        Gather-based window (like _soft_argmax) — NOT a -inf-masked softmax over the full
        grid: the CUDA float64 softmax kernel returns wrong normalization when ~99% of a
        row is -inf (weights summed to exact 1/2 or 1/3 on healthy inputs).
        """
        win_deg = self.refine_win_deg if win_deg is None else win_deg
        temp = F.softplus(self._raw_temp) + self.temp_eps
        if self.training:
            # TRAIN-time temperature floor: at temp~0.05 the window softmax over the normalized
            # spectrum saturates (d(angle)/d(spectrum) ~ 0), so the RMSPE term supplied NO gradient
            # to (lambda_k, p_k, m_k). A 0.3 floor during training keeps the readout informative;
            # EVAL keeps the learned sharp temperature (readout accuracy unchanged).
            temp = torch.clamp(temp, min=self.train_temp_floor)
        grid = self.grid_fine                                                     # [Gf]
        Gf = grid.numel()
        step = float(grid[1] - grid[0])
        K = max(self.pick_min_cols, int(round(np.deg2rad(win_deg) / step)))
        offsets = torch.arange(-K, K + 1, device=spec_f.device)
        out = doa.clone()
        for j in range(doa.shape[1]):
            idx = torch.argmin((grid[None, :] - doa[:, j:j + 1]).abs(), dim=1)    # [B] nearest fine col
            win = (idx.unsqueeze(1) + offsets).clamp(0, Gf - 1)                   # [B, W]
            vals = torch.gather(spec_f, 1, win)                                   # [B, W]
            w = torch.softmax(vals / temp, dim=1)
            out[:, j] = (w * grid[win]).sum(dim=1)
        return out

    def _count_accuracy(self, sources_num, source_estimation):
        """Counts how many estimated source counts match the true count.

        Args:
            sources_num: True number of sources (scalar tensor).
            source_estimation (torch.Tensor): Estimated counts per sample, shape [B].

        Returns:
            float: Number of correct source-count estimates in the batch.
        """
        if source_estimation is None:
            return 0
        return torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation)).item()

    def training_step(self, batch):
        """Runs one training step: forward pass and RMSPE loss.

        Args:
            batch: A tuple of (x, sources_num, angles).

        Returns:
            tuple: (loss, acc, None) where loss is the RMSPE loss and acc is the
                source-count accuracy count.
        """
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, source_estimation, _ = self(x, sources_num)
        loss = self.criterion(doa_prediction, angles)
        # Dense spectrum heat-map auxiliary loss: the soft-argmax readout's window softmax is
        # SATURATED at the reduce-to-MFOCUSS init (measured total grad norm ~5e-15 — literally no
        # training signal; the flat loss curve). Supervising the normalized spectrum toward a
        # Gaussian bump at each GT angle gives dense, informative gradients to (lambda_k, p_k, m_k)
        # and the hyper net — sharpening peak contrast, which is exactly what detection needs.
        spec = getattr(self, "_last_spectrum", None)
        if self.training and spec is not None:
            grid = (self._grid_use if self._grid_use is not None else self.grid).to(spec.device)
            sig = np.deg2rad(self.spectrum_loss_sigma_deg)
            tgt = torch.zeros_like(spec)
            for j in range(angles.shape[1]):
                d = grid[None, :] - angles[:, j:j + 1].to(spec.device)
                tgt = torch.maximum(tgt, torch.exp(-0.5 * (d / sig) ** 2))
            if getattr(self, "spectrum_loss_mode", "bce") == "conv":
                # Convolutional window matching (user-proposed): K * S vs K * T with a triangular
                # window (~8 deg wide). Peak-position error -> overlapping-area difference, so the
                # gradient is informative even when predicted and true peaks do not overlap pointwise.
                step = float(grid[1] - grid[0])
                kh = max(self.conv_kernel_min_cols, int(round(np.deg2rad(self.conv_kernel_halfwidth_deg) / step)))
                ker = 1.0 - torch.arange(-kh, kh + 1, device=spec.device, dtype=spec.dtype).abs() / (kh + 1)
                ker = (ker / ker.sum()).view(1, 1, -1)
                tgt_d = torch.zeros_like(spec)
                idx = torch.argmin((grid[None, None, :] - angles.to(spec.device)[:, :, None]).abs(), dim=2)
                tgt_d.scatter_(1, idx, 1.0)
                Ss = F.conv1d(spec.unsqueeze(1), ker, padding=kh).squeeze(1)
                Ts = F.conv1d(tgt_d.unsqueeze(1), ker, padding=kh).squeeze(1)
                loss = loss + self.spectrum_loss_weight * ((Ss - Ts) ** 2).sum(dim=1).mean() * x.shape[0]
            else:
                loss = loss + self.spectrum_loss_weight * F.binary_cross_entropy(
                    spec.clamp(self.bce_clamp_eps, 1 - self.bce_clamp_eps), tgt.to(spec.dtype)) * x.shape[0]
        if getattr(self, "learn_calibration", False):
            C = (self._cal_re + 1j * self._cal_im)
            eyeC = torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
            loss = loss + self.cal_reg_weight * (C - eyeC).abs().pow(2).sum() * x.shape[0]
        acc = self._count_accuracy(sources_num, source_estimation)
        return loss, acc, None

    @torch.no_grad()
    def validation_step(self, batch):
        """Runs one validation step: forward pass and RMSPE loss.

        Args:
            batch: A tuple of (x, sources_num, angles).

        Returns:
            tuple: (loss, acc) — RMSPE loss and source-count accuracy count.
        """
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, source_estimation, _ = self(x, sources_num)
        loss = self.criterion(doa_prediction, angles)
        acc = self._count_accuracy(sources_num, source_estimation)
        return loss, acc

    @torch.no_grad()
    def test_step(self, batch):
        """Runs one test step (delegates to validation_step).

        Args:
            batch: A tuple of (x, sources_num, angles).

        Returns:
            tuple: (loss, acc) as returned by validation_step.
        """
        return self.validation_step(batch)
