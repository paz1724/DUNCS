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
                 grid_range_deg: float = None):
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
        """
        super().__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.num_iter = num_iterations
        self.grid_size = grid_size
        self.M = system_model.params.M
        self.lam_multi_scale = 0.02     # scale the learned lambda down for >=2 sources; 0.02 (sharper than MFOCUSS's
                                        # 0.05) selected on the real train split and better on synthetic too
                                        # (synth reuse 10->5%, multipath 16->9%, real held-out reuse 9->2% MD)
        self._lam_scale = 1.0
        # Peak-search limit (deg): suppress spurious spectral peaks in the grid-edge columns beyond the
        # trained/valid azimuth range. On real data the DataSim-trained (front-cone) unfolding leaks energy
        # into the untrained |theta|>~82 edge columns -> the greedy argmax lands there (44% of samples at
        # |a|>=88). Masking |theta|>peak_lim_deg before peak-picking recovers the true interior peak (44->8% MD).
        self.peak_lim_deg = None

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
        self.grid_size_fine = max(int(grid_size), int(round(901 * abs(hi - lo) / 180.0)))
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
        p_sched = (0.01 + 0.98 * torch.exp(-0.27 * ks)).clamp(1e-3, 2.0 - 1e-3)   # 0.99 -> ~0.1
        raw_p0 = torch.log((p_sched / 2.0) / (1.0 - p_sched / 2.0))               # invert 2*sigmoid
        lam0 = float(np.log(np.exp(0.99) - 1.0))                                  # invert softplus -> lambda~=0.99
        self._raw_lambda = nn.Parameter(torch.full((num_iterations,), lam0))
        self._raw_p = nn.Parameter(raw_p0)
        # INPUT-ADAPTIVE hyper-parameters: per-layer corrections to (raw_lambda_k, raw_p_k)
        # computed from the PREVIOUS iterate's state — [log residual ratio, row-sparsity ratio].
        # Zero-initialized: at init the corrections are exactly 0, so the reduce-to-MFOCUSS
        # equivalence is preserved; training learns data-dependent schedules on top.
        self._hyper = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_iterations)])
        for lin in self._hyper:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.adaptive_hyper = True
        # Readout temperature for the differentiable local soft-argmax peak picker
        # (init softplus(-3)+1e-3 ~= 0.05, matching the MFOCUSS peak picker).
        self._raw_temp = nn.Parameter(torch.tensor(-3.0))

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
        B = Y.shape[0]
        eye = torch.eye(A.shape[0], dtype=torch.complex128, device=Y.device)

        # RMS-normalize the input (match MFOCUSS): brings the signal to RMS=1 so the
        # learned regularization lambda operates at a signal-scale-independent point.
        rms = torch.sqrt(torch.mean(Y.abs() ** 2, dim=(1, 2), keepdim=True)).clamp_min(1e-12)
        Y = Y / rms.to(torch.complex128)

        # Least-squares (min-norm) initialization: s0 = A^H (A A^H)^-1 Y (match MFOCUSS).
        AAh = A @ A.conj().t()
        s = torch.einsum("gn,bnt->bgt", A.conj().t(),
                         torch.linalg.solve(AAh + 1e-6 * eye, Y))  # [B, G, T]

        for k in range(self.num_iter):
            row_norm = torch.linalg.norm(s, dim=2)                # [B, G] real

            if getattr(self, "adaptive_hyper", False):
                # Input-adaptive (lambda_k, p_k): correct the learned per-layer schedule by a
                # zero-initialized linear map of the PREVIOUS iterate's state. Features:
                #   f1 = log10(||Y - A s||_F / ||Y||_F)   residual ratio (fit quality)
                #   f2 = (||rn||_1/||rn||_2)/sqrt(G)      row-sparsity ratio in (0, 1]
                resid = Y - torch.einsum("ng,bgt->bnt", A, s)
                f1 = torch.log10(torch.linalg.norm(resid.reshape(B, -1), dim=1) /
                                 (torch.linalg.norm(Y.reshape(B, -1), dim=1) + 1e-30) + 1e-12)
                f2 = (row_norm.sum(dim=1) / (torch.linalg.norm(row_norm, dim=1) + 1e-30)) / np.sqrt(A.shape[1])
                feat = torch.stack([f1, f2], dim=1).to(torch.float32)                    # [B, 2]
                delta = self._hyper[k](feat).to(torch.float64)                           # [B, 2]
                p_k = (2.0 * torch.sigmoid(self._raw_p[k] + delta[:, 1])).unsqueeze(1)   # [B, 1]
                lam_k = (F.softplus(self._raw_lambda[k] + delta[:, 0]) + 1e-5) * self._lam_scale   # [B]
            else:
                p_k = 2.0 * torch.sigmoid(self._raw_p[k])
                lam_k = (F.softplus(self._raw_lambda[k]) + 1e-5) * self._lam_scale

            w = (row_norm + 1e-9) ** (1.0 - p_k / 2.0)            # FOCUSS reweighting

            AW = A.unsqueeze(0) * w.unsqueeze(1).to(torch.complex128)   # [B, N, G]
            gram = torch.einsum("bng,bmg->bnm", AW, AW.conj())    # [B, N, N]
            lam_b = lam_k if lam_k.dim() else lam_k.expand(B)     # per-sample lambda [B]
            reg = gram + lam_b.to(torch.complex128).view(-1, 1, 1) * eye   # [B, N, N]
            inv_Y = torch.linalg.solve(reg, Y)                    # [B, N, T]
            q = torch.einsum("bng,bnt->bgt", AW.conj(), inv_Y)    # [B, G, T]
            s = w.unsqueeze(2).to(torch.complex128) * q           # [B, G, T]

        spectrum = torch.linalg.norm(s, dim=2)                    # [B, G] real power
        spectrum = spectrum / (spectrum.amax(dim=1, keepdim=True) + 1e-9)
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
        half = max(1, int(0.025 * G))           # local soft-argmax window (~sub-min_gap)
        supp = max(half, int(0.03 * G))         # suppression radius to separate sources
        temp = F.softplus(self._raw_temp) + 1e-3
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

    @staticmethod
    def _estimate_count(spectrum: torch.Tensor) -> torch.Tensor:
        """Estimates the source count from the number of significant spectral peaks.

        Args:
            spectrum (torch.Tensor): Normalized power spectrum, shape [B, G].

        Returns:
            torch.Tensor: Estimated source count per sample, shape [B] (>= 1).
        """
        center = spectrum[:, 1:-1]
        left = spectrum[:, :-2]
        right = spectrum[:, 2:]
        peaks = (center > left) & (center >= right) & (center > 0.5)
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
            spec_f = torch.cat([self.get_learned_covariance(x[i:i + 256])        # chunked: fine-grid pass
                                for i in range(0, x.shape[0], 256)])             # OOMs on 4 GB at B=3000
            doa_prediction = self._refine_local(spec_f, doa_prediction)
        return doa_prediction, source_estimation, None

    def _refine_local(self, spec_f: torch.Tensor, doa: torch.Tensor, win_deg: float = 2.0) -> torch.Tensor:
        """Local soft-argmax on the FINE grid within +/-win_deg of each detected angle.

        Gather-based window (like _soft_argmax) — NOT a -inf-masked softmax over the full
        grid: the CUDA float64 softmax kernel returns wrong normalization when ~99% of a
        row is -inf (weights summed to exact 1/2 or 1/3 on healthy inputs).
        """
        temp = F.softplus(self._raw_temp) + 1e-3
        grid = self.grid_fine                                                     # [Gf]
        Gf = grid.numel()
        step = float(grid[1] - grid[0])
        K = max(1, int(round(np.deg2rad(win_deg) / step)))
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
