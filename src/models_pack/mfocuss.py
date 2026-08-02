"""
MFOCUSS: classical (non-learned) Multiple-measurement FOCUSS baseline for DoA.

This is the same M-FOCUSS sparse-recovery iteration used by DU-MFOCUSS, but with
FIXED regularization lambda and FIXED lp-diversity p, run for a fixed number of
iterations and NOT trained. It serves as the classical baseline: comparing
DU-MFOCUSS against MFOCUSS isolates the benefit of *learning* lambda and p per
layer (the deep-unfolding gain), while comparing SubspaceNet / DoAFormer against
it measures the gain of the learned-feature approaches over classical sparse
recovery. The steering dictionary is the recorded manifold (steering_vec), so the
estimation manifold matches the data-generation manifold.
"""

import numpy as np
import torch

from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.metrics.criterions import set_criterions
from src.utils import device


class MFOCUSS(ParentModel):
    """Classical M-FOCUSS sparse-recovery DoA baseline (fixed lambda, p; no training)."""

    def __init__(self, system_model: SystemModel, num_iterations: int = 50,
                 grid_size: int = 121, p: float = 0.8, lam: float = 0.05,
                 criterion: str = "rmspe", grid_range_deg=None,
                 p_init: float = 0.99, p_min: float = 0.01, p_decay: float = 0.2,
                 lam_init: float = 0.99, lam_max: float = 0.99, lam_growth: float = 0.2,
                 lam_multi: float = 0.02,
                 coarse_cols_per_180deg: int = 361,
                 pick_temp: float = 0.05,
                 pick_half_frac: float = 0.025,
                 pick_supp_frac: float = 0.03,
                 pick_min_cols: int = 1,
                 refine_win_deg: float = 2.0,
                 eval_chunk: int = 256,
                 ls_init_reg: float = 1e-6,
                 rms_clamp_eps: float = 1e-12,
                 reweight_floor: float = 1e-12,
                 spectrum_norm_eps: float = 1e-9,
                 count_peak_thr: float = 0.5):
        """Initializes the classical MFOCUSS baseline.

        Args:
            system_model (SystemModel): Array geometry and signal parameters.
            num_iterations (int): Number of M-FOCUSS iterations (fixed, not unrolled).
            grid_size (int): Number of angle-grid columns in the FINE (single-source)
                steering dictionary; the multi-source dictionary is COARSE (<=361 cols).
            p (float): Fixed lp-diversity exponent in (0, 2] (smaller = sparser).
            lam (float): Fixed regularization weight (> 0).
            criterion (str): Evaluation criterion name (e.g. "rmspe").
            grid_range_deg (float | [lo, hi]): Optional grid azimuth range override
                (scalar = symmetric); same semantics as DU-MFOCUSS.

        The remaining keyword args are hoisted numeric tunables / thresholds /
        epsilons; their defaults reproduce the historical hard-coded values
        exactly (see the constants block below for what each one controls).
        """
        super().__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.num_iter = num_iterations
        self.grid_size = grid_size
        self.p = float(p)
        self.lam = float(lam)

        # ---- Hoisted tunables / schedule constants / epsilons (defaults == historical literals) ----
        # Hof MFOCUSS_Original annealing schedule (cArray defaults): p anneals from
        # p_init toward p_min (rate p_decay); lambda stays at 0.99 (lam_init==lam_max).
        self.p_init, self.p_min, self.p_decay = p_init, p_min, p_decay
        self.lam_init, self.lam_max, self.lam_growth = lam_init, lam_max, lam_growth
        self.lam_multi = lam_multi  # low regularization for >=2 sources (0.02: sharper than the old 0.05 —
                                    # same value as DU's train-selected scale; better on synthetic too)
        self.coarse_cols_per_180deg = coarse_cols_per_180deg  # coarse (M>=2) grid density cap (cols per 180 deg)
        self.pick_temp = pick_temp                  # peak-pick / refine local softmax temperature
        self.pick_half_frac = pick_half_frac        # peak-pick window half-width (fraction of G)
        self.pick_supp_frac = pick_supp_frac        # greedy peak suppression radius (fraction of G)
        self.pick_min_cols = pick_min_cols          # minimum window/refine half-width (columns)
        self.refine_win_deg = refine_win_deg        # fine-grid local refine window (+/- deg)
        self.eval_chunk = eval_chunk                # fine-grid pass chunk size (memory bound)
        self.ls_init_reg = ls_init_reg              # least-squares init regularization (A A^H + reg I)
        self.rms_clamp_eps = rms_clamp_eps          # RMS normalization clamp floor
        self.reweight_floor = reweight_floor        # FOCUSS reweight row-norm floor
        self.spectrum_norm_eps = spectrum_norm_eps  # spectrum max-normalization epsilon
        self.count_peak_thr = count_peak_thr        # source-count local-max significance threshold

        self.M = system_model.params.M

        # Source-adaptive grid (backported from DU-MFOCUSS): COARSE grid for >=2 sources
        # (broad peaks detect BOTH close sources; a fine grid over-sparsifies and drops the
        # weaker one), FINE grid for a single source (sub-grid precision).
        if grid_range_deg is None:
            lo, hi = system_model.params.doa_range
        elif np.isscalar(grid_range_deg):
            lo, hi = -float(grid_range_deg), float(grid_range_deg)
        else:
            lo, hi = float(grid_range_deg[0]), float(grid_range_deg[1])
        gm = min(int(grid_size), int(round(self.coarse_cols_per_180deg * abs(hi - lo) / 180.0)))
        grid_rad = np.deg2rad(np.linspace(lo, hi, gm))
        A = np.asarray(system_model.steering_vec(grid_rad))      # (N, Gm) coarse (M>=2)
        self.register_buffer("grid", torch.as_tensor(grid_rad, dtype=torch.float64))
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.complex128))
        grid_rad_f = np.deg2rad(np.linspace(lo, hi, grid_size))
        Af = np.asarray(system_model.steering_vec(grid_rad_f))   # (N, G) fine (M=1)
        self.register_buffer("grid_fine", torch.as_tensor(grid_rad_f, dtype=torch.float64))
        self.register_buffer("A_fine", torch.as_tensor(Af, dtype=torch.complex128))
        self._A_use = None
        self._grid_use = None

    def get_model_params(self):
        """Returns a short string summarizing the model's hyper-parameters.

        Returns:
            str: String of the form "K=<iters>_p=<p>_lam=<lam>".
        """
        return f"K={self.num_iter}_p={self.p}_lam={self.lam}"

    def _spectrum(self, x: torch.Tensor, lam_const: float = None) -> torch.Tensor:
        """Runs M-FOCUSS (Hof "Original" MMV variant) and returns the spectrum.

        Faithful port of the Hof cArray.MFOCUSS_Original recovery: the input is
        RMS-normalized, the solution is initialized with the (min-norm) least-
        squares estimate, the lp-diversity exponent p is annealed from pInit
        toward pMinTh and the regularization lambda follows the same schedule
        (constant 0.99 with the Hof defaults). The per-iteration step is the
        weighted-pseudo-inverse update mu = W (Phi W)^+ Y, identical in form to
        the previous fixed-parameter version.

        Args:
            x (torch.Tensor): Input snapshots, shape [B, N, T].
            lam_const (float): If given, use this constant regularization instead of
                the Hof lambda schedule (set per source-count by forward: high lambda
                for a single source = precision, low lambda for >=2 = resolution).

        Returns:
            torch.Tensor: Normalized angular power spectrum, shape [B, G].
        """
        Y = x.to(torch.complex128)
        rms = torch.sqrt(torch.mean(Y.abs() ** 2, dim=(1, 2), keepdim=True)).clamp_min(self.rms_clamp_eps)
        Y = Y / rms.to(torch.complex128)                        # bring signal to RMS = 1
        A = self._A_use if self._A_use is not None else self.A  # source-adaptive dictionary
        N = A.shape[0]
        eye = torch.eye(N, dtype=torch.complex128, device=Y.device)
        # least-squares (min-norm) initialization: mu0 = A^H (A A^H)^-1 Y
        AAh = A @ A.conj().t()
        mu = torch.einsum("gn,bnt->bgt", A.conj().t(),
                          torch.linalg.solve(AAh + self.ls_init_reg * eye, Y))
        li = self.lam_init if lam_const is None else lam_const
        lm = self.lam_max if lam_const is None else lam_const
        p, lam = self.p_init, li
        for it in range(1, self.num_iter + 1):
            gamma = torch.linalg.norm(mu, dim=2)                # sqrt(sum_t |mu|^2)
            w = (gamma + self.reweight_floor) ** (1.0 - p / 2.0)
            AW = A.unsqueeze(0) * w.unsqueeze(1).to(torch.complex128)
            gram = torch.einsum("bng,bmg->bnm", AW, AW.conj())
            lam_t = torch.tensor(lam, dtype=torch.complex128, device=Y.device)
            q = torch.einsum("bng,bnt->bgt", AW.conj(), torch.linalg.solve(gram + lam_t * eye, Y))
            mu = w.unsqueeze(2).to(torch.complex128) * q
            p = self.p_min + (self.p_init - self.p_min) * np.exp(-self.p_decay * it)
            lam = lm + (li - lm) * np.exp(-self.lam_growth * it)
        spectrum = torch.linalg.norm(mu, dim=2)
        return spectrum / (spectrum.amax(dim=1, keepdim=True) + self.spectrum_norm_eps)

    def _pick(self, spectrum: torch.Tensor, source_number: int) -> torch.Tensor:
        """Greedy multi-peak picker with local parabolic-style soft refinement.

        Args:
            spectrum (torch.Tensor): Normalized power spectrum, shape [B, G].
            source_number (int): Number of peaks (sources) to extract.

        Returns:
            torch.Tensor: Estimated angles in radians, shape [B, source_number].
        """
        B, G = spectrum.shape
        grid = self._grid_use if self._grid_use is not None else self.grid
        half = max(self.pick_min_cols, int(self.pick_half_frac * G))
        supp = max(half, int(self.pick_supp_frac * G))
        grid_idx = torch.arange(G, device=spectrum.device)
        offsets = torch.arange(-half, half + 1, device=spectrum.device)
        work = spectrum.clone()
        centers = []
        for _ in range(source_number):
            idx = work.argmax(dim=1)
            centers.append(idx)
            mask = (grid_idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= supp
            work = work.masked_fill(mask, float("-inf"))
        out = torch.zeros(B, source_number, dtype=torch.float64, device=spectrum.device)
        for j, idx in enumerate(centers):
            win = (idx.unsqueeze(1) + offsets).clamp(0, G - 1)
            vals = torch.gather(spectrum, 1, win)
            ang = grid[win]
            w = torch.softmax(vals / self.pick_temp, dim=1)
            out[:, j] = (w * ang).sum(dim=1)
        return out

    def _estimate_count(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Estimates the source count from significant spectral peaks.

        Args:
            spectrum (torch.Tensor): Normalized power spectrum, shape [B, G].

        Returns:
            torch.Tensor: Estimated source count per sample, shape [B] (>= 1).
        """
        c = spectrum[:, 1:-1]
        peaks = (c > spectrum[:, :-2]) & (c >= spectrum[:, 2:]) & (c > self.count_peak_thr)
        return peaks.sum(dim=1).clamp(min=1).float()

    def forward(self, x: torch.Tensor, sources_num: int = None, phase: str = "test"):
        """Recovers the sparse spectrum and reads out the DoAs.

        Args:
            x (torch.Tensor): Input snapshots, shape [B, N, T].
            sources_num (int): Number of sources to estimate.
            phase (str): Unused; kept for interface compatibility.

        Returns:
            tuple: (doa_prediction [B, M], source_estimation [B], None).
        """
        src = int(sources_num) if sources_num is not None else self.M
        # Source-adaptive regularization: a single source uses the Hof high-lambda schedule
        # (smooth -> precise); >=2 sources use a low constant lambda (sharp -> resolves close
        # sources instead of merging them into one peak and missing the other).
        lam_const = None if src <= 1 else self.lam_multi
        # Source-adaptive grid (mirrors DU-MFOCUSS): fine for M=1, coarse for M>=2.
        single = src <= 1
        self._A_use = self.A_fine if single else self.A
        self._grid_use = self.grid_fine if single else self.grid
        spectrum = self._spectrum(x, lam_const)
        doa = self._pick(spectrum, src)
        if not single and getattr(self, "refine_pairs", True):
            # Coarse-detect -> FINE-refine (mirrors DU-MFOCUSS): re-run the recovery on the fine
            # dictionary and take a local soft-argmax around each coarse peak (accuracy only).
            self._A_use = self.A_fine
            self._grid_use = self.grid_fine
            spec_f = torch.cat([self._spectrum(x[i:i + self.eval_chunk], lam_const)   # chunked: fine-grid pass
                                for i in range(0, x.shape[0], self.eval_chunk)])      # OOMs on 4 GB at B=3000
            # Gather-based window (like _pick) — NOT a -inf-masked softmax over the full grid:
            # the CUDA float64 softmax kernel mis-normalizes when ~99% of a row is -inf.
            grid = self.grid_fine
            Gf = grid.numel()
            step = float(grid[1] - grid[0])
            K = max(self.pick_min_cols, int(round(np.deg2rad(self.refine_win_deg) / step)))
            offsets = torch.arange(-K, K + 1, device=spec_f.device)
            out = doa.clone()
            for j in range(doa.shape[1]):
                idx = torch.argmin((grid[None, :] - doa[:, j:j + 1]).abs(), dim=1)
                win = (idx.unsqueeze(1) + offsets).clamp(0, Gf - 1)
                vals = torch.gather(spec_f, 1, win)
                w = torch.softmax(vals / self.pick_temp, dim=1)
                out[:, j] = (w * grid[win]).sum(dim=1)
            doa = out
        return doa, self._estimate_count(spectrum), None

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

    @torch.no_grad()
    def validation_step(self, batch):
        """Runs one evaluation step: forward pass and RMSPE loss.

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
