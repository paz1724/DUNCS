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
                 criterion: str = "rmspe"):
        """Initializes the classical MFOCUSS baseline.

        Args:
            system_model (SystemModel): Array geometry and signal parameters.
            num_iterations (int): Number of M-FOCUSS iterations (fixed, not unrolled).
            grid_size (int): Number of angle-grid columns in the steering dictionary.
            p (float): Fixed lp-diversity exponent in (0, 2] (smaller = sparser).
            lam (float): Fixed regularization weight (> 0).
            criterion (str): Evaluation criterion name (e.g. "rmspe").
        """
        super().__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.num_iter = num_iterations
        self.grid_size = grid_size
        self.p = float(p)
        self.lam = float(lam)
        # Hof MFOCUSS_Original annealing schedule (cArray defaults): p anneals from
        # p_init toward p_min (rate p_decay); lambda stays at 0.99 (lam_init==lam_max).
        self.p_init, self.p_min, self.p_decay = 0.99, 0.01, 0.2
        self.lam_init, self.lam_max, self.lam_growth = 0.99, 0.99, 0.2
        self.lam_multi = 0.05      # low regularization for >=2 sources (resolution over smoothness)
        self.M = system_model.params.M

        lo, hi = system_model.params.doa_range
        grid_rad = np.deg2rad(np.linspace(lo, hi, grid_size))
        A = np.asarray(system_model.steering_vec(grid_rad))      # (N, G)
        self.register_buffer("grid", torch.as_tensor(grid_rad, dtype=torch.float64))
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.complex128))

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
        rms = torch.sqrt(torch.mean(Y.abs() ** 2, dim=(1, 2), keepdim=True)).clamp_min(1e-12)
        Y = Y / rms.to(torch.complex128)                        # bring signal to RMS = 1
        A = self.A
        N = A.shape[0]
        eye = torch.eye(N, dtype=torch.complex128, device=Y.device)
        # least-squares (min-norm) initialization: mu0 = A^H (A A^H)^-1 Y
        AAh = A @ A.conj().t()
        mu = torch.einsum("gn,bnt->bgt", A.conj().t(),
                          torch.linalg.solve(AAh + 1e-6 * eye, Y))
        li = self.lam_init if lam_const is None else lam_const
        lm = self.lam_max if lam_const is None else lam_const
        p, lam = self.p_init, li
        for it in range(1, self.num_iter + 1):
            gamma = torch.linalg.norm(mu, dim=2)                # sqrt(sum_t |mu|^2)
            w = (gamma + 1e-12) ** (1.0 - p / 2.0)
            AW = A.unsqueeze(0) * w.unsqueeze(1).to(torch.complex128)
            gram = torch.einsum("bng,bmg->bnm", AW, AW.conj())
            lam_t = torch.tensor(lam, dtype=torch.complex128, device=Y.device)
            q = torch.einsum("bng,bnt->bgt", AW.conj(), torch.linalg.solve(gram + lam_t * eye, Y))
            mu = w.unsqueeze(2).to(torch.complex128) * q
            p = self.p_min + (self.p_init - self.p_min) * np.exp(-self.p_decay * it)
            lam = lm + (li - lm) * np.exp(-self.lam_growth * it)
        spectrum = torch.linalg.norm(mu, dim=2)
        return spectrum / (spectrum.amax(dim=1, keepdim=True) + 1e-9)

    def _pick(self, spectrum: torch.Tensor, source_number: int) -> torch.Tensor:
        """Greedy multi-peak picker with local parabolic-style soft refinement.

        Args:
            spectrum (torch.Tensor): Normalized power spectrum, shape [B, G].
            source_number (int): Number of peaks (sources) to extract.

        Returns:
            torch.Tensor: Estimated angles in radians, shape [B, source_number].
        """
        B, G = spectrum.shape
        half = max(1, int(0.025 * G))
        supp = max(half, int(0.03 * G))
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
            ang = self.grid[win]
            w = torch.softmax(vals / 0.05, dim=1)
            out[:, j] = (w * ang).sum(dim=1)
        return out

    @staticmethod
    def _estimate_count(spectrum: torch.Tensor) -> torch.Tensor:
        """Estimates the source count from significant spectral peaks.

        Args:
            spectrum (torch.Tensor): Normalized power spectrum, shape [B, G].

        Returns:
            torch.Tensor: Estimated source count per sample, shape [B] (>= 1).
        """
        c = spectrum[:, 1:-1]
        peaks = (c > spectrum[:, :-2]) & (c >= spectrum[:, 2:]) & (c > 0.5)
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
        spectrum = self._spectrum(x, lam_const)
        return self._pick(spectrum, src), self._estimate_count(spectrum), None

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
