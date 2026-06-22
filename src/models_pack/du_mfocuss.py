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
                 grid_size: int = 121, criterion: str = "rmspe"):
        """Initializes the DU-MFOCUSS model.

        Args:
            system_model (SystemModel): Array geometry and signal parameters.
            num_iterations (int): Number of unrolled M-FOCUSS layers (K).
            grid_size (int): Number of angle-grid columns in the steering dictionary.
            criterion (str): Training criterion name (e.g. "rmspe").
        """
        super().__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.num_iter = num_iterations
        self.grid_size = grid_size
        self.M = system_model.params.M

        # Overcomplete steering dictionary on the (recorded or analytic) manifold.
        lo, hi = system_model.params.doa_range
        grid_rad = np.deg2rad(np.linspace(lo, hi, grid_size))
        A = np.asarray(system_model.steering_vec(grid_rad))      # (N, G)
        self.register_buffer("grid", torch.as_tensor(grid_rad, dtype=torch.float64))
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.complex128))

        # Learned per-layer parameters (reparameterized to valid ranges):
        #   lambda_k = softplus(raw_lambda_k) + eps  > 0   (regularization)
        #   p_k      = 2 * sigmoid(raw_p_k) in (0, 2)       (lp diversity)
        self._raw_lambda = nn.Parameter(torch.full((num_iterations,), -2.0))
        self._raw_p = nn.Parameter(torch.zeros(num_iterations))
        # Readout temperature for the differentiable local soft-argmax peak picker.
        self._raw_temp = nn.Parameter(torch.tensor(-2.0))

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
        A = self.A                                                # [N, G]
        B = Y.shape[0]
        eye = torch.eye(A.shape[0], dtype=torch.complex128, device=Y.device)

        # Matched-filter initialization: s0 = A^H Y.
        s = torch.einsum("gn,bnt->bgt", A.conj().t(), Y)          # [B, G, T]

        for k in range(self.num_iter):
            p_k = 2.0 * torch.sigmoid(self._raw_p[k])
            lam_k = F.softplus(self._raw_lambda[k]) + 1e-5

            row_norm = torch.linalg.norm(s, dim=2)                # [B, G] real
            w = (row_norm + 1e-9) ** (1.0 - p_k / 2.0)            # FOCUSS reweighting

            AW = A.unsqueeze(0) * w.unsqueeze(1).to(torch.complex128)   # [B, N, G]
            gram = torch.einsum("bng,bmg->bnm", AW, AW.conj())    # [B, N, N]
            reg = gram + lam_k.to(torch.complex128) * eye         # [B, N, N]
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
            angles_win = self.grid[win]                                        # [B, W]
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
        spectrum = self.get_learned_covariance(x)
        doa_prediction = self._soft_argmax(spectrum, src)
        source_estimation = self._estimate_count(spectrum)
        return doa_prediction, source_estimation, None

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
