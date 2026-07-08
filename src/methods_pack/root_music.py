import torch
import torch.nn as nn

from src.methods_pack.subspace_method import SubspaceMethod
from src.system_model import SystemModel
from src.utils import *


class RootMusic(SubspaceMethod):
    def __init__(self, system_model: SystemModel):
        """Initialize the Root-MUSIC subspace estimator.

        Args:
            system_model (SystemModel): Array geometry and parameters.
        """
        super(RootMusic, self).__init__(system_model)

    def forward(self, cov: torch.Tensor, sources_num: torch.tensor = None):
        """Estimate DoAs from a covariance matrix using Root-MUSIC.

        Args:
            cov (torch.Tensor): Covariance matrix, shape [B, N, N].
            sources_num (int, optional): Number of sources M; defaults to the
                system model's M if None.

        Returns:
            tuple: (angles_prediction [B, M], angles_prediction_all for all roots,
                roots [B, 2N-2]).
        """
        if sources_num is None:
            M = self.system_model.params.M
        else:
            M = sources_num
        # over_model: separate the subspace assuming M+over_model sources so the (coherent) multipath
        # reflections occupy the signal subspace instead of leaking into the noise subspace and corrupting
        # the roots. The strongest root (direct path) is then recovered by the power-pick selection below.
        M_sub = min(int(M) + getattr(self, "over_model", 0), cov.shape[-1] - 1)
        _, noise_subspace, _, self.eigen_regularization = self.subspace_separation(cov, number_of_sources=M_sub)
        poly_generator = torch.einsum("bnk, bkj -> bnj", noise_subspace, noise_subspace.conj().transpose(1, 2))
        diag_sum = self.sum_of_diag(poly_generator)
        roots = self.find_roots(diag_sum)
        angles_prediction_all = self.get_doa_from_roots(roots)

        if getattr(self, "power_pick", False):
            # Multipath-robust selection: among inside-circle roots, pick the M with the highest
            # power a^H R a (the direct path = strongest), instead of the M closest to the unit circle
            # (which can be a reflection root). Mirrors how MUSIC/ESPRIT lock onto the dominant source.
            angles_prediction = self._power_select(roots, cov, M)
        else:
            # the actual prediction is for roots that are closest to the unit circle.
            closest_roots = self.extract_roots_closest_unit_circle(roots, k=M)
            angles_prediction = self.get_doa_from_roots(closest_roots)
        return angles_prediction, angles_prediction_all, roots

    def _power_select(self, roots, cov, k: int):
        """Select roots by steered power a^H cov a (inside the unit circle).

        soft_power (k=1): DIFFERENTIABLE power-weighted average of the root angles -> the CNN can be
        trained end-to-end to put the direct path's root at high power. Else: hard top-k by power.
        """
        angles = self.get_doa_from_roots(roots)                              # [B, R] radians
        rec_sv = getattr(self, "recorded_sv", None)
        if rec_sv is not None:
            # power on the RECORDED manifold a_rec(theta)^H cov a_rec(theta) (no calibration to an ideal ULA)
            grid = self.recorded_grid.to(angles.device); sv = rec_sv.to(cov.dtype).to(angles.device)   # [G],[N,G]
            idx = torch.argmin((grid.view(1, 1, -1) - angles.unsqueeze(-1)).abs(), dim=-1)              # [B,R] nearest
            a = sv[:, idx].permute(1, 2, 0)                                  # [B, R, N]
        else:
            nn = torch.arange(cov.shape[-1], device=cov.device, dtype=torch.float64)
            a = torch.exp(-1j * np.pi * nn.view(1, 1, -1) * torch.sin(angles).unsqueeze(-1).to(torch.complex128))  # [B,R,N]
        power = torch.einsum("brn,bnm,brm->br", a.conj(), cov.to(a.dtype), a).real      # [B, R]
        inside = torch.abs(roots) <= 1.0
        if getattr(self, "soft_power", False) and k == 1:
            p = power.masked_fill(~inside, 0.0)
            p = p / (p.amax(dim=1, keepdim=True) + 1e-9)                     # normalize to [0,1] per sample
            w = torch.softmax(p / getattr(self, "_pp_temp", 0.15), dim=1)    # sharp soft-argmax over roots
            w = w * inside.to(w.dtype); w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
            return (angles * w).sum(dim=1, keepdim=True)                     # [B, 1] differentiable
        power = power.masked_fill(~inside, float("-inf"))
        idx = torch.topk(power, k, dim=1).indices
        return self.get_doa_from_roots(roots.gather(1, idx))

    def get_doa_from_roots(self, roots):
        """Convert polynomial roots to DoA angles (radians).

        Args:
            roots (torch.Tensor): Complex polynomial roots, shape [B, K].

        Returns:
            torch.Tensor: Predicted angles in radians, shape [B, K].
        """
        roots_phase = torch.angle(roots)
        # doa_sign matches ESPRIT's convention: -1 for the recorded ULA3 / a_ideal=exp(-j*pi*n*sin theta)
        # manifold (after calibration), +1 for the analytic (+2j) steering. Default 1 (legacy behaviour).
        sign = getattr(self, "doa_sign", 1.0)
        # d_lambda override: the recorded ULA3's MEASURED effective element spacing (d/lambda) at the
        # operating freq is ~0.389, not the nominal 0.5 -- use it so roots map to the right angle.
        d = getattr(self, "d_lambda", None) or self.system_model.dist_array_elems["NarrowBand"]
        # clamp to [-1,1]: for d/lambda < 0.5 the factor 1/(2*pi*d) > 1/pi, so out-of-visible-region roots
        # (|phase| > 2*pi*d) would push arcsin out of domain -> NaN (crashes the Hungarian RMSPE loss).
        arg = ((1 / (2 * np.pi * d)) * roots_phase).clamp(-1.0, 1.0)
        angle_predicted = sign * torch.arcsin(arg)
        return angle_predicted

    def extract_roots_closest_unit_circle(self, roots, k: int):
        """Select the k INSIDE-the-unit-circle roots closest to it (one per source).

        Polynomial roots come in reciprocal pairs (z inside, 1/z* outside, SAME angle). Picking the k
        globally-closest-to-|z|=1 roots can take both members of one source's pair and miss another source
        entirely (=> ~50% MD for k>=2). Restricting to |z|<=1 yields one root per source.

        Args:
            roots (torch.Tensor): Complex roots, shape [B, R].
            k (int): Number of roots to keep.

        Returns:
            torch.Tensor: The k closest inside-circle roots, shape [B, k].
        """
        distances = torch.abs(torch.abs(roots) - 1)
        distances = distances.masked_fill(torch.abs(roots) > 1.0, float("inf"))   # keep only inside/on the circle
        sorted_indcies = torch.argsort(distances, dim=1)
        k_closest = sorted_indcies[:, :k]
        closest_roots = roots.gather(1, k_closest)
        return closest_roots

    def sum_of_diag(self, tensor: torch.Tensor):
        """Sum the elements along each diagonal of a batch of square matrices.

        Args:
            tensor (torch.Tensor): Square matrices, shape [B, N, N] (or [N, N]).

        Returns:
            torch.Tensor: Diagonal sums (polynomial coefficients), shape [B, 2N-1].
        """
        tensor = self.__check_diag_sums_dim(tensor)
        N = tensor.shape[-1]
        diag_indcies = torch.linspace(-N + 1, N - 1, 2 * N - 1, dtype=torch.int)
        sum_of_diags = torch.zeros(tensor.shape[0], 2 * N - 1, dtype=tensor.dtype, device=tensor.device)
        for idx, diag_idx in enumerate(diag_indcies):
            sum_of_diags[:, idx] = torch.sum(torch.diagonal(tensor, dim1=-2, dim2=-1, offset=diag_idx), dim=-1)
        return sum_of_diags

    def find_roots(self, coeffs: torch.Tensor):
        """Find polynomial roots via the companion-matrix eigenvalues.

        Args:
            coeffs (torch.Tensor): Polynomial coefficients, shape [B, D].

        Returns:
            torch.Tensor: Complex roots, shape [B, D-1].
        """
        A = torch.diag(torch.ones(coeffs.shape[-1] - 2, dtype=coeffs.dtype, device=coeffs.device), -1)  # sub-diagonal
        A = A.repeat(coeffs.shape[0], 1, 1)  # repeat for all elements in the batch
        A[:, 0, :] = -coeffs[:, 1:] / coeffs[:, 0].unsqueeze(1)
        roots = torch.linalg.eigvals(A)
        return roots

    def __check_diag_sums_dim(self, tensor):
        """Validate/normalize the input tensor to a batched square-matrix shape.

        Args:
            tensor (torch.Tensor): Input of shape [N, N] or [B, N, N].

        Returns:
            torch.Tensor: Tensor of shape [B, N, N].
        """
        if len(tensor.shape) != 3:
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError("sum_of_diag: Input tensor shape should be 2 or 3 dim")
        else:
            if tensor.shape[-1] != tensor.shape[-2]:
                raise ValueError("sum_of_diag: input tensor should be square matrices as a batch.")
        return tensor


def root_music(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent tosrc.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """

    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        # Assign noise subspace as the eigenvectors associated with M greatest eigenvalues
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        # Generate hermitian noise subspace matrix
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        # Calculates the sum of F matrix diagonals
        diag_sum = sum_of_diags_torch(F)
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = find_roots_torch(diag_sum)
        # Calculate the phase component of the roots
        roots_angels_all = torch.angle(roots)
        # Calculate doa
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        # Take only roots which inside the unit circle
        roots = roots[
            sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))
        ]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        # Calculate the phase component of the roots
        roots_angels = torch.angle(roots)
        # Calculate doa
        doa_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)

    return (
        torch.stack(doa_batches, dim=0),
        torch.stack(doa_all_batches, dim=0),
        roots_to_return,
    )
