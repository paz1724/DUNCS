from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.methods_pack.subspace_method import SubspaceMethod
from src.system_model import SystemModel


class ESPRIT(SubspaceMethod):
    def __init__(self, system_model: SystemModel, model_order_estimation:str = 'sorte'):
        """Initialize the ESPRIT subspace estimator.

        Args:
            system_model (SystemModel): Array geometry and parameters.
            model_order_estimation (str): Source-number estimation method (default 'sorte').
        """
        super().__init__(system_model, model_order_estimation)
        # DoA readout: theta = doa_sign * arcsin(phase / (2*pi*d_lambda)). d_lambda is the array's
        # element spacing in wavelengths; the arcsin uses it so a NON-ideal spacing reads the right angle.
        # Default d_lambda=0.5 (ideal lambda/2 ULA) reproduces the classic arcsin(phase/pi).
        # For the recorded ULA3, read the RAW covariance at its NATIVE d_lambda (~0.234) with NO
        # recorded->ideal calibration (the calibration scatters the non-Vandermonde manifold -> ~48% MD;
        # native-d/lambda raw reading -> ~0% MD, same fix as Root-MUSIC).
        self.doa_sign = -1.0
        self.d_lambda = 0.5

    def forward(self, cov: torch.Tensor, number_of_sources: torch.tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Estimate DoAs from a covariance matrix using ESPRIT.

        Args:
            cov (torch.Tensor): Covariance matrix, shape [B, N, N].
            number_of_sources (int): Number of sources M (may be None to estimate).

        Returns:
            tuple: (prediction angles [B, M], sources_estimation, regularization term).
        """
        # get the signal subspace
        signal_subspace, _, sources_estimation, regularization = self.subspace_separation(
            cov,
            number_of_sources=number_of_sources
        )
        # create 2 overlapping matrices
        upper = signal_subspace[:, :-1]
        lower = signal_subspace[:, 1:]
        phi = torch.linalg.lstsq(upper, lower)[0]  # identical to pinv(A) @ B but faster and stable.
        eigvalues = torch.linalg.eigvals(phi)
        eigvals_phase = torch.angle(eigvalues)
        prediction = self.doa_sign * torch.arcsin(
            torch.clamp(eigvals_phase / (2 * torch.pi * self.d_lambda), -1.0, 1.0))

        return prediction, sources_estimation, regularization

    def __str__(self):
        """Return the method name string.

        Returns:
            str: "esprit".
        """
        return "esprit"


def esprit(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Esprit algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent to src.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
    """

    doa_batches = []

    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)

        # Get signal subspace
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M]
        # Separate the signal subspace into 2 overlapping subspaces
        Us_upper, Us_lower = (
            Us[0: R.shape[0] - 1],
            Us[1: R.shape[0]],
        )
        # Generate Phi matrix
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        # Find eigenvalues and eigenvectors (EVD) of Phi
        phi_eigenvalues, _ = torch.linalg.eig(phi)
        # Calculate the phase component of the roots
        eigenvalues_angels = torch.angle(phi_eigenvalues)
        # Calculate the DoA out of the phase component
        doa_predictions = self.doa_sign * torch.arcsin(
            torch.clamp(eigenvalues_angels / (2 * torch.pi * self.d_lambda), -1.0, 1.0))
        doa_batches.append(doa_predictions)

    return torch.stack(doa_batches, dim=0)
