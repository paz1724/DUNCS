import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils import *
from src.system_model import SystemModel


class SubspaceMethod(nn.Module):
    def __init__(self, system_model: SystemModel, model_order_estimation:str = None):
        """Initialize the subspace method base module.

        Args:
            system_model (SystemModel): Array geometry and parameters.
            model_order_estimation (str, optional): Source-number estimation
                method ("threshold", "mdl", "aic", "sorte", or None).
        """
        super(SubspaceMethod, self).__init__()
        self.system_model = system_model
        self.eigen_threshold = nn.Parameter(torch.tensor(0.18), requires_grad=True)
        self.normalized_eigenvals = None
        self.model_order_estimation = model_order_estimation
        self.eigen_values_avg = {}
        self._num_sources=0
        self.avg_len = {}

    def subspace_separation(self,
                            covariance: torch.Tensor,
                            number_of_sources: torch.tensor = None) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.tensor):
        """Separate the covariance into signal and noise subspaces via eigendecomposition.

        Args:
            covariance (torch.Tensor): Covariance matrix, shape [B, N, N].
            number_of_sources (int, optional): Number of sources; if None it
                is estimated from the eigenvalues.

        Returns:
            tuple: (signal_subspace [B, N, M], noise_subspace [B, N, N-M],
                source_estimation, l_eig regularization term).
        """
        # ---- 1. Eigendecompose ----
        # The whole subspace family rests on one fact: for M sources in white noise the covariance
        # has M large eigenvalues (signal) and N-M small equal ones (noise), and their eigenvectors
        # span orthogonal subspaces. Diagonal loading keeps the decomposition conditioned at low SNR.
        covariance = diag_loading(covariance, training=self.training)  # For training stability in low SNR
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        # ---- 2. Sort eigenvectors by descending eigenvalue ----
        # eigh returns ASCENDING order, so the split below would take the wrong half without this.
        sorted_idx = torch.argsort(torch.real(eigenvalues), descending=True)
        sorted_eigvectors = torch.gather(eigenvectors, 2,
                                         sorted_idx.unsqueeze(-1).expand(-1, -1, covariance.shape[-1]).transpose(1, 2))
        # ---- 3. Decide where to split ----
        # Either the caller supplies M, or it is inferred from the eigenvalue profile.
        # number of sources estimation
        self._num_sources = number_of_sources
        source_estimation, l_eig = self.estimate_number_of_sources(eigenvalues,
                                                                   number_of_sources=number_of_sources)
        # ---- 4. Split into signal (top M) and noise (remaining N-M) subspaces ----
        if number_of_sources is None:
            warnings.warn("Number of sources is not defined, using the number of sources estimation.")
        # if source_estimation == sorted_eigvectors.shape[2]:
        #     source_estimation -= 1
            signal_subspace = sorted_eigvectors[:, :, :source_estimation]
            noise_subspace = sorted_eigvectors[:, :, source_estimation:]
        else:
            signal_subspace = sorted_eigvectors[:, :, :number_of_sources]
            noise_subspace = sorted_eigvectors[:, :, number_of_sources:]

        return signal_subspace.to(device), noise_subspace.to(device), source_estimation, l_eig

    def estimate_number_of_sources(self, eigenvalues, number_of_sources: int = None):
        """Estimate the number of sources from the eigenvalue spectrum.

        Args:
            eigenvalues (torch.Tensor): Eigenvalues, shape [B, N].
            number_of_sources (int, optional): Ground-truth source count, used
                for the training regularization term.

        Returns:
            tuple: (source_estimation, l_eig) where l_eig is a training-time
                regularization/loss term (None outside training or for None method).
        """
        # ---- 1. Sort the eigenvalue profile ----
        # Every criterion below reads the SHAPE of this descending profile: M sources produce a
        # visible drop between the M-th and (M+1)-th eigenvalue, and they differ only in how they
        # decide where that drop is.
        batch_size = eigenvalues.shape[0]
        sorted_eigenvals = torch.sort(torch.real(eigenvalues), descending=True, dim=1).values
        self.normalized_eigenvals = sorted_eigenvals
        l_eig = None

        # ---- 2a. No estimator requested ----
        if self.model_order_estimation is None:
            return None, None

        # ---- 2b. THRESHOLD: count eigenvalues above a fraction of the largest ----
        # Simplest rule and the only differentiable one, so it can also produce a training
        # regularization term that pushes the eigen-profile toward the true source count.
        elif self.model_order_estimation.lower().startswith("threshold"):
            self.normalized_eigenvals = sorted_eigenvals / sorted_eigenvals[:, 0][:, None]
            source_estimation = torch.linalg.norm(
                nn.functional.relu(
                    self.normalized_eigenvals - self.__get_eigen_threshold() * torch.ones_like(
                        self.normalized_eigenvals)),
                dim=1, ord=0).to(torch.int)
            # return regularization term if training
            if self.training:
                l_eig = self.eigen_regularization(number_of_sources)
        # ---- 2c. MDL / AIC / SORTE: score every hypothesis M = 1..N-1, keep the best ----
        # Information-criterion style: each M is scored, and the minimizing M wins. Vectorized over
        # the batch by masked running-minimum rather than a per-sample argmin.
        elif self.model_order_estimation.lower() in ["mdl", "aic", "sorte"]:
            # mdl -> calculate the value of the mdl test for each number of sources
            # and choose the number of sources that minimizes the mdl test
            optimal_test = torch.ones(eigenvalues.shape[0], device=device) * float("inf")
            optimal_m = torch.zeros(eigenvalues.shape[0], device=device)
            hypothesis_results = []
            for m in range(1, eigenvalues.shape[1]):
                m = torch.tensor(m, device=device)
                # calculate the test
                test = self.hypothesis_testing(sorted_eigenvals, m)
                hypothesis_results.append(test)
                # update the optimal number of sources by masking the current number of sources
                optimal_m = torch.where(test < optimal_test, m, optimal_m)
                # update the optimal mdl value
                optimal_test = torch.where(test < optimal_test, test, optimal_test)
                # if self.training and m == number_of_sources:
                #     # l_eig = torch.sum(test)
                #     l_eig = test
            # ---- 3. Training signal ----
            # Treat the per-hypothesis scores as logits (negated, since lower = better) and apply
            # cross-entropy against the true count, so the model learns to shape its eigen-profile
            # into one whose model-order test picks the right M.
            if self.training:
                hypothesis_results = torch.stack(hypothesis_results, dim=1)  # (B, N-3)
                logits = -hypothesis_results
                labels = torch.full((batch_size,), number_of_sources - 1, dtype=torch.long, device=logits.device)
                l_eig = torch.nn.functional.cross_entropy(logits, labels)

            source_estimation = optimal_m

        else:
            raise ValueError(
                f"SubspaceMethod.estimate_number_of_sources: method {self.model_order_estimation.lower()} is not recognized.")
        return source_estimation, l_eig

    def hypothesis_testing(self, eigenvalues, number_of_sources):
        """Compute the model-order test statistic (MDL/AIC/SORTE) for a hypothesized source count.

        Args:
            eigenvalues (torch.Tensor): Sorted (descending) eigenvalues, shape [B, N].
            number_of_sources (int): Hypothesized number of sources M.

        Returns:
            torch.Tensor: Test value per batch element, shape [B]; lower is better.
        """
        moe = self.model_order_estimation.lower()
        M = number_of_sources
        if self.system_model.is_sparse_array:
            N_eff = self.system_model.virtual_array_ula_seg.shape[0]
        else:
            N_eff = self.system_model.params.N

        # ---- MDL / AIC: log-likelihood + a complexity penalty ----
        # Both trade fit against model size; they differ ONLY in the penalty weight, log(T) vs 2,
        # which is why MDL is the more conservative of the two (it charges more per source).
        if moe in ["mdl", "aic"]:
            # extract the number of snapshots and the number of antennas
            T = self.system_model.params.T
            # calculate the number of degrees of freedom
            dof = (2 * N_eff * M - M ** 2 + 1) / 2
            penalty = dof * (np.log(T) if moe == "mdl" else 2)
            ll = self.get_ll(eigenvalues, M)
            return ll + penalty

        # ---- SORTE: variance ratio of successive eigenvalue GAPS ----
        # Threshold-free and scale-free. Inside the noise floor the gaps are small and similar, so
        # their variance is low; the gap at the true M stands out. The ratio is minimized at the
        # correct source count. Needs at least two gaps on each side, hence the guard.
        elif moe == "sorte":
            if M >= N_eff - 2:
                return torch.full((eigenvalues.shape[0],), float("inf"), device=eigenvalues.device)

            gaps = eigenvalues[:, :-1] - eigenvalues[:, 1:]  # Δ_i, shape (batch, N‑1)
            # Denominator uses gaps_i for i = K…N‑2 ⇒ slice from K‑1
            den_gaps = gaps[:, M - 1:]
            # Numerator uses gaps_i for i = K+1… ⇒ slice from K
            num_gaps = gaps[:, M:]

            den_var = torch.var(den_gaps, dim=1, unbiased=False)
            num_var = torch.var(num_gaps, dim=1, unbiased=False)

            ratio = torch.where(den_var == 0,
                                torch.full_like(den_var, float("inf")),
                                num_var / den_var)
            return ratio

    def snr_estimation(self, eigenvalues, M):
        """Estimate SNR (dB) as the ratio of mean signal to mean noise eigenvalues.

        Args:
            eigenvalues (torch.Tensor): Sorted (descending) eigenvalues, shape [B, N].
            M (int): Number of sources (signal subspace dimension).

        Returns:
            torch.Tensor: Estimated SNR in dB per batch element, shape [B].
        """
        snr = 10 * torch.log10(torch.mean(eigenvalues[:, :M], dim=1) / torch.mean(eigenvalues[:, M:], dim=1))
        return snr

    def get_ll(self, eigenvalues, M):
        """Compute the log-likelihood term used in the MDL/AIC criteria.

        Args:
            eigenvalues (torch.Tensor): Sorted (descending) eigenvalues, shape [B, N].
            M (int): Hypothesized number of sources.

        Returns:
            torch.Tensor: Log-likelihood term per batch element, shape [B].
        """
        T = self.system_model.params.T

        if self.system_model.is_sparse_array:
            N = self.system_model.virtual_array_ula_seg.shape[0]
        else:
            N = self.system_model.params.N

        ll = -T * torch.sum(torch.log(eigenvalues[:, M:]), dim=1) + T * (N - M) * torch.log(
            torch.mean(eigenvalues[:, M:], dim=1))
        return ll

    def get_noise_subspace(self, covariance: torch.Tensor, number_of_sources: int):
        """Return the noise subspace of the covariance matrix.

        Args:
            covariance (torch.Tensor): Covariance matrix, shape [B, N, N].
            number_of_sources (int): Number of sources M.

        Returns:
            torch.Tensor: Noise subspace, shape [B, N, N-M].
        """
        _, noise_subspace, _, _ = self.subspace_separation(covariance, number_of_sources)
        return noise_subspace

    def get_signal_subspace(self, covariance: torch.Tensor, number_of_sources: int):
        """Return the signal subspace of the covariance matrix.

        Args:
            covariance (torch.Tensor): Covariance matrix, shape [B, N, N].
            number_of_sources (int): Number of sources M.

        Returns:
            torch.Tensor: Signal subspace, shape [B, N, M].
        """
        signal_subspace, _, _, _ = self.subspace_separation(covariance, number_of_sources)
        return signal_subspace

    def eigen_regularization(self, number_of_sources: int):
        """Regularization term encouraging the eigen-threshold to separate signal/noise eigenvalues.

        Args:
            number_of_sources (int): Ground-truth number of sources M.

        Returns:
            torch.Tensor: Per-batch regularization term, shape [B].
        """
        l_eig = (self.normalized_eigenvals[:, number_of_sources - 1] - self.__get_eigen_threshold()) * \
                (self.normalized_eigenvals[:, number_of_sources] - self.__get_eigen_threshold())
        # l_eig = torch.sum(l_eig)
        return l_eig

    def __get_eigen_threshold(self):
        """Return the learnable eigenvalue threshold parameter.

        Returns:
            torch.Tensor: Scalar eigen-threshold parameter.
        """
        return self.eigen_threshold

    @staticmethod
    def __spatial_smoothing_coarray_cov(R_coarray: torch.Tensor, sub_array_size: int = None) -> torch.Tensor:
        """
        Perform forward–backward spatial smoothing on the coarray covariance matrix R_coarray.

        Parameters
        ----------
        R_coarray : torch.Tensor
            The coarray covariance, shape = [batch_size, L, L].
            - L is the size of the virtual ULA in the coarray domain.
        sub_array_size : int, optional
            The length of each sub-subarray in the coarray domain.
            If None, it will default to L//2 + 1 (typical choice).

        Returns
        -------
        R_smoothed : torch.Tensor
            The smoothed covariance, shape = [batch_size, sub_array_size, sub_array_size].
            This can then be used in MUSIC/ESPRIT for coherent sources.
        """
        # R_coarray has shape [batch_size, L, L]
        batch_size, L, _ = R_coarray.shape

        # Default subarray size: L//2 + 1 if not provided
        if sub_array_size is None:
            sub_array_size = L // 2 + 1

        # Number of forward subarrays
        number_of_sub_arrays = L - sub_array_size + 1
        if number_of_sub_arrays <= 0:
            raise ValueError("sub_array_size is too large for the given L.")

        # Initialize the smoothed covariance accumulator
        R_smoothed = torch.zeros(
            (batch_size, sub_array_size, sub_array_size),
            dtype=R_coarray.dtype, device=R_coarray.device
        )

        for start_idx in range(number_of_sub_arrays):
            # Extract the forward subarray covariance block
            sub_cov = R_coarray[:, start_idx:start_idx + sub_array_size,
                      start_idx:start_idx + sub_array_size]
            # Compute the backward covariance block by taking the conjugate and flipping along both dimensions
            sub_cov_back = torch.flip(torch.conj(sub_cov), dims=[1, 2])
            # Average the forward and backward covariance blocks
            R_smoothed += 0.5 * (sub_cov + sub_cov_back)

        # Final averaging over the number of subarrays
        R_smoothed /= number_of_sub_arrays

        return R_smoothed

    def save_eigen_values(self):
        """Accumulate normalized eigenvalues per source count for later averaging/plotting."""
        n_sources = self._num_sources.item()
        self.avg_len[n_sources] = self.avg_len.get(n_sources, 0) + 1
        self.eigen_values_avg[n_sources] = self.eigen_values_avg.get(n_sources, torch.zeros_like(self.normalized_eigenvals)) + self.normalized_eigenvals

    def plot_eigen_spectrum(self, batch_idx: int=0):
        """
        Plot the eigenvalues spectrum.

        Args:
        -----
            batch_idx (int): Index of the batch to plot.
        """
        for num_sources in self.eigen_values_avg.keys():
            self.eigen_values_avg[num_sources] /= self.avg_len[num_sources]

            plt.figure()
            plt.stem(self.eigen_values_avg[num_sources].squeeze().cpu().detach().numpy(), label="Normalized Eigenvalues")
            # ADD threshold line
            plt.axhline(y=self.eigen_threshold.detach().numpy(), color='r', linestyle='--', label="Threshold")
            plt.title("Eigenvalues Spectrum - M={}".format(num_sources))
            plt.xlabel("Eigenvalue Index")
            plt.ylabel("Eigenvalue")
            plt.legend()
            plt.grid()
            plt.show()
