from abc import ABC, abstractmethod
import torch
import cvxpy as cp
import numpy as np

from src.utils import build_phi, psd_proj, toeplitz_proj, hermitian_proj, svt, device
from src.system_model import SystemModel


class CovReconstructor(ABC):
    """
    Given observation tensor x from S sensors (B×S×T),
    returns batch of completed coarray covariances (B×U×U).
    """
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a (co)array covariance from observations.

        Args:
            x (torch.Tensor): Observations, shape [B, S, T].

        Returns:
            torch.Tensor: Completed covariance, shape [B, U, U].
        """
        pass


class SampleCov(CovReconstructor):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sample covariance of the observations.

        Args:
            x (torch.Tensor): Observations, shape [B, S, T].

        Returns:
            torch.Tensor: Sample covariance, shape [B, S, S].
        """
        return sample_covariance(x).to(device=device)


class SpatialSmoothingReconstructor(CovReconstructor):
    """
    Calculates the covariance matrix using forward–backward spatial smoothing technique.
    """
    def __init__(self, sub_array_size=None):
        """Initialize the spatial-smoothing reconstructor.

        Args:
            sub_array_size (int, optional): Subarray length; defaults to
                sensor_number // 2 + 1 when None.
        """
        self.sub_array_size = sub_array_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward-backward spatially-smoothed covariance.

        Args:
            x (torch.Tensor): Observations, shape [B, S, T] (or [S, T]).

        Returns:
            torch.Tensor: Smoothed covariance, shape [B, sub_array_size, sub_array_size].
        """
        # Ensure x has three dimensions (batch, sensors, samples)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, sensor_number, samples_number = x.shape

        # Define subarray size and the number of overlapping subarrays
        if self.sub_array_size is None:
            self.sub_array_size = sensor_number // 2 + 1

        number_of_sub_arrays = sensor_number - self.sub_array_size + 1

        # Initialize the smoothed covariance matrix
        Rx_smoothed = torch.zeros(batch_size, self.sub_array_size, self.sub_array_size,
                                  dtype=torch.complex128, device=x.device)

        for j in range(number_of_sub_arrays):
            # Extract the j-th subarray
            x_sub = x[:, j:j + self.sub_array_size, :]

            # Forward covariance calculation
            cov_forward = torch.einsum("bmt, btl -> bml", x_sub,
                                       torch.conj(x_sub).transpose(1, 2)) / (samples_number - 1)

            # backward processing: take the complex conjugate before flipping
            x_sub_back = torch.flip(torch.conj(x_sub), dims=[1])
            cov_backward = torch.einsum("bmt, btl -> bml", x_sub_back,
                                        torch.conj(x_sub_back).transpose(1, 2)) / (samples_number - 1)

            # Average the forward and backward covariances for this subarray
            cov_fb = 0.5 * (cov_forward + cov_backward)

            # Aggregate over all subarrays
            Rx_smoothed += cov_fb / number_of_sub_arrays

        return Rx_smoothed


class AveragingReconstructor(CovReconstructor):
    """
    Calculates the virtual array covariance matrix, based on the paper: "Remarks on the Spatial Smoothing Step in
    Coarray MUSIC"

     Parameters
     ----------
      X (torch.Tensor): Input samples matrix.
      system_model (SystemModel): settings of the system model

    Returns
    -------
    Rx (torch.Tensor): virtual array's covariance matrix
    """
    def __init__(self, sys_model: SystemModel):
        """Initialize the coarray-averaging reconstructor.

        Args:
            sys_model (SystemModel): System model providing the sparse array
                geometry and its virtual ULA segment.
        """
        self.L = len(sys_model.virtual_array_ula_seg)
        self.virtual_array = sys_model.virtual_array_ula_seg
        self.diff_array = sys_model.array[:, None] - sys_model.array[None, :]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the virtual-array covariance by averaging over difference-coarray lags.

        Args:
            x (torch.Tensor): Observations, shape [B, S, T].

        Returns:
            torch.Tensor: Virtual-array covariance, shape [B, L, L].
        """
        R_real_array = sample_covariance(x)

        Rx = torch.zeros(R_real_array.shape[0], self.L, self.L, dtype=torch.complex128)
        x_s_diff = torch.zeros(x.shape[0], 2 * self.L - 1, dtype=torch.complex128)  # x.shape[0] = batch size
        max_sensor = np.max(self.virtual_array)

        for i, lag in enumerate(range(-max_sensor, max_sensor + 1)):
            pairs = torch.from_numpy(self.diff_array) == lag
            if pairs.any():
                x_s_diff[:, i] = torch.mean(R_real_array[:, pairs], dim=1)

        for j in range(self.L):
            start_idx = self.L - 1 - j
            Rx[:, :, j] = x_s_diff[:, start_idx:start_idx + self.L]

        return Rx.to(device=device)


class ADMMReconstructor(CovReconstructor):
    """
    Batched ADMM solver for the nuclear-norm covariance completion

    min ‖ΦRΦᴴ − Rₓₓ‖_F² + μ‖R‖_*
    s.t.  R  Hermitian–Toeplitz – PSD.

    ADMM implementation to the method proposed at:
    'Structured Nyquist Correlation Reconstruction for DOA Estimation With Sparse Arrays, 2023'
    """
    def __init__(self, sys_model: SystemModel, mu: float = 2.5e-3,
                     rho: float = 2, max_iter=400, tol_primal: float = 1e-7,
                     tol_dual: float = 1e-7,
                     verbose: bool = False):
        """Initialize the batched ADMM covariance-completion solver.

        Args:
            sys_model (SystemModel): System model providing the sparse array geometry.
            mu (float): Nuclear-norm weight.
            rho (float): ADMM penalty/step-size parameter.
            max_iter (int): Maximum number of ADMM iterations.
            tol_primal (float): Primal residual tolerance for convergence.
            tol_dual (float): Dual residual tolerance for convergence.
            verbose (bool): If True, print residuals during iterations.
        """
        self.sys = sys_model
        self.phi = build_phi(self.sys.array)  # (|S|,|U|)
        self.phi_H = self.phi.t()
        self.U = self.phi.shape[1]  # |U|

        # mask diagonal  (P = Φᴴ Φ) → 1-D of length |U|²
        m = (self.phi_H @ self.phi).diag()  # (|U|,)
        self.P = (m[:, None] * m[None, :]).flatten()  # (|U|²,)

        self.mu = mu
        self.rho = rho
        self.max_iter = max_iter
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self._verbose = verbose

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run batched ADMM to complete the Hermitian-Toeplitz-PSD coarray covariance.

        Parameters
        ----------
        x : torch.Tensor
            Observations, shape [B, S, T].

        Returns
        -------
        R_tilde : (B,|U|,|U|) complex tensor
        """
        # ---- 1. Sample covariance of the PHYSICAL sparse array ----
        # Only the |S| real sensors are measured, so this covariance is missing the lags the
        # virtual co-array would have.
        Rx = sample_covariance(x)
        B, S, _ = Rx.shape
        U = self.U
        dev, dtype = Rx.device, Rx.dtype

        phi = self.phi.to(dev, dtype)
        phi_H = self.phi_H.to(dev, dtype)

        # ---- 2. Lift measurements onto the virtual co-array grid ----
        # Phi places each measured lag where it belongs in the |U| x |U| virtual covariance; the
        # lags the sparse array never observes stay empty, and filling them is the job below.
        # ------------------------------------------------------------
        # measured part  Phi^H Rxx Phi  ->  (B,|U|,|U|)
        # ------------------------------------------------------------
        meas = phi_H @ Rx @ phi
        vec_meas = meas.reshape(B, -1)  # (B, |U|^2)

        # ---- 3. Precompute the R-update solve ----
        # The data-fit operator is diagonal in this basis (P is the measured-lag mask), so its
        # inverse is an elementwise reciprocal computed ONCE outside the loop instead of a matrix
        # inverse per iteration.
        # ------------------------------------------------------------
        # diagonal coefficients  (mask + 2 rho I)^-1  (1-D then broadcast)
        # ------------------------------------------------------------
        inv_coeff = 1.0 / (self.P.to(dev, dtype) + 2 * self.rho)
        inv_coeff = inv_coeff.expand(B, -1)  # (B, |U|²)

        # ---- 4. Splitting variables ----
        # The answer must satisfy three things at once -- fit the data, be LOW RANK (M sources give
        # a rank-M covariance) and be Hermitian-Toeplitz-PSD. No single projection does all three,
        # so ADMM keeps a copy per property (R / S / T) and drives them together via the duals.
        # ------------------------------------------------------------
        # initial variables  (B,|U|,|U|)
        # ------------------------------------------------------------
        R = hermitian_proj(meas)
        S = R.clone()
        T = toeplitz_proj(R)
        Udual = torch.zeros_like(R)
        Vdual = torch.zeros_like(R)

        S_prev, T_prev = S.clone(), T.clone()

        # ------------------------------------------------------------
        # ADMM iterations
        # ------------------------------------------------------------

        # ---- 5. ADMM iterations ----
        # Classical (non-unrolled) ADMM: fixed rho and mu, run to convergence rather than to a
        # fixed layer count. This is the baseline DUNCS's unrolled version is measured against.
        for k in range(self.max_iter):
            # 5a. R-update: data fit, closed form via the precomputed diagonal inverse.
            # R-update  (diagonal solve, batched)
            rhs = vec_meas + self.rho * (S - Udual + T - Vdual).reshape(B, -1)
            vec_R = inv_coeff * rhs
            R = vec_R.view(B, U, U)

            # 5b. S-update: LOW RANK via singular-value thresholding -- the proximal operator of
            # the nuclear norm, the convex stand-in for "rank = number of sources".
            # S-update  (SVT)
            Z = R + Udual
            S = svt(Z, self.mu / self.rho)

            # 5c. T-update: PHYSICAL STRUCTURE. Hermitian, then Toeplitz (a ULA covariance depends
            # only on sensor SEPARATION), then PSD. The Toeplitz averaging along each diagonal is
            # what actually fills the co-array holes from the measured lags.
            # T-update  (Herm-Toeplitz-PSD)
            W = R + Vdual
            T = psd_proj(toeplitz_proj(hermitian_proj(W)))

            # 5d. Dual ascent: accumulate the disagreement between copies, which is the pressure
            # that eventually makes one matrix satisfy all three properties at once.
            # dual ascent
            Udual += R - S
            Vdual += R - T

            # 5e. Stop when the copies agree (primal) and stop moving (dual).
            # convergence criteria (batch max)
            r_norm = torch.max(
                (R - S).flatten(1).norm(dim=1),
                (R - T).flatten(1).norm(dim=1)
            ).max()  # global primal residual

            s_norm = self.rho * torch.max(
                (S - S_prev).flatten(1).norm(dim=1),
                (T - T_prev).flatten(1).norm(dim=1)
            ).max()  # global dual residual

            if self._verbose and (k % 25 == 0):
                print(f"iter {k:4d} | primal {r_norm:.3e} | dual {s_norm:.3e}")

            if r_norm < self.tol_primal and s_norm < self.tol_dual:
                break

            S_prev.copy_(S)
            T_prev.copy_(T)

        # ---- 6. Return the STRUCTURED copy ----
        # T, not R: only T is guaranteed Hermitian-Toeplitz-PSD, which the subspace method
        # downstream needs to be meaningful.
        return T


class ADMMReconstructorCVXPY(CovReconstructor):
    """
    Solver for the nuclear-norm covariance completion using CVXPY package

    min ‖ΦRΦᴴ − Rₓₓ‖_F² + μ‖R‖_*
    s.t.  R  Hermitian–Toeplitz – PSD.

    ADMM implementation to the method proposed at:
    'Structured Nyquist Correlation Reconstruction for DOA Estimation With Sparse Arrays, 2023'
    """
    def __init__(self, sys_model: SystemModel, mu: float = 2.5e-3, **unused_kwargs):
        """Initialize the CVXPY-based covariance-completion solver.

        Args:
            sys_model (SystemModel): System model providing the sparse array geometry.
            mu (float): Nuclear-norm weight.
            **unused_kwargs: Ignored extra arguments (for interface compatibility).
        """
        self.sys = sys_model
        self.phi = build_phi(self.sys.array)  # (|S|,|U|)
        self.phi_H = self.phi.t()
        self.U = self.phi.shape[1]  # |U|
        self.S = self.phi.shape[0]  # |S|

        self.mu = mu
        self._solver = "SCS"   # Splitting Conic Solver, It solves primal-dual problems

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: observations Tensor
        mu : float
            nuclear-norm weight μ in (17)
        solver : str
            CVXPY solver name (“SCS”, “MOSEK”, "CVXOPT", ...)

        Returns
        -------
        R_tilde : torch.Tensor (|U|, |U|)
            Toeplitz-PSD covariance fitted by nuclear-norm minimisation
        """
        R_xx = sample_covariance(x).cpu().numpy()  # (|S|,|S|)

        # CVXPY variable (complex Hermitian)
        R = cp.Variable((self.U, self.U), hermitian=True)

        # Toeplitz constraint: equality of diagonals
        toeplitz_constraints = []
        for k in range(-self.U + 1, self.U):
            diag = cp.diag(R, k)
            toeplitz_constraints.append(diag == cp.mean(diag))

        # PSD constraint
        constraints = toeplitz_constraints + [R >> 0]

        # Objective
        fit = cp.norm(self.phi @ R @ self.phi_H - R_xx, p='fro')
        obj = cp.Minimize(fit + self.mu * cp.normNuc(R))

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=self._solver, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY failed: {prob.status}")

        return torch.tensor(R.value, dtype=torch.complex64, device=x.device).unsqueeze(0)


def sample_covariance(x: torch.Tensor):
    """
    Calculates the sample covariance matrix.

    Args:
    -----
        X (torch.Tensor): Input samples matrix.

    Returns:
    --------
        Rx (torch.Tensor): Covariance matrix.
    """
    # ---- R = x x^H / T ----
    # The maximum-likelihood covariance estimate for Gaussian data. Dividing by T (not T-1) matches
    # the ML convention every method in this repo assumes. A 2-D input is promoted to a batch of
    # one so callers can pass a single sample.
    if x.dim() == 2:
        x = x[None, :, :]
    batch_size, sensor_number, samples_number = x.shape
    Rx = torch.einsum("bmt, btl -> bml", x, torch.conj(x).transpose(1, 2)) / samples_number
    return Rx


def get_cov_reconstruction_method(method: str, sys_model: SystemModel, **kwargs):
    """Factory returning the covariance reconstructor matching the requested method.

    Args:
        method (str): One of 'sample', 'averaging', 'admm', 'admm_cvxpy'
            (forced to 'sample' for non-sparse arrays).
        sys_model (SystemModel): System model providing array geometry/parameters.
        **kwargs: Extra arguments forwarded to the selected reconstructor.

    Returns:
        CovReconstructor: An instantiated covariance reconstructor.
    """
    method = method.lower()
    if not sys_model.is_sparse_array:
        method = "sample"

    if method == 'sample':
        if sys_model.params.signal_nature == 'coherent':
            recon = SpatialSmoothingReconstructor()
        else:
            recon = SampleCov()
    elif method == 'averaging':
        recon = AveragingReconstructor(sys_model)
    elif method == 'admm':
        recon = ADMMReconstructor(sys_model, **kwargs)
    elif method == 'admm_cvxpy':
        recon = ADMMReconstructorCVXPY(sys_model, **kwargs)
    else:
        raise ValueError(f"The covariance Reconstruction method: {method} isn't supported")

    return recon
