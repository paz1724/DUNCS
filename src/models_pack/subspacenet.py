"""
SubspaceNet: model-based deep learning algorithm as described in:
        [2] "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
"""

from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.utils import *
from src.metrics.criterions import EigenRegularizationLoss

from src.methods_pack.music import MUSIC, MVDR
from src.methods_pack.esprit import ESPRIT
from src.methods_pack.root_music import RootMusic
from src.metrics.criterions import set_criterions

class SubspaceNet(ParentModel):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        N (int):
        diff_method (str):
        field_type (str):
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, diff_method: str = "root_music",
                 system_model: SystemModel = None, field_type: str = "Far", eigen_regularization_weight=0,
                 ideal_ula_cov_weight=0.0, criterion="rmspe"):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.tau = tau
        self.N = self.system_model.params.N
        self._clamp_tau()
        self.diff_method = None
        self.field_type = field_type
        self.p = 0.15
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(self.p)
        self.ReLU = nn.ReLU()
        self.norm1 = SpectralNormalization()
        # Residual/skip: blend the raw sample covariance into the learned one so a degenerate
        # CNN output (gram_diagonal_overload -> ~identity) cannot destroy an already-informative
        # covariance. Starts raw-dominated; the CNN learns a correction on top.
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        # Ideal-ULA covariance supervision (for the ESPRIT readout): when > 0, the learned covariance
        # is trained to match the IDEAL lambda/2-ULA covariance a_ideal(theta) a_ideal(theta)^H so that
        # ESPRIT's theta = -arcsin(phase/pi) readout (which assumes an ideal ULA) reads it correctly,
        # bridging the non-ideal recorded manifold to the ideal one.
        self.ideal_ula_cov_weight = float(ideal_ula_cov_weight)
        self.calibration = None        # optional [N,N] recorded->ideal-ULA calibration for ESPRIT
        self._eigen_regularization = EigenRegularizationLoss(eigen_regularization_weight)
        # Full-azimuth (±180°) support: ESPRIT's arcsin readout is limited to [-90°,90°]; when a
        # recorded-manifold grid is installed (set_full_azimuth) the front/back sin-ambiguity is
        # resolved per source by matching the candidate {θ, 180-θ} against the data covariance.
        self.full_azimuth = False
        self._fa_A = None        # (N, G) recorded manifold at the operating frequency
        self._fa_grid = None     # (G,) grid azimuths in degrees over the full ±180°
        # Set the subspace method for training
        self.set_diff_method(diff_method, system_model)

    def _clamp_tau(self):
        """Clamp tau to T-1 to avoid division by zero in pre_processing.

        When tau >= T, the last lag (i = T-1) produces a zero denominator
        in the auto-correlation normalization: (T - i - 1) = 0.
        """
        T = self.system_model.params.T
        if self.tau >= T:
            clamped = T - 1
            print(f"{self.__class__.__name__}: tau={self.tau} >= T={T}, clamping tau to {clamped}")
            self.tau = clamped

    def get_learned_covariance(self, X: torch.Tensor) -> torch.Tensor:
        """
            This function is the "real" forward pass of the SubspaceNet.
            It receives the input tensor and returns the surrogate covariance matrix.
            Args:
                x: the input tensor of shape [Batch size, N, T]

            Returns:
                Rz: the surrogate covariance matrix of shape [Batch size, N, N]
        """
        x = self.pre_processing(X)
        # Rx_tau shape: [Batch size, tau, 2N, N]
        N = x.shape[-1]
        self.batch_size = x.shape[0]
        ############################
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(x)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)

        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #4
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        Kx_tag = self.norm1(Kx_tag)
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Residual/skip with the raw sample covariance (trace-normalized for comparable scale).
        # Prevents a degenerate CNN output (gram_diagonal_overload -> ~identity, flat spectrum,
        # zero gradient) from collapsing training. The blend is readout-aware:
        #   - MUSIC/MVDR search the recorded manifold grid, so the raw covariance is directly
        #     informative -> let it dominate (CNN adds a learned correction).
        #   - ESPRIT's readout assumes an ideal lambda/2-ULA rotational invariance, so it needs the
        #     CNN to RESHAPE the covariance into that form -> the CNN dominates and the raw term is
        #     just a small anti-collapse regularizer.
        Rraw = torch.einsum("bnt,bmt->bnm", X, X.conj()) / X.shape[-1]

        def _tr_norm(M):
            tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).real.clamp_min(1e-9)
            return M / tr[:, None, None]

        if isinstance(self.diff_method, (ESPRIT, RootMusic)) and self.ideal_ula_cov_weight > 0 \
                and not getattr(self, "cov_raw_anchor", False):
            # Direct ideal-ULA covariance supervision: use the pure CNN output (supervision provides
            # gradient/anti-collapse and the raw recorded cov would bias the signal subspace). Root-MUSIC
            # roots the noise-subspace polynomial, which needs an EXACT Vandermonde signal steering --
            # post-hoc calibration's 0.62 residual is too coarse, so it relies on this supervision instead.
            # cov_raw_anchor: on heavy multipath the pure CNN can't synthesize the angle-conditional cov and
            # regresses to the mean (white) cov -> use the raw-anchored branch below instead; the cov loss then
            # trains the CNN residual to SUBTRACT the multipath while the raw carries the (generalizing) angle.
            Rz = _tr_norm(Rz)
        else:
            # Raw-dominated residual: keeps the signal subspace anchored to the (informative) sample
            # covariance even if the CNN output degenerates. For ESPRIT the forward then applies the
            # recorded->ideal-ULA calibration C so the arcsin readout reads the right angle.
            Rz = _tr_norm(Rraw).to(Rz.dtype) + self.residual_weight.to(Rz.dtype) * _tr_norm(Rz)
        if getattr(self, "fba", False):
            # Forward-backward averaging: Rz <- (Rz + J Rz* J)/2 with J the exchange (anti-identity) matrix.
            # Improves the covariance/subspace estimate on real, snapshot-limited data (measured single-source
            # RMS drops ~15-25% for MUSIC/MVDR). Off by default; enabled per-readout at eval.
            Jm = torch.flip(torch.eye(self.N, device=Rz.device, dtype=Rz.dtype), dims=[0])
            Rz = 0.5 * (Rz + Jm @ Rz.conj() @ Jm)
        return Rz

    def _ideal_ula_cov(self, angles: torch.Tensor, sources_num) -> torch.Tensor:
        """Ideal lambda/2-ULA covariance for the given DoAs, trace-normalized.

        a_ideal(theta)_n = exp(-j*pi*n*sin(theta)) so that ESPRIT's theta = -arcsin(phase/pi)
        recovers theta exactly. R_ideal = (1/M) sum_k a(theta_k) a(theta_k)^H.

        Args:
            angles (torch.Tensor): DoAs in radians, shape [B, M].
            sources_num: number of sources (scalar) used for normalization.

        Returns:
            torch.Tensor: Ideal covariance, shape [B, N, N], complex, trace-normalized.
        """
        ang = angles.to(torch.float64)                                   # [B, M]
        rec_sv = getattr(self, "recorded_sv", None)
        if rec_sv is not None:
            # RECORDED ULA3 covariance (NOT the lambda/2 ideal): R = (1/M) sum_k a_rec(theta_k) a_rec(theta_k)^H,
            # a_rec from the measured manifold. Root-MUSIC then roots this with the ULA3's actual d/lambda.
            grid = self.recorded_grid.to(ang.device)                     # [G] radians
            sv = rec_sv.to(ang.device)                                   # [N, G]
            idx = torch.argmin((grid.view(1, 1, -1) - ang.unsqueeze(-1)).abs(), dim=-1)   # [B, M] nearest grid
            a = sv[:, idx].permute(1, 2, 0)                              # [B, M, N]
            R = torch.einsum("bmn,bmk->bnk", a, a.conj()) / a.shape[1]
            tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1).real.clamp_min(1e-9)
            return R / tr[:, None, None]
        n = torch.arange(self.N, device=angles.device, dtype=torch.float64)
        phase = -np.pi * n.view(1, 1, -1) * torch.sin(ang).unsqueeze(-1)  # [B, M, N]
        a = torch.exp(1j * phase.to(torch.complex128))                   # [B, M, N]
        R = torch.einsum("bmn,bmk->bnk", a, a.conj()) / a.shape[1]        # [B, N, N]
        tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1).real.clamp_min(1e-9)
        return R / tr[:, None, None]

    def set_recorded_manifold(self, sv, grid_rad):
        """Supervise the learned covariance toward the RECORDED ULA3 manifold instead of the lambda/2 ideal.

        Args:
            sv (array): [N, G] complex recorded steering matrix over the angle grid.
            grid_rad (array): [G] grid angles in radians.
        """
        dev = next(self.parameters()).device
        self.recorded_sv = torch.as_tensor(np.asarray(sv), dtype=torch.complex128, device=dev)
        self.recorded_grid = torch.as_tensor(np.asarray(grid_rad), dtype=torch.float64, device=dev)

    def forward(self, x: torch.Tensor, sources_num: torch.tensor = None, known_angles: torch.tensor = None):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            x (torch.Tensor): Input tensor of shape [Batch size, N, T].
            sources_num (torch.Tensor): The number of sources in the signal.
            known_angles (torch.Tensor): The known angles for the near-field scenario.

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """

        # Feed surrogate covariance to the differentiable subspace algorithm
        Rz = self.get_learned_covariance(x)

        # Array calibration: ESPRIT and Root-MUSIC both assume an ideal lambda/2-ULA, but the recorded
        # manifold is non-ideal. A fixed least-squares calibration C (C a_rec(theta) ~ a_ideal(theta))
        # maps the covariance into ideal-ULA coordinates so their arcsin / polynomial-root readout applies.
        if isinstance(self.diff_method, (ESPRIT, RootMusic)) and self.calibration is not None:
            C = self.calibration.to(Rz.dtype)
            Rz = torch.einsum("mn,bnk,lk->bml", C, Rz, C.conj())

        if self.field_type == "Far":
            method_output = self.diff_method(Rz, sources_num)
            if isinstance(self.diff_method, RootMusic):
                doa_prediction, doa_all_predictions, roots = method_output
                if self.full_azimuth:
                    # Root-MUSIC's arcsin readout is front-only [-90,90]; resolve the front/back |sin|
                    # ambiguity per source via the recorded-manifold candidate scoring (same as ESPRIT).
                    doa_prediction = self._resolve_front_back(x, doa_prediction)
                return doa_prediction, doa_all_predictions, roots
            elif isinstance(self.diff_method, ESPRIT):
                # Esprit output
                doa_prediction, sources_estimation, eigen_regularization = method_output
                if self.full_azimuth:
                    doa_prediction = self._resolve_front_back(x, doa_prediction)
                return doa_prediction, sources_estimation, eigen_regularization
            elif isinstance(self.diff_method, MUSIC):
                doa_prediction, sources_estimation, eigen_regularization = method_output
                return doa_prediction, sources_estimation, eigen_regularization
            else:
                raise Exception(f"SubspaceNet.forward: Method {self.diff_method} is not defined for SubspaceNet")

        elif self.field_type == "Near":
            if known_angles is None:
                predictions, sources_estimation, eigen_regularization = self.diff_method(
                    Rz, number_of_sources=sources_num)
                doa_prediction, distance_prediction = predictions
                return doa_prediction, distance_prediction, sources_estimation, eigen_regularization
            else:  # the angles are known
                distance_prediction = self.diff_method(
                    cov=Rz, number_of_sources=sources_num, known_angles=known_angles)
                if isinstance(distance_prediction, tuple):
                    distance_prediction, _, _ = distance_prediction
                return known_angles, distance_prediction, Rz

    def set_calibration(self, C):
        """Installs a fixed recorded->ideal-ULA calibration matrix for the ESPRIT readout.

        Args:
            C (array): [N, N] complex calibration so that C a_rec(theta) ~= a_ideal(theta).
        """
        self.calibration = torch.as_tensor(C, dtype=torch.complex128, device=device)

    def set_full_azimuth(self, A_grid, grid_deg):
        """Installs a recorded-manifold grid so ESPRIT estimates can be extended to ±180°.

        ESPRIT's readout θ = -arcsin(phase/π) only spans [-90°,90°]; a source at |az|>90°
        aliases to its front mirror. The recorded array manifold differs front vs back, so the
        ambiguity is resolved by matching each candidate {θ, 180-θ} against the data covariance.

        Args:
            A_grid (array): Complex recorded steering manifold at the operating frequency,
                shape [N, G].
            grid_deg (array): Grid azimuths in degrees over the full ±180°, shape [G].
        """
        self._fa_A = torch.as_tensor(A_grid, dtype=torch.complex128, device=device)
        self._fa_grid = torch.as_tensor(grid_deg, dtype=torch.float64, device=device)
        self.full_azimuth = True

    @torch.no_grad()
    def _resolve_front_back(self, x, angles):
        """Resolves the front/back sin-ambiguity of ESPRIT estimates over the full ±180°.

        For each estimated (front) angle θ, the sin-ambiguous candidates are θ and wrap(180-θ).
        Each candidate's recorded steering vector is scored as a^H R a against the sample
        covariance R of the raw input; the higher-scoring hemisphere is selected.

        Args:
            x (torch.Tensor): Raw input snapshots, shape [B, N, T].
            angles (torch.Tensor): ESPRIT front-angle estimates (radians), shape [B, M].

        Returns:
            torch.Tensor: Disambiguated angles (radians) over the full ±180°, shape [B, M].
        """
        Xc = x.to(torch.complex128)
        R = torch.einsum("bnt,bmt->bnm", Xc, Xc.conj()) / Xc.shape[-1]      # [B, N, N]
        A = self._fa_A; g = self._fa_grid                                   # [N, G], [G]
        deg = torch.rad2deg(angles.to(torch.float64))                       # [B, M] front, in [-90,90]
        back = ((180.0 - deg + 180.0) % 360.0) - 180.0                      # wrapped back-twin

        def score(cand_deg):
            idx = torch.abs(cand_deg.unsqueeze(-1) - g.view(1, 1, -1)).argmin(-1)  # [B, M] nearest grid cell
            a = A[:, idx].permute(1, 2, 0)                                  # [B, M, N]
            quad = torch.einsum("bmn,bnk,bmk->bm", a.conj(), R, a)          # a^H R a
            return quad.real

        pick_back = score(back) > score(deg)                               # [B, M]
        out_deg = torch.where(pick_back, back, deg)
        return torch.deg2rad(out_deg).to(angles.dtype)

    def pre_processing(self, x):
        """
        The input data is a complex signal of size [batch, N, T] and the input to the model supposed to be complex
         tensors of size [batch, tau, 2N, N].
        """
        batch_size = x.shape[0]
        Rx_tau = torch.zeros(batch_size, self.tau, 2 * self.N, self.N, device=device)
        meu = torch.mean(x, dim=-1, keepdim=True).to(device)
        center_x = x - meu
        if center_x.dim() == 2:
            center_x = center_x[None, :, :]

        for i in range(self.tau):
            x1 = center_x[:, :, :center_x.shape[-1] - i].to(torch.complex128)
            x2 = torch.conj(center_x[:, :, i:]).transpose(1, 2).to(torch.complex128)
            Rx_lag = torch.einsum("BNT, BTM -> BNM", x1, x2) / (center_x.shape[-1] - i - 1)
            Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), dim=1)
            Rx_tau[:, i, :, :] = Rx_lag

        return Rx_tau

    def set_diff_method(self, diff_method: str, system_model):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if self.field_type == "Far":
            if diff_method.startswith("root_music"):
                self.diff_method = RootMusic(system_model=system_model)
            elif diff_method.startswith("esprit"):
                self.diff_method = ESPRIT(system_model=system_model)
            elif diff_method.startswith("music_1D") or diff_method.startswith("music"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="angle")
            elif diff_method.startswith("mvdr"):
                self.diff_method = MVDR(system_model=system_model, estimation_parameter="angle")
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} scenario")
        elif self.field_type == "Near":
            if diff_method.startswith("music_2D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="angle, range")
            elif diff_method.startswith("music_1D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="range")
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} Field scenario")

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def adjust_diff_method_temperature(self, epoch):
        """Reduces the MUSIC spectrum cell size every 20 epochs (temperature annealing).

        Args:
            epoch (int): Current training epoch.
        """
        if isinstance(self.diff_method, MUSIC):
            if epoch % 20 == 0 and epoch != 0:
                self.diff_method.adjust_cell_size()
                print(f"Model temepartue updated --> {self.get_diff_method_temperature()}")

    def get_diff_method_temperature(self):
        """Returns the current MUSIC cell size(s) used as the differentiable temperature.

        Returns:
            int or dict or None: The cell size for angle/range estimation, a dict of
                angle/distance cell sizes for joint estimation, or None if not MUSIC.
        """
        if isinstance(self.diff_method, MUSIC):
            if self.diff_method.estimation_params in ["angle", "range"]:
                return self.diff_method.cell_size
            elif self.diff_method.estimation_params == "angle, range":
                return {"angle_cell_size": self.diff_method.cell_size_angle,
                        "distance_cell_size": self.diff_method.cell_size_distance}

    def get_model_params(self):
        """Returns a string summarizing the model's tau and differentiable method.

        Returns:
            str: String in the form "tau=<tau>_diff_method=<diff_method>".
        """
        tau = self.tau
        diff_method = self.diff_method
        return f"tau={tau}_diff_method={diff_method}"

    def training_step(self, batch):
        """Runs one training step: forward pass, DOA loss, and eigen-regularization.

        Args:
            batch: A tuple of (x, sources_num, angles).

        Returns:
            tuple: (loss, acc, eigen_regularization) where loss is the regularized
                RMSPE loss, acc is the source-count accuracy, and eigen_regularization
                is the eigenvalue regularization term.
        """
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, sources_estimation, eigen_regularization = self(x, sources_num)
        if isinstance(self.diff_method, RootMusic):
            # Root-MUSIC's 2nd/3rd forward outputs are all-root angles / roots, not a source count / reg term.
            # Drop the count, but use the eigen-regularization captured inside RootMusic's subspace separation
            # so training shapes the covariance eigenstructure (as for ESPRIT/MUSIC) -- critical for clean roots.
            sources_estimation = None
            eigen_regularization = getattr(self.diff_method, "eigen_regularization", None)
        loss = self.criterion(doa_prediction, angles)
        acc = self._eigen_regularization.source_estimation_accuracy(sources_num, sources_estimation)
        loss = self._eigen_regularization.get_regularized_loss(loss, eigen_regularization)
        if self.ideal_ula_cov_weight > 0:
            # Supervise the learned covariance toward the ideal lambda/2-ULA covariance so the
            # ESPRIT readout (which assumes an ideal ULA) reads the recorded-data covariance correctly.
            Rz = self.get_learned_covariance(x)
            R_ideal = self._ideal_ula_cov(angles, sources_num)
            cov_loss = (Rz - R_ideal).abs().pow(2).mean()
            loss = loss + self.ideal_ula_cov_weight * cov_loss
        return loss, acc, eigen_regularization

    def validation_step(self, batch):
        """Runs one validation step: forward pass and DOA loss (no regularization).

        Args:
            batch: A tuple of (x, sources_num, angles).

        Returns:
            tuple: (loss, acc) where loss is the RMSPE loss and acc is the
                source-count accuracy.
        """
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, source_estimation, eigen_regularization = self(x, sources_num)
        loss = self.criterion(doa_prediction, angles)
        acc = self._eigen_regularization.source_estimation_accuracy(sources_num, source_estimation)
        return loss, acc

    def test_step(self, batch):
        """Runs one test step (delegates to validation_step).

        Args:
            batch: A tuple of (x, sources_num, angles).

        Returns:
            tuple: (loss, acc) as returned by validation_step.
        """
        return self.validation_step(batch)
