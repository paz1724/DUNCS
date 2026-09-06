"""SPICE: SParse Iterative Covariance-based Estimation — covariance MATCHING without a NN.

Fits the structured covariance model R(p) = A diag(p) Aᴴ + diag(σ) to the sample
covariance R̂ = YYᴴ/T by the SPICE fixed-point iteration (Stoica et al., 2011):

    p_g ← p_g · sqrt(a_gᴴ R⁻¹ R̂ R⁻¹ a_g) / (a_gᴴ R⁻¹ a_g)

(and the analogous update for the per-sensor noise powers σ_n with e_n columns).
Hyperparameter-free, convex criterion, snapshot-efficient (designed for small T) and
coherence-robust (no rank requirement — the power parametrization does not care whether
sources are correlated). The classical covariance-matching sibling of the project's
sparse-covariance ADMM roots.

Interface mirrors MFOCUSS (source-adaptive coarse/fine grids, greedy peak-pick + local
soft-argmax, per-frequency dictionary swap via score()).
"""

import numpy as np
import torch

from src.models_pack.parent_model import ParentModel, local_ml_refine
from src.system_model import SystemModel
from src.metrics.criterions import set_criterions
from src.utils import device


class SPICE(ParentModel):
    """Covariance-matching (SPICE) DoA baseline — no NN, no training, no hyperparameters."""

    def __init__(self, system_model: SystemModel, num_iterations: int = 30,
                 grid_size: int = 121, criterion: str = "rmspe", grid_range_deg=None,
                 coarse_cols_per_180deg: int = 361,
                 loading_factor: float = 1e-6,
                 mf_eps: float = 1e-30,
                 p_floor: float = 1e-14,
                 peak_lim_deg: float = None,   # optional peak-search limit (unused with a data-cone grid)
                 noise_term: bool = True,      # model sigma^2 I explicitly (see the iteration below)
                 refine_min_sep_deg: float = 60.0,  # refine a source only if the nearest neighbour is
                                                   # farther than this. MUST exceed the array Rayleigh
                                                   # limit (~50 deg here) -- inside one beamwidth the
                                                   # beamscan cannot separate a pair and refining it
                                                   # destroys the coarse estimator's resolution.
                 refine_deg: float = 12.0,     # local ML refinement half-window (0/None disables).
                                               # IAA's structured covariance is a poorer fit than the
                                               # sample covariance when T > N, so its peak is coarse:
                                               # refining locally takes single-source 2.03 -> 0.44 deg
                                               # (CRLB 0.39). See parent_model.local_ml_refine.
                 spectrum_norm_eps: float = 1e-30,
                 pick_temp: float = 0.05,
                 pick_min_cols: int = 1,
                 eval_chunk: int = 256,
                 count_peak_thr: float = 0.5):
        """Initializes the SPICE baseline.

        Args:
            system_model (SystemModel): Array geometry and signal parameters.
            num_iterations (int): SPICE fixed-point iterations.
            grid_size (int): FINE (single-source) grid columns; the multi-source grid is
                coarse (<=361 cols), mirroring the MFOCUSS/DU source-adaptive grid.
            criterion (str): Evaluation criterion name.
            grid_range_deg (float | [lo, hi]): Optional grid azimuth range override.

        The remaining keyword args are hoisted numeric tunables / thresholds /
        epsilons; their defaults reproduce the historical hard-coded values
        exactly (see the constants block below for what each one controls).
        """
        super().__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.num_iter = num_iterations
        self.grid_size = grid_size
        self.M = system_model.params.M

        # ---- Hoisted tunables / epsilons (defaults == historical literals) ----
        self.coarse_cols_per_180deg = coarse_cols_per_180deg  # coarse (M>=2) grid density cap (cols per 180 deg)
        self.loading_factor = loading_factor        # relative diagonal (conditioning) load on R
        self.mf_eps = mf_eps                        # matched-filter / IAA denominator clamp epsilon
        self.p_floor = p_floor                      # power-spectrum clamp floor
        self.peak_lim_deg = peak_lim_deg
        self.noise_term = noise_term
        self.refine_deg = refine_deg
        self.refine_min_sep_deg = refine_min_sep_deg
        self.spectrum_norm_eps = spectrum_norm_eps  # spectrum max-normalization epsilon
        self.pick_temp = pick_temp                  # peak-pick local softmax temperature
        self.pick_min_cols = pick_min_cols          # minimum peak-pick window half-width (columns)
        self.eval_chunk = eval_chunk                # spectrum pass chunk size (memory bound)
        self.count_peak_thr = count_peak_thr        # source-count local-max significance threshold

        if grid_range_deg is None:
            # Grid = the DATA CONE. IAA needs its model covariance R to represent the isotropic
            # noise too; the textbook way is a grid spanning the whole field of view, and that does
            # fix single-source accuracy (RMS 1.62 -> 0.65 deg). But on THIS array the back-hemisphere
            # atoms are near-collinear with the front ones, so a 360 deg dictionary SPLITS each
            # source's power between a front atom and its replica and wrecks pair detection
            # (reuse-15 MD 22% -> 31%, multipath-15 48% -> 66%). Restricting the peak search does not
            # recover it -- the split happens in the recovery, not the pick. So keep the data cone and
            # give IAA the noise model EXPLICITLY instead (noise_term below).
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
        return f"K={self.num_iter}_grid={self.grid_size}"

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """IAA (Iterative Adaptive Approach) covariance-fitting spectrum [B, G].

        IAA is the weighted-least-squares covariance-fitting relative of SPICE:
            R(p) = A diag(p) Aᴴ (+ small load);  p_g = mean_t |a_gᴴ R⁻¹ y_t|² / (a_gᴴ R⁻¹ a_g)²
        Hyperparameter-free, snapshot-efficient (works down to a single snapshot), and
        coherence-robust. (A first-cut plain-SPICE fixed point without the criterion's
        proper weighting under-resolved badly — 50% pair MD — and was replaced.)
        """
        Y = x.to(torch.complex128)                                # [B, N, T]
        B, N, T = Y.shape
        A = self._A_use if self._A_use is not None else self.A    # [N, G]
        Rh = torch.einsum("bnt,bmt->bnm", Y, Y.conj()) / T        # sample covariance [B, N, N]
        pwr = torch.diagonal(Rh, dim1=-2, dim2=-1).real.mean(-1).clamp_min(self.mf_eps)   # [B]
        eye = torch.eye(N, dtype=torch.complex128, device=Y.device)
        # init: matched-filter (periodogram) powers
        anorm2 = (A.conj() * A).sum(dim=0).real.clamp_min(self.mf_eps)                    # [G]
        p = (torch.einsum("bnt,ng->bgt", Y, A.conj()).abs() ** 2).mean(dim=2) / anorm2 ** 2   # [B, G]
        p = p.clamp_min(self.p_floor)
        # Explicit noise power: R = A diag(p) A^H + s2 I. Without it the source atoms alone must
        # explain the isotropic noise, which a data-cone dictionary cannot do, and the iteration
        # distorts the source peak to absorb it (measured: single RMS 1.62 -> 1.32, reuse-15 MD
        # 22% -> 20%, multipath-15 MD 48% -> 42%). s2 is initialized from the smallest eigenvalue
        # of the sample covariance and then updated by the SAME IAA rule applied to N orthonormal
        # unit atoms (the standard IAA noise-power update).
        s2 = (torch.linalg.eigvalsh(Rh)[:, 0].real.clamp_min(self.mf_eps) if self.noise_term
              else self.loading_factor * pwr)                                            # [B]
        for _ in range(self.num_iter):
            R = torch.einsum("ng,bg,mg->bnm", A, p.to(torch.complex128), A.conj())
            R = R + s2.to(torch.complex128)[:, None, None] * eye                         # noise / load
            RiY = torch.linalg.solve(R, Y)                                           # [B, N, T]
            RiA = torch.linalg.solve(R, A.unsqueeze(0).expand(B, -1, -1))            # [B, N, G]
            RiA_e = torch.linalg.solve(R, eye.unsqueeze(0).expand(B, -1, -1)) if self.noise_term else None
            denom = torch.einsum("ng,bng->bg", A.conj(), RiA).real.clamp_min(self.mf_eps)  # a^H R^-1 a
            num = (torch.einsum("ng,bnt->bgt", A.conj(), RiY).abs() ** 2).mean(dim=2)  # mean_t |a^H R^-1 y|^2
            p = (num / denom ** 2).clamp_min(self.p_floor)
            if self.noise_term:                       # same update applied to the N unit atoms
                nn = (torch.einsum("nm,bnt->bmt", eye.conj(), RiY).abs() ** 2).mean(dim=2)   # [B, N]
                dd = torch.einsum("nm,bnm->bm", eye.conj(), RiA_e).real.clamp_min(self.mf_eps)
                s2 = (nn / dd ** 2).mean(dim=1).clamp_min(self.p_floor)
        spectrum = p
        return spectrum / (spectrum.amax(dim=1, keepdim=True) + self.spectrum_norm_eps)

    def _pick(self, spectrum: torch.Tensor, source_number: int) -> torch.Tensor:
        """Greedy multi-peak picker with local soft refinement (mirrors MFOCUSS)."""
        B, G = spectrum.shape
        grid = self._grid_use if self._grid_use is not None else self.grid
        half = max(self.pick_min_cols, int(getattr(self, "pick_half_frac", 0.025) * G))
        supp = max(half, int(getattr(self, "pick_supp_frac", 0.03) * G))
        grid_idx = torch.arange(G, device=spectrum.device)
        offsets = torch.arange(-half, half + 1, device=spectrum.device)
        # LOCAL-MAXIMA candidates only: IAA's dense spectrum has fat peaks whose SHOULDERS
        # out-rank the true second peak under plain argmax+suppress (26% pair MD); shoulders
        # are not local maxima, so restricting candidates to peaks fixes it without a large
        # suppression radius (which would mask genuinely close pairs).
        is_peak = torch.zeros_like(spectrum, dtype=torch.bool)
        is_peak[:, 1:-1] = (spectrum[:, 1:-1] >= spectrum[:, :-2]) & (spectrum[:, 1:-1] >= spectrum[:, 2:])
        # The two GRID ENDPOINTS count as peaks. Excluding them was tried (the MFOCUSS edge-pick
        # investigation) and measured IDENTICAL on every scenario -- IAA's dense spectrum puts an
        # estimate on an endpoint in under 1% of scenes -- so the exclusion was dropped rather than
        # kept as a setting that does nothing.
        is_peak[:, 0] = spectrum[:, 0] > spectrum[:, 1]
        is_peak[:, -1] = spectrum[:, -1] > spectrum[:, -2]
        work = spectrum.masked_fill(~is_peak, float("-inf"))
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
        c = spectrum[:, 1:-1]
        peaks = (c > spectrum[:, :-2]) & (c >= spectrum[:, 2:]) & (c > self.count_peak_thr)
        return peaks.sum(dim=1).clamp(min=1).float()

    def forward(self, x: torch.Tensor, sources_num: int = None, phase: str = "test"):
        src = int(sources_num) if sources_num is not None else self.M
        # IAA is a DENSE spectral estimator (like MUSIC/Capon, no over-sparsification), so it
        # searches the FINE grid for ALL source counts — the coarse-pair-grid trick exists for
        # FOCUSS-type methods whose fine-grid recovery drops the weaker source; IAA's doesn't
        # (coarse pairs measured 26-28% MD vs fine-grid pairs far lower).
        self._A_use = self.A_fine
        self._grid_use = self.grid_fine
        spectrum = torch.cat([self._spectrum(x[i:i + self.eval_chunk])
                              for i in range(0, x.shape[0], self.eval_chunk)])
        if self.peak_lim_deg is not None:
            # The DICTIONARY spans the full azimuth so that R = A diag(p) A^H can represent the
            # isotropic noise (see __init__), but sources only ever lie in the data cone. Searching
            # the whole 360 deg lets a back-hemisphere replica outrank the true second source:
            # measured reuse-15 MD 19% -> 39% and multipath-15 45% -> 79% when the grid was widened.
            # Model the noise over the full sphere, but pick peaks only where sources can be.
            edge = (self._grid_use.abs() > np.deg2rad(self.peak_lim_deg)).unsqueeze(0)
            spectrum = spectrum.masked_fill(edge, 0.0)
        doa = self._pick(spectrum, src)
        if self.refine_deg:
            doa = local_ml_refine(x, doa, self.A_fine, self.grid_fine,
                                  (self.A_fine.conj() * self.A_fine).sum(dim=0).real,
                                  float(np.deg2rad(self.refine_deg)),
                                  float(np.deg2rad(self.refine_min_sep_deg)))
        return doa, self._estimate_count(spectrum), None

    def _count_accuracy(self, sources_num, source_estimation):
        if source_estimation is None:
            return 0
        return torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation)).item()

    @torch.no_grad()
    def validation_step(self, batch):
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, source_estimation, _ = self(x, sources_num)
        loss = self.criterion(doa_prediction, angles)
        acc = self._count_accuracy(sources_num, source_estimation)
        return loss, acc

    @torch.no_grad()
    def test_step(self, batch):
        return self.validation_step(batch)
