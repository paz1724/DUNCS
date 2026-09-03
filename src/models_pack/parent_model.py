import torch.nn as nn

from src.system_model import SystemModel
from src.utils import validate_constant_sources_number, device

class ParentModel(nn.Module):
    """Base class for all DUNCS models, providing common naming and batch utilities."""

    def __init__(self, system_model: SystemModel, training_loss_type):
        """Initializes the ParentModel.

        Args:
            system_model (SystemModel): Array geometry and signal parameters.
            training_loss_type: Identifier for the training loss/criterion (unused here).
        """

        super(ParentModel, self).__init__()
        self.system_model = system_model
        # self.criterion = set_criterions(training_loss_type.lower())

    def get_model_name(self):
        """Builds a model name string from the class name and model params.

        Returns:
            str: Name in the form "<ClassName>_<model_params>".
        """
        return f"{self._get_name()}_{self.get_model_params()}"

    def get_model_params(self):
        """Returns a string summary of the model's parameters.

        Returns:
            None: Base implementation; subclasses override with a descriptive string.
        """
        return None

    def training_step(self, batch):
        """Runs a single training step on a batch.

        Args:
            batch: A tuple of (x, sources_num, angles).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def validation_step(self, batch):
        """Runs a single validation step on a batch.

        Args:
            batch: A tuple of (x, sources_num, angles).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def test_step(self, batch):
        """Runs a single test step on a batch.

        Args:
            batch: A tuple of (x, sources_num, angles).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_learned_covariance(self, x):
        """Computes the model's learned/surrogate covariance matrix.

        Args:
            x (torch.Tensor): Input snapshots, shape [Batch size, N, T].

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def _prepare_batch(batch):
        """Unpacks a batch, moves tensors to device, and validates source counts.

        Args:
            batch: A tuple of (x, sources_num, angles), where x may be [N, T] or [B, N, T].

        Returns:
            tuple: (x, source_num, angles) with x on device of shape [B, N, T],
                source_num a single scalar source count, and angles on device.
        """
        x, sources_num, angles = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(device)
        angles = angles.to(device)
        source_num = sources_num.to(device)
        validate_constant_sources_number(sources_num)
        return x, source_num[0], angles

    def get_model_file_name(self):
        """Constructs a unique filename encoding the model and system model parameters.

        Returns:
            str: Filename string including model name, N, M, T, signal type, SNR,
                field type, signal nature, eta, and steering-vector noise variance.
        """
        M = str(self.system_model.params.M).replace(' ', '-')
        snr = str(self.system_model.params.snr).replace(' ', '-')
        return f"{self.get_model_name()}_" + \
            f"N={self.system_model.params.N}_" + \
            f"M={M}_" + \
            f"T={self.system_model.params.T}_" + \
            f"{self.system_model.params.signal_type}_" + \
            f"SNR={snr}_" + \
            f"{self.system_model.params.field_type}_field_" + \
            f"{self.system_model.params.signal_nature}_" + \
            f"eta={self.system_model.params.eta}_" + \
            f"sv_var={self.system_model.params.sv_noise_var}"


if __name__ == "__main__":
    pass


def local_ml_refine(x, doa_rad, A_fine, grid_fine, anorm2, win_rad, min_sep_rad=None, eps=1e-30):
    """Coarse-to-fine readout: local normalized-beamscan (ML) peak + parabolic sub-grid interpolation.

    A coarse estimator fixes WHICH source is where; the accuracy then comes from a local search on
    the recorded manifold. The discriminant is the NORMALIZED beamscan real(a^H R a)/||a||^2, which
    is the efficient one (it reaches the CRLB) and is invariant to the manifold's |a| ripple.

    Measured on single sources at 150 MHz, T=8, SNR U(25,30) (CRLB 0.39 deg):
        DoAFormer 1.27 -> 0.44 deg   (a gridless regression head has no sub-grid refinement of its own)
        SPICE/IAA 2.03 -> 0.44 deg   (its structured covariance is a poorer fit than R_hat when T > N)

    ONLY applied where the beamscan is actually the efficient discriminant, i.e. where no other
    source contaminates the local window. min_sep_rad must therefore EXCEED the array's Rayleigh
    limit (~50 deg for this 5-element array at 150 MHz): below it the beamscan has a single broad
    peak, so refining drags both estimates of a pair onto the midpoint and destroys resolution the
    coarse estimator had already found. Measured on DoAFormer multipath-15: 1.93 deg / 0 % MD
    unrefined, 8.43 deg / 58 % MD if refined unconditionally, and even a 40 deg guard still cost
    reuse-25 12 % -> 21 % MD because 25-55 deg pairs are all inside one beamwidth. With the default
    guard (60 deg > Rayleigh) single sources gain the full accuracy and every pair is left untouched.

    Args:
        x: snapshots [B, N, T]; doa_rad: coarse angles [B, M]; A_fine: [N, G] fine dictionary;
        grid_fine: [G] radians; anorm2: [G] = ||a||^2; win_rad: max half-window (radians).
    Returns:
        Refined angles [B, M] (radians).
    """
    import torch as _t
    B, N, T = x.shape
    R = _t.einsum("bnt,bmt->bnm", x.to(_t.complex128), x.to(_t.complex128).conj()) / T
    RA = _t.einsum("bnm,mg->bng", R, A_fine)
    P = _t.einsum("ng,bng->bg", A_fine.conj(), RA).real / anorm2.clamp_min(eps)      # [B, G]
    G = P.shape[1]
    step = (grid_fine[1] - grid_fine[0]).to(_t.float64)
    out = doa_rad.clone().to(_t.float64)
    M = doa_rad.shape[1]
    for j in range(M):
        d = doa_rad[:, j:j + 1].to(_t.float64)                                        # [B, 1]
        w = _t.full_like(d.squeeze(1), float(win_rad))
        keep = _t.ones_like(w, dtype=_t.bool)        # which samples may be refined at all
        if M > 1:
            others = _t.cat([doa_rad[:, k:k + 1] for k in range(M) if k != j], dim=1).to(_t.float64)
            nn = (others - d).abs().amin(dim=1)      # distance to the nearest other source
            if min_sep_rad is not None:              # unresolvable by a beamscan -> leave alone
                keep = nn >= float(min_sep_rad)
            w = _t.minimum(w, 0.5 * nn)
        sel = (grid_fine.unsqueeze(0) - d).abs() <= w.unsqueeze(1).clamp_min(step)
        Pm = P.masked_fill(~sel, float("-inf"))
        idx = Pm.argmax(dim=1).clamp(1, G - 2)
        y0 = _t.gather(P, 1, (idx - 1).unsqueeze(1)).squeeze(1).to(_t.float64)
        y1 = _t.gather(P, 1, idx.unsqueeze(1)).squeeze(1).to(_t.float64)
        y2 = _t.gather(P, 1, (idx + 1).unsqueeze(1)).squeeze(1).to(_t.float64)
        den = y0 - 2.0 * y1 + y2
        dl = _t.where(den.abs() > eps, 0.5 * (y0 - y2) / den, _t.zeros_like(den)).clamp(-1.0, 1.0)
        refined = grid_fine[idx].to(_t.float64) + dl * step
        out[:, j] = _t.where(keep, refined, doa_rad[:, j].to(_t.float64))
    return out
