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
