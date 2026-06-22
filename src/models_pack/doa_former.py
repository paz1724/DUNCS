"""
DoAFormer: Transformer set-prediction network for DoA estimation.

A single-forward-pass alternative to iterative subspace/peak-search methods.
The sample covariance is tokenized per sensor and encoded with a Transformer
self-attention encoder. A bank of learnable "source query" embeddings is decoded
(DETR-style) into gridless angles, and a separate count head (over the pooled
encoder memory) estimates the number of sources. Trained end-to-end with the
RMSPE criterion (permutation-invariant) plus a count cross-entropy, giving
accurate, real-time DoA without an iterative search. Supports a variable number
of sources M (the angle queries use the known M; the count head predicts it).
"""

import numpy as np
import torch
import torch.nn as nn

from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.metrics.criterions import set_criterions
from src.utils import device


class DoAFormer(ParentModel):
    """Transformer encoder/decoder with learnable source queries + a count head."""

    def __init__(self, system_model: SystemModel, d_model: int = 64, nhead: int = 4,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, criterion: str = "rmspe",
                 input_mode: str = "cov"):
        """Initializes the DoAFormer model.

        Args:
            system_model (SystemModel): Array geometry and signal parameters.
            d_model (int): Transformer embedding dimension.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of Transformer encoder layers.
            num_decoder_layers (int): Number of Transformer decoder layers.
            dim_feedforward (int): Hidden size of the feed-forward sub-layers.
            dropout (float): Dropout probability.
            criterion (str): Training criterion name (e.g. "rmspe").
        """
        super().__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.N = system_model.params.N
        self.T = system_model.params.T
        # Token source: "cov" = covariance rows (N tokens), "snapshots" = raw snapshots
        # (T tokens, preserve front/back phase), "both" = N+T tokens.
        self.input_mode = input_mode
        n_tok = {"cov": self.N, "snapshots": self.T, "both": self.N + self.T}[input_mode]
        # Max number of sources (size of the query bank / count classes).
        M = system_model.params.M
        self.Q = max(M) if isinstance(M, (list, tuple)) else int(M)
        # Angle outputs are scaled by tanh into +/- the configured half-aperture.
        hi = max(abs(system_model.params.doa_range[0]), abs(system_model.params.doa_range[1]))
        self.angle_scale = float(np.deg2rad(hi))

        # Token = [Re(row), Im(row)] -> 2N features (covariance row OR snapshot column).
        self.input_proj = nn.Linear(2 * self.N, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(n_tok, d_model) * 0.02)
        self.query_embed = nn.Parameter(torch.randn(self.Q, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.angle_head = nn.Linear(d_model, 1)
        self.count_head = nn.Linear(d_model, self.Q)     # classifies #sources in {1..Q}
        self.count_loss = nn.CrossEntropyLoss()

    def get_model_params(self):
        """Returns a short string summarizing the model's hyper-parameters.

        Returns:
            str: String of the form "dmodel=<d_model>_Q=<Q>".
        """
        return f"dmodel={self.pos_embed.shape[1]}_Q={self.Q}"

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Builds input tokens from the snapshots (covariance rows and/or raw snapshots).

        Args:
            x (torch.Tensor): Input snapshots, shape [B, N, T].

        Returns:
            torch.Tensor: Real-valued tokens, shape [B, n_tok, 2N].
        """
        x = x.to(torch.complex64)
        toks = []
        if self.input_mode in ("cov", "both"):
            Rx = torch.einsum("bnt,bmt->bnm", x, x.conj()) / x.shape[-1]   # [B, N, N]
            # Per-sample unit-Frobenius normalization keeps the transformer stable.
            Rx = Rx / (torch.linalg.norm(Rx, dim=(1, 2), keepdim=True) + 1e-6)
            toks.append(torch.cat((Rx.real, Rx.imag), dim=2))             # [B, N, 2N]
        if self.input_mode in ("snapshots", "both"):
            # Raw snapshots as T tokens of [Re, Im] over the N sensors — preserve the
            # per-snapshot phase that distinguishes front from back-lobe on a measured array.
            xs = x.transpose(1, 2)                                        # [B, T, N]
            xs = xs / (torch.linalg.norm(xs, dim=(1, 2), keepdim=True) + 1e-6)
            toks.append(torch.cat((xs.real, xs.imag), dim=2))            # [B, T, 2N]
        return torch.cat(toks, dim=1)                                     # [B, n_tok, 2N]

    def forward(self, x: torch.Tensor, sources_num: int = None, phase: str = "train"):
        """Encodes the covariance, decodes M source queries, and predicts the count.

        Args:
            x (torch.Tensor): Input snapshots, shape [B, N, T].
            sources_num (int): Known number of sources; selects how many angle queries
                are decoded. Defaults to the full query bank (Q).
            phase (str): Unused; kept for interface compatibility.

        Returns:
            tuple: (doa_prediction [B, M], source_estimation [B], count_logits [B, Q]).
        """
        B = x.shape[0]
        M = int(sources_num) if sources_num is not None else self.Q
        M = max(1, min(M, self.Q))
        tokens = self.input_norm(self.input_proj(self._tokens(x))) + self.pos_embed.unsqueeze(0)  # [B, N, d]
        memory = self.encoder(tokens)                                            # [B, N, d]
        queries = self.query_embed[:M].unsqueeze(0).expand(B, -1, -1)            # [B, M, d]
        decoded = self.decoder(queries, memory)                                  # [B, M, d]

        angles = torch.tanh(self.angle_head(decoded).squeeze(-1)) * self.angle_scale  # [B, M]
        count_logits = self.count_head(memory.mean(dim=1))                       # [B, Q]
        source_estimation = (count_logits.argmax(dim=1) + 1).float()             # [B]
        return angles.to(torch.float64), source_estimation, count_logits

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
        """Runs one training step: forward pass, RMSPE angle loss + count cross-entropy.

        Args:
            batch: A tuple of (x, sources_num, angles).

        Returns:
            tuple: (loss, acc, None) where loss combines RMSPE and a count
                cross-entropy, and acc is the source-count accuracy count.
        """
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, source_estimation, count_logits = self(x, sources_num)
        loss = self.criterion(doa_prediction, angles)
        target = torch.full((count_logits.shape[0],), int(sources_num) - 1,
                            dtype=torch.long, device=count_logits.device)
        loss = loss + self.count_loss(count_logits, target)
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
