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

from src.models_pack.parent_model import ParentModel, local_ml_refine
from src.system_model import SystemModel
from src.metrics.criterions import set_criterions
from src.utils import device



def _unit_fro(z: torch.Tensor) -> torch.Tensor:
    """Scales each sample to unit Frobenius norm, at any physical signal level.

    Args:
        z (torch.Tensor): Batch of matrices, shape [B, R, C].

    Returns:
        torch.Tensor: Same shape, each sample divided by its own Frobenius norm.
    """
    # Guarding the division with an ABSOLUTE epsilon (the previous `+ 1e-6`) silently destroys the
    # normalization on data that is not already O(1). The recorded instances carry physical levels:
    # ||R||_F spans 5e-12..5e-4, so a 1e-6 guard DOMINATED the denominator on 88% of them and left
    # the covariance tokens at ~1e-3 of unit scale (down to 5e-6) -- near-zero tokens the network
    # never saw in training, while the scale-invariant classical methods read the SAME instances at
    # ~1.2 deg. Dividing by the norm itself, substituting 1 only for an exactly-zero input, is
    # unit-Frobenius at every scale and keeps the model genuinely scale-invariant.
    nrm = torch.linalg.norm(z, dim=(1, 2), keepdim=True)
    return z / torch.where(nrm > 0, nrm, torch.ones_like(nrm))


class DoAFormer(ParentModel):
    """Transformer encoder/decoder with learnable source queries + a count head."""

    def __init__(self, system_model: SystemModel, d_model: int = 64, nhead: int = 4,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, criterion: str = "rmspe",
                 input_mode: str = "cov",
                 refine_min_sep_deg: float = 60.0,  # refine a source only if the nearest neighbour is
                                                   # farther than this. MUST exceed the array Rayleigh
                                                   # limit (~50 deg here) -- inside one beamwidth the
                                                   # beamscan cannot separate a pair and refining it
                                                   # destroys the coarse estimator's resolution.
                 refine_deg: float = 12.0,     # local ML refinement half-window (0/None disables).
                                               # A gridless regression head has no sub-grid refinement, so
                                               # its error FLOORS at ~1.07 deg regardless of SNR (measured
                                               # 5-40 dB) -- model-capacity bias, not noise. Refining
                                               # locally: 1.27 -> 0.44 deg (CRLB 0.39).
                 refine_grid: int = 2801):     # fine grid used only by that refinement
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
        self.refine_deg = refine_deg
        self.refine_min_sep_deg = refine_min_sep_deg
        if refine_deg:                              # fine dictionary used ONLY by the refinement stage
            lo, hi = system_model.params.doa_range
            g = np.deg2rad(np.linspace(float(lo), float(hi), int(refine_grid)))
            Af = np.asarray(system_model.steering_vec(g))
            # persistent=False: these are DERIVED from the system model, not learned. Keeping them
            # out of the state_dict means existing checkpoints still load strictly (a persistent
            # buffer here shows up as a MISSING key and trips the "partially random model" guard).
            self.register_buffer("refine_grid_rad", torch.as_tensor(g, dtype=torch.float64), persistent=False)
            self.register_buffer("refine_A", torch.as_tensor(Af, dtype=torch.complex128), persistent=False)

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
        # ---- Build the token sequence the transformer attends over ----
        # Complex data cannot be fed to a real-valued transformer, so every token is a row of
        # [Re | Im] concatenated over the N sensors. Two token sources, optionally combined.
        x = x.to(torch.complex64)
        toks = []

        # ---- 1. Covariance tokens: N tokens, one per sensor row of R ----
        # R is snapshot-order invariant and summarizes the spatial statistics compactly.
        if self.input_mode in ("cov", "both"):
            Rx = torch.einsum("bnt,bmt->bnm", x, x.conj()) / x.shape[-1]   # [B, N, N]
            # Per-sample unit-Frobenius normalization keeps the transformer stable.
            Rx = _unit_fro(Rx)
            toks.append(torch.cat((Rx.real, Rx.imag), dim=2))             # [B, N, 2N]
        # ---- 2. Snapshot tokens: T tokens, one per time sample ----
        if self.input_mode in ("snapshots", "both"):
            # Raw snapshots as T tokens of [Re, Im] over the N sensors — preserve the
            # per-snapshot phase that distinguishes front from back-lobe on a measured array.
            xs = x.transpose(1, 2)                                        # [B, T, N]
            xs = _unit_fro(xs)
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
        # ---- 1. How many sources to decode ----
        # One query per source, capped by the fixed query bank Q the model was built with.
        B = x.shape[0]
        M = int(sources_num) if sources_num is not None else self.Q
        M = max(1, min(M, self.Q))

        # ---- 2. Encode the measurement ----
        # Project tokens to d_model, add learned positional embeddings (the token order carries
        # sensor identity, which the array geometry depends on), then self-attend.
        tokens = self.input_norm(self.input_proj(self._tokens(x))) + self.pos_embed.unsqueeze(0)  # [B, N, d]
        memory = self.encoder(tokens)                                            # [B, N, d]

        # ---- 3. Decode one query per source ----
        # SET PREDICTION, as in DETR: the M learned queries cross-attend to the encoded measurement
        # and each emerges as one source. This is what makes the model GRIDLESS -- angles are
        # regressed directly, never picked off a spectrum.
        queries = self.query_embed[:M].unsqueeze(0).expand(B, -1, -1)            # [B, M, d]
        decoded = self.decoder(queries, memory)                                  # [B, M, d]

        # ---- 4. Heads ----
        # tanh * angle_scale bounds every prediction inside the field of view by construction, so
        # the model cannot emit an impossible angle. The count head reads the pooled memory.
        angles = torch.tanh(self.angle_head(decoded).squeeze(-1)) * self.angle_scale  # [B, M]
        count_logits = self.count_head(memory.mean(dim=1))                       # [B, Q]
        source_estimation = (count_logits.argmax(dim=1) + 1).float()             # [B]
        angles = angles.to(torch.float64)

        # ---- 5. Eval-only local ML refinement ----
        # A regression head has no sub-grid refinement of its own and floors at ~1.07 deg
        # regardless of SNR (model-capacity bias, not noise). Refining locally: 1.27 -> 0.44 deg.
        # Skipped while training so the graph stays end-to-end differentiable.
        if self.refine_deg and not self.training:   # eval-only: keep training end-to-end differentiable
            angles = local_ml_refine(x, angles, self.refine_A, self.refine_grid_rad,
                                     (self.refine_A.conj() * self.refine_A).sum(dim=0).real,
                                     float(np.deg2rad(self.refine_deg)),
                                     float(np.deg2rad(self.refine_min_sep_deg)))
        return angles, source_estimation, count_logits

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
        # ---- 1. Forward ----
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, source_estimation, count_logits = self(x, sources_num)

        # ---- 2. Angle loss ----
        # RMSPE is permutation-invariant (Hungarian-matched inside), which a SET predictor needs:
        # query k is not tied to source k, so an ordinary MSE would penalise correct-but-permuted
        # predictions.
        loss = self.criterion(doa_prediction, angles)

        # ---- 3. Count loss ----
        # Cross-entropy on the source count, trained jointly so the model can report how many
        # sources it believes are present rather than always emitting M.
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
