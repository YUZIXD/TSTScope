"""
TSTScope Core Module
"""
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from ..base._base import CVAELatentsModelMixin
from ..trvae.losses import mse, nb
from .losses import tcrhsic, cm_loss
from .modules import Decoder, Encoder, ExtEncoder, MaskedLinearDecoder
from .utils import one_hot_encoder

class TSTScopeCore(nn.Module, CVAELatentsModelMixin):
    """
    ScArches model class. This class contains the implementation of Conditional Variational Auto-encoder
    for Gene Expression and TCR data.

    Args:
        input_gene_dim: Number of input features (i.e. gene in case of scRNA-seq).
        input_tcr_dim: Number of input TCR features.
        latent_dim: Bottleneck layer (z) size.
        mask: Tensor of 0s and 1s from utils.add_annotations to create VAE with a masked linear decoder.
        conditions: List of Condition names.
        hidden_layer_sizes: A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
        dr_rate: Dropout rate applied to all layers, if `dr_rate`=0 no dropout will be applied.
        recon_loss: Definition of Reconstruction-Loss-Method, 'mse' or 'nb'.
        use_l_encoder: If True and `decoder_last_layer`='softmax', library size encoder is used.
        use_bn: If `True` batch normalization will be applied to layers.
        use_ln: If `True` layer normalization will be applied to layers.
        decoder_last_layer: The last layer of the decoder. Must be 'softmax', 'identity', 'softplus', 'exp' or 'relu'.
        use_hsic: If True, use HSIC for disentanglement regularization of TCR nodes.
    """

    def __init__(
        self,
        input_gene_dim: int,
        input_tcr_dim: int,
        latent_dim: int,
        mask: torch.Tensor,
        conditions: List[str],
        hidden_layer_sizes: List[int] = [256, 256],
        dr_rate: float = 0.05,
        recon_loss: str = 'nb',
        use_l_encoder: bool = False,
        use_bn: bool = False,
        use_ln: bool = True,
        decoder_last_layer: Optional[str] = None,
        use_hsic: bool = False,
    ):
        super().__init__()

        self._validate_inputs(hidden_layer_sizes, latent_dim, conditions, recon_loss)

        print("\nINITIALIZING NEW NETWORK..............")

        self.input_gene_dim = input_gene_dim
        self.input_tcr_dim = input_tcr_dim
        self.latent_dim = latent_dim
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.condition_encoder = {k: v for v, k in enumerate(conditions)}
        self.recon_gene_loss = recon_loss
        self.freeze = False
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.mask = mask
        self.use_mmd = False
        
        # Not used in hard masking
        self.n_ext_decoder = 0
        self.n_ext_m_decoder = 0

        self.use_hsic = use_hsic

        self.decoder_last_layer = self._resolve_decoder_last_layer(decoder_last_layer, recon_loss)
        self.use_l_encoder = use_l_encoder
        self.dr_rate = dr_rate
        self.use_dr = dr_rate > 0

        # Theta network (dispersion parameters)
        if recon_loss == "nb":
            self.theta = nn.Parameter(torch.randn(self.input_gene_dim, self.n_conditions))
        else:
            self.theta = None

        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_tcr_dim = self.latent_dim
        self.cell_type_encoder = None # Placeholder

        # Build networks
        self.gene_encoder = self._build_gene_encoder()
        self.tcr_encoder = self._build_tcr_encoder()
        self.gene_decoder = self._build_gene_decoder()
        self.tcr_decoder = self._build_tcr_decoder()
        
        self.layer_norm = nn.LayerNorm(self.latent_dim)

        if self.use_l_encoder:
            self.l_encoder = self._build_library_size_encoder()

    def _validate_inputs(self, hidden_layer_sizes, latent_dim, conditions, recon_loss):
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in ["mse", "nb"], "'recon_loss' must be 'mse' or 'nb'"

    def _resolve_decoder_last_layer(self, decoder_last_layer: Optional[str], recon_loss: str) -> str:
        if decoder_last_layer is None:
            return 'softmax' if recon_loss == 'nb' else 'identity'
        return decoder_last_layer

    def _get_layer_sizes(self, input_dim: int) -> List[int]:
        sizes = self.hidden_layer_sizes.copy()
        sizes.insert(0, input_dim)
        return sizes
    
    def _get_decoder_layer_sizes(self, output_dim: int) -> List[int]:
        sizes = self.hidden_layer_sizes.copy()
        sizes.reverse()
        sizes.append(output_dim)
        return sizes

    def _build_gene_encoder(self) -> ExtEncoder:
        return ExtEncoder(
            layer_sizes=self._get_layer_sizes(self.input_gene_dim),
            latent_dim=self.latent_dim,
            use_bn=self.use_bn,
            use_ln=self.use_ln,
            use_dr=self.use_dr,
            dr_rate=self.dr_rate,
            num_classes=None,
            n_expand=0
        )

    def _build_tcr_encoder(self) -> Encoder:
        return Encoder(
            layer_sizes=self._get_layer_sizes(self.input_tcr_dim),
            latent_dim=self.latent_tcr_dim,
            use_bn=self.use_bn,
            use_ln=self.use_ln,
            use_dr=self.use_dr,
            dr_rate=self.dr_rate,
            num_classes=None
        )

    def _build_gene_decoder(self) -> MaskedLinearDecoder:
        return MaskedLinearDecoder(
            in_dim=self.latent_dim,
            out_dim=self.input_gene_dim,
            n_cond=self.n_conditions,
            mask=self.mask,
            recon_loss=self.recon_gene_loss,
            last_layer=self.decoder_last_layer
        )

    def _build_tcr_decoder(self) -> Decoder:
        return Decoder(
            layer_sizes=self._get_decoder_layer_sizes(self.input_tcr_dim),
            latent_dim=self.latent_tcr_dim,
            recon_loss="identity",
            use_bn=self.use_bn,
            use_ln=self.use_ln,
            use_dr=self.use_dr,
            dr_rate=self.dr_rate
        )

    def _build_library_size_encoder(self) -> ExtEncoder:
        return ExtEncoder(
            layer_sizes=[self.input_gene_dim, 128],
            latent_dim=1,
            use_bn=self.use_bn,
            use_ln=self.use_ln,
            use_dr=self.use_dr,
            dr_rate=self.dr_rate,
            num_classes=self.n_conditions
        )

    def forward(
        self, 
        x: Optional[torch.Tensor] = None, 
        tcremb: Optional[torch.Tensor] = None, 
        tcrlabel: Optional[torch.Tensor] = None, 
        batch: Optional[torch.Tensor] = None, 
        sizefactor: Optional[torch.Tensor] = None, 
        labeled: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Ensure inputs are present
        if x is None or tcremb is None:
            raise ValueError("x and tcremb are required for forward pass")

        x_log = torch.log(1 + x)
        if self.recon_gene_loss == 'mse':
            x_log = x

        # Embeddings
        z1_mean_gene, z1_log_var_gene = self.gene_encoder(x_log)
        z2_mean_tcr, z2_log_var_tcr = self.tcr_encoder(tcremb)

        # Sampling & Reconstruction
        z_out_gene = self.sampling(z1_mean_gene, z1_log_var_gene)
        gene_outputs = self.gene_decoder(z_out_gene, batch)

        z_out_tcr = self.sampling(z2_mean_tcr, z2_log_var_tcr)
        tcr_outputs = self.tcr_decoder(z_out_tcr)

        # Losses
        tcr_recon_loss = self._compute_tcr_recon_loss(tcr_outputs, tcremb)
        align_loss = cm_loss(z1_mean_gene, z2_mean_tcr, tcrlabel)
        tcr_hsic_loss = self._compute_hsic_loss(z2_mean_tcr, z_out_gene.device)
        gene_recon_loss = self._compute_gene_recon_loss(gene_outputs, x, x_log, batch, sizefactor)
        kl_div = self._compute_kl_loss(z1_mean_gene, z1_log_var_gene, z2_mean_tcr, z2_log_var_tcr)

        return gene_recon_loss, tcr_recon_loss, tcr_hsic_loss, kl_div, align_loss

    def _compute_tcr_recon_loss(self, tcr_outputs, tcremb):
        recon_tcr, _ = tcr_outputs
        return mse(recon_tcr, tcremb).sum(-1).mean()

    def _compute_hsic_loss(self, z2_mean_tcr, device):
        if self.use_hsic:
            return tcrhsic(z2_mean_tcr)
        return torch.tensor(0.0, device=device)

    def _compute_gene_recon_loss(self, gene_outputs, x, x_log, batch, sizefactor):
        if self.recon_gene_loss == "mse":
            recon_x, _ = gene_outputs
            return mse(recon_x, x_log).sum(dim=-1).mean()
        
        elif self.recon_gene_loss == "nb":
            if self.use_l_encoder and self.decoder_last_layer == "softmax":
                sizefactor = torch.exp(self.sampling(*self.l_encoder(x_log, batch))).flatten()
            
            dec_mean_gamma, _ = gene_outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            
            if self.decoder_last_layer == "softmax":
                dec_mean = dec_mean_gamma * size_factor_view
            else:
                dec_mean = dec_mean_gamma
                
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            return -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
        
        return torch.tensor(0.0)

    def _compute_kl_loss(self, z1_mean, z1_log_var, z2_mean, z2_log_var):
        def _kl(mean, log_var):
            var = torch.exp(log_var) + 1e-4
            return kl_divergence(
                Normal(mean, torch.sqrt(var)),
                Normal(torch.zeros_like(mean), torch.ones_like(var))
            ).sum(dim=1).mean()
            
        return _kl(z1_mean, z1_log_var) + _kl(z2_mean, z2_log_var)
