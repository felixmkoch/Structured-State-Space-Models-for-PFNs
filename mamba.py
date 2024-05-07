from torch import nn
import math
import torch
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    '''
    Create Block function from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    '''

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    '''
    Initialize Weights function from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    '''

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias) 
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaModel(nn.Module):
    '''
    Actual full MAMBA Model used.
    '''

    def __init__(self,
                 d_model: int,
                 num_layers: int = 1,
                 ssm_config = None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_config = None,
                 fused_add_norm = False,
                 residual_in_fp32 = False,
                 device = "cpu",
                 dtype=None
                 ) -> None:

        super().__init__()
        self.model_type = 'mamba-ssm'

        factory_kwargs = {"device": device, "dtype": dtype}

        self.blocks = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_config,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs
                )
                for i in range(num_layers)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_config if initializer_config is not None else {}),
            )
        )


    def forward(self,
                src: tuple  # Inputs (src) have to be given as (x,y) or (style,x,y) tuple'
                ):
        
        if len(src) == 2: src = (None,) + src       # Check whether a style was given

        style_src, x_src, y_src = src               # Split input into style, train (x) and test (y) part.





        

        pass