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


class MambaBackbone(nn.Module):
    '''
    Backbone blocks of the MAMBA Model.
    '''

    def __init__(self,
                ninp: int,
                num_layers: int,
                ssm_config=None,
                norm_epsilon: float = 1e-5,
                rms_norm: bool = False,
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                device=None,
                dtype=None,
                 ):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.blocks = nn.ModuleList(
            [
                create_block(
                    ninp,
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
            ninp, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def forward(self,
                x,
                inference_parameters
                ):
            
        hidden_states = x

        residual = None

        for block in self.blocks: hidden_states, residual = block(hidden_states, residual, inference_params=inference_parameters)

        return hidden_states


class MambaModel(nn.Module):
    '''
    Actual full MAMBA Model used.
    '''

    def __init__(self,
                 encoder,
                 n_out,
                 ninp,
                 nhid,
                 num_layers: int = 1,
                 ssm_config = None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 y_encoder=None,
                 initializer_config = None,
                 fused_add_norm = False,
                 residual_in_fp32 = False,
                 device = "cpu",
                 dtype=None
                 ) -> None:

        super().__init__()
        self.model_type = 'mamba-ssm'
        self.encoder = encoder
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.ssm_config = ssm_config
        self.rms_norm = rms_norm
        self.y_encoder = y_encoder
        self.initializer_config = initializer_config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        factory_kwargs = {"device": device, "dtype": dtype}

        self.mamba_backbone = MambaBackbone(
            ninp=ninp,
            num_layers=self.num_layers,
            ssm_config=self.ssm_config,
            norm_epsilon=1e-5,
            rms_norm=False,     # Doesn't work with true yet.
            initializer_cfg=self.initializer_config,
            fused_add_norm=self.fused_add_norm,
            residual_in_fp32=self.residual_in_fp32,
            device=self.device,
            dtype=self.dtype
        )

        self.decoder = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            ninp, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_config if initializer_config is not None else {}),
            )
        )


    def forward(self,
                src: tuple,  # Inputs (src) have to be given as (x,y) or (style,x,y) tuple'
                single_eval_pos: int
                ):
        
        if len(src) == 2: src = (None,) + src       # Check whether a style was given

        style_src, x_src, y_src = src               # Split input into style, train (x) and test (y) part.

        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)

        hidden_states = self.mamba_backbone(x_src, inference_parameters=None)

        #print(f"Hidden States before decoder: {hidden_states}")

        output = self.decoder(hidden_states)
        
        #print(f"Hidden States after decoder: {output}")

        #output = output + 0.5

        #print(f"Hidden States after Add operation: {output}")

        if not style_src: style_src = [] # To overcome the NoneType has no len() error.

        return output[single_eval_pos+len(style_src):]
