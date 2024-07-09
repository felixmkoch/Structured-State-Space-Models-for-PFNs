from torch import nn
import math
import torch
from functools import partial
from s4.models.s4.s4d import S4D
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    '''
    Function from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    '''
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class S4Model(nn.Module):
    '''
    S4 Model from the S4 Example.
    '''

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()


        dropout_fn = nn.Dropout

        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layers,
                **({}),
                n_residuals_per_layer=1
            )
        )
        

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)   # Don't need encoder as it is encoded before.

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    


class S4Model_Wrap(nn.Module):
    
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
        self.model_type = 's4-ssm'
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

        self.s4_backbone = S4Model(
            d_input=ninp,
            d_output=ninp,
            d_model=ninp,
            n_layers=num_layers,
            dropout=0.0
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

        if not style_src: style_src = torch.tensor([]).to(self.device) # To overcome the NoneType has no len() error.

        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]

        src = torch.cat([style_src, train_x, x_src[single_eval_pos:]], 0)

        print(f"Before: {src.shape}")

        # Shape here: [bptt, batch_size, emsize]
        src = src.permute(1, 0, 2)
        # Shape here: [batch_size, bptt, emsize]

        hidden_states = self.s4_backbone(src)

        hidden_states = hidden_states.permute(1, 0, 2)

        output = self.decoder(hidden_states)

        return output[single_eval_pos+len(style_src):]