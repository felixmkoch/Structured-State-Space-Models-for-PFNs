from torch import nn

class MambaModel(nn.Module):

    def __init__(self):

        super().__init__()
        self.model_type = 'mamba-ssm'

        pass

    def forward(self,
                src: tuple  # Inputs (src) have to be given as (x,y) or (style,x,y) tuple'
                ):
        
        if len(src) == 2: src = (None,) + src       # Check whether a style was given

        style_src, x_src, y_src = src               # Split input into style, train (x) and test (y) part.

        

        pass