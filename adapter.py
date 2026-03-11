import torch 
import torch.nn as nn

class AdapterLayer(nn.Module):

    def __init__(self, original_layer, d_model, adapter_dim=64):
        super().__init__()

        self.original_layer = original_layer

        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, d_model)
        )

        nn.init.zeros_(self.adapter[3].weight)
        nn.init.zeros_(self.adapter[3].bias)
    
    def forward(self, *args, **kwargs):

        out = self.original_layer(*args, **kwargs)
        
        if isinstance(out, tuple):
            x = out[0]
            rest = out[1:]

            x_adapted = x + self.adapter(x)

            return (x_adapted,) + rest

        else:
            return out + self.adapter(out) 