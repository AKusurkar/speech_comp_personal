from validate import ValidatePreds
from data_load import DataLoader
import pandas as pd
import nemo.collections.asr as nemo_asr
from adapter import AdapterLayer
import torch

ckpt_path = "/home/epochvipc1/Documents/Speech_comp_temp/updated_ckpt/epoch_11.ckpt"
adapter_dim = 128

#Model
model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(ckpt_path, strict=False)
d_model = model.cfg.encoder.d_model

for i in range(len(model.encoder.layers)):
    original_layer = model.encoder.layers[i]

    model.encoder.layers[i] = AdapterLayer(
        original_layer=original_layer,
        d_model=d_model,
        adapter_dim=adapter_dim
    )

for param in model.parameters():
    param.requires_grad = False

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=True)
model = model.cuda()
model.eval()

print(model.summarize())

# #Save Model
model.save_to("custom_model_adapters.nemo")
print("Model Saved!")