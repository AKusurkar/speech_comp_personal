from validate import ValidatePreds
from data_load import DataLoader
import pandas as pd
import nemo.collections.asr as nemo_asr
from adapter import AdapterLayer
import torch
import tarfile
import os

nemo_path = "/home/epochvipc1/Documents/Speech_comp_temp/custom_adapters_grad_epoch_16.nemo"
adapter_dim = 128

model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(nemo_path, strict=False)
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

with tarfile.open(nemo_path, "r:*") as tar:
    for file_name in tar.getnames():
        print(f" - {file_name}")

with tarfile.open(nemo_path, "r:*") as tar:
    tar.extract("./model_weights.ckpt", path=".")

weights = torch.load("model_weights.ckpt", map_location="cpu", weights_only=False)
model.load_state_dict(weights, strict=True)

os.remove("model_weights.ckpt")
model = model.cuda()
model.eval()

#Validate

# data_dir = "/home/epochvipc1/Documents/Speech_comp_temp/data/"
data_dir = "/home/epochvipc1/Documents/Speech_comp_temp/data/extra_jibl_data"
train_file = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/train_data_comb.jsonl"
# val_file =  "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/val_data_comb.jsonl"
val_file = "/home/epochvipc1/Documents/Speech_comp_temp/data/extra_jibl_data/manifest.jsonl"
save_preds = "/home/epochvipc1/Documents/Speech_comp_temp/save_scoring"
batch_size = 20

data_load = DataLoader(data_dir, train_file, val_file, batch_size)
train_data, val_data = data_load.load_data()

v = ValidatePreds(save_loc=save_preds, label_loc=val_file, model=model)
score, proper_score = v.val_score(val_data)
# print(score)