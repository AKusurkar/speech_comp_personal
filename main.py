from validate import ValidatePreds
from data_load import DataLoader
import pandas as pd

model_path = "/home/epochvipc1/Documents/speech_comp_pieter/childrens-speech-recognition-runtime/src/assets/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"
# model_path = "/home/epochvipc1/Documents/Speech_comp_temp/custom_model.nemo"

data_dir = "/home/epochvipc1/Documents/Speech_comp_temp/data/"
train_file = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/train_data_comb.jsonl"
val_file =  "/home/epochvipc1/Documents/Speech_comp_temp/dicts_original_only/val_data.jsonl"#"/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/val_data_comb.jsonl"
save_preds = "/home/epochvipc1/Documents/Speech_comp_temp/save_scoring"
label_loc = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_original_only/val_data.jsonl"
batch_size = 50

data_load = DataLoader(data_dir, train_file, val_file, batch_size)
train_data, val_data = data_load.load_data()

v = ValidatePreds(save_loc=save_preds, label_loc=label_loc, model_load_from=model_path)
score, proper_score = v.val_score(val_data)
# print(score)
