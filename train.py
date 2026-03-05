import nemo.collections.asr as nemo_asr
import lightning.pytorch as pyl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

logger_path = "/home/epochvipc1/Documents/Speech_comp_temp/tb_logs"
logger_name = "parakeet_custom_combined"
checkpoint_path = "/home/epochvipc1/Documents/Speech_comp_temp/model_checkpoints"

tb_logger = TensorBoardLogger(
    save_dir=logger_path,
    name=logger_name
)

checkpoint = ModelCheckpoint(
    dirpath=checkpoint_path,
    filename="latest_checkpoint",
    monitor="val_wer",
    mode="min",
    save_top_k=3,
    save_last=True
)

model_path = "/home/epochvipc1/Documents/speech_comp_pieter/childrens-speech-recognition-runtime/src/assets/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"

model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)

for param in model.encoder.parameters():
    param.requires_grad = False

try:
    conformer_layers = model.encoder.encoder.layers 
    num_layers = len(conformer_layers)
    layers_to_unfreeze = 4 
    
    for i in range(num_layers - layers_to_unfreeze, num_layers):
        for param in conformer_layers[i].parameters():
            param.requires_grad = True

except Exception as e:
    print(e)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable_params)

train_path = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/train_data_comb_nemo.jsonl"
val_path = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/val_data_comb_nemo.jsonl"
model_out = "/home/epochvipc1/Documents/Speech_comp_temp/models"

OmegaConf.set_struct(model.cfg, False)

model.cfg.train_ds.manifest_filepath = train_path
model.cfg.validation_ds.manifest_filepath = val_path
model.cfg.train_ds.batch_size = 2
model.cfg.validation_ds.batch_size = 2
model.cfg.train_ds.use_lhotse = False
model.cfg.validation_ds.use_lhotse = False

OmegaConf.set_struct(model.cfg, True)

model.setup_training_data(train_data_config=model.cfg.train_ds)
model.setup_validation_data(val_data_config=model.cfg.validation_ds)

trainer = pyl.Trainer(
    devices=2,
    accelerator='gpu',
    max_epochs=50,
    callbacks=[checkpoint],
    logger=tb_logger
)

trainer.fit(model)

model.save_to("/custom_model_comb.nemo")