import nemo.collections.asr as nemo_asr
import lightning.pytorch as pyl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from adapter import AdapterLayer
from custom_params_override import custom_params
import types

# Setup

logger_path = "/home/epochvipc1/Documents/Speech_comp_temp/tb_logs"
logger_name = "parakeet_adap_diff_grad"
checkpoint_path = "/home/epochvipc1/Documents/Speech_comp_temp/model_checkpoints/adap_diff_grad"
model_path = "/home/epochvipc1/Documents/speech_comp_pieter/childrens-speech-recognition-runtime/src/assets/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"
train_path = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/train_data_comb_nemo.jsonl"
val_path = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/val_data_comb_nemo.jsonl"
model_out = "/home/epochvipc1/Documents/Speech_comp_temp/models"
max_epochs = 50
num_gpus = 1
batch_size = 4
adapter_dim = 128

tb_logger = TensorBoardLogger(
    save_dir=logger_path,
    name=logger_name
)

checkpoint = ModelCheckpoint(

    dirpath=checkpoint_path,
    filename="adap_diff_grad_{epoch:02d}",
    monitor="val_wer",
    mode="min",
    save_top_k=-1,
    every_n_epochs=1,
    save_last=True
)

model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
d_model = model.cfg.encoder.d_model

# Step Params

for param in model.parameters():
    param.requires_grad = False

for i in range(len(model.encoder.layers)):
    original_layer = model.encoder.layers[i]

    model.encoder.layers[i] = AdapterLayer(
        original_layer=original_layer,
        d_model=d_model,
        adapter_dim=adapter_dim
    )

for name, param in model.named_parameters():
    if "adapter" in name:
        param.requires_grad = True

for param in model.decoder.parameters():
    param.requires_grad = True
for param in model.joint.parameters():
    param.requires_grad = True

print(model.summarize())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum (p.numel() for p in model.parameters())
print("Trainable:", trainable_params, "Total", total_params)

# Pass Params

OmegaConf.set_struct(model.cfg, False)

model.cfg.train_ds.manifest_filepath = train_path
model.cfg.validation_ds.manifest_filepath = val_path
model.cfg.train_ds.batch_size = batch_size
model.cfg.validation_ds.batch_size = batch_size
model.cfg.train_ds.use_lhotse = False
model.cfg.validation_ds.use_lhotse = False
model.cfg.train_ds.num_workers = 28
model.cfg.validation_ds.num_workers = 28
model.cfg.optim.sched.name = "CosineAnnealing"
model.cfg.optim.name = "adamw"
model.cfg.optim.sched.warmup_steps = 1000
model.cfg.optim.sched.min_lr = 0.0
model.cfg.optim.lr = 1e-4
model.cfg.optim.weight_decay = 1e-4
model.cfg.encoder.activations_checkpoint_method = "block"
model.cfg.encoder.activations_checkpoint_ratio = 1.0

if "spec_augment" in model.cfg:
    model.cfg.spec_augment.freq_masks = 2
    model.cfg.spec_augment.freq_width = 27
    model.cfg.spec_augment.time_masks = 2
    model.cfg.spec_augment.time_width = 0.05

model.cfg.train_ds.augmentor = {
    "shift": {
        "prob": 0.5, 
        "min_shift_ms": -5.0,
        "max_shift_ms": 5.0
    },
    "speed": {
        "prob": 0.5,
        "sr": 16000,
        "resample_type": "kaiser_fast",
        "min_speed_rate": 0.9,
        "max_speed_rate": 1.1
    }
}

OmegaConf.set_struct(model.cfg, True)

# Train

model.setup_optimizer_param_groups = types.MethodType(custom_params, model)
model.setup_training_data(train_data_config=model.cfg.train_ds)
model.setup_validation_data(val_data_config=model.cfg.validation_ds)


trainer = pyl.Trainer(
    devices=num_gpus,
    accelerator='gpu',
    max_epochs=max_epochs,
    callbacks=[checkpoint],
    logger=tb_logger,
    precision="bf16-mixed",
    # use_distributed_sampler=False,
    accumulate_grad_batches=8
)

trainer.fit(model)

model.save_to("adapters_differential_grad.nemo")