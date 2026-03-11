import torch
import os

# 1. Paths (Update these!)
ckpt_path = "/home/epochvipc1/Documents/Speech_comp_temp/model_checkpoints/custom_model_adapter/adapter_comb_checkpoint_cont_epoch=11.ckpt"
patched_ckpt_path = "/home/epochvipc1/Documents/Speech_comp_temp/updated_ckpt/epoch_11.ckpt"
tokenizer_dir = "/home/epochvipc1/Documents/Speech_comp_temp/tokenizer_files"

# 2. Load the checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
hp = ckpt['hyper_parameters']

# 3. Locate the tokenizer configuration block
if 'cfg' in hp and 'tokenizer' in hp['cfg']:
    tokenizer_cfg = hp['cfg']['tokenizer']
elif 'model' in hp and 'tokenizer' in hp['model']:
    tokenizer_cfg = hp['model']['tokenizer']
elif 'tokenizer' in hp:
    tokenizer_cfg = hp['tokenizer']
else:
    raise KeyError("Could not find the tokenizer block.")

# 4. The Brute Force Patch: Ignore hashes, hardcode the exact plain filenames
print("Overriding all tokenizer paths...")

tokenizer_cfg['dir'] = tokenizer_dir
tokenizer_cfg['model_path'] = os.path.join(tokenizer_dir, "tokenizer.model")
tokenizer_cfg['vocab_path'] = os.path.join(tokenizer_dir, "tokenizer.vocab")

# Also catch the spe_tokenizer_vocab if it exists
if 'spe_tokenizer_vocab' in tokenizer_cfg:
    tokenizer_cfg['spe_tokenizer_vocab'] = os.path.join(tokenizer_dir, "tokenizer.vocab")

# Clean up any remaining rogue keys starting with 'nemo:' just in case
for key, value in list(tokenizer_cfg.items()):
    if isinstance(value, str) and value.startswith("nemo:"):
        if "vocab" in value:
            tokenizer_cfg[key] = os.path.join(tokenizer_dir, "tokenizer.vocab")
        elif "model" in value:
            tokenizer_cfg[key] = os.path.join(tokenizer_dir, "tokenizer.model")

# 5. Save the patched checkpoint
torch.save(ckpt, patched_ckpt_path)
print("Checkpoint successfully brute-forced!")