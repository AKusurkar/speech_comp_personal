import nemo.collections.asr as nemo_asr
from adapter import AdapterLayer
import torch
from itertools import islice
import json
import os
from pathlib import Path
import tarfile

from loguru import logger
import torch
from tqdm import tqdm

BATCH_SIZE = 64
PROGRESS_STEP_DENOM = 100  # Update progress bar every 1 // PROGRESS_STEP_DENOM


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def main():
    # Diagnostics
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())
    logger.info("CUDA device count: {}", torch.cuda.device_count())

    # Load model
    src_root = Path(__file__).parent.resolve()
    nemo_path = src_root / "custom_model_adapters.nemo"
    logger.info(f"Loading model from: {nemo_path}")

    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(nemo_path, strict=False)
    d_model = model.cfg.encoder.d_model
    adapter_dim = 128

    for i in range(len(model.encoder.layers)):
        original_layer = model.encoder.layers[i]

        model.encoder.layers[i] = AdapterLayer(
            original_layer=original_layer,
            d_model=d_model,
            adapter_dim=adapter_dim
        )

    for param in model.parameters():
        param.requires_grad = False

    # with tarfile.open(nemo_path, "r:*") as tar:
    #     for file_name in tar.getnames():
    #         print(f" - {file_name}")

    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extract("./model_weights.ckpt", path=".")

    weights = torch.load("model_weights.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(weights, strict=True)

    os.remove("model_weights.ckpt")
    model = model.cuda()
    model.eval()

    # Load manifest and process data

    data_dir = Path("data")
    manifest_path = data_dir / "utterance_metadata.jsonl"

    # data_dir = Path("data")
    # manifest = Path("dicts_original_only")
    # manifest_path = manifest / "val_data.jsonl"

    with manifest_path.open("r") as fr:
        items = [json.loads(line) for line in fr]

    # Sort by audio duration for better batching
    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)

    logger.info(f"Processing {len(items)} utterances from {manifest_path}")

    step = max(1, len(items) // PROGRESS_STEP_DENOM)

    # Predict
    predictions = {}
    next_log = step
    processed = 0
    logger.info("Starting transcription...")
    with open(os.devnull, "w") as devnull:
        with tqdm(total=len(items), file=devnull) as pbar:
            for batch in batched(items, BATCH_SIZE):
                preds = model.transcribe(
                    # audio_path includes audio/ prefix
                    [str(data_dir / item["audio_path"]) for item in batch],
                    batch_size=len(batch),
                    verbose=False
                )
                for item, pred in zip(batch, preds):
                    predictions[item["utterance_id"]] = pred.text
                this_batch_size = len(batch)
                pbar.update(this_batch_size)
                processed += this_batch_size
                while processed >= next_log:
                    logger.info(str(pbar))
                    next_log += step

    logger.success("Transcription complete.")

    # Write submission file
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    logger.info(f"Writing submission file to {submission_path}")
    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["orthographic_text"] = predictions[item["utterance_id"]]
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    main()