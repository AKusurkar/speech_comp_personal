import json
import glob 
import nemo.collections.asr as nemo_asr

# data_files = glob.glob("/home/epochvipc1/Documents/Speech_comp_temp/data/audio" + "/*.flac")
# print(len(data_files))

train_file = "/home/epochvipc1/Documents/Speech_comp_temp/data/train_word_transcripts.jsonl"
data_dir = "/home/epochvipc1/Documents/Speech_comp_temp/data/"
model_path = "/home/epochvipc1/Documents/speech_comp_pieter/childrens-speech-recognition-runtime/src/assets/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"

with open(train_file) as f:

    train_data = [json.loads(line) for line in f]

# print(len(train_data))

batch1 = train_data[:50]
# print(len(batch1))

asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)

output = asr_model.transcribe(
    [data_dir + ind["audio_path"] for ind in batch1],
    batch_size=len(batch1)
)

for m in output:

    print(m.text)

