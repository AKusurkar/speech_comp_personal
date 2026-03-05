# # import nemo.collections.asr as nemo_asr
import json

# # file = "/home/epochvipc1/Documents/Speech_comp_temp/data/audio_part_0/audio/U_0a00a59928a27e26.flac"

# # asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

# # output = asr_model.transcribe([file], verbose=False)

# # print(output[0].text)

# train_labels_file = "/home/epochvipc1/Documents/Speech_comp_temp/data/train_word_transcripts.jsonl"
# train_labels = []

# with open(train_labels_file) as f:

#     for line in f:

#         train_labels.append(json.loads(line))

# ut_ids_train = []

# for ind in train_labels:

#     ut_ids_train.append(ind['utterance_id'])

# smoke_file = "/home/epochvipc1/Documents/Speech_comp_temp/data/submission_format_z2HCh3r_SMOKE.jsonl"

# smoke_labels = []

# with open(smoke_file) as f:

#     for line in f:

#         smoke_labels.append(json.loads(line))

# test_file = "/home/epochvipc1/Documents/Speech_comp_temp/data/submission_format_aqPHQ8m_TEST.jsonl"
# test_labels = []

# with open(test_file) as f:

#     for line in f:

#         test_labels.append(json.loads(line))

# ut_ids_test = []

# for ind in test_labels:

#     ut_ids_test.append(ind["utterance_id"])

# count_train = 0 
# count_test = 0

# for ind in smoke_labels:

#     ut_sm = ind["utterance_id"]

#     if ut_sm in ut_ids_train:

#         count_train +=1

#     elif ut_sm in ut_ids_test:

#         count_test += 1

# print(count_train, count_test)

# l = [(1,4), (5, 8.5), (9, 3.6)]

# def avg_lsts(lst):

#     tot_num = 0
#     tot_val = 0 

#     for num, val in lst:

#         tot_num += num
#         tot_val += val
    
#     return tot_val/tot_num

# m = avg_lsts(l)

# print(m)

with open("/home/epochvipc1/Documents/Speech_comp_temp/data/train_word_transcripts.jsonl") as f:

    original = [json.loads(i) for i in f]

with open("/home/epochvipc1/Documents/Speech_comp_temp/data/train_word_transcripts_talkbank.jsonl") as f:

    new = [json.loads(i) for i in f]

# comb = original + new

# with open("train_word_transcripts_combined.jsonl", 'w') as f:

#     for item in comb:
#         f.write(json.dumps(item) + "\n")

import glob

files = glob.glob("/home/epochvipc1/Documents/Speech_comp_temp/data/audio" + "/*.flac")
filenames = [i.split('/')[-1].split('.')[0] for i in files]

for i in new:

    if i['utterance_id'] not in filenames:
        print(i)

