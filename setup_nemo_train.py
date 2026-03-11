import json

train_file = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/train_data_comb.jsonl"
val_file = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/val_data_comb.jsonl"
data_path = "/home/epochvipc1/Documents/Speech_comp_temp/data"
save_loc = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined"

with open(train_file) as f:

    train_data = [json.loads(line) for line in f]

with open(val_file) as f:

    val_data = [json.loads(line) for line in f]

def create_file(data_lst, data_path, save_path, new_filename):

    full_path = save_path + '/' + new_filename

    with open(full_path, 'w') as f:

        for data in data_lst:

            new_dict = {
                "audio_filepath": data_path + "/" + data["audio_path"],
                "duration": data["audio_duration_sec"],
                "text": data["orthographic_text"]
            }

            f.write(json.dumps(new_dict) + "\n")

create_file(train_data, data_path, save_loc, "train_data_comb_nemo.jsonl")
create_file(val_data, data_path, save_loc, "val_data_comb_nemo.jsonl")

