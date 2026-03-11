import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class SplitData:
    
    def __init__(self, original_train_file, save_loc):

        self.original_train_file = original_train_file
        self.save_loc = save_loc

    def examine_str(self, inp_str):

        filler_words = ["hmm", "mm", "mhm", "mmm", "uh", "um"]

        inp_str_lst = inp_str.lower().strip().split(" ")

        if all(i in filler_words for i in inp_str_lst):

            return True   

        return False

    def drop_filler_words(self, df):

        df['drop'] = df["orthographic_text"].map(self.examine_str)
        df = df[df['drop'] == False]

        df = df.drop(columns=["drop"])

        return df

    def set_max_time(self, df, max_time):

        df = df[df["audio_duration_sec"] < max_time]

        return df
    
    def drop_age_bucket(self, df, drop_bucket:str):

        df = df[df["age_bucket"] != drop_bucket]
        return df
    
    def drop_specific_file(self, df, id):

        df = df[df["utterance_id"] != id]
        return df        

    def train_val_split(self, df, test_size=0.2):

        bins = [0, 0.61, 1.05, 1.89, 3.29, 6.52, 12.51, float('inf')]
        val_proportions = [0.05, 0.20, 0.25, 0.25, 0.20, 0.04, 0.01]

        df["dur_bin"] = pd.cut(df["audio_duration_sec"], bins=bins, labels=False, include_lowest=True)

        total_val_size = int(len(df) * test_size)
        target_counts = np.array([int(total_val_size * p) for p in val_proportions])
        current_counts = np.zeros(len(val_proportions))

        child_groups = df.groupby('child_id')
        child_bin_counts = {}
        
        for child_id, group in child_groups:
            counts = group['dur_bin'].value_counts()
            bin_arr = np.zeros(len(val_proportions))
            for b_idx, count in counts.items():
                bin_arr[int(b_idx)] = count
            child_bin_counts[child_id] = bin_arr

        val_children = set()
        available_children = list(child_bin_counts.keys())
        
        np.random.seed(42)
        np.random.shuffle(available_children)
        
        while current_counts.sum() < total_val_size and available_children:

            deficit = target_counts - current_counts
            deficit[deficit < 0] = 0  

            best_child = None
            best_score = -float('inf')

            for child in available_children:
                counts = child_bin_counts[child]
                
                overflow = (current_counts + counts) - target_counts
                overflow_penalty = np.sum(overflow[overflow > 0]) * 2 
                
                useful_fill = np.minimum(counts, deficit)
                score = np.sum(useful_fill) - overflow_penalty

                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                print("Failed to find best child")
                break

            val_children.add(best_child)
            current_counts += child_bin_counts[best_child]
            available_children.remove(best_child)

        val_df = df[df['child_id'].isin(val_children)].copy()
        train_df = df[~df['child_id'].isin(val_children)].copy()

        train_df = train_df.drop(columns=["dur_bin"])
        val_df = val_df.drop(columns=["dur_bin"])
        
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return train_df, val_df

    def save_to_jsonl(self, insert_lst, filename):

        with open(filename, 'w') as f:

            for item in insert_lst:
                f.write(json.dumps(item) + "\n")

    def generate_split(self):

        with open(self.original_train_file) as f:

            train_data = [json.loads(line) for line in f]

        df = pd.DataFrame(train_data)

        df = self.drop_filler_words(df)
        df = self.drop_age_bucket(df, "12+")
        df = self.drop_specific_file(df, "U_b8a4e8220e65219b")
        df = self.set_max_time(df, 60)

        train_df, val_df = self.train_val_split(df, 0.2)

        self.train_lst = train_df.to_dict(orient='records')
        self.val_lst = val_df.to_dict(orient='records')

    def __len__(self):

        self.generate_split()
        
        return len(self.val_lst)

    def save(self):

        self.generate_split()

        self.save_to_jsonl(self.train_lst, self.save_loc + '/train_data_comb.jsonl')
        self.save_to_jsonl(self.val_lst, self.save_loc + "/val_data_comb.jsonl")


if __name__ == "__main__":
 
    train_file = "/home/epochvipc1/Documents/Speech_comp_temp/data/train_word_transcripts_combined.jsonl"
    save_loc = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined"
    splitter = SplitData(train_file, save_loc)

    print(len(splitter))
    splitter.save()