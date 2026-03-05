import json
import pandas as pd
from sklearn.model_selection import train_test_split

class SplitData:
    
    def __init__(self, original_train_file):

        self.original_train_file = original_train_file

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
        #S_6438aac83bbbd772
        df = df[df["utterance_id"] != id]
        return df        
    
    def train_val_split(self, df, num_buckets):

        df["dur_bin"] = pd.qcut(df["audio_duration_sec"], q=num_buckets, labels=False, duplicates="drop")

        df["stratify_key"] = df["age_bucket"].astype(str) + "_" + df["dur_bin"].astype(str)

        train_df, val_df = train_test_split(
            df, 
            test_size=0.2,
            stratify=df["stratify_key"],
            random_state=42
        )

        train_df = train_df.drop(columns=["stratify_key", "dur_bin"])
        val_df = val_df.drop(columns=["stratify_key", "dur_bin"])

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

        train_df, val_df = self.train_val_split(df, 10)

        self.train_lst = train_df.to_dict(orient='records')
        self.val_lst = val_df.to_dict(orient='records')

    def __len__(self):

        self.generate_split()
        
        return len(self.val_lst)

    def save(self):

        self.generate_split()

        self.save_to_jsonl(self.train_lst, 'train_data_comb.jsonl')
        self.save_to_jsonl(self.val_lst, "val_data_comb.jsonl")


if __name__ == "__main__":
 
    train_file = "/home/epochvipc1/Documents/Speech_comp_temp/data/train_word_transcripts_combined.jsonl"
    splitter = SplitData(train_file)

    print(len(splitter))
    splitter.save()