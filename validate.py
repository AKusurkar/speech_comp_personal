from metric import score_wer_local, score_jsonl
import nemo.collections.asr as nemo_asr
import torch
from tqdm import tqdm
import json

class ValidatePreds:

    def __init__(self, save_loc, label_loc, model=None, model_load_from=None,):

        if model_load_from:
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_load_from)

        elif model:
            self.model = model

        else:

            print("Error: No Model Loaded")
        
        self.save_loc = save_loc 
        self.save_file = save_loc + "/saved_preds.jsonl"
        self.label_loc = label_loc

    def avg_lsts(self, lst):

        tot_num = 0
        tot_val = 0 

        for num, val in lst:

            tot_num += num
            tot_val += val
        
        return tot_val/tot_num

    def save_to_jsonl(self, insert_dict, filename, type):

        with open(filename, type) as f:
            f.write(json.dumps(insert_dict) + "\n")

    @torch.no_grad()
    def val_score(self, batched_val_data):

        score = 0
        long_lst = []
        short_lst = []

        for j in tqdm(range(len(batched_val_data)), desc="Batch Progress"):
            
            batch = batched_val_data[j]

            print(f'Starting Batch {j}')
            
            labels = [i["metadata"]["orthographic_text"] for i in batch] 
            length = [i["metadata"]["audio_duration_sec"] for i in batch]
            utterance = [i["metadata"]["utterance_id"] for i in batch]

            preds = self.model.transcribe([i["path"] for i in batch]) 
            
            score_batch = 0
            score_short = 0
            score_long = 0
            num_short = 0
            num_long = 0

            for i in range(len(labels)):

                if j == 0 and i == 0:
                    ty = 'w'
                else:
                    ty = 'a'

                lab = labels[i]
                pred = preds[i].text
                l = length[i]
                ut = utterance[i]

                dict_save = {"utterance_id": ut, "orthographic_text": pred}

                self.save_to_jsonl(dict_save, self.save_file, ty)

                s_raw = score_wer_local(lab, pred)
                score_batch += s_raw

                if float(l) <= 1.5:
                    score_short += s_raw
                    num_short += 1
                    short_lst.append((num_short, score_short))
                else:
                    score_long += s_raw
                    num_long += 1
                    long_lst.append((num_long, score_long))

            score_batch = score_batch/len(batch)

            if num_long == 0 or num_short == 0:
                
                print("Only long or short")
            
            else:
                score_short = score_short/num_short
                score_long = score_long/num_long

            score += score_batch

            print(f"Batch score: {score_batch} \n Short: {num_short}, {score_short} \n Long: {num_long}, {score_long}")
        
        score = score/len(batched_val_data)

        avg_short = self.avg_lsts(short_lst)
        avg_long = self.avg_lsts(long_lst)

        print("Short:", avg_short)
        print("Long:", avg_long)
        print("All:", score)

        proper_score = score_jsonl(self.save_file, self.label_loc, metric="wer")
        print("Actual Score:", proper_score)

        return score, proper_score