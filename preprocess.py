import glob
from tqdm import tqdm
import librosa
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import random

class PreprocessData:

    def __init__(self, data_path, save_path):

        self.data_path = data_path
        self.save_path = save_path

    def preprocess(self):

        data_lst = glob.glob(self.data_path + "/*.flac")
        
        for data in tqdm(data_lst):
        
            try:
            
                data_save_name = data.split("/")[-1]
                
                #Normalize to 16 kHz and convert to mono
                audio_arr, samp_rate = librosa.load(data, sr=16000, mono=True)
                sf.write(self.save_path + "/" + data_save_name, audio_arr, samp_rate)
            
            except Exception as e:

                print(e)
                print("Error with file:", data)

    def apply_pitch_shift(waveform, sample_rate=16000, max_steps=4):
        
        n_steps = random.randint(-max_steps, max_steps)
        
        if n_steps == 0:
            n_steps = 1

        shifted = F.pitch_shift(waveform, sample_rate, n_steps)
        
        return shifted
    
    def apply_time_shift(waveform, shift_limit = 0.1):

        _, num_samples = waveform.shape

        shift_amt = int(random.uniform(-shift_limit, shift_limit) * num_samples)
        shifted = torch.roll(waveform, shifts=shift_amt, dims=1)

        return shifted
    
    def apply_frequency_masking(spectogram, max_mask_pct=0.15):

        n_mels = spectogram.shape[1]
        mask_param = int(n_mels * max_mask_pct)

        freq_mask = T.FrequencyMasking(freq_mask_param=mask_param)
        masked = freq_mask(spectogram)

        return masked

    # def augment(self, json_inp):


if __name__ == "__main__":

    data_dir = "/home/epochvipc1/Documents/Speech_comp_temp/data/audio_extra"
    save_path = "/home/epochvipc1/Documents/Speech_comp_temp/data/audio"

    preproc = PreprocessData(data_dir, save_path)
    # preproc.preprocess()