import glob
from tqdm import tqdm
import librosa
import soundfile as sf

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


if __name__ == "__main__":

    data_dir = "/home/epochvipc1/Documents/Speech_comp_temp/data/audio_extra"
    save_path = "/home/epochvipc1/Documents/Speech_comp_temp/data/audio"

    preproc = PreprocessData(data_dir, save_path)
    preproc.preprocess()