import json

class DataLoader:

    def __init__(self, data_path, labels_train_path, labels_val_path, batch_size):

        self.data_path = data_path
        self.labels_train_path = labels_train_path
        self.labels_val_path = labels_val_path
        self.batch_size = batch_size

    def load_data(self):

        with open(self.labels_train_path) as f:

            labels_train_lst = [json.loads(line) for line in f]
        
        with open(self.labels_val_path) as f:

            labels_val_lst = [json.loads(line) for line in f]

        labels_train_path_lst = [{"path": self.data_path + "/" + ind["audio_path"], "metadata": ind} for ind in labels_train_lst]
        train_batched_data = [labels_train_path_lst[i:i+self.batch_size] for i in range(0, len(labels_train_path_lst), self.batch_size)] 

        labels_val_path_lst = [{"path": self.data_path + "/" + ind["audio_path"], "metadata": ind} for ind in labels_val_lst]
        val_batched_data = [labels_val_path_lst[i:i+self.batch_size] for i in range(0, len(labels_val_path_lst), self.batch_size)] 
        
        return train_batched_data, val_batched_data