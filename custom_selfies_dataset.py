import torch
from torch.utils.data import Dataset

class CustomSELFIESDataset(Dataset):
    def __init__(self, encoded_selfies, label):
        self.input_feature = encoded_selfies
        self.label = label
    def __len__(self):
        return len(self.input_feature)
    def __getitem__(self, idx):
        x = torch.tensor(self.input_feature[idx])
        y = torch.tensor(self.label[idx]).type(torch.FloatTensor)
        return x, y