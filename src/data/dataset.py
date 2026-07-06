import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

class SP500Dataset(Dataset):

    def __init__(self, csv_path:pathlib.Path, lookback:int=60, target_col:str="gspc_target_up_5d"):
        
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        #df = df.dropna().reset_index(drop=True)

        self.feature_cols = [
            col for col in df.columns
            if col not in ["Date", target_col]
        ]

        self.target_col = target_col
        self.input_size = len(self.feature_cols)
        self.lookback = lookback

        self.x = df[self.feature_cols].values.astype(np.float32) #args
        self.y = df[target_col].values.astype(np.float32) #pred
        #"astype is just to make sure pytorch aint gonna blow up" ~ William Shakespeare
        self.indices = np.arange(self.lookback - 1, len(self.x))

        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x)

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx): # the first sample will be +60days

        prediction = self.indices[idx]
        end_idx = prediction+1

        x = self.x[end_idx - self.lookback : end_idx]
        y = self.y[prediction]

        return (torch.tensor(x), torch.tensor(y))

