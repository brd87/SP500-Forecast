from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class SP500Dataset(Dataset):

    def __init__(self, csv_path:Path, lookback:int=60, target_col:str="gspc_target_up_5d"):
        
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        #df = df.dropna().reset_index(drop=True)

        self.feature_cols = [
            col for col in df.columns
            if col not in ["Date", target_col]
        ]

        self.target_col = target_col
        self.lookback = lookback
        self.valid_length = len(df) - lookback #minus becasue the first sample needs some space

        self.X = df[self.feature_cols].values.astype(np.float32) #arg
        self.y = df[target_col].values.astype(np.float32) #res
        #"astype is just to make sure pytorch aint gonna blow up" ~ William Shakespeare

    def __len__(self):

        return self.valid_length

    def __getitem__(self, idx): # the first sample will be +60days

        start = idx
        end = idx + self.lookback

        x = self.X[start:end]
        y = self.y[end - 1]

        return (torch.tensor(x), torch.tensor(y))
    
#sp500 = SP500Dataset(csv_path="D:/projectsGYM/SP500FC/SP500-Forecast/data/trainready/features10_1999-10-07_2026-06-05_id20260614134039.csv");
#print(sp500.__getitem__(0))

