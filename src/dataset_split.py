import os

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
import joblib

import config

def __make_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, eval_loader, test_loader


# def __time_split(dataset, train_ratio, val_ratio):
#     n = len(dataset)

#     train_end = int(n * train_ratio)
#     val_end = int(n * (train_ratio + val_ratio))

#     train_indices = list(range(0, train_end))
#     val_indices = list(range(train_end, val_end))
#     test_indices = list(range(val_end, n))

#     return train_indices, val_indices, test_indices

def __time_split(dataset):
    n = len(dataset)

    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, n))

    return train_indices, val_indices, test_indices

def __scaling(dataset:Dataset, train_indices, val_indices, test_indices, scaler_path):

    scaler = StandardScaler()
    scaler.fit(dataset.x[train_indices])
    joblib.dump(scaler, scaler_path)

    dataset.x[train_indices] = scaler.transform(
        dataset.x[train_indices]
    )
    dataset.x[val_indices] = scaler.transform(
        dataset.x[val_indices]
    )
    dataset.x[test_indices] = scaler.transform(
        dataset.x[test_indices]
    )

    return dataset

def subset(dataset:Dataset, scaler_path):
    dataset.x = dataset.x.copy()
    train_indices, val_indices, test_indices = __time_split(dataset)

    dataset = __scaling(dataset, train_indices, val_indices, test_indices, scaler_path)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader, val_loader, test_loader = __make_loaders(train_dataset, val_dataset, test_dataset)

    return train_loader, val_loader, test_loader