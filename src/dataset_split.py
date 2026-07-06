import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

def __make_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loader, test_loader


def __time_split(dataset, train_ratio, val_ratio):
    n = len(dataset)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, n))

    return train_indices, val_indices, test_indices


def subset(dataset:Dataset, train_ratio, val_ratio, batch_size):
    train_indices, val_indices, test_indices = __time_split(dataset, train_ratio, val_ratio)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader, val_loader, test_loader = __make_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    return train_loader, val_loader, test_loader


