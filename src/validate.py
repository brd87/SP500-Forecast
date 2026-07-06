import torch
from torch import nn
from torch.utils.data import DataLoader

def one_epoch(model:nn.Module, val_loader:DataLoader, criterion, device:torch.device):
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device).float()
            y = y.to(device).float().unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y)
            eval_loss += loss.item()

    eval_avg_loss = eval_loss / len(val_loader)
    return eval_avg_loss