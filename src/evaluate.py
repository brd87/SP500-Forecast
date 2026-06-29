import torch
from torch import nn

def evaluate_one_epoch(model:nn.Module, eval_loader, criterion, device):
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(device).float()
            y = y.to(device).float().unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y)
            eval_loss += loss.item()

    eval_avg_loss = eval_loss / len(eval_loader)
    return eval_avg_loss