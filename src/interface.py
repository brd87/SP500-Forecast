import torch
from torch import nn
from torch.utils.data import DataLoader

def run_epoch(model:nn.Module, device:torch.device, loader:DataLoader, criterion, optimizer=None):
    training = optimizer is not None
    if training: 
        model.train()
        context = torch.enable_grad()
    else: 
        model.eval()
        context = torch.no_grad()

    total_loss = 0.0
    all_logits = []
    all_targets = []
    
    with context:
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float().unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    result = {
        "avg_loss": avg_loss,
        "logits": all_logits,
        "targets": all_targets
        }
    
    return result