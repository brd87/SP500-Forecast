import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

def test(model:nn.Module, device:torch.device, test_loader:DataLoader, criterion, best_ckpt_path):
    #model.load_state_dict(torch.load(best_ckpt_path)["model_state"])
    checkpoint_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.float()

            logits = model(x)
            #prob = torch.sigmoid(logits).cpu().squeeze()
            prob = torch.sigmoid(logits).cpu().view(-1)
            pred = (prob > 0.5).float()
            loss = criterion(logits.squeeze(), y.to(device))
            total_loss += loss.item() * len(y)

            all_probs.append(prob)
            all_preds.append(pred)
            all_targets.append(y)

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    avg_loss = total_loss / len(test_loader.dataset)

    # ----------------- METRICS -----------------
    acc = (all_preds == all_targets).float().mean().item()
    auc = roc_auc_score(all_targets.numpy(), all_probs.numpy())
    
    baseline = max(all_targets.mean().item(), 1 - all_targets.mean().item())

    print("\n--- FINAL METRICS ---")
    print("Accuracy:", acc)
    print("Area Under the Curve:", auc)
    print("Baseline:", baseline)
    print("Total Avg Loss: ", avg_loss)
    