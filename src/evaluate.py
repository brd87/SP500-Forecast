import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

# def windowed_eval(model:nn.Module, dataset, criterion, device, window_size=500):

#     model.eval()
#     losses = []

#     with torch.no_grad():

#         for start in range(0, len(dataset), window_size):
#             chunk = range(start, min(start + window_size, len(dataset)))

#             batch_loss = 0.0
#             count = 0

#             for i in chunk:
#                 x, y = dataset[i]

#                 x = x.unsqueeze(0).to(device)
#                 y = y.to(device).unsqueeze(0)

#                 logits = model(x)
#                 loss = criterion(logits, y)

#                 batch_loss += loss.item()
#                 count += 1

#             losses.append(batch_loss / max(count, 1))

#     return float(np.mean(losses))

def test(model:nn.Module, device:torch.device, test_loader:DataLoader, best_ckpt_path):
    #model.load_state_dict(torch.load(best_ckpt_path)["model_state"])
    checkpoint_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.float()

            logits = model(x)
            #prob = torch.sigmoid(logits).cpu().squeeze()
            prob = torch.sigmoid(logits).cpu().view(-1)
            pred = (prob > 0.5).float()

            all_probs.append(prob)
            all_preds.append(pred)
            all_targets.append(y)

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # ----------------- METRICS -----------------
    acc = (all_preds == all_targets).float().mean().item()
    auc = roc_auc_score(all_targets.numpy(), all_probs.numpy())
    
    baseline = max(all_targets.mean().item(), 1 - all_targets.mean().item())

    print("\n--- FINAL METRICS ---")
    print("Accuracy:", acc)
    print("Area Under the Curve:", auc)
    print("Baseline:", baseline)