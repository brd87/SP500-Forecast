import os
import torch
from torch import nn

import config

def last_and_best(model, optimizer, input_size, device, zero_epoch=True):
    model_nameid = model.nameid
    ckpt_dir_path = f"checkpoints/{model_nameid}/{config.EXPERIMENT_NAME}"
    os.makedirs(ckpt_dir_path, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir_path, "last.pt")
    best_ckpt_path = os.path.join(ckpt_dir_path, "best.pt")
    scaler_path = os.path.join(ckpt_dir_path, "scaler.pkl")

    best_valid_avg_loss = float("inf")
    start_epoch = 0

    if os.path.exists(best_ckpt_path):
        best_checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)

        if model_nameid == best_checkpoint["model_nameid"] and input_size == best_checkpoint["input_size"]:
            best_valid_avg_loss = best_checkpoint.get("valid_loss", float("inf"))
        else:
            raise Exception('ERROR: model_nameid and/or input_size are mismatched with the (BEST) saved checkpoint')
        
        print(f"Loaded best checkpoint metadata: epoch={best_checkpoint['epoch']}, val_loss={best_valid_avg_loss}")

    if (not zero_epoch) and os.path.exists(ckpt_path):
        last_model_nameid, model, optimizer, last_epoch, last_input_size, scaler_path = load(
          ckpt_path, model, optimizer, device
        )

        if model_nameid == last_model_nameid and input_size == last_input_size:
            start_epoch = last_epoch + 1
        else:
            raise Exception('ERROR: model_nameid and/or input_size are mismatched with the (LAST) saved checkpoint')
    
    return ckpt_path, best_ckpt_path, scaler_path, best_valid_avg_loss, start_epoch, model, optimizer


def load(path, model:nn.Module, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    model_nameid = checkpoint["model_nameid"]
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    input_size = checkpoint["input_size"]
    scaler_path = checkpoint["scaler_path"]
    print(f"Loaded checkpoint from epoch {epoch}, train_loss={train_loss}, valid_loss={valid_loss}")

    return model_nameid, model, optimizer, epoch, input_size, scaler_path


def save(ckpt_path, model:nn.Model, epoch, train_loss, valid_loss, input_size, optimizer, scaler_path):
    torch.save({
        "epoch": epoch,
        "model_nameid": model.nameid,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "input_size": input_size,
        "scaler_path": scaler_path
    }, ckpt_path)
    #print(f"Saved checkpoint: {ckpt_path}")