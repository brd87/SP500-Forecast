import torch
from torch import nn

def load(path, model:nn.Module, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint.get("epoch", -1)
    loss = checkpoint.get("train_loss", None)

    print(f"Loaded checkpoint from epoch {epoch}, loss={loss}")

    return model, optimizer, epoch

def save(ckpt_path, model:nn.Model, epoch, avg_loss, input_size, optimizer):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "avg_loss": avg_loss,
        "input_size": input_size
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")