import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data.dataset import SP500Dataset
from data.download import RawData
from data.processing import DataProcessing
from src.models.lstm import LSTMModel
from torch.utils.tensorboard import SummaryWriter

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint.get("epoch", -1)
    loss = checkpoint.get("train_loss", None)

    print(f"Loaded checkpoint from epoch {epoch}, loss={loss}")

    return model, optimizer, epoch

# ----------------- CONFIG -----------------
CKPT_DIR = "checkpoints"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.2

print("START")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs(CKPT_DIR, exist_ok=True)

writer = SummaryWriter("runs/exp1") # tensorboard --logdir=runs

# ----------------- DATA -----------------
rawdata = RawData()
dataprocessing = DataProcessing(data_df=rawdata.data)
#dataprocessing = DataProcessing(csv_path=rawdata.save_path)
dataset = SP500Dataset(dataprocessing.save_path)

input_size = dataset.input_size

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------- MODEL -----------------
model = LSTMModel(input_size=input_size).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

sample_x, _ = next(iter(train_loader))
writer.add_graph(model, sample_x.to(device).float())

# ----------------- CHECKPOINT PREP -----------------
ckpt_path = os.path.join(CKPT_DIR, "last.pt")
best_ckpt_path = os.path.join(CKPT_DIR, "best.pt")
best_loss = float("inf")
start_epoch = 0
if os.path.exists(ckpt_path):
    model, optimizer, last_epoch = load_checkpoint(ckpt_path, model, optimizer, device)
    start_epoch = last_epoch + 1

# ----------------- THE LOOP -----------------
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device).float()
        y = y.to(device).float().unsqueeze(1)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_avg_loss = train_loss / len(train_loader)

    #save
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "input_size": input_size
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    if train_loss < best_loss:
        best_loss = train_loss
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss
        }, best_ckpt_path)

    #evaluation
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

    #log
    writer.add_scalar("Loss/train", train_avg_loss, epoch)
    writer.add_scalar("Loss/val", eval_loss, epoch)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss: {train_loss:.6f} | val_loss: {eval_loss:.6f}")
writer.close()

# ----------------- EVAL -----------------
model.eval()
with torch.no_grad():
    x, y = next(iter(eval_loader))
    x = x.float().to(device)

    logits = model(x)
    prob = torch.sigmoid(logits)
    preds = (prob > 0.5).float()

print("Sample preds:", preds[:10].cpu().numpy())