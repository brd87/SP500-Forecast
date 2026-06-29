import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data.dataset import SP500Dataset
from data.download import RawData
from data.processing import DataProcessing
from src.models.lstm import LSTMModel
from torch.utils.tensorboard import SummaryWriter

import config
import evaluate
import checkpoint
import train

def main():
    print("START")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------- DATA -----------------
    rawdata = RawData()
    dataprocessing = DataProcessing(data_df=rawdata.data)
    #dataprocessing = DataProcessing(csv_path=rawdata.save_path)
    dataset = SP500Dataset(dataprocessing.save_trainready_path)

    input_size = dataset.input_size

    val_size = int(len(dataset) * config.VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # ----------------- MODEL -----------------
    model = LSTMModel(input_size=input_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # ----------------- CHECKPOINT PREP -----------------
    writer = SummaryWriter(f"runs/{model.name}/exp_{config.EXPERIMENT_NAME}") # tensorboard --logdir=runs
    sample_x, _ = next(iter(train_loader))
    writer.add_graph(model, sample_x.to(device).float())

    ckpt_dir_path = f"checkpoints/{model.name}"
    os.makedirs(ckpt_dir_path, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir_path, "last.pt")
    best_ckpt_path = os.path.join(ckpt_dir_path, "best.pt")
    best_loss = float("inf")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        model, optimizer, last_epoch = checkpoint.load(ckpt_path, model, optimizer, device)
        start_epoch = last_epoch + 1

    # ----------------- THE LOOP -----------------
    for epoch in range(start_epoch, config.EPOCHS):

        train_avg_loss = train.train_one_epoch(model, train_loader, optimizer, criterion, device)

        eval_avg_loss = evaluate.evaluate_one_epoch(model, eval_loader, criterion, device)

        #save
        checkpoint.save(ckpt_path, model, epoch, train_avg_loss, input_size, optimizer)
        if train_avg_loss < best_loss:
            checkpoint.save(best_ckpt_path, model, epoch, train_avg_loss, input_size, optimizer)

        #log
        writer.add_scalar("LOSS/TRAIN", train_avg_loss, epoch)
        writer.add_scalar("LOSS/VAL", eval_avg_loss, epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_train_loss: {train_avg_loss:.6f} | avg_val_loss: {eval_avg_loss:.6f}")
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

if __name__ == "__main__":
    main()