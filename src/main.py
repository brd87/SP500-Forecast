import os
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from data.dataset import SP500Dataset
from data.download import RawData
from data.processing import DataProcessing
from models.lstm import LSTMModel
from torch.utils.tensorboard import SummaryWriter

import config
import validate
import checkpoint
import train
import evaluate
import dataset_split

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

    train_loader, val_loader, test_loader = dataset_split.subset(dataset, config.TRAIN_RATIO, config.VAL_RATIO, config.BATCH_SIZE)

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

    best_valid_avg_loss = float("inf")
    start_epoch = 0

    # if os.path.exists(ckpt_path):
    #     model, optimizer, last_epoch = checkpoint.load(ckpt_path, model, optimizer, device)
    #     start_epoch = last_epoch + 1

    # ----------------- THE LOOP -----------------
    for epoch in range(start_epoch, config.EPOCHS):

        train_avg_loss = train.one_epoch(model, train_loader, optimizer, criterion, device)
        valid_avg_loss = validate.one_epoch(model, val_loader, criterion, device)

        #save
        checkpoint.save(ckpt_path, model, epoch, train_avg_loss, input_size, optimizer)

        if valid_avg_loss < best_valid_avg_loss:
            best_valid_avg_loss = valid_avg_loss
            checkpoint.save(best_ckpt_path, model, epoch, valid_avg_loss, input_size, optimizer)

        #log
        writer.add_scalar("LOSS/TRAIN", train_avg_loss, epoch)
        writer.add_scalar("LOSS/VAL", valid_avg_loss, epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_train_loss: {train_avg_loss:.6f} | avg_val_loss: {valid_avg_loss:.6f}")
    
    writer.close()

    # ----------------- EVAL -----------------
    evaluate.test(model, device, test_loader, best_ckpt_path)


if __name__ == "__main__":
    main()