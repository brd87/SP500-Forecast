import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from data.dataset import SP500Dataset
from data.download import RawData
from data.processing import DataProcessing
from models.lstm import LSTMModel
from models.rnn import RNNModel
from models.gru import GRUModel 
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
    #dataprocessing = DataProcessing(csv_path="D:/projectsGYM/SP500FC/SP500-Forecast/data/raw/tick10_1998-12-22_2026-07-07_id20260707142125.csv")
    dataset = SP500Dataset(dataprocessing.save_dsready_path)
    #dataset = SP500Dataset("D:/projectsGYM/SP500FC/SP500-Forecast/data/dsready/features10_1999-10-07_2026-06-26_id20260707142125.csv")
    
    print(f"SIZES | preprocessed: {len(dataprocessing.preprocessed)} , trainready: {len(dataprocessing.dsready)}")
    print("Class balance")
    print(np.unique(dataset.y, return_counts=True))
    print("Positive ratio:", np.nanmean(dataset.y))
    
    input_size = dataset.input_size

    
    # ----------------- MODEL -----------------
    # model = LSTMModel(input_size=input_size).to(device)
    # model = RNNModel(input_size=input_size).to(device)
    model = GRUModel(input_size=input_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)


    # ----------------- LOADERS & CHECKPOINT PREP -----------------
    ckpt_path, best_ckpt_path, scaler_path, best_valid_avg_loss, start_epoch, model, optimizer = checkpoint.last_best(
        model.nameid, input_size, device
        )

    train_loader, val_loader, test_loader = dataset_split.subset(dataset, scaler_path)

    writer = SummaryWriter(f"runs/{model.name}/exp_{config.EXPERIMENT_NAME}") # tensorboard --logdir=runs
    sample_x, _ = next(iter(train_loader))
    writer.add_graph(model, sample_x.to(device).float())


    # ----------------- THE LOOP -----------------
    for epoch in range(start_epoch, config.EPOCHS):

        train_avg_loss = train.one_epoch(model, train_loader, optimizer, criterion, device)
        valid_avg_loss = validate.one_epoch(model, val_loader, criterion, device)

        #save
        checkpoint.save(ckpt_path, model, epoch, train_avg_loss, input_size, optimizer, scaler_path)

        if valid_avg_loss < best_valid_avg_loss:
            best_valid_avg_loss = valid_avg_loss
            checkpoint.save(best_ckpt_path, model, epoch, valid_avg_loss, input_size, optimizer, scaler_path)

        #log
        writer.add_scalar("LOSS/TRAIN", train_avg_loss, epoch)
        writer.add_scalar("LOSS/VAL", valid_avg_loss, epoch)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_train_loss: {train_avg_loss:.6f} | avg_val_loss: {valid_avg_loss:.6f}")
    
    writer.close()


    # ----------------- EVAL -----------------
    evaluate.test(model, device, test_loader, criterion, best_ckpt_path)


if __name__ == "__main__":
    main()