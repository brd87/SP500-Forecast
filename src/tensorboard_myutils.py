def add_scalars(writer, metrics, val_avg_loss, train_avg_loss, epoch):
    writer.add_scalar("Loss/TRAIN", train_avg_loss, epoch)
    writer.add_scalar("Loss/VAL", val_avg_loss, epoch)
    writer.add_scalar("Accuracy/VAL", metrics["accuracy"], epoch)
    writer.add_scalar("F1/VAL", metrics["f1"], epoch)
    writer.add_scalar("AUC/VAL", metrics["auc"], epoch)