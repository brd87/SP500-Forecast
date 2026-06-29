from torch import nn

def train_one_epoch(model:nn.Module, train_loader, optimizer, criterion, device):
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
    return train_avg_loss