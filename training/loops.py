import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / len(loader)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy  = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average="macro")
    recall    = recall_score(all_labels, all_preds, average="macro")
    f1        = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, precision, recall, f1