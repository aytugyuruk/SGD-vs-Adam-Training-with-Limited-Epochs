import torch.optim as optim

def build_optimizer(model, lr, optimizer_type):
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")