from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR

def build_scheduler(optimizer, lr_type='step'):
    if lr_type == 'step':
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif lr_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=50)
    elif lr_type == "exponential":
        return ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unsupported lr_type: {lr_type}")