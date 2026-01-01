import torch.nn as nn

def initialize_weights(model, init_type: str):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "he":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown init type: {init_type}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)