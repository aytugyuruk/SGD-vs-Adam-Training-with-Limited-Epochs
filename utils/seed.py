# utils/seed.py
import random
import numpy as np
import torch
import os

def seed_everything(seed: int = 3):
    import random
    import numpy as np
    import torch
    import os

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Python hash
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"All sources seeded with seed={seed}")