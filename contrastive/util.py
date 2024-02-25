import random
import numpy as np
import torch

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # if torch.backends.mps.is_available():
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

def generate_run_name(args):
    run_name = "cl_transform"
    run_name += f"-{args.csv_path.split('.')[0]}"
    run_name += f"-b{args.batch_size}"
    run_name += f"-e{args.epochs}"
    run_name += f"-lr{args.lr:.6f}"
    run_name += f"-{args.loss_type}"
    run_name += f"-lt{args.loss_temp:.2f}"
    return run_name

