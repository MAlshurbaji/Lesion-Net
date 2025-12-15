import torch
import pandas as pd
import matplotlib.pyplot as plt
import os, csv
import random
import numpy as np
import logging
from thop import profile, clever_format

def save_training_plots_from_csv_G1(metrics_csv_path, output_dir):
    df = pd.read_csv(metrics_csv_path)

    # Single plot with G1_train_loss, G1_val_loss, G1_3d_dsc
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(df['epoch'], df['G1_train_loss'], label='G1 Train Loss', color='navy')
    ax1.plot(df['epoch'], df['G1_val_loss'], label='G1 Val Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['G1_3d_dsc'], 'r--', label='G1 3D-DSC')
    ax2.set_ylabel('G1 3D-DSC (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle('G1 Training Loss, Validation Loss, and 3D-DSC per Epoch', fontsize=14)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'G1_train_val_loss_and_3d_dsc.png'), dpi=300)
    plt.close()

def log_epoch_metrics(metrics_file_path, epoch, G1_train_loss, G1_val_loss, G1_3d_dsc):
    with open(metrics_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, G1_train_loss, G1_val_loss, G1_3d_dsc])

def compute_model_complexity(model, input_size, device):
    """
    Compute FLOPs and number of parameters for a PyTorch model using THOP.
    """
    dummy_input = torch.randn(*input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.2f")
    return flops, params


def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e

    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic may slow down; keep benchmark True for speed unless strict reproducibility is needed
    torch.backends.cudnn.benchmark = True

def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def get_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        return torch.device(device_str)
    return torch.device("cpu")
