import os
import csv
import time
import shutil
import logging
import datetime
from typing import Dict
from collections import defaultdict
import numpy as np
import imageio
import torch
import torch.optim as optim
from tqdm import tqdm
from networks.lesion_net import LesionNet
from utils.dataload import BrainSegmentationDataset
from utils.loss import dice_bce_loss
from utils.helpers import (
    save_training_plots_from_csv_G1,
    log_epoch_metrics,
    compute_model_complexity,
    load_yaml_config,
    set_seed,
    setup_logger,
    get_device,
)
from utils.metrics import test_single_case, calculate_metric_percase


# ============================================================
# MODEL / OPTIMIZER
# ============================================================
def get_input_size(h: int, w: int):
    return (1, 1, h, w)  # batch=1, channel=1 (DWI only)


def build_model(cfg: dict, device: torch.device):
    model_cfg = cfg["model"]

    model = LesionNet(
        in_channels=model_cfg["in_channels"],
        widths=model_cfg["widths"],
        depths=model_cfg["depths"],
        all_num_heads=model_cfg["all_num_heads"],
        patch_sizes=model_cfg["patch_sizes"],
        overlap_sizes=model_cfg["overlap_sizes"],
        reduction_ratios=model_cfg["reduction_ratios"],
        mlp_expansions=model_cfg["mlp_expansions"],
        decoder_channels=model_cfg["decoder_channels"],
        scale_factors=model_cfg["scale_factors"],
        num_classes=model_cfg["num_classes"],
        drop_prob=model_cfg["drop_prob"],
    ).to(device)

    return model


def build_optimizer(model, lr: float):
    return optim.Adam(model.parameters(), lr=lr)


# ============================================================
# LOGGING
# ============================================================
def write_run_header(
    logger: logging.Logger,
    eval_log_path: str,
    cfg: dict,
    device: torch.device,
    model_name: str,
    params: str,
    flops: str,
):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_cfg = cfg["training"]
    sys_cfg = cfg["system"]

    with open(eval_log_path, "a") as f:
        f.write(f"Start Time: {t}\n")
        f.write(f"Experiment: {cfg['experiment']['name']}\n")
        f.write(f"Batch Size: {train_cfg['batch_size']}\n")
        f.write(f"Epochs: {train_cfg['epochs']}\n")
        f.write(f"LR: {train_cfg['lr']}\n")
        f.write(f"Device: {device}\n")
        f.write(f"AMP: {sys_cfg.get('amp', False)}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Params = {params}, FLOPs = {flops}\n\n")

    logger.info("Run initialized. Logging header written.")


def init_metrics_csv(metrics_file_path: str):
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
    with open(metrics_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "G1_train_loss", "G1_val_loss", "G1_3d_dsc"])


# ============================================================
# DATA
# ============================================================
def build_datasets_and_loaders(cfg: dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    train_dataset = BrainSegmentationDataset(
        os.path.join(data_cfg["dwi_dir"], data_cfg["split_train"]),
        os.path.join(data_cfg["mask_dir"], data_cfg["split_train"]),
    )

    val_dataset = BrainSegmentationDataset(
        os.path.join(data_cfg["dwi_dir"], data_cfg["split_val"]),
        os.path.join(data_cfg["mask_dir"], data_cfg["split_val"]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


# ============================================================
# TRAINING
# ============================================================
def train_one_epoch(epoch, model, optimizer, train_loader, device, logger):
    model.train()
    epoch_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Train {epoch+1}"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_bce_loss(torch.sigmoid(outputs), masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    logger.info(f"[Epoch {epoch+1}] Train Dice Loss: {avg_loss:.4f}")
    return avg_loss


# ============================================================
# EVALUATION (3D)
# ============================================================
def evaluate_3d_metrics(
    cfg: dict,
    device: torch.device,
    val_dataset,
    val_loader,
    model,
    logger: logging.Logger,
):
    threshold = float(cfg["training"]["threshold"])
    model.eval()

    patient_slices_pred = defaultdict(dict)
    patient_slices_gt = defaultdict(dict)
    epoch_pred_masks: Dict[str, np.ndarray] = {}
    total_slices = 0

    with torch.no_grad():
        for idx, (image, mask) in enumerate(val_loader):
            name = os.path.splitext(val_dataset.image_filenames[idx])[0]

            image = image.to(device)   # [1, 1, H, W]
            mask = mask.to(device)

            output = model(image)
            probs = torch.sigmoid(output)
            pred_bin = (probs > threshold).float()
            gt_bin = (mask > 0.5).float()

            pred_np = pred_bin.cpu().numpy().squeeze().astype(np.uint8)
            gt_np = gt_bin.cpu().numpy().squeeze().astype(np.uint8)

            epoch_pred_masks[name] = pred_np
            total_slices += 1

            patient_id, slice_idx = name.split("_")
            patient_id, slice_idx = int(patient_id), int(slice_idx)

            patient_slices_pred[patient_id][slice_idx] = pred_np
            patient_slices_gt[patient_id][slice_idx] = gt_np

    # --- 3D metrics ---
    total_dice, total_jc, total_hd, total_asd = [], [], [], []

    for pid in sorted(patient_slices_pred):
        pred_slices = patient_slices_pred[pid]
        gt_slices = patient_slices_gt[pid]
        max_slice = max(pred_slices.keys())

        ref_zero = np.zeros_like(next(iter(pred_slices.values())), dtype=np.uint8)

        pred_stack = np.stack(
            [pred_slices.get(i, ref_zero) for i in range(max_slice + 1)], axis=0
        )
        gt_stack = np.stack(
            [gt_slices.get(i, ref_zero) for i in range(max_slice + 1)], axis=0
        )

        score_map = np.zeros((2,) + pred_stack.shape, dtype=np.float32)
        score_map[1] = pred_stack.astype(np.float32)
        score_map[0] = 1.0 - score_map[1]

        pred_3d = test_single_case(score_map)
        gt_3d = gt_stack.astype(np.uint8)

        dice, jc, hd, asd = calculate_metric_percase(pred_3d, gt_3d)
        total_dice.append(dice)
        total_jc.append(jc)
        total_hd.append(hd)
        total_asd.append(asd)

    mean_3d_dsc = float(np.mean(total_dice)) if total_dice else 0.0
    mean_3d_jc = float(np.mean(total_jc)) if total_jc else 0.0
    mean_3d_hd = float(np.mean(total_hd)) if total_hd else 0.0
    mean_3d_asd = float(np.mean(total_asd)) if total_asd else 0.0

    logger.info(
        f"Eval | slices={total_slices}, patients={len(patient_slices_pred)} "
        f"DSC={mean_3d_dsc*100:.2f}% IoU={mean_3d_jc*100:.2f}% "
        f"HD95={mean_3d_hd:.2f} ASD={mean_3d_asd:.2f}"
    )

    return mean_3d_dsc, mean_3d_jc, mean_3d_hd, mean_3d_asd, epoch_pred_masks, total_slices, len(patient_slices_pred)


def save_pred_masks_png(epoch_pred_masks: Dict[str, np.ndarray], out_dir: str):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for name, pred_np in epoch_pred_masks.items():
        imageio.imwrite(
            os.path.join(out_dir, f"{name}.png"),
            (pred_np.astype(np.uint8) * 255),
        )


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = load_yaml_config("config/config_train.yaml")

    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    eval_log_path = os.path.join(out_dir, "val_metrics.txt")
    metrics_file_path = os.path.join(out_dir, "loss_metrics.csv")
    log_file = os.path.join(out_dir, "train.log")

    logger = setup_logger(log_file)
    set_seed(int(cfg["experiment"]["seed"]))

    device = get_device(cfg["system"]["device"])
    logger.info(f"Using device: {device}")

    train_dataset, val_dataset, train_loader, val_loader = build_datasets_and_loaders(cfg)

    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg["training"]["lr"])

    h, w = cfg["model"]["image_size"]
    flops, params = compute_model_complexity(
        model, get_input_size(h, w), device
    )

    write_run_header(
        logger, eval_log_path, cfg, device,
        model.__class__.__name__, params, flops
    )

    init_metrics_csv(metrics_file_path)

    best_3d_dsc = -1.0
    num_epochs = int(cfg["training"]["epochs"])
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            epoch, model, optimizer, train_loader, device, logger
        )

        mean_3d_dsc, mean_3d_jc, mean_3d_hd, mean_3d_asd, epoch_pred_masks, total_slices, total_patients = (
            evaluate_3d_metrics(cfg, device, val_dataset, val_loader, model, logger)
        )

        with open(eval_log_path, "a") as f:
            f.write(
                f"[Epoch {epoch+1}] Slices={total_slices}, Patients={total_patients} || "
                f"DSC={mean_3d_dsc*100:.2f}%, IoU={mean_3d_jc*100:.2f}%, "
                f"HD95={mean_3d_hd:.2f}, ASD={mean_3d_asd:.2f}\n"
            )

        if mean_3d_dsc > best_3d_dsc:
            best_3d_dsc = mean_3d_dsc
            logger.info(f"[New Best] 3D DSC = {best_3d_dsc*100:.2f}%")

            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
            save_pred_masks_png(
                epoch_pred_masks,
                os.path.join(out_dir, "segmentation_outputs", "best_epoch"),
            )

        log_epoch_metrics(
            metrics_file_path,
            epoch + 1,
            G1_train_loss=float(train_loss),
            G1_val_loss=0.0,
            G1_3d_dsc=float(mean_3d_dsc * 100.0),
        )

    hours = (time.time() - start_time) / 3600.0
    logger.info(f"Training finished in {hours:.2f} hours.")

    save_training_plots_from_csv_G1(metrics_file_path, out_dir)

    with open(eval_log_path, "a") as f:
        f.write(f"\nTotal Training Duration: {hours:.2f} hours.\n")
        f.write(f"Experiment: {cfg['experiment']['name']}\n")


if __name__ == "__main__":
    main()
