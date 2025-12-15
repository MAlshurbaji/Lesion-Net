import os
import time
import shutil
import logging
from collections import defaultdict
from typing import Dict
import numpy as np
import torch
import imageio
from networks.lesion_net import LesionNet
from utils.dataload import BrainSegmentationDataset
from utils.metrics import test_single_case, calculate_metric_percase
from utils.helpers import load_yaml_config, set_seed, setup_logger, get_device

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

    # ----------------------------
    # Per-patient 3D metrics
    # ----------------------------
    total_dice, total_jc, total_hd, total_asd = [], [], [], []

    for pid in sorted(patient_slices_pred):
        pred_slices = patient_slices_pred[pid]
        gt_slices = patient_slices_gt[pid]
        max_slice = max(pred_slices.keys())

        ref_zero = np.zeros_like(next(iter(pred_slices.values())), dtype=np.uint8)

        pred_stack = np.stack(
            [pred_slices.get(i, ref_zero) for i in range(max_slice + 1)],
            axis=0,
        )
        gt_stack = np.stack(
            [gt_slices.get(i, ref_zero) for i in range(max_slice + 1)],
            axis=0,
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
        f"Eval summary | slices={total_slices} patients={len(patient_slices_pred)} "
        f"3D: DSC={mean_3d_dsc*100:.2f}% IoU={mean_3d_jc*100:.2f}% "
        f"HD95={mean_3d_hd:.2f} ASD={mean_3d_asd:.2f}"
    )

    return (
        mean_3d_dsc,
        mean_3d_jc,
        mean_3d_hd,
        mean_3d_asd,
        epoch_pred_masks,
        total_slices,
        len(patient_slices_pred),
    )


def save_pred_masks_png(epoch_pred_masks: Dict[str, np.ndarray], out_dir: str):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    for name, pred_np in epoch_pred_masks.items():
        imageio.imwrite(
            os.path.join(out_dir, f"{name}.png"),
            pred_np.astype(np.uint8) * 255,
        )


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = load_yaml_config("config/config_eval.yaml")

    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "evaluate.log")
    logger = setup_logger(log_file)

    set_seed(int(cfg["experiment"]["seed"]))
    device = get_device(cfg["system"]["device"])
    logger.info(f"Using device: {device}")

    # ----------------------------
    # Dataset
    # ----------------------------
    data_cfg = cfg["data"]

    val_dataset = BrainSegmentationDataset(
        os.path.join(data_cfg["dwi_dir"], data_cfg["split_val"]),
        os.path.join(data_cfg["mask_dir"], data_cfg["split_val"]),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["system"]["num_workers"],
        pin_memory=True,
    )

    # ----------------------------
    # Load trained model
    # ----------------------------
    model = LesionNet(in_chns=1, class_num=1).to(device)

    checkpoint_path = os.path.join(out_dir, "model_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # ----------------------------
    # Evaluation
    # ----------------------------
    start_time = time.time()
    mean_dsc, mean_jc, mean_hd, mean_asd, epoch_pred_masks, total_slices, total_patients = (
        evaluate_3d_metrics(
            cfg=cfg,
            device=device,
            val_dataset=val_dataset,
            val_loader=val_loader,
            model=model,
            logger=logger,
        )
    )

    # ----------------------------
    # Save predictions
    # ----------------------------
    out_pred_dir = os.path.join(out_dir, "segmentation_outputs", "evaluation")
    save_pred_masks_png(epoch_pred_masks, out_pred_dir)

    elapsed = (time.time() - start_time) / 60.0
    logger.info(f"Evaluation finished in {elapsed:.2f} minutes")
    logger.info(
        f"Final 3D Metrics | slices={total_slices} patients={total_patients} | "
        f"DSC={mean_dsc*100:.2f}% IoU={mean_jc*100:.2f}% "
        f"HD95={mean_hd:.2f} ASD={mean_asd:.2f}"
    )


if __name__ == "__main__":
    main()
