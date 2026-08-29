from medpy import metric
import numpy as np

def test_single_case(score_map):
    """
    Converts a softmax probability score_map (shape: [2, H, W, D]) into a hard segmentation using argmax.
    """
    label_map = np.argmax(score_map, axis=0)
    return label_map

from medpy import metric
import numpy as np


def calculate_metric_percase(pred, gt, voxelspacing=None):
    """
    Computes case-wise 3D binary segmentation metrics.

    When exactly one mask is empty, HD95 and ASD are assigned
    the spatial diagonal of the evaluated volume.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"
        )

    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)

    if pred_empty and gt_empty:
        return 1.0, 1.0, 0.0, 0.0

    if pred_empty or gt_empty:
        extent = np.asarray(gt.shape, dtype=np.float64) - 1.0

        if voxelspacing is not None:
            spacing = np.asarray(voxelspacing, dtype=np.float64)
            extent = extent * spacing

        maximum_distance = float(np.linalg.norm(extent))
        return 0.0, 0.0, maximum_distance, maximum_distance

    dice = metric.binary.dc(pred, gt)
    iou = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(
        pred,
        gt,
        voxelspacing=voxelspacing,
    )
    asd = metric.binary.assd(
        pred,
        gt,
        voxelspacing=voxelspacing,
    )

    return dice, iou, hd95, asd
