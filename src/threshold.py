"""
Threshold selection for the change head.

The model outputs a change probability per pixel. We need to pick the
threshold that turns those probabilities into 0/1 decisions.

Method (no retraining):
  1. Run the model on val_tuning, collect change probabilities + targets.
  2. Sweep thresholds, compute BCD IoU for each, keep the best.
  3. Freeze that threshold, apply it later on val_test.

The threshold is chosen on val_tuning and applied as-is on val_test
(choosing it on val_test would be cheating).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.hiucd_dataset import HiUCDDataset
from src.utils.device import get_device


def collect_change_probs(model, data_root, tile_list=None, debug_n=None,
                         batch_size=8):
    """Run the model on validation and collect change probs + targets.

    Returns three flat arrays (over all valid pixels of all images):
      probs  : predicted change probability (0..1)
      target : true change (0 or 1)
    Only valid pixels are kept (unlabeled excluded).
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    dataset = HiUCDDataset(data_root, split="val", debug_n=debug_n)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            img_2018 = batch["image_2018"].to(device)
            img_2019 = batch["image_2019"].to(device)

            _, _, out_change = model(img_2018, img_2019)

            prob = torch.sigmoid(out_change.squeeze(1)).cpu().numpy()  # (B,H,W)
            target = batch["change"].numpy()                           # (B,H,W)
            valid = batch["valid"].numpy()                             # (B,H,W)

            # Keep only valid pixels, flatten.
            all_probs.append(prob[valid])
            all_targets.append(target[valid])

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    return probs, targets


def sweep_thresholds(probs, targets, n_steps=100):
    """Sweep thresholds, compute BCD IoU for each. Returns a list of rows."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    rows = []
    P = (targets == 1)   # true change mask
    for t in thresholds:
        pred = probs >= t
        tp = np.sum(pred & P)
        fp = np.sum(pred & ~P)
        fn = np.sum(~pred & P)

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append((t, iou, precision, recall))

    return rows


def best_threshold(rows):
    """Return the threshold that maximizes BCD IoU."""
    best = max(rows, key=lambda r: r[1])   # r[1] is IoU
    t, iou, precision, recall = best
    return t, iou, precision, recall