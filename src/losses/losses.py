"""
Loss functions for the multi-task model.

Assignment (from the project slides):
  - segmentation 2018 / 2019 : CrossEntropy + Dice
  - change (binary)          : Focal

The unlabeled class (0) is excluded everywhere:
  - segmentation : CrossEntropy uses ignore_index=0, Dice skips class 0
  - change       : we only compute the loss on valid pixels (valid mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# CrossEntropy that ignores the unlabeled class (0).
# PyTorch handles this natively with ignore_index.
cross_entropy = nn.CrossEntropyLoss(ignore_index=0)


def dice_loss(logits, target, num_classes=10, ignore_index=0, eps=1e-6):
    """Dice loss for semantic segmentation.

    logits : raw model output (B, num_classes, H, W)
    target : ground truth labels (B, H, W), values 0..num_classes-1
    ignore_index : class to skip (0 = unlabeled)

    Dice looks at the overlap between prediction and target per class,
    which balances rare classes against dominant ones.
    """
    # Turn logits into per-class probabilities (softmax over the class axis).
    probs = F.softmax(logits, dim=1)   # (B, num_classes, H, W)

    # Turn the target labels into one-hot form, same shape as probs.
    # target (B, H, W) -> one_hot (B, H, W, num_classes) -> (B, num_classes, H, W)
    target_one_hot = F.one_hot(target, num_classes=num_classes)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

    # Compute Dice per class, then average over the classes we keep.
    dice_per_class = []
    for c in range(num_classes):
        if c == ignore_index:
            continue   # skip the unlabeled class

        pred_c = probs[:, c, :, :]              # predicted prob for class c
        target_c = target_one_hot[:, c, :, :]   # 1 where true class is c

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice_c = (2 * intersection + eps) / (union + eps)
        dice_per_class.append(dice_c)

    # Loss = 1 - mean Dice (so that better overlap = lower loss).
    mean_dice = torch.stack(dice_per_class).mean()
    return 1 - mean_dice

def focal_loss(logits, target, valid, alpha=0.75, gamma=2.0, eps=1e-6):
    """Focal loss for the binary change task.

    logits : raw model output (B, 1, H, W)
    target : ground truth change (B, H, W), values 0 or 1
    valid  : validity mask (B, H, W), True where the pixel is labeled

    Focal loss tackles the heavy imbalance (~6% change) two ways:
      - gamma : down-weights easy, well-classified pixels
      - alpha : up-weights the rare class (change)
    Only valid pixels contribute to the loss.
    """
    # Remove the channel dimension: (B, 1, H, W) -> (B, H, W)
    logits = logits.squeeze(1)

    # Turn the raw score into a probability of change (sigmoid -> 0..1).
    prob = torch.sigmoid(logits)

    # target as float for the math below.
    target = target.float()

    # p_t = probability the model assigns to the TRUE class of each pixel.
    #   where target == 1 : p_t = prob
    #   where target == 0 : p_t = 1 - prob
    p_t = prob * target + (1 - prob) * (1 - target)

    # alpha_t : the alpha weight applied to the true class of each pixel.
    #   change pixels get alpha, no-change pixels get (1 - alpha).
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    # Standard binary cross-entropy per pixel, then the focal modulation.
    bce = -torch.log(p_t + eps)             # base loss per pixel
    focal = alpha_t * (1 - p_t) ** gamma * bce   # focal weighting

    # Keep only valid pixels (ignore unlabeled), then average over them.
    focal = focal[valid]
    if focal.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    return focal.mean()

def compute_total_loss(outputs, batch):
    """Compute the full multi-task loss.

    outputs : tuple (out_sem_2018, out_sem_2019, out_change) from the model
    batch   : dict from the dataloader, with keys
              sem_2018, sem_2019, change, valid

    Returns:
      total : the scalar loss to optimize
      parts : a dict with each term, for logging
    """
    out_sem_2018, out_sem_2019, out_change = outputs

    # --- Segmentation 2018 : CE + Dice ---
    ce_2018 = cross_entropy(out_sem_2018, batch["sem_2018"])
    dice_2018 = dice_loss(out_sem_2018, batch["sem_2018"])

    # --- Segmentation 2019 : CE + Dice ---
    ce_2019 = cross_entropy(out_sem_2019, batch["sem_2019"])
    dice_2019 = dice_loss(out_sem_2019, batch["sem_2019"])

    # --- Change : Focal ---
    focal_change = focal_loss(out_change, batch["change"], batch["valid"])

    # Sum of all terms (simple sum for now; weighting comes later).
    total = ce_2018 + dice_2018 + ce_2019 + dice_2019 + focal_change

    # Keep each term separately so we can watch them during training.
    parts = {
        "ce_2018": ce_2018.item(),
        "dice_2018": dice_2018.item(),
        "ce_2019": ce_2019.item(),
        "dice_2019": dice_2019.item(),
        "focal_change": focal_change.item(),
        "total": total.item(),
    }

    return total, parts