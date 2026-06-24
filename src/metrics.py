"""
Metrics for change detection.

IoU = Intersection over Union. For a class, it is the overlap between
predicted pixels and true pixels, divided by their union. It is robust to
class imbalance, unlike plain accuracy.

IMPORTANT: IoU must NOT be averaged per image. We ACCUMULATE intersection
and union over the whole dataset, then compute the ratio ONCE at the end.
That is why these are classes that accumulate, then report.

Unlabeled pixels are always excluded (via the valid mask or ignore_index).
"""

import torch


class BinaryChangeIoU:
    """BCD IoU : IoU of the 'change' class in the binary change task.

    Usage:
        metric = BinaryChangeIoU()
        for each batch:
            metric.add(pred_change, target_change, valid)
        score = metric.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Running totals over the whole dataset.
        self.intersection = 0
        self.union = 0

    def add(self, pred, target, valid):
        """Accumulate one batch.

        pred   : predicted change (B, H, W), values 0 or 1
        target : true change (B, H, W), values 0 or 1
        valid  : validity mask (B, H, W), True where the pixel is labeled
        """
        # Keep only valid (labeled) pixels.
        pred = pred[valid]
        target = target[valid]

        # Change class = value 1.
        pred_is_change = (pred == 1)
        target_is_change = (target == 1)

        # Intersection: both agree it changed.
        inter = (pred_is_change & target_is_change).sum().item()
        # Union: either says it changed.
        uni = (pred_is_change | target_is_change).sum().item()

        self.intersection += inter
        self.union += uni

    def compute(self):
        """Return the accumulated IoU. Call once at the end."""
        if self.union == 0:
            return 0.0   # no change anywhere, avoid division by zero
        return self.intersection / self.union
    
class SemanticMIoU:
    """mIoU for semantic segmentation: mean IoU over the 9 classes.

    We accumulate a confusion matrix over the whole dataset, then derive
    the IoU of each class from it and average them. Class 0 (unlabeled)
    is excluded from the mean.

    Usage:
        metric = SemanticMIoU(num_classes=10, ignore_index=0)
        for each batch:
            metric.add(pred_labels, target_labels)
        score = metric.compute()                # the mean IoU
        per_class = metric.compute_per_class()   # IoU of each class
    """

    def __init__(self, num_classes=10, ignore_index=0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # Confusion matrix: rows = true class, columns = predicted class.
        self.confusion = torch.zeros(self.num_classes, self.num_classes,
                                     dtype=torch.long)

    def add(self, pred, target):
        """Accumulate one batch into the confusion matrix.

        pred   : predicted labels (B, H, W), values 0..num_classes-1
        target : true labels (B, H, W), values 0..num_classes-1
        """
        pred = pred.flatten()
        target = target.flatten()

        # For each pixel, increment the cell (true class, predicted class).
        # We do this efficiently with bincount on a combined index.
        index = target * self.num_classes + pred
        counts = torch.bincount(index,
                                minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute_per_class(self):
        """Return the IoU of each class as a list."""
        ious = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                ious.append(None)   # skipped class
                continue

            # Intersection: the diagonal cell.
            intersection = self.confusion[c, c].item()
            # Union: everything in row c + everything in column c - intersection.
            row = self.confusion[c, :].sum().item()    # true class c
            col = self.confusion[:, c].sum().item()     # predicted class c
            union = row + col - intersection

            if union == 0:
                ious.append(None)   # class absent, no score
            else:
                ious.append(intersection / union)
        return ious

    def compute(self):
        """Return the mean IoU over the kept classes."""
        per_class = self.compute_per_class()
        valid_ious = [iou for iou in per_class if iou is not None]
        if len(valid_ious) == 0:
            return 0.0
        return sum(valid_ious) / len(valid_ious)