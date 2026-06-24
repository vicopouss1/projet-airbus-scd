"""
Visualization of model predictions vs ground truth.

Displays, for one image, a 3x3 grid:
  rows    = 2018, 2019, change
  columns = input image, prediction, ground truth

Used to produce result figures and to diagnose where the model fails.
Run from a notebook (needs a display).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.models.model import ChangeDetectionModel
from src.data.hiucd_dataset import HiUCDDataset


# Official Hi-UCD palette (10 classes, 0=unlabeled .. 9=woodland).
PALETTE = [
    [255, 255, 255],  # 0 unlabeled
    [0, 153, 255],    # 1 water
    [202, 255, 122],  # 2 grass
    [230, 0, 0],      # 3 building
    [230, 0, 255],    # 4 green house
    [255, 230, 0],    # 5 road
    [255, 181, 197],  # 6 bridge
    [0, 255, 230],    # 7 others
    [175, 122, 255],  # 8 bare land
    [26, 255, 0],     # 9 woodland
]
CMAP_SEM = ListedColormap(np.array(PALETTE) / 255.0)


def load_model(checkpoint_path, num_classes=10, device="cpu"):
    """Rebuild the architecture and load trained weights."""
    model = ChangeDetectionModel(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def show_prediction(model, dataset, index, change_threshold=0.5):
    """Display predictions vs GT for one image, as a 3x3 grid."""
    sample = dataset[index]

    img_2018 = sample["image_2018"].unsqueeze(0)
    img_2019 = sample["image_2019"].unsqueeze(0)
    with torch.no_grad():
        out_sem_2018, out_sem_2019, out_change = model(img_2018, img_2019)

    # Predictions
    pred_sem_2018 = out_sem_2018.argmax(dim=1).squeeze(0).numpy()
    pred_sem_2019 = out_sem_2019.argmax(dim=1).squeeze(0).numpy()
    prob_change = torch.sigmoid(out_change.squeeze(1)).squeeze(0).numpy()
    pred_change = (prob_change >= change_threshold).astype(int)

    # Ground truth
    gt_sem_2018 = sample["sem_2018"].numpy()
    gt_sem_2019 = sample["sem_2019"].numpy()
    gt_change = sample["change"].numpy()

    # Display images: (3,H,W) float -> (H,W,3) uint8
    disp_2018 = sample["image_2018"].permute(1, 2, 0).numpy().astype(np.uint8)
    disp_2019 = sample["image_2019"].permute(1, 2, 0).numpy().astype(np.uint8)

    fig, ax = plt.subplots(3, 3, figsize=(13, 13))

    # Row 0 : 2018
    ax[0, 0].imshow(disp_2018);                                    ax[0, 0].set_title("Image 2018")
    ax[0, 1].imshow(pred_sem_2018, cmap=CMAP_SEM, vmin=0, vmax=9); ax[0, 1].set_title("Pred sem 2018")
    ax[0, 2].imshow(gt_sem_2018, cmap=CMAP_SEM, vmin=0, vmax=9);   ax[0, 2].set_title("GT sem 2018")

    # Row 1 : 2019
    ax[1, 0].imshow(disp_2019);                                    ax[1, 0].set_title("Image 2019")
    ax[1, 1].imshow(pred_sem_2019, cmap=CMAP_SEM, vmin=0, vmax=9); ax[1, 1].set_title("Pred sem 2019")
    ax[1, 2].imshow(gt_sem_2019, cmap=CMAP_SEM, vmin=0, vmax=9);   ax[1, 2].set_title("GT sem 2019")

    # Row 2 : change
    ax[2, 0].imshow(disp_2019);                              ax[2, 0].set_title("Image 2019 (ref)")
    ax[2, 1].imshow(pred_change, cmap="gray", vmin=0, vmax=1); ax[2, 1].set_title("Pred change")
    ax[2, 2].imshow(gt_change, cmap="gray", vmin=0, vmax=1);   ax[2, 2].set_title("GT change")

    for row in ax:
        for a in row:
            a.axis("off")
    plt.suptitle(f"Image {sample['name']}", y=1.0, fontsize=14)
    plt.tight_layout()
    plt.show()


def find_images_with_change(dataset, min_change_pct=1.0, max_scan=1000):
    """Return indices of images that contain at least min_change_pct% change.

    Useful because change is ultra-rare in Hi-UCD (most images have none).
    """
    found = []
    n = min(len(dataset), max_scan)
    for i in range(n):
        sample = dataset[i]
        pct = (sample["change"].numpy() == 1).mean() * 100
        if pct >= min_change_pct:
            found.append((i, sample["name"], pct))
    found.sort(key=lambda x: -x[2])
    return found