"""
Hi-UCD dataset for multi-task semantic change detection.

For each image, returns:
  - image_2018 : raw RGB tensor   (3, H, W)
  - image_2019 : raw RGB tensor   (3, H, W)
  - sem_2018   : 2018 labels       (H, W), values 0-9
  - sem_2019   : 2019 labels       (H, W), values 0-9
  - change     : binary change     (H, W), values 0 or 1
  - valid      : validity mask     (H, W), True where the pixel is labeled

The 3 TASKS of the multi-task model:
  task 1 = segmentation 2018  -> sem_2018  (loss ignores class 0)
  task 2 = segmentation 2019  -> sem_2019  (loss ignores class 0)
  task 3 = binary change      -> change + valid mask

Channel 3 of the mask is recoded:
  raw channel 3: 0 = unlabeled, 1 = no-change, 2 = change
  -> change = 1 where channel 3 == 2, else 0
  -> valid  = False where channel 3 == 0 (unlabeled pixels are ignored)

Images are returned RAW (no normalization here). Transforms come later,
so we can adapt normalization to the backbone (ResNet vs DinoV3).
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HiUCDDataset(Dataset):

    def __init__(self, data_root, split="train", debug_n=None):
        # split is "train", "val" or "test"
        # debug_n: if set, only load the first N images (fast local test)
        self.split = split

        self.dir_2018 = os.path.join(data_root, split, "image", "2018")
        self.dir_2019 = os.path.join(data_root, split, "image", "2019")
        self.dir_mask = os.path.join(data_root, split, "mask", "2018_2019")

        # test/ has no masks
        self.has_masks = os.path.exists(self.dir_mask)

        # Build the list of image names (filter out macOS ._ parasite files).
        # We list masks if they exist, otherwise the 2018 images (test set).
        if self.has_masks:
            folder = self.dir_mask
        else:
            folder = self.dir_2018

        self.names = []
        for filename in os.listdir(folder):
            if filename.endswith(".png") and not filename.startswith("._"):
                name_without_extension = filename[:-4]   # remove ".png"
                self.names.append(name_without_extension)
        self.names = sorted(self.names)

        if debug_n is not None:
            self.names = self.names[:debug_n]
            print("[debug] loading only", len(self.names), "images")

        if len(self.names) == 0:
            raise RuntimeError(
                "No images found for split '" + split + "' in " + data_root
            )

    def __len__(self):
        return len(self.names)

    def load_image(self, path):
        # Open as RGB, turn into a (3, H, W) float tensor, no normalization.
        array = np.array(Image.open(path).convert("RGB"))   # (H, W, 3)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        return tensor

    def __getitem__(self, index):
        name = self.names[index]

        path_2018 = os.path.join(self.dir_2018, name + ".png")
        path_2019 = os.path.join(self.dir_2019, name + ".png")
        image_2018 = self.load_image(path_2018)
        image_2019 = self.load_image(path_2019)

        sample = {
            "name": name,
            "image_2018": image_2018,
            "image_2019": image_2019,
        }

        # The test split has no masks: stop here.
        if not self.has_masks:
            return sample

        path_mask = os.path.join(self.dir_mask, name + ".png")
        mask = np.array(Image.open(path_mask))   # (H, W, 3)
        labels_2018 = mask[:, :, 0]   # values 0-9
        labels_2019 = mask[:, :, 1]   # values 0-9
        change_raw = mask[:, :, 2]    # values 0 / 1 / 2

        # Semantic targets: kept as-is, the loss will ignore class 0.
        sample["sem_2018"] = torch.from_numpy(labels_2018).long()
        sample["sem_2019"] = torch.from_numpy(labels_2019).long()

        # Recode the change channel.
        change = (change_raw == 2).astype(np.int64)   # 1 if change, else 0
        valid = (change_raw != 0)                      # False if unlabeled
        sample["change"] = torch.from_numpy(change).long()
        sample["valid"] = torch.from_numpy(valid)

        return sample