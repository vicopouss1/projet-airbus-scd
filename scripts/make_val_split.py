"""
Splits the Hi-UCD validation set into two parts:
  - val_tuning (70%) : to choose the threshold and hyperparameters
  - val_test   (30%) : for the final measurement, used only once

The test set of Hi-UCD has no labels (reserved for Codabench), so we carve
our own internal test set out of the validation set.

The split uses a fixed seed, so it is always the same. Run it once.
"""

import random
from pathlib import Path

# --- Settings: edit these two lines if needed ---
DATA_ROOT = Path("/Volumes/SSD_VICO/hi-ucd-complete")
TUNING_RATIO = 0.70
SEED = 42
OUT_DIR = Path("configs/splits")


# 1. List all validation image names (based on the masks).
mask_dir = DATA_ROOT / "val" / "mask" / "2018_2019"

names = []
for path in mask_dir.glob("*.png"):
    if not path.name.startswith("._"):   # skip macOS parasite files
        names.append(path.stem)          # ".stem" = file name without ".png"
names = sorted(names)

print("Validation images found:", len(names))


# 2. Shuffle with a fixed seed, then cut into two parts.
random.seed(SEED)
random.shuffle(names)

cut = round(len(names) * TUNING_RATIO)
tuning = sorted(names[:cut])    # first 70%
test = sorted(names[cut:])      # remaining 30%


# 3. Save each part to a text file (one name per line).
OUT_DIR.mkdir(parents=True, exist_ok=True)

(OUT_DIR / "val_tuning.txt").write_text("\n".join(tuning) + "\n")
(OUT_DIR / "val_test.txt").write_text("\n".join(test) + "\n")

print("Wrote", len(tuning), "-> val_tuning.txt")
print("Wrote", len(test), "-> val_test.txt")