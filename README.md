# Semantic Change Detection in Aerial Imagery

Team project at Télécom Paris, supervised by Airbus (2026).

Given two aerial images of the same area taken one year apart, the model predicts:

- a land-cover map for each date (semantic segmentation, 9 classes plus `unlabeled`);
- a binary map of what changed between the two dates.

## Dataset

Hi-UCD (Tallinn, 0.1 m aerial imagery, 2018 and 2019). Each mask is a 3-channel PNG:

| Channel | Content |
|---|---|
| 1 | land-cover classes, 2018 (0–9) |
| 2 | land-cover classes, 2019 (0–9) |
| 3 | change: 0 = unlabeled, 1 = no change, 2 = change |

`unlabeled` pixels are excluded from the losses and the metrics. The official test set has no public labels, so `scripts/make_val_split.py` splits the validation set into `val_tuning` (70 %, threshold and hyperparameter selection) and `val_test` (30 %), with a fixed seed. Data are not included in the repo.

## Model

A multi-task Siamese network:

1. **Shared frozen encoder**, applied to both dates:
   - ResNet-50, ImageNet weights (`src/models/encoders/resnet_encoder.py`);
   - DINOv3 ViT-L/16 pretrained on satellite imagery (SAT-493M), loaded with HuggingFace Transformers (`src/models/encoders/dino_encoder.py`). Patch tokens are reshaped into a 32×32 feature map.
2. **Fusion**: `[z_2018 ; z_2019 ; |z_2018 − z_2019|]` along the channel axis.
3. **Multi-task decoder**: shared upsampling trunk with three heads (segmentation 2018, segmentation 2019, change). The number of upsampling steps adapts to the encoder resolution.

Only the decoder and heads are trained.

**Losses**: cross-entropy + Dice for segmentation, Focal loss for the change head (strong class imbalance).
**Metrics**: mean IoU over land-cover classes, IoU of the `change` class. The change threshold is selected on `val_tuning` (`src/threshold.py`).

## Results

Validation change IoU, frozen encoders:

| Encoder | Change IoU (val) |
|---|---|
| ResNet-50 (ImageNet) | 0.26 |
| DINOv3 ViT-L (satellite) | **0.41** |

With DINOv3, validation scores plateau after about 10 epochs while training scores keep increasing, so the model overfits. Semantic segmentation (val mIoU ≈ 0.33) is limited by the frozen backbone. Next steps: early stopping, partial fine-tuning of the encoder (e.g. LoRA), stronger data augmentation.

## Repository structure

```
src/
├── data/hiucd_dataset.py   # dataset and 3-channel mask decoding
├── models/                 # encoders, fusion, decoder, full model
├── losses/losses.py        # CE, Dice, Focal, multi-task loss
├── metrics.py              # mIoU, change IoU
├── train.py                # training and evaluation loop
├── threshold.py            # change-threshold sweep on validation
└── utils/                  # device selection, plots, prediction visualization
scripts/make_val_split.py   # val_tuning / val_test split
notebooks/                  # data exploration and experiments
```

## Usage

```bash
pip install -r requirements.txt   # torch / torchvision installed separately (see below)
python -m src.utils.device        # checks which accelerator is available
```

```python
from src.train import train

# smoke test on 20 tiles
model, history = train("data/hi-ucd-complete", epochs=2, debug_n=20)

# full run with the DINOv3 encoder
model, history = train("data/hi-ucd-complete", epochs=30, backbone="dino",
                       save_path="outputs/checkpoints/dino.pt")
```

The same code runs on Apple Silicon (MPS) for development and on NVIDIA GPUs (CUDA) for full training. `torch` is not pinned in `requirements.txt` because the right build depends on the machine.
