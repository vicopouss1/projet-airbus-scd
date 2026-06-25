"""
Training loop for the multi-task change detection model.

Same function for smoke tests and real runs:
  - smoke test : call with debug_n=20, epochs=2, save_path=None
  - real run   : call with debug_n=None, more epochs, a save_path

The frozen backbone is not optimized: only the decoder and heads learn.
Only the trainable weights are saved (the frozen ResNet reloads itself).

train() returns the model and a history dict, so the loss and metric
curves can be plotted afterwards.
"""
import os
import torch
from torch.utils.data import DataLoader

from src.data.hiucd_dataset import HiUCDDataset
from src.models.model import ChangeDetectionModel
from src.losses.losses import compute_total_loss
from src.metrics import SemanticMIoU, BinaryChangeIoU
from src.utils.device import get_device


def evaluate(model, loader, device, change_threshold=0.5):
    """Run the model on a loader and compute loss + metrics.

    No learning happens here: eval mode + no gradients.
    Works on any set (train sample or validation).
    Returns (loss, mIoU, BCD IoU).
    """
    model.eval()   # evaluation mode

    miou_2018 = SemanticMIoU(num_classes=10, ignore_index=0)
    miou_2019 = SemanticMIoU(num_classes=10, ignore_index=0)
    bcd = BinaryChangeIoU()

    sum_loss = 0.0
    num_batches = 0

    with torch.no_grad():   # no gradients during evaluation
        for batch in loader:
            image_2018 = batch["image_2018"].to(device)
            image_2019 = batch["image_2019"].to(device)
            targets = {
                "sem_2018": batch["sem_2018"].to(device),
                "sem_2019": batch["sem_2019"].to(device),
                "change": batch["change"].to(device),
                "valid": batch["valid"].to(device),
            }

            outputs = model(image_2018, image_2019)

            # Loss (same function as training, but no backward here).
            total, parts = compute_total_loss(outputs, targets)
            sum_loss += parts["total"]
            num_batches += 1

            out_sem_2018, out_sem_2019, out_change = outputs

            # Decisions for the metrics.
            pred_2018 = out_sem_2018.argmax(dim=1).cpu()
            pred_2019 = out_sem_2019.argmax(dim=1).cpu()
            prob_change = torch.sigmoid(out_change.squeeze(1)).cpu()
            pred_change = (prob_change >= change_threshold).long()

            miou_2018.add(pred_2018, batch["sem_2018"])
            miou_2019.add(pred_2019, batch["sem_2019"])
            bcd.add(pred_change, batch["change"], batch["valid"])

    avg_loss = sum_loss / num_batches
    mean_miou = (miou_2018.compute() + miou_2019.compute()) / 2
    return avg_loss, mean_miou, bcd.compute()


def train(data_root, epochs=2, batch_size=4, lr=1e-4,
          debug_n=None, save_path=None, val_debug_n=200, train_eval_n=200,
          backbone="resnet"):
    """Train the model and return (model, history).

    data_root    : dataset root
    epochs       : number of passes over the data
    batch_size   : images per batch
    lr           : learning rate
    debug_n      : if set, only use the first N train images (smoke test)
    save_path    : if set, save trainable weights there at the end
    val_debug_n  : number of validation images used to measure each epoch
    train_eval_n : number of train images used to measure each epoch
    """

    # 1. Device.
    device = get_device()

    # 2. Data.
    dataset = HiUCDDataset(data_root, split="train", debug_n=debug_n)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loader to MEASURE on a train sample (no shuffle, just for evaluation).
    train_eval_dataset = HiUCDDataset(data_root, split="train", debug_n=train_eval_n)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)

    # Loader to measure on validation.
    val_dataset = HiUCDDataset(data_root, split="val", debug_n=val_debug_n)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model.
    model = ChangeDetectionModel(num_classes=10, backbone=backbone)
    model = model.to(device)

    # 4. Optimizer: only trainable parameters.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # 5. History: train AND validation for each quantity.
    history = {
        "train_loss": [], "train_miou": [], "train_bcd": [],
        "val_loss": [], "val_miou": [], "val_bcd": [],
    }

    # 6. Training loop.
    for epoch in range(epochs):
        model.train()   # training mode

        for batch in loader:
            image_2018 = batch["image_2018"].to(device)
            image_2019 = batch["image_2019"].to(device)
            targets = {
                "sem_2018": batch["sem_2018"].to(device),
                "sem_2019": batch["sem_2019"].to(device),
                "change": batch["change"].to(device),
                "valid": batch["valid"].to(device),
            }

            outputs = model(image_2018, image_2019)
            total, parts = compute_total_loss(outputs, targets)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        # After the epoch: measure on a train sample AND on validation,
        # the same way (eval mode, frozen weights), so the curves compare.
        train_loss, train_miou, train_bcd = evaluate(model, train_eval_loader, device)
        val_loss, val_miou, val_bcd = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["train_bcd"].append(train_bcd)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_bcd"].append(val_bcd)

        print(f"epoch {epoch+1}/{epochs} | "
              f"train: loss {train_loss:.4f} mIoU {train_miou:.4f} BCD {train_bcd:.4f} | "
              f"val: loss {val_loss:.4f} mIoU {val_miou:.4f} BCD {val_bcd:.4f}")

    # 7. Optionally save the model weights.
    if save_path is not None:
        # Make sure the destination folder exists (create it if needed).
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print("Saved model to", save_path)

    return model, history