"""
Plot training curves from the history returned by train().

Two figures:
  - loss terms (total, CE, Dice, Focal) over the epochs
  - validation metrics (mIoU, BCD IoU) over the epochs

Plotting the loss terms separately lets you see whether the change task
(Focal) is being learned or crushed while segmentation improves.
"""

import matplotlib.pyplot as plt


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Three plots: loss, mIoU, BCD IoU. Each shows train vs validation.
    fig, (ax_loss, ax_miou, ax_bcd) = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax_loss.plot(epochs, history["train_loss"], label="train", marker="o")
    ax_loss.plot(epochs, history["val_loss"], label="validation",
                 marker="o", linestyle="--")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    # mIoU
    ax_miou.plot(epochs, history["train_miou"], label="train", marker="o")
    ax_miou.plot(epochs, history["val_miou"], label="validation",
                 marker="o", linestyle="--")
    ax_miou.set_xlabel("epoch"); ax_miou.set_ylabel("mIoU")
    ax_miou.set_title("Semantic mIoU"); ax_miou.legend(); ax_miou.grid(alpha=0.3)

    # BCD IoU
    ax_bcd.plot(epochs, history["train_bcd"], label="train", marker="o")
    ax_bcd.plot(epochs, history["val_bcd"], label="validation",
                marker="o", linestyle="--")
    ax_bcd.set_xlabel("epoch"); ax_bcd.set_ylabel("BCD IoU")
    ax_bcd.set_title("Change BCD IoU"); ax_bcd.legend(); ax_bcd.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()