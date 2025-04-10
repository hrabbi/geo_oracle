from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_history(history, save_folder: Path, filename: str):
    """
    Plot training & validation loss and accuracy, then save the plot.

    Args:
        history (_type_): _description_
        save_folder (Path): _description_
        filename (str): _description_
    """

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", marker="o")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss", marker="o")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    save_path = save_folder / filename
    plt.savefig(save_path)
    plt.close()
    print(f"Training plot saved to {save_path}")
