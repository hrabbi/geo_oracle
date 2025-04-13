import json

from pathlib import Path
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def plot_model_f1_score_per_class(report, save_folder: Path, filename: str):
    labels = list(report.keys())
    labels = [label for label in labels if label not in ["accuracy", "macro avg", "weighted avg"]]

    f1_scores = [report[label]['f1-score'] for label in labels]

    # Plotting
    plt.figure(figsize=(20, 6))
    plt.bar(labels, f1_scores, color='blue')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.show()

    save_path = save_folder / filename #   "f1_barchart.png"
    plt.savefig(save_path)
    plt.close()


def save_report(y_true, y_pred, folder: Path, name: str, class_names: list, save_f1_score_per_class: bool = False):
    """
    Creates a sklearn classification report and saves to a json file

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        folder (Path): _description_
        name (str): _description_
        class_names (list): _description_
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    with open(folder / f"{name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if save_f1_score_per_class:
        plot_model_f1_score_per_class(report, folder, f"{name}_f1_barchart.png")
