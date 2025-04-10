import json

from pathlib import Path
from sklearn.metrics import classification_report


def save_report(y_true, y_pred, folder: Path, name: str, class_names: list):
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
