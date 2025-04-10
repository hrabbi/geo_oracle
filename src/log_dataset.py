import json
from pathlib import Path
from config import Config


def log_dataset_to_file(datasets: dict, class_names: list, run_folder: Path):
    log_data = {
        "metadata": {
            "classes": class_names,
            "image_size": (Config.TARGET_HEIGHT, Config.TARGET_WIDTH),
            "batch_size": Config.BATCH_SIZE,
        },
        "splits": {},
    }

    for split_name, dataset in datasets.items():
        total_samples = 0
        class_counts = {cls: 0 for cls in class_names}

        # Iterate through batches
        for _, labels in dataset:
            batch_size = labels.shape[0]
            total_samples += batch_size

            for label in labels.numpy():
                class_counts[class_names[label]] += 1

        log_data["splits"][split_name] = {
            "total_samples": total_samples,
            "class_distribution": class_counts,
        }

    with open(run_folder / "dataset_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
