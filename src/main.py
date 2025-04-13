from config import Config
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path

from image_dataset import get_image_dataset
from collections import Counter

from utils import save_report, plot_training_history, log_dataset_to_file

from models import (
    get_most_common_predictions,
    get_random_predictions,
    create_panorama_resnet,
    get_clip_predictions,
    get_street_clip_predictions,
    get_geo_oracle_predictions,
)


def main(run_folder: Path):

    # Load and split dataset
    train_ds, val_ds, test_ds, class_names = get_image_dataset()

    num_classes = len(class_names)

    print(f"Training batches: {train_ds.cardinality().numpy()}")
    print(f"Validation batches: {val_ds.cardinality().numpy()}")
    print(f"Test batches: {test_ds.cardinality().numpy()}")
    print(f"Class names: {class_names}")
    print(f"Num classes: {num_classes}")

    log_dataset_to_file(
        {"training": train_ds, "validation": val_ds, "test": test_ds},
        class_names,
        run_folder,
    )

    if Config.RUN_RESNET:
        model = create_panorama_resnet(
            input_shape=(Config.TARGET_HEIGHT, Config.TARGET_WIDTH, 3),
            num_classes=num_classes,
        )
        print(model.summary())

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            ),  # We softmax at the end
            metrics=["accuracy"],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )

        history = model.fit(
            train_ds,
            epochs=Config.NUM_EPOCHS,
            validation_data=val_ds,
            callbacks=[early_stopping],
        )

        model_save_path = run_folder / "resnet_model.keras"
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
        plot_training_history(history, run_folder, "model_training_plot.png")

    # Create classification reports
    test_ds = test_ds.cache()  # Important!

    # Random
    if Config.RUN_RANDOM:
        print("Generating report for random guess")
        y_random_true, y_random_pred = get_random_predictions(test_ds, class_names)
        save_report(y_random_true, y_random_pred, run_folder, "random", class_names)

    # Most common
    if Config.RUN_COMMON:
        print("Generating report for most common guess")
        train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
        most_common_class, count = Counter(train_labels).most_common(1)[0]
        print(
            f"Most common class in training set: {class_names[most_common_class]}, index: {most_common_class}, count: {count}"
        )
        y_common_true, y_common_pred = get_most_common_predictions(
            test_ds, most_common_class
        )
        save_report(y_common_true, y_common_pred, run_folder, "common", class_names)

    # Evaluate model and get report
    if Config.RUN_RESNET:
        print("Generating report for model")
        y_test_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
        y_pred_model = model.predict(test_ds)
        y_pred_model = np.argmax(y_pred_model, axis=1)  # TODO: fix
        save_report(y_test_true, y_pred_model, run_folder, "model", class_names)

    # Evaluate on CLIP
    if Config.RUN_CLIP:
        print("Generating report for CLIP")
        y_true_clip, y_pred_clip = get_clip_predictions(test_ds, class_names)
        save_report(y_true_clip, y_pred_clip, run_folder, "CLIP_224x224", class_names)

    # Evaluate on StreetCLIP
    if Config.RUN_STREET_CLIP:
        print("Generating report for StreetCLIP")
        y_true_street_clip, y_pred_street_clip = get_street_clip_predictions(
            test_ds, class_names
        )
        save_report(
            y_true_street_clip,
            y_pred_street_clip,
            run_folder,
            "STREET_CLIP_336x336",
            class_names,
        )

    # Evaluate on GeoOracle
    if Config.RUN_GEO_ORACLE:
        print("Generating report for GeoOracle")
        y_true_geo_oracle, y_pred_geo_oracle, y_true_country_name, y_pred_top3_names = (
            get_geo_oracle_predictions(test_ds, class_names)
        )
        save_report(
            y_true_geo_oracle,
            y_pred_geo_oracle,
            run_folder,
            "GeoOracle",
            class_names,
            True,
        )

        top_3_true = []
        top_3_pred = []
        for true_country_name, top_3_country_names in zip(
            y_true_country_name, y_pred_top3_names
        ):
            top_3_true.append(1)
            if true_country_name in top_3_country_names:
                top_3_pred.append(1)
            else:
                top_3_pred.append(0)
        save_report(
            top_3_true, top_3_pred, run_folder, "GeoOracleTop3", ["False", "True"]
        )


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y-%#m-%#d-%H-%M")
    run_folder = Config.RUN_PATH / time_stamp
    run_folder.mkdir()

    # print("GPUs Available:", tf.config.list_physical_devices("GPU"))
    main(run_folder)
