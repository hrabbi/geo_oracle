from config import Config
import numpy as np
import tensorflow as tf
from resnet_model import create_panorama_resnet
from datetime import datetime
from pathlib import Path
from report import save_report
from plot import plot_training_history
from clip import get_clip_predictions
from image_dataset import get_image_dataset
from collections import Counter
from street_clip import get_street_clip_predictions


def main(run_folder: Path):

    # Load and split dataset
    train_ds, val_ds, test_ds, class_names = get_image_dataset()

    num_classes = len(class_names)

    print(f"Training batches: {train_ds.cardinality().numpy()}")
    print(f"Validation batches: {val_ds.cardinality().numpy()}")
    print(f"Test batches: {test_ds.cardinality().numpy()}")
    print(f"Class names: {class_names}")
    print(f"Num classes: {num_classes}")

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

        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_accuracy",
        #     patience=3,
        #     restore_best_weights=True,
        #     verbose=1,
        # )

        history = model.fit(
            train_ds,
            epochs=Config.NUM_EPOCHS,
            validation_data=val_ds,
            # callbacks=[early_stopping],
        )

        model_save_path = run_folder / "resnet_model.keras"
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
        plot_training_history(history, run_folder, "model_training_plot.png")

    # Create classification reports
    y_test_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    # Random
    if Config.RUN_RANDOM:
        print("Generating report for random guess")
        y_random_pred = np.random.randint(num_classes, size=len(y_test_true))
        save_report(y_test_true, y_random_pred, run_folder, "random", class_names)

    # Most common
    if Config.RUN_COMMON:
        print("Generating report for most common guess")
        train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
        most_common_class = Counter(train_labels).most_common(1)[0][0]
        y_common_pred = np.full_like(y_test_true, most_common_class)
        save_report(y_test_true, y_common_pred, run_folder, "common", class_names)

    # Evaluate model and get report
    if Config.RUN_RESNET:
        print("Generating report for model")
        y_pred_model = model.predict(test_ds)
        y_pred_model = np.argmax(y_pred_model, axis=1)
        save_report(y_test_true, y_pred_model, run_folder, "model", class_names)

    # Evaluate on CLIP
    if Config.RUN_CLIP:
        print("Generating report for CLIP")
        y_true_clip, y_pred_clip = get_clip_predictions(test_ds, class_names)
        save_report(y_true_clip, y_pred_clip, run_folder, "CLIP", class_names)

    # Evaluate on StreetCLIP
    if Config.RUN_STREET_CLIP:
        print("Generating report for StreetCLIP")
        y_true_street_clip, y_pred_street_clip = get_street_clip_predictions(test_ds, class_names)
        save_report(y_true_street_clip, y_pred_street_clip, run_folder, "STREET_CLIP", class_names)


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y-%#m-%#d-%H-%M")
    run_folder = Config.RUN_PATH / time_stamp
    run_folder.mkdir()

    # print("GPUs Available:", tf.config.list_physical_devices("GPU"))
    main(run_folder)
