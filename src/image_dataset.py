import tensorflow as tf

from config import Config


def get_image_dataset():
    """
    Load images from Config.DATA_PATH folder into a dataset and split into train, val, test

    Returns:
        _type_: _description_
    """

    train_ds, val_test_ds = tf.keras.utils.image_dataset_from_directory(
        Config.DATA_PATH,
        validation_split=(Config.VAL_SIZE + Config.TEST_SIZE),
        subset="both",
        seed=Config.SEED,
        image_size=(Config.TARGET_HEIGHT, Config.TARGET_WIDTH),
        batch_size=Config.BATCH_SIZE,
    )

    class_names = train_ds.class_names

    # Get the number of batches
    train_samples = train_ds.cardinality().numpy()
    val_test_samples = val_test_ds.cardinality().numpy()

    print(f"Total batches: {train_samples + val_test_samples}")

    val_split = Config.VAL_SIZE / (Config.VAL_SIZE + Config.TEST_SIZE)
    # Calculate sizes
    val_size = int(val_split * val_test_samples)

    # Split the dataset
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)

    for image_batch, labels_batch in train_ds:
        print(f"Image training batch shape: {image_batch.shape}")
        print(f"Label training batch shape: {labels_batch.shape}")
        break

    return train_ds, val_ds, test_ds, class_names
