import tensorflow as tf
from config import Config


def get_image_dataset():
    """
    Load images from Config.DATA_PATH folder into a dataset and split into train, val, test

    Returns:
        _type_: _description_
    """

    full_ds = tf.keras.utils.image_dataset_from_directory(
        Config.DATA_PATH,
        seed=Config.SEED,
        image_size=(Config.TARGET_HEIGHT, Config.TARGET_WIDTH),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
    )

    class_names = full_ds.class_names

    # Get the number of batches
    ds_size = full_ds.cardinality().numpy()
    print(f"Number of total batches: {ds_size}")

    # Define the split ratios
    train_split = 1 - (Config.VAL_SIZE + Config.TEST_SIZE)
    val_split = Config.VAL_SIZE

    # Calculate sizes
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    # Split the dataset
    train_ds = full_ds.take(train_size)
    remaining = full_ds.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)

    for image_batch, labels_batch in train_ds:
        print(f"Image training batch shape: {image_batch.shape}")
        print(f"Label training batch shape: {labels_batch.shape}")
        break

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
