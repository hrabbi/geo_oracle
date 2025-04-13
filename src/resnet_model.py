import tensorflow as tf


def residual_block(x, filters, stride=1):
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same")(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Shortcut connection, adjust dimensions if necessary.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, kernel_size=1, strides=stride, padding="same"
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = shortcut

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def process_segment(segment):
    x = tf.keras.layers.Conv2D(32, kernel_size=7, strides=2, padding="same")(segment)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = residual_block(x, 32, stride=1)
    return x


def create_panorama_resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocess
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)  # Scale to [-1, 1]

    # Split each panorama image into three side by side images and process them individually
    split_axis = 2  # Width dimension
    left, middle, right = tf.keras.layers.Lambda(
        lambda x: tf.split(x, num_or_size_splits=3, axis=split_axis),
        output_shape=lambda s: [(s[0], s[1], s[2] // 3, s[3])] * 3,
    )(x)

    left = process_segment(left)
    middle = process_segment(middle)
    right = process_segment(right)

    # Join segments
    x = tf.keras.layers.Concatenate(axis=split_axis)([left, middle, right])

    x = residual_block(x, 32, stride=2)
    x = residual_block(x, 16, stride=1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
