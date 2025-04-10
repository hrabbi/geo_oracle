# Load model directly
from transformers import CLIPProcessor, TFCLIPModel
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def tensor_to_cropped_pil_image(image: tf.Tensor) -> Image.Image:
    """
    Crop 1008x336 panorama image to 336x336 (take center image)

    Args:
        image (tf.Tensor): _description_

    Returns:
        Image.Image: _description_
    """
    image = tf.cast(image, tf.uint8).numpy()  # Convert to NumPy
    img = Image.fromarray(image)

    width, height = img.size
    crop_width = width // 3
    left = crop_width
    right = 2 * crop_width

    # Area (left, upper, right, lower)
    area = (left, 0, right, height)
    img = img.crop(area)

    return img


def get_clip_predictions(test_ds: tf.data.Dataset, class_labels: list) -> np.ndarray:
    """
    Run OpenAi CLIP (large patch 14) on test data

    Args:
        test_ds (tf.data.Dataset): _description_
        class_labels (list): _description_

    Returns:
        np.ndarray: _description_
    """

    text_prompt = "A street view image in "
    text_labels = []
    for label in class_labels:
        text_labels.append(text_prompt + label)

    # Load CLIP model and processor
    model = TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    y_pred = []
    y_true = []

    for image_batch, true_labels in tqdm(test_ds):
        batch_images = []

        for image in image_batch:
            pil_image = tensor_to_cropped_pil_image(
                image
            )  # Convert TensorFlow tensor to PIL Image
            batch_images.append(pil_image)

        # Process images using CLIP's processor
        inputs = processor(
            text=text_labels, images=batch_images, return_tensors="tf", padding=True
        )

        # Run inference
        outputs = model(**inputs)

        # Get predicted class indices
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = tf.nn.softmax(logits_per_image, axis=1)  # Convert to probabilities
        predicted_idx = tf.argmax(probs, axis=1).numpy()

        y_pred.extend(predicted_idx)
        y_true.extend(true_labels.numpy())

    return np.array(y_true), np.array(y_pred)
