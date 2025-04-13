import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def tensor_to_cropped_pil_336x336_image(image: tf.Tensor) -> Image.Image:
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


def get_street_clip_predictions(
    test_ds: tf.data.Dataset, class_labels: list
) -> np.ndarray:
    """
    Run StreetCLIP on test data (336x336 images)
    Prompt supplied to StreetCLIP: "{country}"

    Args:
        test_ds (tf.data.Dataset): _description_
        class_labels (list): _description_

    Returns:
        np.ndarray: _description_
    """

    choices = class_labels

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_pred = []
    y_true = []

    for image_batch, true_labels in tqdm(test_ds):
        batch_images = []

        for image in image_batch:
            pil_image = tensor_to_cropped_pil_336x336_image(
                image
            )  # Convert TensorFlow tensor to PIL Image
            batch_images.append(pil_image)

        # Process images using CLIP's processor
        inputs = processor(
            text=choices, images=batch_images, return_tensors="pt", padding=True
        ).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted class indices
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = torch.softmax(logits_per_image, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).cpu().numpy()

        y_pred.extend(predicted_idx)
        y_true.extend(true_labels.numpy())

    return np.array(y_true), np.array(y_pred)
