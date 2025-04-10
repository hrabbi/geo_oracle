from transformers import CLIPProcessor, CLIPModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import torch

from clip import tensor_to_cropped_pil_image


def get_street_clip_predictions(
    test_ds: tf.data.Dataset, class_labels: list
) -> np.ndarray:
    """
    Run StreetCLIP on test data (336x336 images)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_pred = []
    y_true = []

    for image_batch, true_labels in tqdm(test_ds):
        batch_images = []

        # TODO: We crop the images
        for image in image_batch:
            pil_image = tensor_to_cropped_pil_image(
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
        # probs = tf.nn.softmax(logits_per_image, axis=1)  # Convert to probabilities
        # predicted_idx = tf.argmax(probs, axis=1).numpy()
        probs = torch.softmax(logits_per_image, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).cpu().numpy()

        y_pred.extend(predicted_idx)
        y_true.extend(true_labels.numpy())

    return np.array(y_true), np.array(y_pred)
