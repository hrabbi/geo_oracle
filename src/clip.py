# Load model directly
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import torch


def tensor_to_cropped_pil_224x224_image(image: tf.Tensor) -> Image.Image:
    """
    Crop 1008x336 panorama image to 336x336 (take center image) and resize to 224x224

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
    img = img.resize((224, 224), resample=Image.BILINEAR)
    return img


def get_clip_predictions(test_ds: tf.data.Dataset, class_labels: list) -> np.ndarray:
    """
    Run OpenAi CLIP (openai/clip-vit-large-patch14) on test data
    Model takes in 224x224 images
    Prompt supplied to CLIP: "A street view image in {country}"

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
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_pred = []
    y_true = []

    for image_batch, true_labels in tqdm(test_ds):
        batch_images = []

        for image in image_batch:
            pil_image = tensor_to_cropped_pil_224x224_image(
                image
            )  # Convert TensorFlow tensor to PIL Image
            batch_images.append(pil_image)

        # Process images using CLIP's processor
        inputs = processor(
            text=text_labels, images=batch_images, return_tensors="pt", padding=True
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
