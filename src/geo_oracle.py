# Load model directly
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle
from config import Config


def tensor_to_split_pil_336x336_images(image: tf.Tensor) -> list[Image.Image]:
    """
    Split a 1008x336 panorama image into three 336x336 images (left, center, right).
    Args:
        image (tf.Tensor): _description_
    Returns:
        list[Image.Image]: _description_
    """
    image = tf.cast(image, tf.uint8).numpy()  # Convert to NumPy
    img = Image.fromarray(image)
    width, height = img.size
    crop_width = width // 3
    images = []
    for i in range(3):
        left = i * crop_width
        right = (i + 1) * crop_width
        area = (left, 0, right, height)
        images.append(img.crop(area))
    return images


def get_geo_oracle_predictions(
    test_ds: tf.data.Dataset, class_labels: list
) -> np.ndarray:
    """
    Run OpenAi CLIP (openai/clip-vit-large-patch14-336) on test data
    Model takes in 336x336 images
    Split 1008x336 panorama images into three 336x336 images and run each image through CLIP
    Compare each image embedding to the precomputed text embeddings from plonkit.csv
    Text embeddings created as: "A street view image in {country} featuring distinct {raw_text}"
    Take average/max/topk from each image and add together
    Predicts the country with the highest score

    Args:
        test_ds (tf.data.Dataset): _description_
        class_labels (list): _description_

    Returns:
        np.ndarray: _description_
    """

    with open(Config.COUNTRY_EMBEDDINGS_PATH, "rb") as f:
        country_text_embeddings = pickle.load(f)

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained(Config.HUGGING_FACE_CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(Config.HUGGING_FACE_CLIP_MODEL)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true = []
    y_pred = []

    y_true_country_name = []
    y_pred_top3_names = []

    # Split batches
    for image_batch, true_labels in tqdm(test_ds):

        batch_true_labels = true_labels.numpy()

        # Run on each image
        for image, label in zip(image_batch, true_labels):
            # Split each panorama image
            split_images = tensor_to_split_pil_336x336_images(image)

            country_scores = defaultdict(float)

            # Get score for each image
            for img in split_images:
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=-1)[0]

                for country in class_labels:
                    if country in country_text_embeddings:
                        text_embs = torch.tensor(
                            np.stack(country_text_embeddings[country])
                        ).to(device)
                        text_embs = F.normalize(text_embs, p=2, dim=-1)
                        sims = torch.matmul(image_features.unsqueeze(0), text_embs.T)
                        avg_sim = sims.mean().item()
                        # max_sim = sims.max().item()
                        # top_k = 3
                        # topk_sims, _ = torch.topk(sims, k=top_k, dim=1)  # shape: (1, k)
                        # score = topk_sims.mean().item()
                        country_scores[country] += avg_sim
                    else:
                        country_scores[country] = 0

            predicted_country = max(country_scores, key=country_scores.get)
            # Convert the predicted country string to its corresponding index using the class_names list.
            predicted_idx = class_labels.index(predicted_country)
            y_pred.append(predicted_idx)

            # Also return a list for top 3 results
            y_true_country_name.append(class_labels[int(label.numpy())])
            top_3 = sorted(country_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_3_country_names = [x[0] for x in top_3]
            y_pred_top3_names.append(top_3_country_names)

        y_true.extend(batch_true_labels)

    return np.array(y_true), np.array(y_pred), y_true_country_name, y_pred_top3_names
