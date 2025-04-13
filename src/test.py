import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import Config

if __name__ == "__main__":
    with open(Config.COUNTRY_EMBEDDINGS_PATH, "rb") as f:
        country_text_embeddings = pickle.load(f)

    # test_image_path = r"random_streetview_preprocessed_1008x336/Iceland/image_1604.jpg"
    test_image_path = r"random_streetview_preprocessed_1008x336/Sweden/image_528.jpg"
    # test_image_path = r"random_streetview_preprocessed_1008x336/Netherlands/image_336.jpg"
    # test_image_path = r"random_streetview_preprocessed_1008x336/Greece/image_534.jpg"

    img = Image.open(test_image_path)
    width, height = img.size
    crop_width = width // 3
    images = []
    for i in range(3):
        left = i * crop_width
        right = (i + 1) * crop_width
        area = (left, 0, right, height)
        images.append(img.crop(area))

    model = CLIPModel.from_pretrained(Config.HUGGING_FACE_CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(Config.HUGGING_FACE_CLIP_MODEL)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    country_scores = defaultdict(float)

    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        image_features = F.normalize(image_features, p=2, dim=-1)[0]

        for country, embeddings_list in country_text_embeddings.items():
            # Convert list of numpy arrays to a torch tensor (shape: [num_texts, embedding_dim])
            text_embs = torch.tensor(np.stack(embeddings_list))

            # Normalize text embeddings
            text_embs = F.normalize(text_embs, p=2, dim=-1)

            # Compute cosine similarity between the image embedding and each text embedding.
            # image_embedding has shape (embedding_dim,), add a batch dimension
            image_emb = image_features.unsqueeze(0)  # shape: (1, embedding_dim)
            sims = torch.matmul(image_emb, text_embs.T)  # shape: (1, num_texts)

            top_k = 5
            topk_sims, _ = torch.topk(sims, k=top_k, dim=1)  # shape: (1, k)
            score = topk_sims.mean().item()
            # Average the similarity scores for this country.
            # alpha = 0.7
            # avg_sim = sims.mean().item()
            # max_sim = sims.max().item()
            country_scores[country] += (
                score  # (alpha * max_sim + (1 - alpha) * avg_sim) # ** 2
            )

    sorted_scores = sorted(country_scores.items(), key=lambda x: x[1], reverse=True)

    print("Top country predictions for the image:")
    for country, score in sorted_scores[:5]:
        print(f"{country}: Score = {score:.4f}")
    print("")
