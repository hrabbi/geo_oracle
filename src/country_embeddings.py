import pickle
import re
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from config import Config
from utils import links


def is_shared_feature(text: str, current_country: str, countries: list) -> bool:
    """
    Returns true or false if another country other than the current conutry is named in text

    Args:
        text (str): _description_
        current_country (str): _description_
        countries (list): _description_

    Returns:
        bool: _description_
    """
    other_countries = [c for c in countries if c != current_country]
    escaped_countries = map(re.escape, other_countries)
    pattern = r"\b(" + "|".join(escaped_countries) + r")\b"
    is_another_country_name_present_in_text = bool(
        re.search(pattern, text, flags=re.IGNORECASE)
    )
    return is_another_country_name_present_in_text


if __name__ == "__main__":
    """
    Precompute text embeddings for each country and dump to .pickle file
    Prompt supplied to CLIP: "A street view image in {country} featuring distinct {raw_text}"

    Returns:
        _type_: _description_
    """
    url_to_country = {v: k for k, v in links.items()}
    countries = [x for x in links.keys()]

    df = pd.read_csv(Config.PLONKIT_CSV_PATH)

    model = CLIPModel.from_pretrained(Config.HUGGING_FACE_CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(Config.HUGGING_FACE_CLIP_MODEL)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def get_text_embedding(text: str):
        inputs = processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features[0].cpu().numpy()

    country_text_embeddings = {}
    country_raw_texts = defaultdict(list)  # Store raw texts first

    for _, row in tqdm(df.iterrows()):
        url = row["web-scraper-start-url"]
        text = row["text"]

        country = url_to_country.get(url)
        if not country or not isinstance(text, str) or len(text) < 5:
            continue

        if is_shared_feature(text, country, countries):
            # print(f"Shared feature in text (removed): {text}")
            continue
        country_raw_texts[country].append(text)

    # Option for more text processing

    # Now generate embeddings for prompts
    for country, texts in tqdm(country_raw_texts.items()):
        prompt = f"A street view image in {country} featuring distinct "
        embeddings = []
        for text in texts:
            full_text = prompt + text
            embedding = get_text_embedding(full_text)
            embeddings.append(embedding)

        country_text_embeddings[country] = embeddings

    with open(Config.COUNTRY_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(country_text_embeddings, f)
