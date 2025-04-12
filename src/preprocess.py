from pathlib import Path
import pandas as pd
from tqdm import tqdm
from iso_to_country_name import iso_to_country_name
from PIL import Image
import io
from config import Config


def main(dataset_path: Path, preprocessed_dataset_path: Path):
    """
    Preprocess dataset comprised of .parquet files.
    Resizes the jpeg images to desired dimensions and outputs them to a folder under its country

    Args:
        dataset_path (Path): _description_
        preprocessed_dataset_path (Path): _description_
    """
    files = dataset_path.glob("*.parquet")
    df = pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True
    )

    preprocessed_dataset_path.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Extract image data and country code
            image_bytes = row["image"]["bytes"]
            country_code = row["country_iso_alpha2"]

            # Get country name
            country_name = iso_to_country_name[country_code]

            country_dir = preprocessed_dataset_path / country_name
            country_dir.mkdir(exist_ok=True)

            img = Image.open(io.BytesIO(image_bytes))

            # Resize to (Config.TARGET_WIDTH, Config.TARGET_HEIGHT)
            img = img.resize(
                (Config.TARGET_WIDTH, Config.TARGET_HEIGHT), resample=Image.BILINEAR
            )

            filename = f"image_{idx}.jpg"
            img.save(country_dir / filename, "JPEG")

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue


if __name__ == "__main__":
    dataset_path = Config.DATASET_PATH
    preprocessed_dataset_path = Config.DATA_PATH
    main(dataset_path, preprocessed_dataset_path)
