from pathlib import Path


class Config:
    # Save model and reports in folder
    RUN_PATH = Path(r"runs")

    # Preprocessing
    PLONKIT_CSV_PATH: Path = Path(r"plonkit.csv")
    COUNTRY_EMBEDDINGS_PATH: Path = Path(r"country_text_embeddings_336x336.pkl")
    DATASET_PATH: Path = Path(r"random_streetview_images_pano_v0.0.2/data")
    DATA_PATH: Path = Path(r"random_streetview_preprocessed_1008x336")

    # Resize panorama image to target height and width
    TARGET_HEIGHT: int = 336  # Original about 560
    TARGET_WIDTH: int = 336 * 3  # Original about 3030

    # Dataset
    BATCH_SIZE: int = 16  # DO NOT CHANGE
    TEST_SIZE: float = 0.15  # DO NOT CHANGE
    VAL_SIZE: float = 0.15  # DO NOT CHANGE
    SEED: int = 42

    # ResNet model training
    NUM_EPOCHS: int = 35

    #
    RUN_RESNET: bool = False
    RUN_RANDOM: bool = False
    RUN_COMMON: bool = False
    RUN_CLIP: bool = False
    RUN_STREET_CLIP: bool = False
    RUN_GEO_ORACLE: bool = False

    # GeoOracle CLIP model
    HUGGING_FACE_CLIP_MODEL: str = "openai/clip-vit-large-patch14-336"
