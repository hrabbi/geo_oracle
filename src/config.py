from pathlib import Path


class Config:
    # Save model and reports in folder
    RUN_PATH = Path(r"runs")

    SEED: int = 42

    # Data
    DATA_PATH: Path = Path(r"random_streetview_preprocessed_1008x336")
    BATCH_SIZE: int = 16  # DO NOT CHANGE
    TEST_SIZE: float = 0.15  # DO NOT CHANGE
    VAL_SIZE: float = 0.15  # DO NOT CHANGE

    # Model
    NUM_EPOCHS: int = 50

    # Resize panorama image to target height and width
    TARGET_HEIGHT: int = 336  # Original about 560
    TARGET_WIDTH: int = 336 * 3  # Original about 3030

    #
    RUN_RESNET: bool = False
    RUN_RANDOM: bool = False
    RUN_COMMON: bool = False
    RUN_CLIP: bool = False
    RUN_STREET_CLIP: bool = False
