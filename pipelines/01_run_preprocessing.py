"""
python -m pipelines.01_run_preprocessing
"""
from imdb.data.preprocessor import TextPreprocessor
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
import logging
import structlog
import pandas as pd
import os

setup_logger(
    log_file="logs/01_run_preprocessing.log",
    terminal_level=logging.INFO,
    file_level=logging.DEBUG
)
logger = structlog.get_logger(__name__)

def preprocess_data(paths: dict, params: dict) -> None:
    try: 
        df = pd.read_csv(paths["data"]["raw"])
        
        # We pass a combined config to TextPreprocessor so it has what it needs
        combined_cfg = {"data": params["data"]}
        processor = TextPreprocessor(combined_cfg)
        
        logger.info("Preprocessing pipeline started", input_file=paths["data"]["raw"])
        df = processor.level1_cleaning(df)
        df = processor.add_features1(df)
        df = processor.level2_cleaning(df)
        df = processor.add_features2(df)
        df = processor.level3_cleaning(df)

        os.makedirs(os.path.dirname(paths["data"]["processed"]), exist_ok=True)
        df.to_parquet(paths["data"]["processed"])
        logger.info("Processed dataset saved successfully", path=paths["data"]["processed"])
    except Exception:
        logger.exception("Something went wrong.")

if __name__ == "__main__":
    # Load the newly split configs
    paths_cfg = load_config("configs/paths.yaml")
    params_cfg = load_config("configs/params.yaml")
    preprocess_data(paths_cfg, params_cfg)
