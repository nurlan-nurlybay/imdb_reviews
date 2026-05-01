"""
python -m pipelines.run_preprocessing
"""
from imdb.data.preprocessor import TextPreprocessor
from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
import logging
import structlog
import pandas as pd
import os

setup_logger(
    log_file="logs/run_preprocessing.log",
    terminal_level=logging.INFO,
    file_level=logging.DEBUG
)
logger = structlog.get_logger(__name__)

cfg = load_config("configs/data.yaml")
df = pd.read_csv(cfg["paths"]["raw"])
processor = TextPreprocessor(cfg)

try: 
    logger.info("Preprocessing pipeline started", **cfg)
    df = processor.level1_cleaning(df)
    df = processor.add_features1(df)
    df = processor.level2_cleaning(df)
    df = processor.add_features2(df)
    df = processor.level3_cleaning(df)

    os.makedirs(os.path.dirname(cfg["paths"]["processed"]), exist_ok=True)
    df.to_parquet(cfg["paths"]["processed"])
    logger.info("Processed dataset saved successfully", path=cfg["paths"]["processed"])
except Exception:
    logger.exception("Something went wrong.")
