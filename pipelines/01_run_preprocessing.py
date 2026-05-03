"""
Pipeline 01: Preprocessing
"""
import os
import logging
import pandas as pd
import structlog

from imdb.utils.config import load_config
from imdb.utils.logging import setup_logger
from imdb.data.preprocessor import TextPreprocessor

setup_logger("logs/01_run_preprocessing.log", terminal_level=logging.INFO, file_level=logging.DEBUG)
logger = structlog.get_logger(__name__)

def main() -> None:
    logger.info("pipeline_start", pipeline="01_run_preprocessing")
    cfg_paths = load_config("configs/paths.yaml")
    cfg_params = load_config("configs/params.yaml")
    
    try: 
        df = pd.read_csv(cfg_paths["data"]["raw"])
        
        processor = TextPreprocessor({"data": cfg_params["data"]})
        
        logger.info("cleaning_started", input_file=cfg_paths["data"]["raw"])
        df = processor.level1_cleaning(df)
        df = processor.add_features1(df)
        df = processor.level2_cleaning(df)
        df = processor.add_features2(df)
        df = processor.level3_cleaning(df)

        out_path = cfg_paths["data"]["processed"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path)
        
        logger.info("pipeline_done", path=out_path)
    except Exception:
        logger.exception("pipeline_failed")

if __name__ == "__main__":
    main()
