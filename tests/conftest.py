import pytest
import pandas as pd
from imdb.data.preprocessor import TextPreprocessor
from imdb.utils.config import load_config

@pytest.fixture(scope="module")
def preprocessor():
    cfg = load_config("configs/data.yaml")
    return TextPreprocessor(cfg)

@pytest.fixture(scope="module")
def raw_df(preprocessor):
    cfg = load_config("configs/data.yaml")
    df = pd.read_csv(cfg["paths"]["raw"], nrows=500)
    return df

@pytest.fixture(scope="module")
def level1_df(preprocessor, raw_df):
    return preprocessor.level1_cleaning(raw_df)

@pytest.fixture(scope="module")
def features1_df(preprocessor, level1_df):
    return preprocessor.add_features1(level1_df)

@pytest.fixture(scope="module")
def level2_df(preprocessor, features1_df):
    """Depends on Phase 1 features"""
    return preprocessor.level2_cleaning(features1_df)

@pytest.fixture(scope="module")
def features2_df(preprocessor, level2_df):
    """Depends on Phase 2 cleaning"""
    return preprocessor.add_features2(level2_df)

@pytest.fixture(scope="module")
def level3_df(preprocessor, features2_df):
    return preprocessor.level3_cleaning(features2_df)