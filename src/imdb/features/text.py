from typing import Tuple
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_features(
    df: pd.DataFrame, 
    text_col: str, 
    max_features: int, 
    ngram_range: Tuple[int, int],
    use_idf: bool = True
) -> Tuple[TfidfVectorizer, spmatrix]:
    """
    Builds TF-IDF features from a specified text column in a DataFrame.
    
    Args:
        df: The pandas DataFrame containing the text.
        text_col: The name of the column containing the text.
        max_features: Maximum number of features to extract.
        ngram_range: The lower and upper boundary of the range of n-values for different n-grams.
        use_idf: Enable inverse-document-frequency reweighting.
        
    Returns:
        A tuple containing the fitted TfidfVectorizer and the resulting sparse matrix.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        ngram_range=ngram_range, 
        use_idf=use_idf
    )
    
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    
    return vectorizer, tfidf_matrix
