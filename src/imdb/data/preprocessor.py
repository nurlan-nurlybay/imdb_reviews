import pandas as pd
import numpy as np
# from imdb.utils.logging import setup_logger
import structlog
# import logging
# import re
# from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
logger = structlog.get_logger(__name__)

class TextPreprocessor:
    """
    1. level1_cleaning
    2. add_features1
    3. level2_cleaning
    4. add_features2
    5. level3_cleaning
    """
    def __init__(self, cfg: dict):
        self.target_col = cfg['data']['target_col']
        logger.debug("Initialized TextPreprocessor", target_col=self.target_col)

    def level1_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Remove html tags
        2. Remove urls
        3. sentiment "positive"/"negative" -> 1/0
        to handle html tags we could use BeautifulSoup(text, "html.parser"), 
        but regex is faster and gets the job done if the pattern is right

        must first handle tags then urls because url pattern matches tags
        """
        logger.info("Starting Level 1 Cleaning (HTML/URLs/Target)")
        df = df.copy()
        tag_pattern = r'<(?:[a-zA-Z/][^>]*|!--.*?--)>'
        url_pattern = r'https?://\S+?(?=[.,!?;:\)]?(?:\s|$))'

        # df["review"] = df["review"].apply(lambda x: BeautifulSoup(x, "html.parser").get_text(separator=" "))
        # Slightly faster than .apply() for single-column string work
        # df["review"] = df["review"].map(lambda x: BeautifulSoup(x, "html.parser").get_text())
        df["review"] = df["review"].str.replace(tag_pattern, '', regex=True)
        df["review"] = df["review"].str.replace(url_pattern, '', regex=True)
        df["sentiment"] = np.where(df["sentiment"] == "positive", 1, 0)

        logger.debug("Level 1 Cleaning complete", shape=df.shape)
        return df

    def add_features1(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        add vader: negative <0 positive 0> score, accounts for negation 'not'
        and amlifiers like all caps, exclamation marks, or adjectives
        Struggles with sarcasm

        add upper_word_count
        save excl_count and question_count for feature
        """
        logger.info("Extracting Phase 1 Features (VADER, Uppercase, Punctuation)")
        df = df.copy()
        
        df["qm_count"] = df["review"].str.count(r'\?')
        df["em_count"] = df["review"].str.count(r'!')

        all_caps = df["review"].str.findall(r'\b[A-Z]{2,}\b')

        # "SIGNAL AHEAD" signal ignored but not ahead - some noise is okay
        ignore_set = {
            "IMDB", "IMDB", "TV", "UK", "BBC", "DVD", "CGI", "VHS", "MGM", 
            "SPOILER", "SPOILERS"
        }

        df["uppercase_count"] = [
            len([word for word in word_list if word not in ignore_set]) 
            for word_list in all_caps
        ]
        
        analyzer = SentimentIntensityAnalyzer()
        # list comprehension, faster than both map and apply
        df['vader_compound'] = [analyzer.polarity_scores(t)['compound'] for t in df['review']]
        
        logger.debug("Phase 1 Features complete")
        return df

    def level2_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting Level 2 Cleaning (Lowercasing/Punctuation removal)")

        df = df.copy()
        df["review"] = df["review"].str.lower()
        df["review"] = df["review"].str.replace(r'[^a-z]', ' ', regex=True)
        df["review"] = df["review"].str.replace(r'\s+', ' ', regex=True).str.strip()

        logger.debug("Level 2 Cleaning complete")
        return df

    def add_features2(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting Phase 2 Features (Densities, TTR)")
        df = df.copy()

        df["word_count"] = df["review"].str.split().str.len()
        safe_word_count = df["word_count"].replace(0, 1)
        df["qm_count"] = df["qm_count"] / safe_word_count
        df["em_count"] = df["em_count"] / safe_word_count
        df = df.rename(columns={'qm_count': 'qm_density', 'em_count': 'em_density'})
        df["TTR"] = [
            len(set(review.split(' '))) / len(review.split(' ')) if len(review) > 0 else 0
            for review in df["review"]
        ]

        logger.debug("Phase 2 Features complete")
        return df

    def level3_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting Level 3 Cleaning (Stopwords/Lemmatization)")
        df = df.copy()

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        df = df.rename(columns={"review": "review_standardized"})

        df["review_lemmatized"] = [
            ' '.join([
                lemmatizer.lemmatize(word) 
                for word in text.split() 
                if word not in stop_words
            ])
            for text in df["review_standardized"]
        ]

        logger.debug("Level 3 Cleaning complete")
        return df
