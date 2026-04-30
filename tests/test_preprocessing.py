def test_level1_cleaning(level1_df):
    def tags_count(df):
        return df["review"].str.findall(r'<(?:[a-zA-Z/][^>]*|!--.*?--)>').explode().value_counts().sum()
    
    assert tags_count(level1_df) == 0
    assert level1_df["sentiment"].dtype == 'int64'

def test_add_features1(features1_df):
    assert "vader_compound" in features1_df.columns
    assert "uppercase_count" in features1_df.columns
    
    # Ensure our vader scores are bounded correctly
    assert features1_df["vader_compound"].max() <= 1.0
    assert features1_df["vader_compound"].min() >= -1.0

def test_level2_cleaning(level2_df):
    assert not level2_df["review"].str.contains(r'[A-Z]').any()
    assert not level2_df["review"].str.contains(r'[^a-z\s]').any()
    assert not level2_df["review"].str.contains(r'\s{2,}').any()

def test_add_features2(features2_df):
    expected_cols = ["word_count", "qm_density", "em_density", "TTR"]
    for col in expected_cols:
        assert col in features2_df.columns

    assert features2_df["qm_density"].min() >= 0
    assert features2_df["em_density"].min() >= 0
    assert features2_df["TTR"].min() >= 0
    assert features2_df["TTR"].max() <= 1.0
    
    assert "qm_count" not in features2_df.columns
    assert "em_count" not in features2_df.columns

def test_level3_cleaning(level3_df):
    assert "review_standardized" in level3_df.columns
    assert "review_lemmatized" in level3_df.columns

    test_stopwords = {"the", "is", "in", "and"}
    for word in test_stopwords:
        pattern = rf'\b{word}\b'
        assert not level3_df["review_lemmatized"].str.contains(pattern).any()

    # Assert stopwords STILL EXIST in the standardized column (DistilBERT needs them!)
    assert level3_df["review_standardized"].str.contains(r'\bthe\b').any()