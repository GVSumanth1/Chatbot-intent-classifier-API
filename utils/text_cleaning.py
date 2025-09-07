
"""
utils/text_cleaning.py

Reusable data cleaning functions for Amazon chatbot project.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd

def clean_review_df(df):
    """
    Clean the Amazon reviews dataframe for chatbot modeling.

    Steps:
    - Fills missing reviewText with summary or a placeholder string.
    - Fills missing summary with an empty string.
    - Adds a reviewText_imputed flag (True if reviewText was filled).
    - Drops low-value columns: vote, style, image, reviewerName.
    - Returns a cleaned copy of the DataFrame.

    Parameters:
    df (pd.DataFrame): Raw reviews DataFrame.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()  # Work on a copy for safety

    # 1. Fill missing reviewText with summary, then with placeholder
    df['reviewText'] = df['reviewText'].fillna(df['summary'])
    df['reviewText'] = df['reviewText'].fillna("No review text provided.")

    # 2. Fill missing summary values with empty string
    df['summary'] = df['summary'].fillna("")

    # 3. Add flag for imputed reviewText
    df['reviewText_imputed'] = False
    df.loc[df['reviewText'] == df['summary'], 'reviewText_imputed'] = True
    df.loc[df['reviewText'] == "No review text provided.", 'reviewText_imputed'] = True

    # 4. Drop low-value columns (if they exist)
    cols_to_drop = ['vote', 'style', 'image', 'reviewerName']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 5. Normalize reviewText and summary
    def clean_text(text):
        return text.lower().strip() if isinstance(text, str) else text

    df['reviewText'] = df['reviewText'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)

    return df
