import os
import sys

# Dynamically add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers import Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict

from collections import Counter

from utils.path_config import get_data_path
from utils.text_cleaning import clean_review_df
from utils.data_saving import save_with_rolling_backup
from utils.intent_mapping import apply_intent_mapping, inspect_potential_fp
import notebook_setup

# Global rare threshold
RARE_CLASS_THRESHOLD = 11

# Function: Load and prepare the dataset
def load_and_prepare_data():
    df = pd.read_csv(get_data_path("chatbot_amazon_cleaned_v1.csv"))
    df = apply_intent_mapping(df)

    # Merge review and summary
    def merge_review_and_summary(row):
        review = str(row['reviewText']) if pd.notnull(row['reviewText']) else ""
        summary = str(row['summary']) if pd.notnull(row['summary']) else ""
        if review and summary:
            return f"{review.strip()}. {summary.strip()}"
        elif review:
            return review.strip()
        elif summary:
            return summary.strip()
        else:
            return ""

    df['merged_text'] = df.apply(merge_review_and_summary, axis=1)

    # Filter high confidence samples
    gold_df = df[df['confidence_tag'] == "tier1_both"].copy()

    # Find rare classes and handle imbalance
    class_counts = gold_df['intent_label'].value_counts()
    rare_classes = class_counts[class_counts < RARE_CLASS_THRESHOLD].index.tolist()

    # Remap rare to OTHER
    gold_df['intent_label_clean'] = gold_df['intent_label'].apply(
        lambda x: "OTHER" if x in rare_classes else x
    )

    # Label encoding
    label_encoder = LabelEncoder()
    gold_df['intent_label_encoded'] = label_encoder.fit_transform(gold_df['intent_label_clean'])

    return gold_df, label_encoder


def split_data(gold_df):
    X = gold_df['merged_text']
    y = gold_df['intent_label_encoded']

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_label_names(label_encoder, show_indices=False):
    labels = list(label_encoder.classes_)
    if show_indices:
        return {i: label for i, label in enumerate(labels)}
    return labels

if __name__ == "__main__":
    print("Testing shared.py...")

    gold_df, label_encoder = load_and_prepare_data()
    print(f"Gold dataset shape: {gold_df.shape}")
    print(f"Label classes: {get_label_names(label_encoder)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(gold_df)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
