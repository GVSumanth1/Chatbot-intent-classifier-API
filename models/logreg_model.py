# logreg_train.py

import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.path_config import get_data_path
from utils.text_cleaning import clean_review_df
from utils.data_saving import save_with_rolling_backup
from utils.intent_mapping import apply_intent_mapping, inspect_potential_fp
import notebook_setup

from models.shared import load_and_prepare_data, split_data, get_label_names

save_path = "saved_models/roberta"
os.makedirs(save_path, exist_ok=True)

# Load and prepare data
gold_df, label_encoder = load_and_prepare_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(gold_df)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)

# Hyperparameter tuning
tuned_params = {
    'C': [0.01, 0.1, 1, 5, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight': ['balanced'],
    'max_iter': [1000]
}

gs = GridSearchCV(
    LogisticRegression(random_state=42),
    tuned_params,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
gs.fit(X_train_vec, y_train)
best_lr = gs.best_estimator_
print("Best Logistic Regression params:", gs.best_params_)

# Validation evaluation
val_preds = best_lr.predict(X_val_vec)
print("\nClassification Report (Validation Set):")
print(classification_report(y_val, val_preds, target_names=get_label_names(label_encoder)))

cm = confusion_matrix(y_val, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=get_label_names(label_encoder),
            yticklabels=get_label_names(label_encoder), cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Validation Confusion Matrix – Logistic Regression")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
plt.close()

# Test evaluation
test_preds = best_lr.predict(X_test_vec)
print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_preds, target_names=get_label_names(label_encoder)))

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=get_label_names(label_encoder),
            yticklabels=get_label_names(label_encoder), cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix – Logistic Regression")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
plt.close()

# === SAVE LOGIC WRAPPED SAFELY ===
if __name__ == "__main__":
    os.makedirs("saved_models/logreg", exist_ok=True)

    joblib.dump(best_lr, "saved_models/logreg/logreg_model.pkl")
    joblib.dump(vectorizer, "saved_models/logreg/logreg_vectorizer.pkl")
    joblib.dump(label_encoder, "saved_models/logreg/logreg_label_encoder.pkl")

    print("\nLogistic Regression model, vectorizer, and label encoder saved to 'saved_models/logreg/'")
