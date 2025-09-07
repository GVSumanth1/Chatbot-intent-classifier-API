# save_roberta_final.py

import os
import sys
import pickle
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

# Set project root
project_root = os.path.abspath(".")  # Points to case_studyfiles/
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root set to:", project_root)

# import os
# import sys
# import pickle
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports from your shared module
from models.shared import load_and_prepare_data, split_data, get_label_names

# Load and prepare data
gold_df, label_encoder = load_and_prepare_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(gold_df)

# Create test dataset
test_df = pd.DataFrame({"merged_text": X_test, "label": y_test})
raw_datasets = DatasetDict({"test": Dataset.from_pandas(test_df)})

# Load tokenizer and tokenize test set
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(example):
    return tokenizer(
        example["merged_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset = tokenized_datasets["test"]

# Find best checkpoint
checkpoint_dirs = glob.glob("./models/roberta-base-intent/checkpoint-*")
if not checkpoint_dirs:
    raise ValueError("No checkpoint directories found in roberta-base-intent.")
best_model_path = checkpoint_dirs[-1]
print("Best checkpoint:", best_model_path)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(best_model_path, num_labels=len(label_encoder.classes_))

# Run predictions
model.eval()
from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=32)
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        all_logits.append(logits)
        all_labels.append(labels)

logits = torch.cat(all_logits)
labels = torch.cat(all_labels)
preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

# Decode
true_labels = label_encoder.inverse_transform(labels.numpy())
predicted_labels = label_encoder.inverse_transform(preds.numpy())

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=get_label_names(label_encoder)))

# Save model and tokenizer
save_path = "saved_models/roberta/"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
with open(os.path.join(save_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
print("Saved model, tokenizer, label encoder to", save_path)

# Plot and save confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=get_label_names(label_encoder))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=get_label_names(label_encoder),
            yticklabels=get_label_names(label_encoder))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("RoBERTa Confusion Matrix – Test Set")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
plt.close()
print("Saved confusion matrix to", os.path.join(save_path, "confusion_matrix.png"))
