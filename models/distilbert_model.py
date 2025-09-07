# distilbert_model.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, EarlyStoppingCallback

from utils.path_config import get_data_path
from utils.text_cleaning import clean_review_df
from utils.data_saving import save_with_rolling_backup
from utils.intent_mapping import apply_intent_mapping, inspect_potential_fp
import notebook_setup

from models.shared import load_and_prepare_data, split_data, get_label_names

save_path = "saved_models/roberta"
os.makedirs(save_path, exist_ok=True)

# Load and split data
gold_df, label_encoder = load_and_prepare_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(gold_df)

# Confirm splits
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Define model-specific tokenizer
tokenizer_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["merged_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Combine text and label into new dataframes
train_df = pd.DataFrame({"merged_text": X_train, "label": y_train})
val_df   = pd.DataFrame({"merged_text": X_val, "label": y_val})
test_df  = pd.DataFrame({"merged_text": X_test, "label": y_test})

# Create Hugging Face DatasetDict
raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
})

# Tokenize all splits
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    load_from_cache_file=False  # Ensures fresh tokenization
)

# Format for PyTorch (input_ids, attention_mask, label)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Extract individual datasets
train_dataset = tokenized_datasets["train"]
val_dataset   = tokenized_datasets["validation"]
test_dataset  = tokenized_datasets["test"]

from transformers import AutoModelForSequenceClassification, TrainingArguments

# Define number of labels
num_labels = len(label_encoder.classes_)

# Load DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/distilbert-intent",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs/distilbert"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


# Traniner setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)

print("\nTest Set Metrics:")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# Predict on test set
preds_output = trainer.predict(test_dataset)
y_true = preds_output.label_ids
y_pred = preds_output.predictions.argmax(axis=1)

# Decode predictions
true_labels = label_encoder.inverse_transform(y_true)
predicted_labels = label_encoder.inverse_transform(y_pred)

# Classification report
print("\nClassification Report (Test Set):")
print(classification_report(true_labels, predicted_labels, target_names=get_label_names(label_encoder)))

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=get_label_names(label_encoder))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=get_label_names(label_encoder),
            yticklabels=get_label_names(label_encoder))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Test Set")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
plt.close()


# Print best checkpoint info
print("\nTraining completed.")
print("Best model checkpoint path:", trainer.state.best_model_checkpoint)
print(f"Stopped at epoch: {trainer.state.epoch:.1f}")

# Optional: reload best model checkpoint
from transformers import AutoModelForSequenceClassification

best_model_path = trainer.state.best_model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(best_model_path, num_labels=num_labels)
trainer.model = model
