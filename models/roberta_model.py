# roberta_train.py
# Setting up environment and importing necessary libraries for data handling etc.
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)

from utils.path_config import get_data_path
from utils.text_cleaning import clean_review_df
from utils.data_saving import save_with_rolling_backup
from utils.intent_mapping import apply_intent_mapping, inspect_potential_fp
import notebook_setup

from models.shared import load_and_prepare_data, split_data, get_label_names

save_path = "saved_models/roberta"
os.makedirs(save_path, exist_ok=True)

# Load cleaned and labeled data, spliting into train, validation, and test sets.

# === Core Logic ===
def train_roberta():
    gold_df, label_encoder = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(gold_df)
    
    save_path = "saved_models/roberta"
    os.makedirs(save_path, exist_ok=True)

    # Initialize the tokenizer with the roberta-base model
    tokenizer_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define tokenization function
    def tokenize_function(example):
        return tokenizer(
            example["merged_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # Converting data to Hugging Face Dataset format
    train_df = pd.DataFrame({"merged_text": X_train, "label": y_train})
    val_df   = pd.DataFrame({"merged_text": X_val, "label": y_val})
    test_df  = pd.DataFrame({"merged_text": X_test, "label": y_test})

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    # Tokenizing all splits and sets format to pytorch for training
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, load_from_cache_file=False)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_datasets["train"]
    val_dataset   = tokenized_datasets["validation"]
    test_dataset  = tokenized_datasets["test"]

    # Loads classified head on top of RoBeRta Model for multiple intent  classes
    num_labels = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=num_labels)

    # Training Parameters for the Trainer API
    training_args = TrainingArguments(
        output_dir="./models/roberta-base-intent",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs/roberta-base",
        logging_steps=20,
        save_total_limit=2
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

    # initialize Hugging Face Trainer and train the model using early stopping.
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

    best_model_path = trainer.state.best_model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path, num_labels=num_labels)
    trainer.model = model

    # Evaluate the model on the test and validation set
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Set Metrics:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

    # Predictions
    preds_output = trainer.predict(test_dataset)
    y_true = preds_output.label_ids
    y_pred = preds_output.predictions.argmax(axis=1)
    true_labels = label_encoder.inverse_transform(y_true)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    print("\nClassification Report (Test Set):")
    print(classification_report(true_labels, predicted_labels, target_names=get_label_names(label_encoder)))

    # === End of Core Logic ===

    # Save the model, tokenizer, and label encoder
    save_path = "saved_models/roberta/"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

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

    

    with open(os.path.join(save_path, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nFinal RoBERTa model and components saved to", save_path)

# Ensures the training function runs when this script is executed directly
if __name__ == "__main__":
    train_roberta()
