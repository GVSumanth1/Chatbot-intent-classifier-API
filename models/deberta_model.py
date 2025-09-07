# deberta_train.py

import pandas as pd
import numpy as np
import torch
import pickle
import os

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DebertaV2Tokenizer
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

save_path = "saved_models/roberta"
os.makedirs(save_path, exist_ok=True)

from shared import load_and_prepare_data, split_data, get_label_names

def train_deberta():
    # Load and prepare data
    gold_df, label_encoder = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(gold_df)

    # Combine into DataFrames
    train_df = pd.DataFrame({"merged_text": X_train, "label": y_train})
    val_df = pd.DataFrame({"merged_text": X_val, "label": y_val})
    test_df = pd.DataFrame({"merged_text": X_test, "label": y_test})

    # Create Hugging Face DatasetDict
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    # Load DeBERTa tokenizer
    model_name = "microsoft/deberta-v3-small"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(
            example["merged_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # Tokenize and format
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load DeBERTa model
    label_names = list(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_names)
    )

    # Define metric computation function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir="./results_deberta",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # Evaluate
    preds_output = trainer.predict(tokenized_datasets["test"])
    y_true = preds_output.label_ids
    y_pred = preds_output.predictions.argmax(axis=1)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DeBERTa Confusion Matrix")
    # plt.show()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()


    # Save model and tokenizer
    save_dir = "saved_models/deberta/"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save label encoder
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"DeBERTa model, tokenizer, and label encoder saved to {save_dir}")

    return model, tokenizer, label_encoder


if __name__ == "__main__":
    train_deberta()
