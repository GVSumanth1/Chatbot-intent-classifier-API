# rnn_train.py
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

from shared import load_and_prepare_data

save_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "rnn")
os.makedirs(save_path, exist_ok=True)


# Tokenizer and vocab
def simple_tokenizer(text):
    return text.lower().split()

def build_vocab(gold_df):
    all_tokens = [token for text in gold_df['merged_text'] for token in simple_tokenizer(text)]
    token_freqs = Counter(all_tokens)
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(token_freqs.items())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def text_to_sequence(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenizer(text)]


# Dataset and collate
class RNNTextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.stack(labels)


# Model
class BiLSTMWithAttention(nn.Module): # Attention Weights
    def __init__(self, embedding_matrix, hidden_dim, output_dim, dropout=0.3, num_layers=2):
        super(BiLSTMWithAttention, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(context)
        return self.fc(out)


# Training loop

def train_phd_rnn_model():
    gold_df, _ = load_and_prepare_data()

    vocab = build_vocab(gold_df)
    gold_df["token_ids"] = gold_df["merged_text"].apply(lambda x: text_to_sequence(x, vocab))

    label_encoder = LabelEncoder()
    gold_df["label_encoded"] = label_encoder.fit_transform(gold_df["intent_label_clean"])

    # Load GloVe
    glove_path = os.path.join(os.path.dirname(__file__), "..", "glove.6B.100d.txt")

    embedding_dim = 100
    glove_embeddings = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove_embeddings[word] = vec

    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        embedding_matrix[idx] = glove_embeddings.get(word, np.random.normal(scale=0.6, size=(embedding_dim,)))
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    np.save(os.path.join(save_path, "embedding_matrix.npy"), embedding_matrix.numpy())

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(label_encoder.classes_)
    model = BiLSTMWithAttention(embedding_matrix, HIDDEN_DIM, OUTPUT_DIM).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Split data
    sequences = gold_df["token_ids"].tolist()
    labels = gold_df["label_encoded"].tolist()
    X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    BATCH_SIZE = 32
    train_loader = DataLoader(RNNTextDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(RNNTextDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(RNNTextDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(15):
        model.train()
        total_loss, total_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/15 | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save model, vocab, label encoder
    torch.save(model.state_dict(), os.path.join(save_path, "rnn_model.pt"))
    with open(os.path.join(save_path, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(save_path, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print("RNN model, vocab, and label encoder saved to", save_path)

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    print("\nClassification Report (Test Set):")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("RNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()

    return model, test_loader, label_encoder


if __name__ == "__main__":
    train_phd_rnn_model()
