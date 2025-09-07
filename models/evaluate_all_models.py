# evaluate_all_models.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from shared import load_and_prepare_data, split_data, get_label_names

gold_df, label_encoder = load_and_prepare_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(gold_df)
label_names = get_label_names(label_encoder)

os.makedirs("logs", exist_ok=True)

def save_reports(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=90)
    plt.tight_layout()
    plt.savefig(f"logs/{name}_confusion_matrix.png")
    plt.close()

    report = classification_report(y_true, y_pred, target_names=label_names)
    with open(f"logs/{name}_classification_report.txt", "w") as f:
        f.write(report)

    print(f"{name.upper()} → Done. Reports saved in logs/")
"""
def evaluate_logreg():
    import joblib
    model = joblib.load("saved_models/logreg/logreg_model.pkl")
    vectorizer = joblib.load("saved_models/logreg/logreg_vectorizer.pkl")
    X_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_vec)
    save_reports("logreg", y_test, y_pred)

def evaluate_roberta():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("saved_models/roberta")
    tokenizer = AutoTokenizer.from_pretrained("saved_models/roberta")
    with open("saved_models/roberta/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    y_true, y_pred = [], []
    for text, label in zip(X_test, y_test):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        y_pred.append(pred)
        y_true.append(label)

    save_reports("roberta", y_true, y_pred)

def evaluate_deberta():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("saved_models/deberta")
    tokenizer = AutoTokenizer.from_pretrained("saved_models/deberta")
    with open("saved_models/deberta/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    y_true, y_pred = [], []
    for text, label in zip(X_test, y_test):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        y_pred.append(pred)
        y_true.append(label)

    save_reports("deberta", y_true, y_pred)
"""
def evaluate_rnn():
    import torch
    import torch.nn as nn

    with open("saved_models/rnn/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("saved_models/rnn/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    embedding_matrix = np.load("saved_models/rnn/embedding_matrix.npy")

    class BiLSTMWithAttention(nn.Module):
        def __init__(self, embedding_matrix, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False)
            self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attention(out), dim=1)
            context = torch.sum(attn_weights * out, dim=1)
            return self.fc(context)

    model = BiLSTMWithAttention(embedding_matrix, 128, len(le.classes_))
    model.load_state_dict(torch.load("saved_models/rnn/rnn_model.pt", map_location=torch.device("cpu")))
    model.eval()

    def simple_tokenizer(text):
        return text.lower().split()

    def text_to_seq(text):
        return [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenizer(text)]

    y_true, y_pred = [], []
    for text, label in zip(X_test, y_test):
        seq = torch.tensor(text_to_seq(text)).unsqueeze(0)
        with torch.no_grad():
            out = model(seq)
            pred = torch.argmax(out, dim=1).item()
        y_pred.append(pred)
        y_true.append(label)

    save_reports("rnn", y_true, y_pred)

if __name__ == "__main__":
    print("\nRunning RNN Evaluation Only")
    evaluate_rnn()

""" 
   evaluate_logreg()
    evaluate_roberta()
    evaluate_deberta()
    """