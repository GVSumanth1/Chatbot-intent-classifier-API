# chatbot_core.py
"""
# Dataset Source: https://nijianmo.github.io/amazon/index.html
# Justifying recommendations using distantly-labeled reviews and fined-grained aspects
# Jianmo Ni, Jiacheng Li, Julian McAuley
# Empirical Methods in Natural Language Processing (EMNLP), 2019

"""
import os
import sys
import csv
import joblib
import pickle
import torch
import numpy as np
from datetime import datetime
from functools import lru_cache

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Ensure project root and utils path are accessible
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)

models_path = os.path.abspath("../models")
if models_path not in sys.path:
    sys.path.append(models_path)

from utils.intent_mapping import map_intent_conservative_contextual, intent_keywords

# ---- Cached Model Loaders ----

@lru_cache(maxsize=4)
def load_logreg():
    model = joblib.load("saved_models/logreg/logreg_model.pkl")
    vectorizer = joblib.load("saved_models/logreg/logreg_vectorizer.pkl")
    label_encoder = joblib.load("saved_models/logreg/logreg_label_encoder.pkl")
    print("Logistic Regression loaded")
    return model, vectorizer, label_encoder

@lru_cache(maxsize=4)
def load_roberta():
    model = AutoModelForSequenceClassification.from_pretrained("saved_models/roberta/")
    tokenizer = AutoTokenizer.from_pretrained("saved_models/roberta/")
    with open("saved_models/roberta/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("RoBERTa model loaded")
    return model, tokenizer, label_encoder

@lru_cache(maxsize=4)
def load_deberta():
    model = AutoModelForSequenceClassification.from_pretrained("saved_models/deberta/")
    tokenizer = AutoTokenizer.from_pretrained("saved_models/deberta/")
    with open("saved_models/deberta/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("DeBERTa model loaded")
    return model, tokenizer, label_encoder

@lru_cache(maxsize=4)
def load_rnn():
    import torch.nn as nn

    class BiLSTMWithAttention(nn.Module):
        def __init__(self, embedding_matrix, hidden_dim, output_dim, dropout=0.3, num_layers=2):
            super().__init__()
            vocab_size, embed_dim = embedding_matrix.shape
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
            return self.fc(context)

    with open("saved_models/rnn/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open("saved_models/rnn/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    embedding_matrix = np.load("saved_models/rnn/embedding_matrix.npy") \
        if os.path.exists("saved_models/rnn/embedding_matrix.npy") \
        else np.random.normal(0, 1, (len(vocab), 100)).astype(np.float32)

    model = BiLSTMWithAttention(
        embedding_matrix=torch.tensor(embedding_matrix),
        hidden_dim=128,
        output_dim=len(label_encoder.classes_)
    )
    model.load_state_dict(torch.load("saved_models/rnn/rnn_model.pt", map_location=torch.device("cpu")))
    model.eval()

    print("RNN model loaded")
    return model, vocab, label_encoder

# ---- Logging Function ----
def log_prediction(user_input, intent, model_name, confidence=None, log_path="logs/prediction_logs.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().isoformat()
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, user_input, intent, model_name, confidence])


# ---- Prediction Logic ----
def predict_intent(user_input):
    timestamp = datetime.now().isoformat()
    rule_intent = map_intent_conservative_contextual(user_input, intent_keywords)
    if rule_intent not in ["OTHER", "AMBIGUOUS"]:
        print("Rule-based match:", rule_intent)
        log_prediction(user_input, rule_intent, "rule_based", confidence=None)
        return rule_intent, "rule_based", None, timestamp

    model_results = []

    try:
        logreg_model, logreg_vectorizer, logreg_label_encoder = load_logreg()
        X_input = logreg_vectorizer.transform([user_input])
        probs = logreg_model.predict_proba(X_input)
        max_prob = probs.max()
        y_pred = logreg_model.predict(X_input)
        intent = logreg_label_encoder.inverse_transform(y_pred)[0]
        print(f"Logistic Regression predicted: {intent} (confidence: {max_prob:.2f})")
        if max_prob >= 0.7 and intent != "OTHER":
            log_prediction(user_input, intent, "logreg", max_prob)
            return intent, "logreg", max_prob, timestamp
        else:
            model_results.append(("logreg", "OTHER"))
    except Exception as e:
        print("Logistic Regression failed:", e)

    try:
        roberta_model, roberta_tokenizer, roberta_label_encoder = load_roberta()
        inputs = roberta_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            max_prob = probs.max().item()
            pred_label = torch.argmax(probs, dim=1).item()
        intent = roberta_label_encoder.inverse_transform([pred_label])[0]
        print(f"RoBERTa predicted: {intent} (confidence: {max_prob:.2f})")
        if max_prob >= 0.75 and intent != "OTHER":
            log_prediction(user_input, intent, "roberta", max_prob)
            return intent, "roberta", max_prob, timestamp
        else:
            model_results.append(("roberta", "OTHER"))
    except Exception as e:
        print("RoBERTa failed:", e)

    try:
        deberta_model, deberta_tokenizer, deberta_label_encoder = load_deberta()
        inputs = deberta_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = deberta_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            max_prob = probs.max().item()
            pred_label = torch.argmax(probs, dim=1).item()
        intent = deberta_label_encoder.inverse_transform([pred_label])[0]
        print(f"DeBERTa predicted: {intent} (confidence: {max_prob:.2f})")
        if max_prob >= 0.75 and intent != "OTHER":
            log_prediction(user_input, intent, "deberta", max_prob)
            return intent, "deberta", max_prob, timestamp
        else:
            model_results.append(("deberta", "OTHER"))
    except Exception as e:
        print("DeBERTa failed:", e)

    try:
        rnn_model, rnn_vocab, rnn_label_encoder = load_rnn()

        def simple_tokenizer(text):
            return text.lower().split()

        def text_to_sequence(text, vocab):
            return [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenizer(text)]

        sequence = text_to_sequence(user_input, rnn_vocab)
        sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            rnn_model.eval()
            output = rnn_model(sequence_tensor)
            probs = torch.softmax(output, dim=1)
            max_prob = probs.max().item()
            pred_label = torch.argmax(probs, dim=1).item()

        intent = rnn_label_encoder.inverse_transform([pred_label])[0]
        print(f"RNN predicted: {intent} (confidence: {max_prob:.2f})")
        if max_prob >= 0.75 and intent != "OTHER":
            log_prediction(user_input, intent, "rnn",max_prob)
            return intent, "rnn", max_prob, timestamp
        else:
            model_results.append(("rnn", "OTHER"))
    except Exception as e:
        print("RNN failed:", e)

    print("All models returned 'OTHER' or failed.")
    clarification_prompt = (
        "I'm not sure I understood that. "
        "Could you rephrase your request or would you like to connect with a support agent?"
    )
    log_prediction(user_input, "OTHER", "fallback", confidence=None)
    return clarification_prompt, "fallback", None, timestamp
