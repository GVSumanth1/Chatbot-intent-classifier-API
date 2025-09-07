# Chatbot Intent Classifier API

Most chatbots fail to understand real user intent, especially when queries are complex.  
I wanted to build a backend system that is **fast, intelligent, and reliable** using a mix of rule-based methods, ML models, and transformers.  
This project uses **FastAPI** to serve predictions in real time.

---

## What this project provides

This project provides a **FastAPI-based backend** for detecting user intent using a hybrid approach:
- Rule-based intent recognition
- Logistic Regression (TF-IDF)
- Transformer models (RoBERTa, DeBERTa)
- BiLSTM with attention (RNN)
- Confidence thresholding and intelligent fallback
- Logging of predictions to CSV

---

## Features

- Fast intent prediction using multiple models  
- Rule-based override for high-precision intents like greetings and farewells  
- Confidence-aware fallback handling  
- RESTful API endpoint (`/predict`)  
- CSV logging of every prediction  
- Easy integration with frontends or chat apps  

---

## Why I thought of this project
- Businesses face a huge volume of customer queries daily.  
- Rule-based bots are simple but break when queries are slightly different.  
- ML models are powerful but need fallback when unsure.  
- My idea was to build a **hybrid chatbot engine**:  
  - Rule-based for high precision intents (greetings, farewells)  
  - Logistic Regression & RNN for generalization  
  - Transformers (RoBERTa, DeBERTa) for deeper understanding  
  - Confidence thresholding + fallback for reliability  

---

## Dataset I used
- **Source**: [Amazon Product Review Dataset](https://nijianmo.github.io/amazon/index.html) (EMNLP 2019)  
- **Categories**: Beauty, Electronics, Clothing & Shoes  
- **Size**: 75,000 rows (25k from each category, 13 columns)  
- **Cleaning**:  
  - Removed null/missing values  
  - Mapped reviews into intent categories  
  - Combined everything into `chatbot_amazon.csv`  

---

## Project Structure

```

project/
│
├── api/
│   └── app.py              # FastAPI app with /predict endpoint
│
├── saved\_models/           # Contains saved models for logreg, roberta, deberta, rnn
│
├── utils/
│   └── intent\_mapping.py   # Rule-based keyword mapping logic
│
├── chatbot\_core.py         # Core logic: model loading, prediction, fallback, logging
│
├── logs/
│   └── prediction\_logs.csv # Auto-created log file for predictions
│
└── README.md               # You're reading it now!

````

---
---

## What I did

### Setup
- Built a **FastAPI backend** with `/predict` endpoint  
- Structured project into clear modules (`api`, `utils`, `logs`, etc.)  

### Models Implemented
- **Rule-based Classifier** → keyword mapping for simple intents  
- **Logistic Regression (TF-IDF)** → baseline ML model  
- **RNN with Attention** → sequence learning with GloVe embeddings  
- **RoBERTa Transformer** → refined text understanding  
- **DeBERTa Transformer** → better position/context separation  

### Prediction Flow
1. Rule-based (priority)  
2. Logistic Regression (≥ 0.70 confidence)  
3. RoBERTa / DeBERTa / RNN (≥ 0.75 confidence)  
4. If none pass → fallback response  

### Logging
- Every request gets logged with: timestamp, input, predicted intent, and model used  

---
## Project Structure

```

project/
│
├── api/
│   └── app.py              # FastAPI app with /predict endpoint
│
├── saved\_models/           # Contains saved models for logreg, roberta, deberta, rnn
│
├── utils/
│   └── intent\_mapping.py   # Rule-based keyword mapping logic
│
├── chatbot\_core.py         # Core logic: model loading, prediction, fallback, logging
│
├── logs/
│   └── prediction\_logs.csv # Auto-created log file for predictions
│
└── README.md               # You're reading it now!

````


---

## How to Run

1. **Install dependencies**

   ```bash
   ! pip install fastapi uvicorn transformers scikit-learn torch pandas joblib

````

2. **Start the API**
   From your notebook or terminal:

   ```bash
   uvicorn api.app:app --reload
   ```

3. **Open in browser**
   Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   Use the Swagger UI to test the `/predict` endpoint.

---
`````

---

## Example Request & Response

### Endpoint
`POST /predict`

---

### Request Body
```json
{
  "message": "I want to buy a new laptop"
}

```
### Sample Response:
```json
{
  "input": "I want to buy a new laptop",
  "predicted_intent": "PRODUCT_SEARCH",
  "source": "roberta",
  "confidence": 0.985,
  "confidence_level": "high",
  "timestamp": "2025-07-21T15:20:17.123Z"
}

````

## Intent Detection Logic

1. **Rule-based** match (priority)
2. Logistic Regression (TF-IDF + sklearn)
3. RoBERTa transformer (Hugging Face)
4. DeBERTa transformer
5. RNN with attention
6. Fallback: Ask user to rephrase or connect to agent

Each model must exceed a confidence threshold:

* `LogReg`: ≥ 0.70
* `All others`: ≥ 0.75

If all models fail → fallback message is triggered.

---

## Prediction Logs

Every prediction is saved to `logs/prediction_logs.csv` with:

* Timestamp
* User input
* Predicted intent
* Model used

---
