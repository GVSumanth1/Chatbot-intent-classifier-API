---

### `README.md`

```markdown
# Chatbot Intent Classifier API

This project provides a **FastAPI-based backend** for detecting user intent using a hybrid approach:
- Rule-based intent recognition
- Logistic Regression (TF-IDF)
- Transformer models (RoBERTa, DeBERTa)
- BiLSTM with attention (RNN)
- Confidence thresholding and intelligent fallback
- Logging of predictions to CSV

---

# Features

- Fast intent prediction using multiple models
- Rule-based override for high-precision intents like greetings and farewells
- Confidence-aware fallback handling
- RESTful API endpoint (`/predict`)
- CSV logging of every prediction
- Easy integration with frontends or chat apps

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

## Example Request (via Swagger or curl)

### Endpoint:

`POST /predict`

### Request Body:

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
```

---

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

## Future Improvements (Optional)

* Add authentication or rate-limiting to API
* Connect to a frontend (chat widget or web app)
* Add unit tests or batch intent prediction support
* Deploy to Hugging Face Spaces, Railway, or Fly.io

---

## Credits

* Developed using Python, FastAPI, Hugging Face Transformers, and PyTorch.
* Inspired by real-world chatbot use cases for e-commerce and customer support.

---

## Citation: 
* Justifying recommendations using distantly-labeled reviews and fined-grained aspects
* Jianmo Ni, Jiacheng Li, Julian McAuley
* Empirical Methods in Natural Language Processing (EMNLP), 2019
* Dataset Source: https://nijianmo.github.io/amazon/index.html

