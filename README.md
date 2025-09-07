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

