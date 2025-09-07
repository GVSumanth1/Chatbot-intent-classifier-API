
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio

from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
from datetime import datetime

# Ensure chatbot_core can be imported
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)

from chatbot_core import predict_intent

app = FastAPI(title="Chatbot Intent API")

class UserInput(BaseModel):
    message: str

# @app.get("/")
# def root():
#     return {"message": "Welcome to the chatbot API!"}

@app.post("/predict")
async def predict(user_input: UserInput):
    message = user_input.message
    predicted_intent, source, confidence, timestamp = predict_intent(message)
    
    # Add confidence label
    if confidence is None:
        confidence_label = "N/A"
    elif confidence >= 0.85:
        confidence_label = "high"
    elif confidence >= 0.65:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    # # Add timestamp
    # timestamp = datetime.now().isoformat()

    return {
        "input": message,
        "predicted_intent": predicted_intent,
        "source": source,
        "confidence": round(confidence, 3) if confidence is not None else None,
        "confidence_level": confidence_label,
        "timestamp": timestamp
    }
