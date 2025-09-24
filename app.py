from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
import nltk, string
from nltk.stem.porter import PorterStemmer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True) 
nltk.download("stopwords", quiet=True)

MODEL_PATH = Path("clf_model.pkl")
VEC_PATH = Path("vectorizer_tfidf.pkl")

class Request(BaseModel):
    text: str
    

app = FastAPI()

# Preprocessing (same as training)
ps = PorterStemmer()

def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # remove stopwords/punctuation
    tokens = [tok for tok in tokens if tok not in string.punctuation]
    
    # stemming
    tokens = [ps.stem(tok) for tok in tokens]

    return " ".join(tokens)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception:
    model = None
    vectorizer = None
    
# Label mapping
label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
@app.get("/")
def root():
    return {"message": "Welcome to SvaraAI Reply Classifier (Baseline with Pickle). Visit /docs for API documentation."}

@app.get("/ping")
def ping():
    return {"status": "ok" if model is not None else "not-ready"}

@app.post("/predict")
def predict(req: Request):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not available.")
    
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        processed = transform_text(text)
        # Convert sparse -> dense
        X = vectorizer.transform([processed]).toarray()

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            label = model.classes_[idx]
            confidence = float(proba[idx])
        else:
            label = model.predict(X)[0]
            confidence = None

        # Map numeric label -> string
        if isinstance(label, (int, np.integer)) and label in label_map:
            label_str = label_map[label]
        else:
            label_str = str(label)

        return {"label": label_str, "confidence": confidence}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")