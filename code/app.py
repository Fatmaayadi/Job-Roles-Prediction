import pickle
import re
import io
import os
import numpy as np
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix

import spacy

app = FastAPI(title="Job Role Prediction API")

# Allow React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths (works both locally and inside Docker)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_DIR  = os.path.join(BASE_DIR, "data", "pkl")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load ML model & vectorizers
with open(os.path.join(PKL_DIR, "modeling_results_gridsearch.pkl"), "rb") as f:
    results = pickle.load(f)

with open(os.path.join(PKL_DIR, "feature_sets.pkl"), "rb") as f:
    features = pickle.load(f)

model         = results["best_model"]
label_encoder = results["label_encoder"]
tfidf         = features["tfidf_vectorizer"]

# Text cleaning 
def clean(text):
    text = text.lower().replace(";", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Build features 
def build_features(skills, description, certifications):
    text  = clean(f"{skills} {description} {certifications}")
    words = text.split()
    stats = np.array([[
        len(text), len(words),
        np.mean([len(w) for w in words]) if words else 0,
        len(set(words)),
        len(set(words)) / (len(words) + 1)
    ]])
    return hstack([tfidf.transform([text]), csr_matrix(stats)])

# Extract skills & certifications from raw CV text
def extract_from_cv(text):
    doc = nlp(text)

    # certifications — find sentences with cert keywords
    cert_pattern = re.compile(
        r"(certified|certification|certificate|associate|professional|"
        r"specialist|expert|foundation|practitioner|cka|ckad|cissp|ceh|"
        r"pmp|pcap|pcep|csm|comptia)[^\n.]{0,80}",
        re.IGNORECASE
    )
    certs = list(set(m.group().strip() for m in cert_pattern.finditer(text)))

    # skills — named entities + noun chunks + capitalised words
    ent_skills   = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT") and len(ent.text) > 1]
    chunk_skills = [chunk.text.strip() for chunk in doc.noun_chunks
                    if 1 <= len(chunk.text.split()) <= 3 and chunk.text[0].isupper()]
    cap_words    = re.findall(r"\b[A-Z][a-zA-Z0-9+#.]*\b", text)

    stopwords = {"I", "A", "The", "My", "We", "In", "At", "Of", "To",
                 "And", "Or", "For", "With", "On", "Is", "Are", "Was",
                 "Be", "An", "As", "It", "By", "He", "She", "His", "Her",
                 "This", "That", "From", "Has", "Have", "Been", "Will"}

    all_skills = list(set(ent_skills + chunk_skills + cap_words))
    skills     = [s for s in all_skills if s not in stopwords and len(s) > 1]

    return (
        ";".join(skills[:40]),
        ";".join(certs[:10]),
    )

# Schemas
class PredictRequest(BaseModel):
    skills: str
    job_description: str = ""
    certifications: str 

# Endpoints
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/model-info")
def model_info():
    return {
        "model":       results["best_model_key"],
        "num_classes": len(label_encoder.classes_),
        "metrics":     {k: round(v, 4) for k, v in results["best_metrics"].items()
                        if isinstance(v, float)}
    }

@app.post("/predict")
def predict(req: PredictRequest):
    X     = build_features(req.skills, req.job_description, req.certifications)
    proba = model.predict_proba(X)[0]
    top3  = np.argsort(proba)[::-1][:3]
    return {
        "predicted_job": label_encoder.classes_[top3[0]],
        "confidence":    round(float(proba[top3[0]]), 4),
        "top_3": [
            {"job": label_encoder.classes_[i], "probability": round(float(proba[i]), 4)}
            for i in top3
        ]
    }

@app.post("/predict-cv")
async def predict_cv(file: UploadFile = File(...)):
    content = await file.read()

    # read text from file
    if file.filename.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            raise HTTPException(status_code=422, detail="Could not read PDF. Try a TXT file.")
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=422, detail="The file appears to be empty.")

    # extract skills & certs directly from CV text
    skills, certifications = extract_from_cv(text)

    # predict
    X     = build_features(skills, text[:500], certifications)
    proba = model.predict_proba(X)[0]
    top3  = np.argsort(proba)[::-1][:3]

    return {
        "predicted_job": label_encoder.classes_[top3[0]],
        "confidence":    round(float(proba[top3[0]]), 4),
        "top_3": [
            {"job": label_encoder.classes_[i], "probability": round(float(proba[i]), 4)}
            for i in top3
        ],
        "extracted": {
            "skills":         skills,
            "certifications": certifications,
        }
    }