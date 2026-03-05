import pickle
import re
import io
import os
import json
import numpy as np
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix

from jose import JWTError, jwt
import bcrypt

import spacy

app = FastAPI(title="Job Role Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── JWT CONFIG ────────────────────────────────────────────────────────────────
SECRET_KEY                  = "change-this-to-a-long-random-secret-in-production"
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ── STORAGE PATHS ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_DIR      = os.path.join(BASE_DIR, "data", "pkl")
DATA_DIR     = os.path.join(BASE_DIR, "data")
USERS_FILE   = os.path.join(DATA_DIR, "users.json")    # persiste les comptes
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")  # persiste l'historique

# ── JSON HELPERS ──────────────────────────────────────────────────────────────
def load_json(path: str, default):
    """Charge un fichier JSON, retourne default si inexistant."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: str, data):
    """Sauvegarde data dans un fichier JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── IN-MEMORY STORES (initialisés depuis les fichiers JSON) ───────────────────
# { email: { username, email, hashed_password } }
USERS_DB: dict = load_json(USERS_FILE, {})

# { email: [ { date, filename, predicted_job, confidence, top_3, skills, certs } ] }
HISTORY_DB: dict = load_json(HISTORY_FILE, {})

# ── PASSWORD HELPERS ──────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# ── JWT HELPERS ───────────────────────────────────────────────────────────────
def create_access_token(email: str, name: str) -> str:
    expire  = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "name": name, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email or email not in USERS_DB:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return USERS_DB[email]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── HISTORY HELPERS ───────────────────────────────────────────────────────────
def add_to_history(email: str, entry: dict):
    """Ajoute une prédiction à l'historique de l'utilisateur et sauvegarde."""
    if email not in HISTORY_DB:
        HISTORY_DB[email] = []
    HISTORY_DB[email].insert(0, entry)   # plus récent en premier
    save_json(HISTORY_FILE, HISTORY_DB)

# ── AUTH SCHEMAS ──────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# ── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.post("/auth/register")
def register(body: RegisterRequest):
    if body.email in USERS_DB:
        raise HTTPException(status_code=400, detail="Email already registered")
    USERS_DB[body.email] = {
        "username":        body.name,
        "email":           body.email,
        "hashed_password": hash_password(body.password),
        "created_at":      datetime.utcnow().isoformat(),
    }
    save_json(USERS_FILE, USERS_DB)   # ← persiste le nouvel utilisateur
    return {
        "access_token": create_access_token(body.email, body.name),
        "token_type":   "bearer",
        "username":     body.name,
        "email":        body.email,
    }

@app.post("/auth/login")
def login(body: LoginRequest):
    user = USERS_DB.get(body.email)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {
        "access_token": create_access_token(body.email, user["username"]),
        "token_type":   "bearer",
        "username":     user["username"],
        "email":        body.email,
    }

# ── PROFILE ROUTE ─────────────────────────────────────────────────────────────
@app.get("/profile")
def get_profile(current_user: dict = Depends(get_current_user)):
    """
    Retourne les infos du compte + tout l'historique des prédictions.
    Calcule aussi des statistiques à la volée.
    """
    email   = current_user["email"]
    history = HISTORY_DB.get(email, [])

    # ── statistiques ──
    total_analyses = len(history)

    # job le plus prédit
    if history:
        from collections import Counter
        job_counts   = Counter(h["predicted_job"] for h in history)
        top_job      = job_counts.most_common(1)[0][0]
        avg_conf     = round(sum(h["confidence"] for h in history) / total_analyses * 100, 1)
    else:
        top_job  = None
        avg_conf = 0

    return {
        "username":       current_user["username"],
        "email":          email,
        "member_since":   current_user.get("created_at", "N/A"),
        "stats": {
            "total_analyses": total_analyses,
            "top_job":        top_job,
            "avg_confidence": avg_conf,
        },
        "history": history,   # liste complète des prédictions
    }

# ── ML SETUP ──────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

with open(os.path.join(PKL_DIR, "modeling_results_gridsearch.pkl"), "rb") as f:
    results = pickle.load(f)

with open(os.path.join(PKL_DIR, "feature_sets.pkl"), "rb") as f:
    features = pickle.load(f)

model         = results["best_model"]
label_encoder = results["label_encoder"]
vectorizer    = features["count_vectorizer"]   # ← Count (3000 features, sans stats)

# ── ML HELPERS ────────────────────────────────────────────────────────────────
def clean(text):
    text = text.lower().replace(";", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_features(skills, description, certifications):
    text = clean(f"{skills} {description} {certifications}")
    return vectorizer.transform([text])   # 3000 features exactement

def extract_from_cv(text):
    doc = nlp(text)

    cert_pattern = re.compile(
        r"(certified|certification|certificate|associate|professional|"
        r"specialist|expert|foundation|practitioner|cka|ckad|cissp|ceh|"
        r"pmp|pcap|pcep|csm|comptia)[^\n.]{0,80}",
        re.IGNORECASE
    )
    certs = list(set(m.group().strip() for m in cert_pattern.finditer(text)))

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

    return ";".join(skills[:40]), ";".join(certs[:10])

# ── ML SCHEMAS ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    skills: str
    job_description: str = ""
    certifications: str

# ── ML ROUTES ─────────────────────────────────────────────────────────────────
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
def predict(req: PredictRequest, current_user: dict = Depends(get_current_user)):
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
async def predict_cv(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    content = await file.read()

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

    skills, certifications = extract_from_cv(text)

    X     = build_features(skills, text[:500], certifications)
    proba = model.predict_proba(X)[0]
    top3  = np.argsort(proba)[::-1][:3]

    predicted_job = label_encoder.classes_[top3[0]]
    confidence    = round(float(proba[top3[0]]), 4)
    top_3_result  = [
        {"job": label_encoder.classes_[i], "probability": round(float(proba[i]), 4)}
        for i in top3
    ]

    # ── sauvegarder dans l'historique ──
    add_to_history(current_user["email"], {
        "date":          datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "filename":      file.filename,
        "predicted_job": predicted_job,
        "confidence":    confidence,
        "top_3":         top_3_result,
        "skills":        skills,
        "certifications": certifications,
    })

    return {
        "predicted_job": predicted_job,
        "confidence":    confidence,
        "top_3":         top_3_result,
        "extracted": {
            "skills":         skills,
            "certifications": certifications,
        }
    }