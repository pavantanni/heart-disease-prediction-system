# FastAPI backend for heart disease prediction
# Now with SQLite to store every prediction permanently
# Start with: uvicorn heart_disease_api:app --reload --port 8000

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import joblib
import sqlite3
import json
import uuid
import io
import os
from datetime import datetime

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Heart disease risk prediction using a 5-model soft voting ensemble. AUC ~0.96.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Database setup ──────────────────────────────────────────

DB_PATH = "predictions.db"

def init_db():
    """Create the predictions table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT NOT NULL,
            age         REAL,
            sex         INTEGER,
            cp          INTEGER,
            trestbps    REAL,
            chol        REAL,
            fbs         INTEGER,
            restecg     INTEGER,
            thalach     REAL,
            exang       INTEGER,
            oldpeak     REAL,
            slope       INTEGER,
            ca          INTEGER,
            thal        INTEGER,
            prediction  INTEGER,
            probability REAL,
            risk_level  TEXT,
            confidence  REAL,
            recommendation TEXT,
            timestamp   TEXT
        )
    """)
    conn.commit()
    conn.close()

# create the table on startup
init_db()


def save_to_db(patient_id: str, features: dict, result: dict):
    """Save one prediction record to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            patient_id, age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            prediction, probability, risk_level, confidence, recommendation, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        features["age"], features["sex"], features["cp"],
        features["trestbps"], features["chol"], features["fbs"],
        features["restecg"], features["thalach"], features["exang"],
        features["oldpeak"], features["slope"], features["ca"], features["thal"],
        result["prediction"], result["probability"], result["risk_level"],
        result["confidence"], result["recommendation"], result["timestamp"]
    ))
    conn.commit()
    conn.close()


def fetch_all_predictions():
    """Return all stored predictions as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def fetch_patient_by_id(patient_id: str):
    """Fetch a single prediction by patient_id."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_stats():
    """Aggregate stats directly from the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    if total == 0:
        conn.close()
        return None

    cursor.execute("SELECT AVG(probability) FROM predictions")
    avg_prob = cursor.fetchone()[0]

    cursor.execute("SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level")
    risk_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    for risk_level, count in cursor.fetchall():
        risk_counts[risk_level] = count

    conn.close()
    return {
        "total_predictions": total,
        "average_probability": round(avg_prob, 4),
        "risk_distribution": risk_counts,
        "high_risk_percentage": round(risk_counts["HIGH"] / total * 100, 1),
    }


# ── Pydantic schemas ────────────────────────────────────────

class PatientFeatures(BaseModel):
    age:      float = Field(..., ge=20,  le=100, example=55)
    sex:      int   = Field(..., ge=0,   le=1,   example=1)
    cp:       int   = Field(..., ge=0,   le=3,   example=1)
    trestbps: float = Field(..., ge=80,  le=220, example=130)
    chol:     float = Field(..., ge=100, le=600, example=250)
    fbs:      int   = Field(..., ge=0,   le=1,   example=0)
    restecg:  int   = Field(..., ge=0,   le=2,   example=0)
    thalach:  float = Field(..., ge=60,  le=220, example=150)
    exang:    int   = Field(..., ge=0,   le=1,   example=0)
    oldpeak:  float = Field(..., ge=0,   le=7,   example=1.5)
    slope:    int   = Field(..., ge=0,   le=2,   example=1)
    ca:       int   = Field(..., ge=0,   le=3,   example=0)
    thal:     int   = Field(..., ge=0,   le=3,   example=2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 55, "sex": 1, "cp": 1, "trestbps": 130,
                "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2
            }
        }
    }


class PredictionResponse(BaseModel):
    patient_id:     str
    prediction:     int    # 0 = No Disease, 1 = Heart Disease
    probability:    float
    risk_level:     str    # LOW / MODERATE / HIGH
    confidence:     float
    risk_factors:   List[dict]
    recommendation: str
    timestamp:      str


class BatchPredictionResponse(BaseModel):
    total_patients: int
    high_risk:      int
    moderate_risk:  int
    low_risk:       int
    predictions:    List[dict]


# ── Model loading ───────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

print("MODEL DIR:", MODEL_DIR)
print("FILES:", os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else "NOT FOUND")

def load_artifacts():
    try:
        model   = joblib.load(f"{MODEL_DIR}/ensemble_model.pkl")
        scaler  = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        imputer = joblib.load(f"{MODEL_DIR}/imputer.pkl")
        return model, scaler, imputer
    except FileNotFoundError:
        raise RuntimeError("Model artifacts not found. Run heart_disease_ml_pipeline.py first.")

try:
    model, scaler, imputer = load_artifacts()
    print("Model artifacts loaded successfully")
    model_loaded = True
except RuntimeError as e:
    print(f"Warning: {e}")
    model_loaded = False

# 13 base features from UCI dataset
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

RISK_THRESHOLDS = {"HIGH": 0.7, "MODERATE": 0.4, "LOW": 0.0}


# ── Helper functions ────────────────────────────────────────

def get_risk_level(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif probability >= RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "LOW"


def get_recommendation(risk_level: str) -> str:
    recs = {
        "HIGH":     "High cardiovascular risk. Immediate cardiology consultation recommended.",
        "MODERATE": "Moderate risk. Schedule a cardiology review within 2-4 weeks.",
        "LOW":      "Low risk. Continue healthy lifestyle and annual checkups.",
    }
    return recs[risk_level]


def get_top_risk_factors(patient_data: dict, n: int = 5) -> List[dict]:
    # weights based on clinical literature and observed feature importances
    risk_weights = {
        "cp":       (3.0,  "Chest pain type is a strong predictor"),
        "thal":     (2.8,  "Thalassemia type indicates blood flow issues"),
        "ca":       (2.5,  "Number of blocked major vessels"),
        "exang":    (2.2,  "Exercise-induced angina suggests ischemia"),
        "oldpeak":  (2.0,  "ST depression indicates myocardial stress"),
        "slope":    (1.8,  "Peak exercise ST segment slope"),
        "thalach":  (-1.5, "Lower max heart rate indicates poorer fitness"),
        "age":      (1.2,  "Age is a key cardiovascular risk factor"),
        "sex":      (1.0,  "Males have higher heart disease prevalence"),
        "trestbps": (0.9,  "Elevated resting blood pressure"),
        "chol":     (0.7,  "High cholesterol contributes to arterial plaque"),
        "fbs":      (0.5,  "High fasting blood sugar damages blood vessels"),
        "restecg":  (0.4,  "Resting ECG abnormalities"),
    }

    factors = []
    for feature, (weight, description) in risk_weights.items():
        value  = patient_data.get(feature, 0)
        impact = float(weight * value / 10)
        factors.append({
            "feature":     feature,
            "value":       value,
            "impact":      round(impact, 3),
            "description": description,
            "direction":   "increases" if impact > 0 else "decreases",
        })

    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return factors[:n]


def preprocess_patient(patient: PatientFeatures) -> np.ndarray:
    d = patient.dict()

    # impute the 13 base features first
    base_values = np.array([[d[f] for f in FEATURE_NAMES]])
    imputed = imputer.transform(base_values)
    df = pd.DataFrame(imputed, columns=FEATURE_NAMES)

    # add the same engineered features used during training
    df["age_thalach_ratio"] = df["age"] / (df["thalach"] + 1)
    df["bp_age_interaction"] = df["trestbps"] * df["age"]
    df["chol_risk"]          = (df["chol"] > 200).astype(int)
    df["severe_chest_pain"]  = (df["cp"] == 3).astype(int)

    # scaler expects all 17 features
    scaled = scaler.transform(df)
    return scaled


# ── API routes ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "message":     "Heart Disease Prediction API",
        "status":      "operational",
        "model_loaded": model_loaded,
        "version":     "1.0.0",
        "docs":        "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status":      "healthy",
        "model_loaded": model_loaded,
        "timestamp":   datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(patient: PatientFeatures):
    """Single patient prediction — saves result to SQLite automatically."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    try:
        patient_id  = str(uuid.uuid4())[:8]
        X           = preprocess_patient(patient)
        probability = float(model.predict_proba(X)[0][1])
        prediction  = int(probability >= 0.5)
        risk_level  = get_risk_level(probability)

        result = {
            "prediction":     prediction,
            "probability":    round(probability, 4),
            "risk_level":     risk_level,
            "confidence":     round(max(probability, 1 - probability), 4),
            "recommendation": get_recommendation(risk_level),
            "timestamp":      datetime.now().isoformat(),
        }

        # save to SQLite — survives API restarts
        save_to_db(patient_id, patient.dict(), result)

        return PredictionResponse(
            patient_id=patient_id,
            risk_factors=get_top_risk_factors(patient.dict()),
            **result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV — also saves each row to SQLite."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        X_imputed = imputer.transform(df[FEATURE_NAMES])
        X_scaled  = scaler.transform(X_imputed)

        probabilities = model.predict_proba(X_scaled)[:, 1]
        predictions   = (probabilities >= 0.5).astype(int)
        risk_levels   = [get_risk_level(p) for p in probabilities]

        results = []
        for i, (pred, prob, risk) in enumerate(zip(predictions, probabilities, risk_levels)):
            pid    = str(uuid.uuid4())[:8]
            rec    = get_recommendation(risk)
            ts     = datetime.now().isoformat()
            record = {
                "patient_index": i,
                "patient_id":    pid,
                "prediction":    int(pred),
                "probability":   round(float(prob), 4),
                "risk_level":    risk,
                "recommendation": rec,
            }

            # save each batch row to DB as well
            save_to_db(pid, df.iloc[i][FEATURE_NAMES].to_dict(), {
                "prediction":     int(pred),
                "probability":    round(float(prob), 4),
                "risk_level":     risk,
                "confidence":     round(float(max(prob, 1 - prob)), 4),
                "recommendation": rec,
                "timestamp":      ts,
            })

            results.append(record)

        return BatchPredictionResponse(
            total_patients=len(results),
            high_risk=    sum(1 for r in results if r["risk_level"] == "HIGH"),
            moderate_risk=sum(1 for r in results if r["risk_level"] == "MODERATE"),
            low_risk=     sum(1 for r in results if r["risk_level"] == "LOW"),
            predictions=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/patient/{patient_id}", tags=["Patient History"])
async def get_patient(patient_id: str):
    """Fetch a specific patient's prediction from the database."""
    row = fetch_patient_by_id(patient_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return row


@app.get("/patients", tags=["Patient History"])
async def list_patients():
    """List all stored predictions from SQLite."""
    rows = fetch_all_predictions()
    return {
        "total":    len(rows),
        "patients": rows,
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    return {
        "model_type":      "Soft Voting Ensemble",
        "components":      ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "Neural Network"],
        "dataset":         "UCI Cleveland Heart Disease (303 patients)",
        "features":        len(FEATURE_NAMES),
        "target":          "Binary (0=No Disease, 1=Disease)",
        "performance":     {"accuracy": "~0.93", "roc_auc": "~0.96", "f1_score": "~0.93"},
        "explainability":  ["SHAP", "LIME", "Feature Importance"],
        "storage":         "SQLite (predictions.db)",
    }


@app.get("/stats", tags=["Analytics"])
async def prediction_stats():
    """Aggregate stats pulled directly from the database."""
    stats = fetch_stats()
    if not stats:
        return {"message": "No predictions made yet"}
    return stats


@app.delete("/patient/{patient_id}", tags=["Patient History"])
async def delete_patient(patient_id: str):
    """Delete a specific prediction record from the database."""
    row = fetch_patient_by_id(patient_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return {"message": f"Patient {patient_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
