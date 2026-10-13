from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import numpy as np
import pandas as pd
import cv2
import pytesseract
import re

app = FastAPI(title="NAFLD AI System - RandomForest")

# ===============================
# Load Model
# ===============================
model = joblib.load("RandomForestClassifier_model.pkl")
print("Feature names:", model.feature_names_in_)
print("Feature dtypes expected:", model.n_features_in_)

# ===============================
# OCR
# ===============================
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

# ===============================
# Feature Columns 
# ===============================
FEATURE_COLUMNS = [
    'id', 'age', 'gender', 'bmi',
    'alcohol_status', 'smoking_status',
    'alt', 'ast', 'ggt', 'bilirubin_total',
    'triglycerides', 'hdl_cholestrol', 'ldl_cholestrol',
    'platelet_count', 'fasting_blood_sugar',
    'tsh', 'diabetes_type_2', 'hypertension',
    'liver_cirrhosis'
]


# ===============================
# Feature Extraction (Improved)
# ===============================
import re

def extract_features(text):

    # ===============================
    # Text Cleaning 
    # ===============================
    text_lower = text.lower()

    # Fix OCR broken words 
    text_lower = re.sub(r'\b([a-z])\s+([a-z])\b', r'\1\2', text_lower)

    # Normalize symbols
    text_lower = text_lower.replace("=", ":")
    text_lower = re.sub(r'[^a-z0-9:\.\s]', ' ', text_lower)


    # ===============================
    # Safe Search Function
    # ===============================
    def safe_search(patterns, default=0):
        if isinstance(patterns, str):
            patterns = [patterns]

        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        return default

    def has_keyword(word):
        return int(bool(re.search(rf"\b{re.escape(word)}\b", text_lower)))

    # ===============================
    # Feature Extraction
    # ===============================
    return {
        "id": 0,

        "age": safe_search([
               r"(\d+)\s*years",
               r"age.*?(\d+)",
                              ]),

        "gender": 1 if re.search(r"\bmale\b", text_lower) else 0,

        "bmi": safe_search(r"bmi\s*[:=\-]?\s*(\d+\.?\d*)"),

        "alcohol_status": 0,
        "smoking_status": has_keyword("smoking"),

        "alt": safe_search([
               r"alt.*?(\d+\.?\d*)",
               r"sgpt.*?(\d+\.?\d*)",
                              ]),


        "ast": safe_search([
               r"ast.*?(\d+\.?\d*)",
               r"sgot.*?(\d+\.?\d*)",
                             ]),

        "ggt": safe_search([
            r"g\s*g\s*t\s*[:=\-]?\s*(\d+\.?\d*)",
            r"ggt\s*[:=\-]?\s*(\d+\.?\d*)"
        ]),

        "bilirubin_total": safe_search([
            r"bilirubin\s*[:=\-]?\s*(\d+\.?\d*)"
        ]),

        "triglycerides": safe_search(
            r"triglycerides\s*[:=\-]?\s*(\d+\.?\d*)", 150
        ),

        "hdl_cholestrol": safe_search(
            r"h\s*d\s*l\s*[:=\-]?\s*(\d+\.?\d*)", 50
        ),

        "ldl_cholestrol": safe_search(
            r"l\s*d\s*l\s*[:=\-]?\s*(\d+\.?\d*)", 100
        ),

        "platelet_count": 250,

        "fasting_blood_sugar": safe_search(
            r"fasting.*?(\d+\.?\d*)", 90
        ),

        "tsh": 2.5,
        "diabetes_type_2": 0,
        "hypertension": 0,
        "liver_cirrhosis": 0,

        # Risk
        "waist": 90
    }



# ===============================
# Prediction
# ===============================

def predict_patient(patient):
    try:
        df = pd.DataFrame([patient])
        
        # Align columns with training data
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # FIX 1: Handle the 'object' dtype for gender to prevent the 'isnan' pipeline crash.
        # Assuming your training data used 'Male'/'Female' strings. 
        # (If it used 'M'/'F', simply change the mapped strings below).
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({1: 'Male', 0: 'Female'}).astype(object)
            
        # FIX 2: Cast ONLY the remaining columns to float, leaving gender as an object.
        numeric_cols = [col for col in df.columns if col != 'gender']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        
        # FIX 3: Pass the DataFrame directly (Do NOT use .to_numpy())
        pred = model.predict(df)[0]
        prob = 1.0
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0].max()
            
        return {
            "prediction": "Positive" if pred == 1 else "Negative",
            "probability": round(prob * 100, 2)
        }
    except Exception as e:
        return {"prediction": "Error", "detail": str(e)}

    
# ===============================
# Risk Score 
# ===============================

def calculate_metabolic_risk(p):

    score = 0
    
    # INDENTATION FIXED: These must be pushed in!
    if p["bmi"] >= 30: score += 2
    if p["fasting_blood_sugar"] > 100: score += 1
    if p["triglycerides"] > 150: score += 1
    if p["hdl_cholestrol"] < 40: score += 1
    if p["alt"] > 41: score += 1

    level = "Low" if score < 2 else "Moderate" if score < 4 else "High"

    return score, level

# ===============================
# Explainability
# ===============================
def explain_patient(p):
    reasons = []

    # Fix the indentation here
    if p["bmi"] >= 30:
        reasons.append("Obesity")

    if p["alt"] > 41:
        reasons.append("High ALT")

    if p["ast"] > 40:
        reasons.append("High AST")

    if p["fasting_blood_sugar"] > 100:
        reasons.append("High blood sugar")

    if p["triglycerides"] > 150:
        reasons.append("High triglycerides")

    if p["hdl_cholestrol"] < 40:
        reasons.append("Low HDL")

    return list(set(reasons))

# ===============================
# Recommendations
# ===============================
def generate_recommendations(p):

    rec = []

    if p["bmi"] >= 30:
        rec.append("Lose weight")

    if p["fasting_blood_sugar"] > 100:
        rec.append("Reduce sugar intake")

    if p["triglycerides"] > 150:
        rec.append("Reduce fats")

    if p["hdl_cholestrol"] < 40:
        rec.append("Exercise regularly")

    return list(set(rec))

# ===============================
# Lifestyle Plan
# ===============================
def generate_lifestyle_plan(p):

    return {
        "Exercise": "60 min/day" if p["bmi"] >= 30 else "30 min/day",
        "Diet": "Low Carb" if p["fasting_blood_sugar"] > 100 else "Balanced",
        "Water": "2-3L/day",
        "Sleep": "7-8h/day"
    }

# ===============================
# Future Risk Factors 
# ===============================
def future_risk_factors(p):

    risks = []

    if p["bmi"] >= 25:
        risks.append("Overweight or obesity increases NAFLD risk")

    if p.get("waist", 0) > 102:
        risks.append("Abdominal obesity detected")

    if p["fasting_blood_sugar"] >= 100:
        risks.append("Prediabetes detected")

    if p["triglycerides"] > 150:
        risks.append("Elevated triglycerides")

    if p["hdl_cholestrol"] < 40:
        risks.append("Low HDL cholesterol")

    if p["ldl_cholestrol"] > 130:
        risks.append("Elevated LDL cholesterol")

    hsi = (8 * (p["alt"] / (p["ast"] + 1))) + p["bmi"] + (2 if p["gender"] == 1 else 0)

    if hsi > 36:
        risks.append("High Hepatic Steatosis Index")

    return list(set(risks))
    

# ===============================
# Future Risk Analysis 
# ===============================
import math

def future_risk_analysis(p):

    score = 0

    
    if p["bmi"] >= 25:
        score += 2

    if p.get("waist", 0) > 102:
        score += 2

    if p["fasting_blood_sugar"] >= 100:
        score += 2

    if p["triglycerides"] > 150:
        score += 2

    if p["hdl_cholestrol"] < 40:
        score += 2

    if p["ldl_cholestrol"] > 130:
        score += 1

    if p["alt"] > 40 or p["ast"] > 40:
        score += 2

    # ===============================
    # Level
    # ===============================
    if score <= 2:
        level = "Low"
    elif score <= 5:
        level = "Moderate"
    else:
        level = "High"

    # ===============================
    # Probability
    # ===============================
    prob = 1 / (1 + math.exp(-0.5 * (score - 5)))
    prob = round(prob * 100, 2)

    return score, level, prob
    

# ===============================
# Final Analysis 
# ===============================

def analyze_patient(patient):
    pred = predict_patient(patient)
    score, level = calculate_metabolic_risk(patient)

    analysis = {
        "Prediction": pred["prediction"],
        "Probability": f"{pred.get('probability', 0)}%",
        "Prediction_Error": pred.get("detail", None),
        "Risk Level": level,
        "Risk Score": score,
        "Risk Factors": explain_patient(patient),
        "Recommendations": generate_recommendations(patient),
        "Lifestyle Plan": generate_lifestyle_plan(patient),
    }

    # ===============================
    # Future Risk 
    # ===============================
    if pred["prediction"] == "Negative":

        future_score, future_level, future_prob = future_risk_analysis(patient)

        analysis["Future Risk Level"] = future_level
        analysis["Future Risk Probability"] = f"{future_prob}%"
        analysis["Future Risk Score"] = future_score
        analysis["Future Risk Factors"] = future_risk_factors(patient)

    return analysis

    
# ===============================
# API Endpoint
# ===============================
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        text = extract_text_from_image(image)
        features = extract_features(text)
        analysis = analyze_patient(features)

        return {
            "success": True,
            "features": features,
            "analysis": analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))