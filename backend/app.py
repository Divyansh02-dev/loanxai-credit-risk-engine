"""
LoanXAI — Explainable AI Loan Approval System
Backend: Flask REST API
Model:   XGBoost Classifier (228 features, Home Credit Default Risk dataset)
XAI:     SHAP TreeExplainer

Run:
    cd backend
    python app.py

API Endpoints:
    GET  /health     -> model status
    POST /predict    -> loan decision + SHAP explanation
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

app = Flask(__name__, static_folder=None)
CORS(app)  # Allow requests from frontend (index.html)

print("Frontend dir:", FRONTEND_DIR)
print("Static dir:", STATIC_DIR)

# ─────────────────────────────────────────────
# Load Model & SHAP Explainer (once at startup)
# ─────────────────────────────────────────────
print("[LoanXAI] Loading model ...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
print(f"[LoanXAI] Model loaded — {len(FEATURES)} features | Classes: {model.classes_}")

print("[LoanXAI] Building SHAP TreeExplainer ...")
explainer = shap.TreeExplainer(model)
print("[LoanXAI] Ready.\n")


# ─────────────────────────────────────────────
# Human-Readable Feature Labels
# ─────────────────────────────────────────────
FEATURE_LABELS = {
    "EXT_SOURCE_1":                         "Credit Bureau Score 1",
    "EXT_SOURCE_2":                         "Credit Bureau Score 2",
    "EXT_SOURCE_3":                         "Credit Bureau Score 3",
    "AMT_INCOME_TOTAL":                     "Annual Income",
    "AMT_CREDIT":                           "Loan Amount Requested",
    "AMT_ANNUITY":                          "Monthly EMI (Annuity)",
    "AMT_GOODS_PRICE":                      "Price of Goods / Property",
    "DAYS_BIRTH":                           "Age of Applicant",
    "DAYS_EMPLOYED":                        "Employment Duration",
    "DAYS_ID_PUBLISH":                      "ID Document Age",
    "DAYS_REGISTRATION":                    "Address Registration Age",
    "DAYS_LAST_PHONE_CHANGE":               "Last Phone Change",
    "CNT_CHILDREN":                         "Number of Children",
    "CNT_FAM_MEMBERS":                      "Family Size",
    "REGION_POPULATION_RELATIVE":           "Region Population Density",
    "REGION_RATING_CLIENT":                 "Region Risk Rating",
    "REGION_RATING_CLIENT_W_CITY":          "City Risk Rating",
    "OWN_CAR_AGE":                          "Age of Owned Car",
    "FLAG_OWN_CAR_Y":                       "Owns a Car",
    "FLAG_OWN_REALTY_Y":                    "Owns Property / House",
    "CODE_GENDER_M":                        "Gender (Male)",
    "NAME_INCOME_TYPE_Working":             "Employment: Salaried",
    "NAME_INCOME_TYPE_Pensioner":           "Employment: Pensioner",
    "NAME_INCOME_TYPE_State servant":       "Employment: Govt Servant",
    "NAME_INCOME_TYPE_Commercial associate":"Employment: Commercial",
    "NAME_EDUCATION_TYPE_Higher education": "Education: Graduate+",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": "Education: 12th Pass",
    "NAME_FAMILY_STATUS_Married":           "Marital Status: Married",
    "NAME_FAMILY_STATUS_Single / not married": "Marital Status: Single",
    "NAME_HOUSING_TYPE_House / apartment":  "Housing: Own House",
    "NAME_HOUSING_TYPE_Rented apartment":   "Housing: Rented",
    "FLAG_DOCUMENT_3":                      "Key Document Submitted",
    "DEF_30_CNT_SOCIAL_CIRCLE":             "Social Circle Defaults (30d)",
    "DEF_60_CNT_SOCIAL_CIRCLE":             "Social Circle Defaults (60d)",
    "AMT_REQ_CREDIT_BUREAU_YEAR":           "Credit Enquiries (Last Year)",
    "ORGANIZATION_TYPE_Self-employed":      "Self-Employed",
    "ORGANIZATION_TYPE_Government":         "Works in Govt Org",
    "OCCUPATION_TYPE_Laborers":             "Occupation: Labourer",
    "OCCUPATION_TYPE_Managers":             "Occupation: Manager",
    "OCCUPATION_TYPE_High skill tech staff":"Occupation: IT / Tech",
}

def get_label(feature):
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


# ─────────────────────────────────────────────
# Plain-English SHAP Factor Explanations
# ─────────────────────────────────────────────
def explain_factor(feature, shap_val, feature_val):
    """
    Returns a plain-English sentence explaining why a feature
    helped or hurt the loan application.
    Negative SHAP = lowers default risk = GOOD for applicant.
    Positive SHAP = raises default risk = BAD for applicant.
    """
    is_positive = shap_val < 0  # good for applicant

    templates = {
        "EXT_SOURCE_2": {
            True:  f"Your credit bureau score ({feature_val:.2f}/1.0) is strong. Lenders see you as a reliable borrower.",
            False: f"Your credit bureau score ({feature_val:.2f}/1.0) is low. Past repayment issues or no credit history is a concern.",
        },
        "EXT_SOURCE_3": {
            True:  f"Third-party credit check ({feature_val:.2f}/1.0) is good — solid repayment track record.",
            False: f"Third-party credit check ({feature_val:.2f}/1.0) is weak — raises the lender's risk concern.",
        },
        "EXT_SOURCE_1": {
            True:  f"External credit score 1 ({feature_val:.2f}/1.0) supports your application.",
            False: f"External credit score 1 ({feature_val:.2f}/1.0) is below average — increases perceived risk.",
        },
        "AMT_INCOME_TOTAL": {
            True:  f"Annual income of ₹{feature_val:,.0f} is sufficient to handle the EMIs comfortably.",
            False: f"Annual income of ₹{feature_val:,.0f} may not be enough to cover the requested loan EMIs reliably.",
        },
        "AMT_CREDIT": {
            True:  f"Loan amount of ₹{feature_val:,.0f} is proportionate to your income and credit profile.",
            False: f"Loan amount of ₹{feature_val:,.0f} is high relative to your profile — consider requesting less.",
        },
        "AMT_ANNUITY": {
            True:  f"Monthly EMI of ₹{feature_val:,.0f} is manageable given your income.",
            False: f"Monthly EMI of ₹{feature_val:,.0f} is high and may strain your monthly budget significantly.",
        },
        "DAYS_BIRTH": {
            True:  f"Your age ({int(abs(feature_val)/365)} years) is in a favourable range — enough earning years to repay.",
            False: f"Your age ({int(abs(feature_val)/365)} years) is a mild concern — either very young or nearing retirement.",
        },
        "DAYS_EMPLOYED": {
            True:  f"You have been employed for {int(abs(feature_val)/365)} years — stable employment reassures lenders.",
            False: f"Employment history of only {int(abs(feature_val)/365)} years is short — lenders prefer longer job stability.",
        },
        "REGION_POPULATION_RELATIVE": {
            True:  "Your region has historically good loan repayment statistics.",
            False: "Your region has a higher historical default rate compared to the national average.",
        },
        "REGION_RATING_CLIENT": {
            True:  "Your region/city has a low credit risk rating — positive indicator.",
            False: "Your region/city carries a higher credit risk rating in the dataset.",
        },
        "FLAG_OWN_REALTY_Y": {
            True:  "You own a property — this shows financial stability and can serve as collateral.",
            False: "Not owning property means less financial security to back the loan.",
        },
        "FLAG_OWN_CAR_Y": {
            True:  "Car ownership indicates a certain level of financial stability.",
            False: "No car ownership is a minor negative signal in the historical data patterns.",
        },
        "CNT_CHILDREN": {
            True:  f"{int(feature_val)} children — financial obligations are within a manageable range.",
            False: f"{int(feature_val)} children — higher dependents increase monthly expenses, reducing repayment capacity.",
        },
        "FLAG_DOCUMENT_3": {
            True:  "Key document (ID/address proof) submitted — complete documentation helps your application.",
            False: "Key document not submitted — lenders strongly prefer complete paperwork.",
        },
        "DEF_30_CNT_SOCIAL_CIRCLE": {
            True:  "People in your social circle have a clean repayment record.",
            False: f"{int(feature_val)} person(s) in your social circle have defaulted recently — a risk signal for lenders.",
        },
        "AMT_REQ_CREDIT_BUREAU_YEAR": {
            True:  f"Only {int(feature_val)} credit enquir{'y' if feature_val==1 else 'ies'} this year — shows you are not credit-hungry.",
            False: f"{int(feature_val)} credit enquiries this year — too many applications signal financial stress.",
        },
        "CODE_GENDER_M": {
            True:  "Statistical gender profile aligns with lower-risk historical patterns.",
            False: "Statistical gender profile shows a slightly higher default tendency in historical data.",
        },
    }

    if feature in templates:
        return templates[feature][is_positive]

    label = get_label(feature)
    if is_positive:
        return f"{label} is favourable and supports your loan application."
    else:
        return f"{label} raised the AI's risk assessment for this application."


# ─────────────────────────────────────────────
# Feature Vector Builder
# Maps simple UI form → all 228 model features
# ─────────────────────────────────────────────
def build_feature_vector(form):
    v = {f: 0.0 for f in FEATURES}

    # ── Core numeric fields ──
    v["AMT_INCOME_TOTAL"]            = float(form.get("annual_income",    0))
    v["AMT_CREDIT"]                  = float(form.get("loan_amount",       0))
    v["AMT_ANNUITY"]                 = float(form.get("monthly_emi",       0))
    v["AMT_GOODS_PRICE"]             = float(form.get("goods_price", 0)) or float(form.get("loan_amount", 0)) * 0.9
    v["CNT_CHILDREN"]                = float(form.get("children",          0))
    v["CNT_FAM_MEMBERS"]             = float(form.get("family_members",    2))
    v["DAYS_BIRTH"]                  = -abs(float(form.get("age",         30))) * 365
    v["DAYS_EMPLOYED"]               = -abs(float(form.get("employment_years", 1))) * 365
    v["DAYS_REGISTRATION"]           = -abs(float(form.get("residence_years",  5))) * 365
    v["DAYS_ID_PUBLISH"]             = -abs(float(form.get("id_age_years",     3))) * 365
    v["DAYS_LAST_PHONE_CHANGE"]      = -abs(float(form.get("phone_change_days", 300)))
    v["OWN_CAR_AGE"]                 = float(form.get("car_age",          0))
    v["REGION_RATING_CLIENT"]        = float(form.get("region_rating",    2))
    v["REGION_RATING_CLIENT_W_CITY"] = float(form.get("region_rating",    2))
    v["REGION_POPULATION_RELATIVE"]  = float(form.get("region_pop",    0.02))

    # ── CIBIL → External credit scores (normalised 0–1) ──
    cibil      = float(form.get("cibil_score", 650))
    cibil_norm = (cibil - 300) / 600.0   # maps 300→0, 900→1
    v["EXT_SOURCE_1"] = round(max(0.0, min(1.0, cibil_norm)), 4)
    v["EXT_SOURCE_2"] = round(max(0.0, min(1.0, cibil_norm)), 4)
    v["EXT_SOURCE_3"] = round(max(0.0, min(1.0, cibil_norm)), 4)

    # ── Contact flags ──
    v["FLAG_MOBIL"]       = 1.0
    v["FLAG_CONT_MOBILE"] = 1.0
    v["FLAG_PHONE"]       = 1.0
    v["FLAG_EMP_PHONE"]   = float(form.get("has_work_phone", 0))
    v["FLAG_WORK_PHONE"]  = float(form.get("has_work_phone", 0))
    v["FLAG_EMAIL"]       = float(form.get("has_email",      1))

    # ── Documents ──
    v["FLAG_DOCUMENT_3"]  = float(form.get("doc3_submitted", 1))

    # ── Asset flags ──
    v["FLAG_OWN_CAR_Y"]    = float(form.get("owns_car",      0))
    v["FLAG_OWN_REALTY_Y"] = float(form.get("owns_property", 0))

    # ── Social circle ──
    v["DEF_30_CNT_SOCIAL_CIRCLE"] = float(form.get("social_defaults", 0))
    v["DEF_60_CNT_SOCIAL_CIRCLE"] = float(form.get("social_defaults", 0))
    v["OBS_30_CNT_SOCIAL_CIRCLE"] = 3.0
    v["OBS_60_CNT_SOCIAL_CIRCLE"] = 3.0

    # ── Credit bureau enquiries ──
    yearly = float(form.get("credit_enquiries_year", 1))
    v["AMT_REQ_CREDIT_BUREAU_YEAR"] = yearly
    v["AMT_REQ_CREDIT_BUREAU_QRT"]  = round(yearly / 4, 2)
    v["AMT_REQ_CREDIT_BUREAU_MON"]  = 0.0
    v["AMT_REQ_CREDIT_BUREAU_WEEK"] = 0.0
    v["AMT_REQ_CREDIT_BUREAU_DAY"]  = 0.0
    v["AMT_REQ_CREDIT_BUREAU_HOUR"] = 0.0

    # ── Gender ──
    v["CODE_GENDER_M"] = 1.0 if form.get("gender", "male").lower() == "male" else 0.0

    # ── Employment / Income type (one-hot) ──
    income_map = {
        "Working":              "NAME_INCOME_TYPE_Working",
        "Commercial associate": "NAME_INCOME_TYPE_Commercial associate",
        "Pensioner":            "NAME_INCOME_TYPE_Pensioner",
        "State servant":        "NAME_INCOME_TYPE_State servant",
        "Student":              "NAME_INCOME_TYPE_Student",
        "Unemployed":           "NAME_INCOME_TYPE_Unemployed",
        "Maternity leave":      "NAME_INCOME_TYPE_Maternity leave",
    }
    key = income_map.get(form.get("employment_type", "Working"))
    if key and key in FEATURES:
        v[key] = 1.0

    # ── Education (one-hot) ──
    edu_map = {
        "Higher education":              "NAME_EDUCATION_TYPE_Higher education",
        "Secondary / secondary special": "NAME_EDUCATION_TYPE_Secondary / secondary special",
        "Incomplete higher":             "NAME_EDUCATION_TYPE_Incomplete higher",
        "Lower secondary":               "NAME_EDUCATION_TYPE_Lower secondary",
    }
    key = edu_map.get(form.get("education", "Secondary / secondary special"))
    if key and key in FEATURES:
        v[key] = 1.0

    # ── Family status (one-hot) ──
    fam_map = {
        "Married":              "NAME_FAMILY_STATUS_Married",
        "Single / not married": "NAME_FAMILY_STATUS_Single / not married",
        "Separated":            "NAME_FAMILY_STATUS_Separated",
        "Widow":                "NAME_FAMILY_STATUS_Widow",
    }
    key = fam_map.get(form.get("family_status", "Married"))
    if key and key in FEATURES:
        v[key] = 1.0

    # ── Housing type (one-hot) ──
    housing_map = {
        "House / apartment":   "NAME_HOUSING_TYPE_House / apartment",
        "Rented apartment":    "NAME_HOUSING_TYPE_Rented apartment",
        "With parents":        "NAME_HOUSING_TYPE_With parents",
        "Municipal apartment": "NAME_HOUSING_TYPE_Municipal apartment",
        "Office apartment":    "NAME_HOUSING_TYPE_Office apartment",
    }
    key = housing_map.get(form.get("housing_type", "House / apartment"))
    if key and key in FEATURES:
        v[key] = 1.0

    # ── Organization type (one-hot) ──
    org_key = f"ORGANIZATION_TYPE_{form.get('organization_type', 'Business Entity Type 3')}"
    if org_key in FEATURES:
        v[org_key] = 1.0

    # ── Occupation type (one-hot) ──
    occ_key = f"OCCUPATION_TYPE_{form.get('occupation', 'Laborers')}"
    if occ_key in FEATURES:
        v[occ_key] = 1.0

    return v


# ─────────────────────────────────────────────
# Action Plan Generator
# ─────────────────────────────────────────────
def generate_action_plan(form, decision):
    actions = []
    cibil   = float(form.get("cibil_score",   650))
    income  = float(form.get("annual_income",   0))
    loan    = float(form.get("loan_amount",     0))
    enq     = float(form.get("credit_enquiries_year", 1))
    soc_def = float(form.get("social_defaults", 0))
    emp_yrs = float(form.get("employment_years", 0))
    own_prop= float(form.get("owns_property",   0))

    if cibil < 700:
        actions.append({
            "title":    "Improve your CIBIL Score",
            "detail":   (f"Your score ({int(cibil)}) is below the preferred 700+. Pay every EMI and credit card bill "
                         "on time for 6–12 consecutive months. Set up auto-pay to avoid accidental misses."),
            "impact":   "High",
            "timeline": "6–12 months",
        })

    if income > 0 and loan / income > 4:
        actions.append({
            "title":    "Reduce the Loan Amount Requested",
            "detail":   (f"Your loan (₹{loan:,.0f}) is {loan/income:.1f}× your annual income (₹{income:,.0f}). "
                         f"Try requesting ₹{loan * 0.65:,.0f} instead — it is more proportionate to your income."),
            "impact":   "High",
            "timeline": "Immediate",
        })

    if enq > 4:
        actions.append({
            "title":    "Stop Applying to Multiple Lenders Simultaneously",
            "detail":   (f"You have {int(enq)} credit enquiries this year. Every application triggers a hard pull "
                         "that lowers your CIBIL score. Apply to 1–2 lenders at a time, not all at once."),
            "impact":   "Medium",
            "timeline": "Immediate",
        })

    if not own_prop:
        actions.append({
            "title":    "Add a Guarantor or Offer Collateral",
            "detail":   ("You don't own property. Adding a guarantor (someone with CIBIL 750+) "
                         "or pledging gold / FD as collateral can significantly improve approval chances."),
            "impact":   "Medium",
            "timeline": "Immediate",
        })

    if emp_yrs < 2:
        actions.append({
            "title":    "Build Employment Stability",
            "detail":   (f"Only {emp_yrs} year(s) in current job. Lenders prefer 2+ years of stable employment. "
                         "Avoid job changes before applying. A government or large corporate job is viewed more favourably."),
            "impact":   "Medium",
            "timeline": "6–24 months",
        })

    if soc_def > 0:
        actions.append({
            "title":    "Be Cautious About Being a Guarantor for Others",
            "detail":   (f"{int(soc_def)} person(s) in your social circle have payment defaults. "
                         "If you are a guarantor on someone else's loan, their missed payments will hurt your score too."),
            "impact":   "Medium",
            "timeline": "Ongoing",
        })

    if decision == "APPROVED" and not actions:
        actions.append({
            "title":    "Maintain Your Strong Financial Habits",
            "detail":   ("Your profile is solid. Keep paying all EMIs and bills on time, avoid taking multiple "
                         "new loans at once, and keep your credit utilisation below 30%."),
            "impact":   "Maintain",
            "timeline": "Ongoing",
        })

    return actions[:4]


# ─────────────────────────────────────────────
# Dynamic SHAP-Based Suggestion Engine
# ─────────────────────────────────────────────
SUGGESTION_MAP = {
    "EXT_SOURCE_2": {
        "reason":  "Your credit bureau score is negatively impacting approval.",
        "action":  "Pay all EMIs and credit card bills on time for 6-12 months. "
                   "Clear any outstanding defaults or overdue amounts first.",
    },
    "EXT_SOURCE_3": {
        "reason":  "Third-party credit assessment shows repayment risk.",
        "action":  "Build a positive credit trail by using a credit card responsibly "
                   "(keep utilisation below 30%) and repaying on time.",
    },
    "EXT_SOURCE_1": {
        "reason":  "External credit score is below the preferred threshold.",
        "action":  "Avoid taking new loans for 6 months. Set up auto-pay for all "
                   "existing EMIs to prevent missed payments.",
    },
    "AMT_CREDIT": {
        "reason":  "Requested loan amount is high relative to your income profile.",
        "action":  "Consider reducing the loan amount by 30-40%. A smaller loan "
                   "is more likely to be approved and easier to repay.",
    },
    "AMT_ANNUITY": {
        "reason":  "Monthly EMI burden is too high compared to your income.",
        "action":  "Request a longer tenure to reduce EMI, or reduce the loan "
                   "amount. Ideal EMI should be below 40% of monthly income.",
    },
    "AMT_INCOME_TOTAL": {
        "reason":  "Income level is low for the requested loan amount.",
        "action":  "Show additional income sources (rental, spouse, FD interest) "
                   "or apply jointly with a co-applicant who has stable income.",
    },
    "DAYS_EMPLOYED": {
        "reason":  "Short employment duration raises stability concerns.",
        "action":  "Wait until you complete 2+ years at your current job. Avoid "
                   "job changes before applying. Government jobs are viewed favourably.",
    },
    "DAYS_BIRTH": {
        "reason":  "Your age profile is a mild concern for lenders.",
        "action":  "Young applicants should build credit history first (get a "
                   "secured credit card). Older applicants should show stable income.",
    },
    "AMT_REQ_CREDIT_BUREAU_YEAR": {
        "reason":  "Too many credit enquiries reduce your CIBIL score.",
        "action":  "Avoid applying to multiple lenders simultaneously. Wait 6 months "
                   "between applications. Each hard pull costs 5-10 CIBIL points.",
    },
    "FLAG_OWN_REALTY_Y": {
        "reason":  "No property ownership reduces collateral strength.",
        "action":  "Add a guarantor with CIBIL 750+ or pledge assets (gold, FD, "
                   "insurance policy) as collateral to strengthen the application.",
    },
    "FLAG_OWN_CAR_Y": {
        "reason":  "No vehicle ownership is a minor negative signal.",
        "action":  "Consider adding any existing assets to the application. Even "
                   "a two-wheeler loan repaid on time builds positive credit history.",
    },
    "DEF_30_CNT_SOCIAL_CIRCLE": {
        "reason":  "People in your social circle have recent loan defaults.",
        "action":  "If you are a guarantor on anyone's loan, ensure they are paying "
                   "on time. Their defaults can drag your CIBIL score down.",
    },
    "CNT_CHILDREN": {
        "reason":  "High number of dependents reduces repayment capacity.",
        "action":  "Show that your expenses are managed well. Provide bank statements "
                   "showing regular savings even with family obligations.",
    },
    "REGION_RATING_CLIENT": {
        "reason":  "Your region has a higher historical default rate.",
        "action":  "This factor is outside your control. Strengthen other areas like "
                   "CIBIL score and income documentation to compensate.",
    },
    "FLAG_DOCUMENT_3": {
        "reason":  "Key identity documents were not submitted.",
        "action":  "Submit Aadhaar, PAN, and address proof. Complete documentation "
                   "signals seriousness and improves approval chances significantly.",
    },
}


def generate_dynamic_suggestions(shap_series, X, form):
    """
    Generates personalized suggestions based on which features have
    negative SHAP impact (increase default risk).
    Returns (top_negative_factors, top_positive_factors, suggestions).
    """
    # Separate positive and negative SHAP contributors
    # Note: negative SHAP = helps approval, positive SHAP = hurts (increases default risk)
    negative_factors = []   # features that HURT the application (positive SHAP)
    positive_factors = []   # features that HELP the application (negative SHAP)

    for feat in FEATURES:
        sv = float(shap_series[feat])
        if abs(sv) < 0.005:
            continue  # skip negligible contributions

        fv = float(X[feat].iloc[0])
        entry = {
            "feature":       feat,
            "label":         get_label(feat),
            "shap_value":    round(sv, 4),
            "feature_value": round(fv, 4),
            "impact":        round(abs(sv), 4),
            "explanation":   explain_factor(feat, sv, fv),
        }

        if sv > 0:  # positive SHAP = increases default risk = BAD
            negative_factors.append(entry)
        else:       # negative SHAP = decreases default risk = GOOD
            positive_factors.append(entry)

    # Sort: negatives by impact descending (most harmful first)
    negative_factors.sort(key=lambda x: x["impact"], reverse=True)
    # Sort: positives by impact descending (most helpful first)
    positive_factors.sort(key=lambda x: x["impact"], reverse=True)

    # Take top 3 each
    top_negative = negative_factors[:3]
    top_positive = positive_factors[:3]

    # Generate personalized suggestions from negative factors
    suggestions = []
    seen_suggestions = set()
    for factor in negative_factors[:5]:  # check top 5 negatives
        feat = factor["feature"]
        if feat in SUGGESTION_MAP and feat not in seen_suggestions:
            sm = SUGGESTION_MAP[feat]
            suggestions.append({
                "feature":    feat,
                "label":      factor["label"],
                "reason":     sm["reason"],
                "action":     sm["action"],
                "priority":   "High" if factor["impact"] > 0.15 else "Medium" if factor["impact"] > 0.05 else "Low",
            })
            seen_suggestions.add(feat)

    # Add employment-specific suggestion if applicable
    emp_type = form.get("employment_type", "Working")
    if emp_type == "Unemployed" and "employment" not in seen_suggestions:
        suggestions.append({
            "feature":    "employment_type",
            "label":      "Employment Status",
            "reason":     "No current employment — a stable income source is required for loan approval.",
            "action":     "Secure employment first, or apply with a co-applicant/guarantor who has stable income. "
                          "Consider PM Mudra Yojana for self-employment loans.",
            "priority":   "High",
        })

    # Add property-specific suggestion if applicable
    if not float(form.get("owns_property", 0)) and "FLAG_OWN_REALTY_Y" not in seen_suggestions:
        suggestions.append({
            "feature":    "owns_property",
            "label":      "Property Ownership",
            "reason":     "No property to offer as collateral reduces loan security.",
            "action":     "Adding a guarantor with property or pledging gold/FD as "
                          "collateral can significantly improve approval chances.",
            "priority":   "Medium",
        })

    return top_negative, top_positive, suggestions[:5]

# ─────────────────────────────────────────────
# Hard Rule Engine (RBI-aligned policy rules)
# ─────────────────────────────────────────────
def apply_hard_rules(form):
    """
    Enforces non-negotiable lending policies inspired by
    Indian banking / RBI guidelines. These override ML
    predictions when triggered.
    """
    cibil       = float(form.get("cibil_score", 650))
    emp_type    = form.get("employment_type", "Working")
    loan_amount = float(form.get("loan_amount", 0))
    enquiries   = float(form.get("credit_enquiries_year", 0))
    income      = float(form.get("annual_income", 0))

    # Rule 1: Very low CIBIL score
    if cibil < 550:
        return {
            "override": True,
            "decision": "REJECTED",
            "reason":   (f"CIBIL score of {int(cibil)} is below the minimum threshold of 550. "
                         "Most Indian banks and NBFCs will not process applications with a "
                         "score this low. Focus on timely bill payments and clearing any "
                         "outstanding defaults to improve your score over 6-12 months."),
            "rule":     "CIBIL_BELOW_550",
        }

    # Rule 2: Unemployed with high loan request
    if emp_type == "Unemployed" and loan_amount > 200000:
        return {
            "override": True,
            "decision": "REJECTED",
            "reason":   (f"Loan amount of Rs.{loan_amount:,.0f} cannot be sanctioned without "
                         "a verified income source. Applicants with no current employment "
                         "are limited to loans under Rs.2,00,000. Consider applying under "
                         "PM Mudra Yojana (Shishu tier) for amounts up to Rs.50,000."),
            "rule":     "UNEMPLOYED_HIGH_LOAN",
        }

    # Rule 3: Excessive credit enquiries
    if enquiries > 8:
        return {
            "override": True,
            "decision": "REJECTED",
            "reason":   (f"You have {int(enquiries)} credit enquiries in the last year. "
                         "More than 8 enquiries signals credit-hungry behaviour and each "
                         "hard pull lowers your CIBIL score. Wait 6 months before applying "
                         "again, and limit applications to 1-2 lenders at a time."),
            "rule":     "EXCESSIVE_ENQUIRIES",
        }

    # Rule 4: Loan-to-income ratio too extreme
    if income > 0 and loan_amount / income > 6:
        return {
            "override": True,
            "decision": "REJECTED",
            "reason":   (f"Loan amount (Rs.{loan_amount:,.0f}) is {loan_amount/income:.1f}x your "
                         f"annual income (Rs.{income:,.0f}). Indian banks typically cap loans at "
                         "4-5x annual income. Reduce the requested amount or show additional "
                         "income sources such as rental income, spouse income, or FD interest."),
            "rule":     "EXTREME_LTI_RATIO",
        }

    return {"override": False}


def get_confidence_level(probability):
    """Classify prediction confidence into tiers."""
    if probability >= 90:
        return {"level": "High",   "color": "green",  "description": "Model is highly confident in this decision"}
    elif probability >= 70:
        return {"level": "Medium", "color": "amber",  "description": "Model has moderate confidence; edge case"}
    else:
        return {"level": "Low",    "color": "red",    "description": "Model has low confidence; manual review recommended"}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "model":    type(model).__name__,
        "features": len(FEATURES),
        "xai":      "SHAP TreeExplainer",
        "engine":   "Hybrid (Rule + ML)",
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.get_json(force=True)
        if not form:
            return jsonify({"error": "No JSON body received"}), 400

        requested = float(form.get("loan_amount", 0))

        # ── STEP 1: Hard Rule Check ──
        rule_result = apply_hard_rules(form)

        if rule_result["override"]:
            # Generate rule-specific actions
            actions = generate_action_plan(form, "REJECTED")

            return jsonify({
                "decision":             "REJECTED",
                "risk_score":           80,
                "default_probability":  80,
                "approval_probability": 20,
                "source":               "RULE_BASED",
                "rule":                 rule_result.get("rule", "POLICY"),
                "rule_reason":          rule_result["reason"],
                "confidence":           {"level": "High", "color": "green",
                                         "description": "Policy rule triggered with certainty"},
                "recommended_loan":     round(requested * 0.40),
                "recommended_note":     f"Consider applying for Rs.{requested * 0.40:,.0f} after addressing the policy concern.",
                "factors":              [],
                "actions":              actions,
                "model_info": {
                    "type":           type(model).__name__,
                    "dataset":        "Home Credit Default Risk",
                    "features_used":  len(FEATURES),
                    "explainability": "Rule-Based Override (ML skipped)",
                    "engine":         "RULE_BASED",
                },
            })

        # ── STEP 2: ML Prediction (no rule override) ──
        feat_dict = build_feature_vector(form)
        X = pd.DataFrame([feat_dict])[FEATURES]

        # Raw model probability
        proba      = model.predict_proba(X)[0]
        raw_prob   = float(proba[0])  # probability of class 0 (no default / repayment)

        # Calibrated probability: prevents unrealistic 98%+ approvals
        calibrated_prob = 0.6 * raw_prob + 0.2
        calibrated_prob = max(0.05, min(0.95, calibrated_prob))  # clamp to 5%-95%

        # Risk score (higher = worse)
        risk_score = round((1 - calibrated_prob) * 100, 1)

        # Approval probability (in %)
        approval_pct = round(calibrated_prob * 100, 1)

        # Decision based on calibrated probability
        if calibrated_prob > 0.65:
            decision = "APPROVED"
        elif calibrated_prob > 0.45:
            decision = "MANUAL REVIEW"
        else:
            decision = "REJECTED"

        # Confidence level
        confidence = get_confidence_level(approval_pct if decision == "APPROVED" else risk_score)

        # ── STEP 3: SHAP Explanations ──
        shap_vals    = explainer.shap_values(X)[0]
        shap_series  = pd.Series(shap_vals, index=FEATURES)
        top_features = shap_series.abs().nlargest(8).index.tolist()

        factors = []
        for feat in top_features:
            sv = float(shap_series[feat])
            fv = float(X[feat].iloc[0])
            factors.append({
                "feature":       feat,
                "label":         get_label(feat),
                "shap_value":    round(sv, 4),
                "feature_value": round(fv, 4),
                "direction":     "positive" if sv < 0 else "negative",
                "impact":        round(abs(sv), 4),
                "explanation":   explain_factor(feat, sv, fv),
            })

        # ── STEP 3b: Dynamic Suggestions Engine ──
        top_neg, top_pos, suggestions = generate_dynamic_suggestions(shap_series, X, form)

        # ── STEP 4: Recommended loan ──
        if decision == "REJECTED":
            rec_loan = round(requested * 0.50)
            rec_note = f"A reduced amount of Rs.{rec_loan:,.0f} may improve your chances."
        elif decision == "MANUAL REVIEW":
            rec_loan = round(requested * 0.75)
            rec_note = f"Consider applying for Rs.{rec_loan:,.0f} to strengthen approval odds."
        else:
            rec_loan = int(requested)
            rec_note = f"Your requested amount of Rs.{rec_loan:,.0f} looks appropriate for your profile."

        # ── STEP 5: Action plan ──
        actions = generate_action_plan(form, decision)

        return jsonify({
            "decision":             decision,
            "risk_score":           risk_score,
            "default_probability":  risk_score,
            "approval_probability": approval_pct,
            "source":               "ML_MODEL",
            "confidence":           confidence,
            "calibration_note":     f"Raw model: {round(raw_prob*100,1)}% -> Calibrated: {approval_pct}%",
            "recommended_loan":     rec_loan,
            "recommended_note":     rec_note,
            "factors":              factors,
            "top_negative_factors": top_neg,
            "top_positive_factors": top_pos,
            "suggestions":          suggestions,
            "actions":              actions,
            "model_info": {
                "type":           type(model).__name__,
                "dataset":        "Home Credit Default Risk",
                "features_used":  len(FEATURES),
                "explainability": "SHAP TreeExplainer",
                "engine":         "ML_MODEL",
            },
        })

    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print("=" * 52)
    print(f"  LoanXAI Backend  ->  http://0.0.0.0:{port}")
    print("  Press Ctrl+C to stop")
    print("=" * 52)
    app.run(host="0.0.0.0", port=port)
