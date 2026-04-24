# LoanXAI — Explainable AI Loan Approval System
### Final Year Project | Indian Credit Market

---

## Project Structure

```
LoanXAI/
│
├── backend/                        # Python Flask REST API
│   ├── app.py                      # Main server — prediction + SHAP
│   ├── model.pkl                   # Trained XGBoost model (228 features)
│   └── requirements.txt            # Python dependencies
│
├── frontend/                       # Web UI (open in browser — no npm needed)
│   ├── index.html                  # Main HTML page (4 tabs)
│   └── static/
│       ├── css/
│       │   └── styles.css          # All styles
│       └── js/
│           └── main.js             # All frontend logic
│
├── glossary/
│   └── GLOSSARY.md                 # All loan + ML terms explained
│
├── docs/
│   └── DATASETS.md                 # Dataset links + download commands
│
├── notebooks/
│   └── model_training.ipynb        # Jupyter notebook: train model + SHAP plots
│
├── sample_data/
│   └── sample_requests.json        # 3 test cases (approved/rejected/review)
│
├── .vscode/
│   ├── settings.json               # Python interpreter + editor config
│   ├── launch.json                 # One-click Flask debug run
│   └── extensions.json             # Recommended VS Code extensions
│
└── README.md                       # This file
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | XGBoost Classifier |
| Explainability | SHAP TreeExplainer |
| Backend | Python 3 + Flask |
| Frontend | Plain HTML + CSS + JavaScript |
| Dataset | Home Credit Default Risk (Kaggle) |

---

## Setup in VS Code — Step by Step

### Prerequisites
- Python 3.8+ installed
- VS Code installed
- model.pkl in the `backend/` folder

---

### Step 1: Open the Project
```
File → Open Folder → select the LoanXAI folder
```

When prompted, click **"Yes, I trust the authors"**.

---

### Step 2: Install Recommended Extensions
VS Code will show a popup: *"Do you want to install the recommended extensions?"*
Click **Install All**. This installs:
- Python (ms-python.python)
- Debugpy (for breakpoints)
- Live Server (to serve the frontend)
- Jupyter (for the notebook)

---

### Step 3: Create a Virtual Environment (Recommended)
Open the integrated terminal (`Ctrl + \`` or Terminal → New Terminal):

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

---

### Step 4: Start the Flask Backend
**Option A — Using VS Code Run button:**
- Open `backend/app.py`
- Press `F5` (or Run → Start Debugging)
- Select "Run Flask Backend" from the dropdown

**Option B — Using terminal:**
```bash
cd backend
python app.py
```

You should see:
```
[LoanXAI] Loading model ...
[LoanXAI] Model loaded — 228 features | Classes: [0 1]
[LoanXAI] Building SHAP TreeExplainer ...
[LoanXAI] Ready.

====================================================
  LoanXAI Backend  →  http://localhost:5000
  Press Ctrl+C to stop
====================================================
```

**Verify it's working:**
Open your browser and go to: `http://localhost:5000/health`

You should see:
```json
{"status": "ok", "model": "XGBClassifier", "features": 228, "xai": "SHAP TreeExplainer"}
```

---

### Step 5: Open the Frontend

**Option A — Live Server (Recommended):**
- In VS Code Explorer, right-click `frontend/index.html`
- Click **"Open with Live Server"**
- Browser opens automatically at `http://127.0.0.1:5500/frontend/index.html`

**Option B — Direct file open:**
- Double-click `frontend/index.html` in your file manager
- Or drag it into your browser

> **Note:** If you open the file directly (file:// protocol), the API calls still work as long as the backend is running on localhost:5000.

---

### Step 6: Use the Application

1. The app opens on the **Application Form** tab
2. Fill in the applicant's details (a sample is pre-filled)
3. Click **🔍 Run AI Analysis**
4. The **AI Result** tab opens automatically showing:
   - Decision: APPROVED / REJECTED / MANUAL REVIEW
   - Risk gauge (0–100 score)
   - SHAP factor bars with plain-English explanations
   - Action plan for the applicant
5. Explore **Loan Glossary** and **Datasets & ML Guide** tabs

---

## API Reference

### Health Check
```
GET http://localhost:5000/health
```

### Predict
```
POST http://localhost:5000/predict
Content-Type: application/json

{
  "name": "Ramesh Kumar",
  "age": 34,
  "gender": "male",
  "annual_income": 480000,
  "cibil_score": 710,
  "loan_amount": 600000,
  "monthly_emi": 18000,
  "employment_type": "Working",
  "employment_years": 4,
  "children": 1,
  "family_members": 3,
  "education": "Secondary / secondary special",
  "occupation": "Laborers",
  "owns_property": 1,
  "owns_car": 0,
  "housing_type": "House / apartment",
  "family_status": "Married",
  "credit_enquiries_year": 1,
  "social_defaults": 0,
  "doc3_submitted": 1,
  "region_rating": 2,
  "goods_price": 550000
}
```

### Test with sample_requests.json
```bash
# Test Case 1 (should be APPROVED)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_data/sample_requests.json

# Or use VS Code REST Client extension with a .http file
```

---

## Decision Thresholds

| Default Probability | Decision |
|---|---|
| < 30% | ✅ APPROVED |
| 30% – 55% | 🟡 MANUAL REVIEW |
| > 55% | ❌ REJECTED |

---

## Model Information

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Dataset | Home Credit Default Risk (Kaggle) |
| Features | 228 |
| Target | 1 = Default, 0 = No Default |
| Typical AUC-ROC | ~0.91 |
| Explainability | SHAP TreeExplainer |

### Top 10 Most Important Features (SHAP)

1. EXT_SOURCE_2 — External credit bureau score 2
2. EXT_SOURCE_3 — External credit bureau score 3
3. EXT_SOURCE_1 — External credit bureau score 1
4. AMT_CREDIT — Loan amount
5. AMT_ANNUITY — Monthly EMI
6. DAYS_BIRTH — Age of applicant
7. DAYS_EMPLOYED — Employment duration
8. AMT_INCOME_TOTAL — Annual income
9. REGION_POPULATION_RELATIVE — Geographic risk
10. DAYS_ID_PUBLISH — ID document age

---

## Troubleshooting

### "Could not connect to backend"
- Make sure `python app.py` is running in the terminal
- Check that the terminal shows "Running on http://localhost:5000"
- Try visiting `http://localhost:5000/health` in your browser

### "ModuleNotFoundError: No module named 'xgboost'"
```bash
pip install -r backend/requirements.txt
```

### CORS error in browser console
- This should not happen as flask-cors is configured
- If it does, make sure you are using the correct version: `pip install flask-cors==5.0.1`

### Model takes too long to load
- Normal — SHAP TreeExplainer builds an internal tree structure on first load
- Takes 5–15 seconds. After that, each prediction is instant.

---

## Limitations

1. This is a final year project demo — not a production financial product
2. CIBIL score is approximated via normalised EXT_SOURCE features — real deployment needs CIBIL API
3. The model reflects biases present in the training data
4. Real loan decisions must comply with RBI Fair Practice Code

---

*LoanXAI — Final Year Project*
*XGBoost + SHAP + Flask + HTML/CSS/JS*
