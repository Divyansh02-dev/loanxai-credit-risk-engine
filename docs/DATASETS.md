# LoanXAI — Datasets Guide

Training datasets for Indian credit risk models with download instructions.

---

## Datasets Used / Recommended

### 1. Home Credit Default Risk ⭐ (Used in this project)
- **URL**: https://www.kaggle.com/competitions/home-credit-default-risk
- **Rows**: 307,511  |  **Features**: 122 raw (228 after engineering)
- **India Fit**: ★★★★★
- **Why**: Home Credit operates in India via HDB Financial Services. Borrower profile closely matches Indian NBFC customers including thin-file, semi-formal, and rural applicants.

### 2. Give Me Some Credit
- **URL**: https://www.kaggle.com/c/GiveMeSomeCredit/data
- **Rows**: 150,000  |  **Features**: 11
- **India Fit**: ★★★
- **Why**: Clean, beginner-friendly. Adapt income to ₹ and map US credit score to CIBIL range.

### 3. German Credit (UCI)
- **URL**: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- **Rows**: 1,000  |  **Features**: 20
- **India Fit**: ★★
- **Why**: Classic academic dataset. Perfect for SHAP/LIME demos and XAI research papers.

### 4. Lending Club
- **URL**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Rows**: 2.2M  |  **Features**: 150+
- **India Fit**: ★★★
- **Why**: Large-scale P2P lending data. Map FICO → CIBIL range for Indian experiments.

### 5. RBI / data.gov.in (India)
- **URL**: https://data.gov.in/sector/finance
- **India Fit**: ★★★★★
- **Why**: Official Indian government data. MSME loans, priority sector lending, state-wise credit flow.

### 6. SIDBI MSME Pulse
- **URL**: https://www.sidbi.in/en/msme-pulse
- **India Fit**: ★★★★
- **Why**: India's most authoritative MSME credit data. Quarterly reports with sector and geographic breakdown.

---

## Download Commands

```bash
# Kaggle setup (one-time)
pip install kaggle
# Place your kaggle.json in ~/.kaggle/

# Home Credit Default Risk (main dataset)
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/

# Give Me Some Credit
kaggle competitions download -c GiveMeSomeCredit

# German Credit (no login needed)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
```
