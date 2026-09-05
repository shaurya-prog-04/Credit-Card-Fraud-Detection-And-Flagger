# Credit Card Fraud Detection & Flagger

A fraud detection system built for Razorpay's AI Risk Manager track. It detects fraudulent credit card transactions using a stacking ensemble of XGBoost and Random Forest, with every flagged transaction explained via SHAP — not a black box.

**Live Demo:** https://credit-card-fraud-detection-and-flagger-dvf5qhdk8yfqjyfnbchpd9.streamlit.app/
*(Note: free-tier hosting may take ~30-60 seconds to wake up if it hasn't been visited recently.)*

---

## The Problem

Fraud makes up just **0.17%** of transactions in this dataset. A naive model can predict "not fraud" every time and still be 99.8% accurate while catching zero fraud — so accuracy is a useless metric here. The real challenge is catching fraud **without punishing honest customers** with false alarms, since every wrongly blocked transaction has a real cost: lost trust, support tickets, churn.

This project is built around measuring and minimizing that false-positive cost honestly, end to end.

---

## Approach

1. **Data audit & cleaning** — Found and removed 1,081 duplicate rows (including 32 duplicated fraud cases) that could otherwise leak between train and test sets and inflate results.
2. **Preprocessing** — Scaled Amount and Time; split into train/test *before* balancing, so the test set stays realistic. Used SMOTE to balance only the training data.
3. **Baseline → stronger models** — Logistic Regression baseline, then Random Forest and XGBoost, each evaluated on precision, recall, and false-positive cost (not accuracy).
4. **Two ensembling strategies compared:**
   - **Fast Filter + Second Opinion** — XGBoost handles confident cases directly; Random Forest is only consulted for the "unsure" middle-ground transactions.
   - **Stacking** — A meta-model learns the optimal way to combine both base models' confidence scores, trained on a dedicated validation split so it never sees data the base models trained on (avoiding leakage).
5. **Explainability (SHAP)** — Every flagged transaction can be explained: which specific features pushed the fraud score up or down.
6. **Interactive dashboard** — Adjust the decision threshold live and watch precision/recall/false-positives update; click any flagged transaction to see its SHAP explanation.

This system is strictly **detection and alerting only** — it never autonomously moves funds or takes irreversible action. Every flag is meant to go to a human analyst, with full visibility into why it was raised.

---

## Results

| Model | Precision | Recall | False Positives | False Negatives |
|---|---|---|---|---|
| Logistic Regression (baseline) | 5.3% | 87.4% | 1,479 | 12 |
| Random Forest | 67.3% | 80.0% | 37 | 19 |
| XGBoost | 70.5% | 77.9% | 31 | 21 |
| Ensemble (Fast Filter + Second Opinion) | 75.3% | 80.0% | 25 | 19 |
| **Ensemble (Stacking)** | **92.5%** | **77.9%** | **6** | 21 |

The stacking ensemble reduced false positives by **~99%** compared to the baseline (1,479 → 6), while still catching nearly the same amount of fraud.

*Note: the stacking ensemble was evaluated on a separate train/validation/test split from the other models, since it required a dedicated validation set to train the meta-model without leakage. Both splits use the same stratification approach, so the comparison is fair, though not on literally identical test rows.*

---

## Explainability

Every flagged transaction can be broken down by feature contribution using SHAP (SHapley Additive exPlanations). See `shap_explanation_row_*.png` for individual transaction examples, and `shap_global_importance.png` for which features matter most overall — V14 and V4 emerged as the two strongest fraud signals across the dataset.

---

## Repository Structure

| File | Purpose |
|---|---|
| `Code.py` | Full pipeline: EDA, preprocessing, baseline model, Random Forest, XGBoost |
| `app.py` | Interactive Streamlit dashboard |
| `creditcard.csv` / `creditcard_clean.csv` | Raw and deduplicated dataset |
| `X_train_balanced.csv`, `y_train_balanced.csv`, `X_test.csv`, `y_test.csv` | Preprocessed data for baseline/RF/XGBoost |
| `stack_X_test.csv`, `stack_y_test.csv` | Held-out test set for the stacking ensemble |
| `model_xgb.joblib`, `model_rf.joblib`, `model_meta.joblib` | Trained model artifacts used by the dashboard |
| `model_results.csv` | Full metrics comparison table across all models |
| `*.png` | EDA charts, precision-recall curves, and SHAP explanations |
| `requirements.txt` | Python dependencies |

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard loads the pre-trained `.joblib` models directly — no retraining needed.

---

## Limitations

- **Compute constraints:** Random Forest was trained on a 40,000-row subsample of the balanced training data rather than the full ~450,000 rows, due to limited local compute. XGBoost (trained on the full dataset) performed at least as well, so this tradeoff didn't meaningfully affect conclusions.
- **Separate test sets:** The stacking ensemble's numbers come from a different train/validation/test split than the other models, since stacking specifically required a held-out validation set. Both splits use identical stratification, but aren't the same literal rows.
- **AI-assisted UI:** The Streamlit dashboard interface was scaffolded with AI assistance, since it was a new tool for this project. The underlying logic — model outputs, threshold handling, SHAP wiring — was designed and implemented by me.

## What's Next

- Calibrate the decision threshold using a real cost function (assigning actual cost to false positives vs. false negatives)
- Test against a live/streaming transaction feed rather than a static historical dataset
- Investigate feature engineering on Time (e.g., hour-of-day) given fraud's observed clustering in low-traffic windows

---

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — anonymized European cardholder transactions, September 2013.