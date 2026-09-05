import pandas as pd
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection — Risk Dashboard")
st.caption("Stacking ensemble: XGBoost + Random Forest, combined by a meta-model")

@st.cache_resource
def load_models():
    xgb_model = joblib.load("model_xgb.joblib")
    rf_model = joblib.load("model_rf.joblib")
    meta_model = joblib.load("model_meta.joblib")
    return xgb_model, rf_model, meta_model

@st.cache_data
def load_test_data():
    X_test = pd.read_csv("stack_X_test.csv")
    y_test = pd.read_csv("stack_y_test.csv").squeeze()
    return X_test, y_test

@st.cache_resource
def get_shap_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)

xgb_model, rf_model, meta_model = load_models()
X_test, y_test = load_test_data()
explainer = get_shap_explainer(xgb_model)

@st.cache_data
def get_predictions(_xgb_model, _rf_model, _meta_model, X_test):
    xgb_prob = _xgb_model.predict_proba(X_test)[:, 1]
    rf_prob = _rf_model.predict_proba(X_test)[:, 1]
    meta_features = pd.DataFrame({"xgb_probability": xgb_prob, "rf_probability": rf_prob})
    final_probability = _meta_model.predict_proba(meta_features)[:, 1]
    return final_probability

final_probability = get_predictions(xgb_model, rf_model, meta_model, X_test)

st.sidebar.header("Decision Threshold")
st.sidebar.write("Lower = catches more fraud, more false alarms. Higher = fewer false alarms, might miss fraud.")
threshold = st.sidebar.slider("Fraud probability threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
final_prediction = (final_probability >= threshold).astype(int)

precision = precision_score(y_test, final_prediction, zero_division=0)
recall = recall_score(y_test, final_prediction, zero_division=0)
cm = confusion_matrix(y_test, final_prediction)
false_positive = cm[0][1]
false_negative = cm[1][0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{precision:.1%}")
col2.metric("Recall", f"{recall:.1%}")
col3.metric("False Positives", false_positive)
col4.metric("False Negatives", false_negative)

st.divider()
st.subheader("Flagged Transactions")

results_table = X_test.copy()
results_table["fraud_probability"] = final_probability
results_table["flagged"] = final_prediction
results_table["actual_class"] = y_test.values
results_table["row_id"] = results_table.index

flagged_only = results_table[results_table["flagged"] == 1].sort_values("fraud_probability", ascending=False)

st.write(f"{len(flagged_only)} transactions flagged at this threshold.")
st.dataframe(flagged_only[["row_id", "fraud_probability", "actual_class", "Amount_scaled"]].head(50), use_container_width=True)

st.divider()
st.subheader("Why was a transaction flagged?")

if len(flagged_only) > 0:
    selected_row_id = st.selectbox("Pick a flagged transaction to explain (shown by row ID):", flagged_only["row_id"].tolist())
    selected_row = X_test.loc[[selected_row_id]]
    shap_values = explainer.shap_values(selected_row)
    contributions = pd.Series(shap_values[0], index=X_test.columns).sort_values(key=abs, ascending=False)
    top_contributions = contributions.head(8)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_contributions.values]
    ax.barh(top_contributions.index[::-1], top_contributions.values[::-1], color=colors[::-1])
    ax.set_xlabel("Contribution to fraud score")
    ax.set_title(f"SHAP explanation for transaction (row {selected_row_id})")
    st.pyplot(fig)
else:
    st.info("No transactions are flagged at this threshold. Try lowering it.")