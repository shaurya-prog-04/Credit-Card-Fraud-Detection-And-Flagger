'''
Credit Card Fraud Detection — Project Overview
This project detects fraudulent credit card transactions using a stacking ensemble of XGBoost and Random Forest, combined by a meta-model that learns the optimal way to weigh their opinions. It's built for Razorpay's AI Risk Manager track and is strictly defense-only — it flags and explains risk, it never takes autonomous action on funds.
The core challenge: fraud makes up only 0.17% of transactions, so the project is built around handling that imbalance honestly rather than chasing misleading accuracy numbers.
Approach:
1) Cleaned the data (found and removed 1,081 duplicate rows, including 32 duplicated fraud cases, to prevent train/test leakage)
2) Balanced the training data with SMOTE — test data stays untouched and reflects real-world imbalance
3) Benchmarked Logistic Regression → Random Forest → XGBoost, each evaluated on precision, recall, and false-positive cost (not accuracy)
4) Built and compared two ensembling strategies: a fast-filter + second-opinion system, and a stacking meta-model trained on a separate validation split to avoid leakage
5) Added SHAP explainability so every flagged transaction shows why it was flagged
'''



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD THE DATA

df = pd.read_csv("creditcard.csv")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows of data:")
print(df.head())

# CHECK MISSIN VALUES

missing_values_per_column = df.isnull().sum()
total_missing = missing_values_per_column.sum()
print("\nTotal missing values in the whole dataset:", total_missing)


# CHECK DUPLICATE ROWS

duplicate_mask = df.duplicated(keep=False)
all_duplicate_rows = df[duplicate_mask]
extra_duplicate_count = df.duplicated(keep="first").sum()

print("\nNumber of extra duplicate rows:", extra_duplicate_count)
print("Duplicate rows split by Class:")
print(all_duplicate_rows["Class"].value_counts())

fraud_duplicates = all_duplicate_rows[all_duplicate_rows["Class"] == 1]
print("\nFraud transactions involved in duplication:", len(fraud_duplicates))
print("This matters because: if not removed before splitting into")
print("train/test, the same fraud transaction could end up in BOTH sets,")
print("which would make our test results look better than they really are.")


# CLASS DISTRIBUTION [Normal(0) vs Fraud(1)]

class_counts = df["Class"].value_counts()
number_of_fraud = class_counts[1]
number_of_normal = class_counts[0]
total_transactions = number_of_fraud + number_of_normal
fraud_percentage = (number_of_fraud / total_transactions) * 100

print("\nClass distribution (0 = Normal, 1 = Fraud):")
print(class_counts)
print("Fraud makes up", round(fraud_percentage, 3), "% of all transactions")

plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.savefig("class_distribution.png", bbox_inches="tight")
plt.close()
print("Saved chart: class_distribution.png")

# AMOUNT COMPARISON

normal_transactions = df[df["Class"] == 0]
fraud_transactions = df[df["Class"] == 1]

print("\nAmount statistics for NORMAL transactions:")
print(normal_transactions["Amount"].describe())
print("\nAmount statistics for FRAUD transactions:")
print(fraud_transactions["Amount"].describe())

plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df, showfliers=False)
plt.title("Transaction Amount by Class (extreme outliers hidden)")
plt.savefig("amount_by_class.png", bbox_inches="tight")
plt.close()
print("Saved chart: amount_by_class.png")

# TIMING PATTERNS 

plt.figure(figsize=(10, 5))
plt.hist(normal_transactions["Time"], bins=50, alpha=0.5, label="Normal", density=True)
plt.hist(fraud_transactions["Time"], bins=50, alpha=0.5, label="Fraud", density=True)
plt.legend()
plt.title("When Do Transactions Happen? Normal vs Fraud")
plt.xlabel("Time (seconds since first transaction)")
plt.savefig("time_distribution.png", bbox_inches="tight")
plt.close()
print("Saved chart: time_distribution.png")

# EXTREME END AMOUNT CHECK

zero_amount_rows = df[df["Amount"] == 0]
print("\nTransactions with Amount = 0:", len(zero_amount_rows))
print("Of those, how many are fraud:", len(zero_amount_rows[zero_amount_rows["Class"] == 1]))

max_amount = df["Amount"].max()
percentile_99 = df["Amount"].quantile(0.99)
print("\nHighest transaction amount:", round(max_amount, 2))
print("99th percentile amount:", round(percentile_99, 2))
print("The gap between these tells us there are a few extreme outliers,")
print("worth being aware of, though not necessarily removing.")

# CLEAN THE DATA (REV DUPES)

df_clean = df.drop_duplicates()

print("\nShape before removing duplicates:", df.shape)
print("Shape after removing duplicates:", df_clean.shape)

fraud_percent_after = (df_clean["Class"].sum() / len(df_clean)) * 100
print("Fraud percentage after cleaning:", round(fraud_percent_after, 4), "%")

df_clean.to_csv("creditcard_clean.csv", index=False)
print("\nSaved cleaned dataset as creditcard_clean.csv")
print("New shape:", df_clean.shape)

print("EDA REPORT")
print("Total transactions:", total_transactions)
print("Fraud transactions:", number_of_fraud, "(", round(fraud_percentage, 3), "%)")
print("Missing values:", total_missing)
print("Duplicate rows removed:", extra_duplicate_count, "(including", len(fraud_duplicates), "fraud cases)")
print("Zero-amount transactions:", len(zero_amount_rows))
print("Final clean dataset shape:", df_clean.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard_clean.csv")
print("Loaded data. Shape:", df.shape)


# SCALE THE AMT AND TIME

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

df = df.drop(columns=["Amount", "Time"])
print("\nAmount and Time have been scaled.")
print("New Amount_scaled range: min =", round(df["Amount_scaled"].min(), 2),
      ", max =", round(df["Amount_scaled"].max(), 2))
print("New Time_scaled range: min =", round(df["Time_scaled"].min(), 2),
      ", max =", round(df["Time_scaled"].max(), 2))

X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# SPLIT INTO TTS (BEFORE SMOTE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nFraud percentage in training set:", round((y_train.sum() / len(y_train)) * 100, 4), "%")
print("Fraud percentage in test set:", round((y_test.sum() / len(y_test)) * 100, 4), "%")

# APPLY SMOTE ON T

print("\nBefore SMOTE")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE")
print(y_train_balanced.value_counts())

X_train_balanced.to_csv("X_train_balanced.csv", index=False)
y_train_balanced.to_csv("y_train_balanced.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

RESULTS = []  
X_train = pd.read_csv("X_train_balanced.csv")
y_train = pd.read_csv("y_train_balanced.csv").squeeze()  
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

def evaluate_model(model_name, model, X_test, y_test, save_pr_curve_as=None):
    y_pred = model.predict(X_test)
    y_pred_probability = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred_probability)

    cm = confusion_matrix(y_test, y_pred)
    true_negative = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_positive = cm[1][1]

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Average Precision (area under PR curve):", round(avg_precision, 4))
    print("\nConfusion Matrix:")
    print(cm)
    print("True Negatives  (correctly said 'not fraud'):", true_negative)
    print("False Positives (innocent customer wrongly flagged):", false_positive)
    print("False Negatives (missed real fraud):", false_negative)
    print("True Positives  (correctly caught fraud):", true_positive)

    if save_pr_curve_as:
        precisions, recalls, _ = precision_recall_curve(y_test, y_pred_probability)
        plt.figure(figsize=(7, 5))
        plt.plot(recalls, precisions, label=f"{model_name} (AP = {avg_precision:.3f})")
        plt.xlabel("Recall (how much fraud we catch)")
        plt.ylabel("Precision (how many flags are real fraud)")
        plt.title(f"Precision-Recall Curve — {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_pr_curve_as, bbox_inches="tight")
        plt.close()

    result = {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_precision": avg_precision,
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
    }
    return result, y_pred_probability


log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)
print("Model training complete.")

log_reg_result, log_reg_probability = evaluate_model(
    "Logistic Regression (baseline)", log_reg_model, X_test, y_test,
    save_pr_curve_as="baseline_precision_recall_curve.png"
)
RESULTS.append(log_reg_result)

print("\nFull classification report (baseline):")
print(classification_report(y_test, log_reg_model.predict(X_test), target_names=["Normal", "Fraud"]))


X_train_rf = X_train.sample(n=40000, random_state=42)
y_train_rf = y_train.loc[X_train_rf.index]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf, y_train_rf)

rf_result, rf_probability = evaluate_model("Random Forest", rf_model, X_test, y_test)
RESULTS.append(rf_result)

xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)

xgb_result, xgb_probability = evaluate_model("XGBoost", xgb_model, X_test, y_test)
RESULTS.append(xgb_result)

rf_precisions, rf_recalls, _ = precision_recall_curve(y_test, rf_probability)
xgb_precisions, xgb_recalls, _ = precision_recall_curve(y_test, xgb_probability)

plt.figure(figsize=(7, 5))
plt.plot(rf_recalls, rf_precisions, label=f"Random Forest (AP = {rf_result['average_precision']:.3f})")
plt.plot(xgb_recalls, xgb_precisions, label=f"XGBoost (AP = {xgb_result['average_precision']:.3f})")
plt.xlabel("Recall (how much fraud we catch)")
plt.ylabel("Precision (how many flags are real fraud)")
plt.title("Precision-Recall Curve — Random Forest vs XGBoost")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("rf_xgb_precision_recall_curve.png", bbox_inches="tight")
plt.close()

results_df = pd.DataFrame(RESULTS)
results_df.to_csv("model_results.csv", index=False)

print("\n FULL MODEL COMPARISON TABLE ")
print(results_df.to_string(index=False))

# FAST FILTER + SECOND OPINION 
print("=" * 60)
print("ENSEMBLE - FAST FILTER + SECOND OPINION")
print("=" * 60)

X_train = pd.read_csv("X_train_balanced.csv")
y_train = pd.read_csv("y_train_balanced.csv").squeeze()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)
print("XGBoost (fast filter) trained.")

X_train_rf = X_train.sample(n=40000, random_state=42)
y_train_rf = y_train.loc[X_train_rf.index]
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf, y_train_rf)
print("Random Forest (second opinion) trained.")

xgb_probability = xgb_model.predict_proba(X_test)[:, 1]

LOWER_THRESHOLD = 0.10
UPPER_THRESHOLD = 0.90
is_confident_normal = xgb_probability < LOWER_THRESHOLD
is_confident_fraud = xgb_probability > UPPER_THRESHOLD
is_unsure = (~is_confident_normal) & (~is_confident_fraud)

print("\nConfident NOT fraud:", is_confident_normal.sum())
print("Confident IS fraud:", is_confident_fraud.sum())
print("Unsure (sent to Random Forest):", is_unsure.sum())

final_prediction_fastfilter = xgb_probability.copy()
final_prediction_fastfilter[is_confident_normal] = 0
final_prediction_fastfilter[is_confident_fraud] = 1
if is_unsure.sum() > 0:
    final_prediction_fastfilter[is_unsure] = rf_model.predict(X_test[is_unsure])
final_prediction_fastfilter = final_prediction_fastfilter.astype(int)

precision = precision_score(y_test, final_prediction_fastfilter)
recall = recall_score(y_test, final_prediction_fastfilter)
f1 = f1_score(y_test, final_prediction_fastfilter)
cm = confusion_matrix(y_test, final_prediction_fastfilter)

print(f"\nFast Filter Ensemble -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print("False Positives:", cm[0][1], "| False Negatives:", cm[1][0])

fastfilter_result = {
    "model": "Ensemble (Fast Filter + Second Opinion)",
    "precision": precision, "recall": recall, "f1_score": f1,
    "average_precision": None,
    "false_positives": int(cm[0][1]), "false_negatives": int(cm[1][0]),
}

print("\n" + "=" * 60)
print("ENSEMBLE - STACKING")
print("=" * 60)

# Build a fresh train,validation,test split
df = pd.read_csv("creditcard_clean.csv")

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
df = df.drop(columns=["Amount", "Time"])

X = df.drop(columns=["Class"])
y = df["Class"]

X_temp, X_test_stack, y_temp, y_test_stack = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train_stack, X_val_stack, y_train_stack, y_val_stack = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print("Stacking train:", X_train_stack.shape, "| val:", X_val_stack.shape, "| test:", X_test_stack.shape)

smote = SMOTE(random_state=42)
X_train_stack_bal, y_train_stack_bal = smote.fit_resample(X_train_stack, y_train_stack)

xgb_stack = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_stack.fit(X_train_stack_bal, y_train_stack_bal)

X_train_rf_stack = X_train_stack_bal.sample(n=40000, random_state=42)
y_train_rf_stack = y_train_stack_bal.loc[X_train_rf_stack.index]
rf_stack = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_stack.fit(X_train_rf_stack, y_train_rf_stack)

print("Base models for stacking trained.")

xgb_val_prob = xgb_stack.predict_proba(X_val_stack)[:, 1]
rf_val_prob = rf_stack.predict_proba(X_val_stack)[:, 1]
meta_features_val = pd.DataFrame({"xgb_probability": xgb_val_prob, "rf_probability": rf_val_prob})

meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_features_val, y_val_stack)

print("Meta-model trained.")
print("Learned weight for XGBoost's opinion:", round(meta_model.coef_[0][0], 4))
print("Learned weight for Random Forest's opinion:", round(meta_model.coef_[0][1], 4))

xgb_test_prob = xgb_stack.predict_proba(X_test_stack)[:, 1]
rf_test_prob = rf_stack.predict_proba(X_test_stack)[:, 1]
meta_features_test = pd.DataFrame({"xgb_probability": xgb_test_prob, "rf_probability": rf_test_prob})

final_prediction_stack = meta_model.predict(meta_features_test)

precision = precision_score(y_test_stack, final_prediction_stack)
recall = recall_score(y_test_stack, final_prediction_stack)
f1 = f1_score(y_test_stack, final_prediction_stack)
cm = confusion_matrix(y_test_stack, final_prediction_stack)

print(f"\nStacking Ensemble -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print("False Positives:", cm[0][1], "| False Negatives:", cm[1][0])

stacking_result = {
    "model": "Ensemble (Stacking)",
    "precision": precision, "recall": recall, "f1_score": f1,
    "average_precision": None,
    "false_positives": int(cm[0][1]), "false_negatives": int(cm[1][0]),
}

X_test_stack.to_csv("stack_X_test.csv", index=False)
y_test_stack.to_csv("stack_y_test.csv", index=False)

import joblib
joblib.dump(xgb_stack, "model_xgb.joblib")
joblib.dump(rf_stack, "model_rf.joblib")
joblib.dump(meta_model, "model_meta.joblib")

existing_results = pd.read_csv("model_results.csv")
new_rows = pd.DataFrame([fastfilter_result, stacking_result])
combined = pd.concat([existing_results, new_rows], ignore_index=True)
combined.to_csv("model_results.csv", index=False)
 
print("\n    UPDATED COMPARISON TABLE (ALL MODELS) ")
print(combined.to_string(index=False))

import shap

xgb_model = joblib.load("model_xgb.joblib")
X_test = pd.read_csv("stack_X_test.csv")
y_test = pd.read_csv("stack_y_test.csv").squeeze()

print("Test data shape:", X_test.shape)

explainer = shap.TreeExplainer(xgb_model)

predicted_probability = xgb_model.predict_proba(X_test)[:, 1]
predicted_class = (predicted_probability >= 0.5).astype(int)

flagged_indices = X_test[predicted_class == 1].index
print("\nNumber of transactions flagged by XGBoost:", len(flagged_indices))

NUMBER_OF_EXAMPLES = 4
example_ids = flagged_indices[:NUMBER_OF_EXAMPLES]

for row_id in example_ids:
    row = X_test.loc[[row_id]]
    shap_values = explainer.shap_values(row)

    contributions = pd.Series(
        shap_values[0], index=X_test.columns
    ).sort_values(key=abs, ascending=False)

    top_5 = contributions.head(5)

    print(f"\n--- Transaction (row {row_id}) ---")
    print("Fraud probability:", round(predicted_probability[row_id], 4))
    print("Actual label:", "Fraud" if y_test.loc[row_id] == 1 else "Normal")
    print("Top 5 features driving this decision:")
    print(top_5)

    plt.figure(figsize=(7, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_5.values]
    plt.barh(top_5.index[::-1], top_5.values[::-1], color=colors[::-1])
    plt.xlabel("Contribution to fraud score (SHAP value)")
    plt.title(f"Why transaction {row_id} was flagged")
    plt.tight_layout()
    plt.savefig(f"shap_explanation_row_{row_id}.png")
    plt.close()

    
sample_for_summary = X_test.sample(n=min(500, len(X_test)), random_state=42)
shap_values_summary = explainer.shap_values(sample_for_summary)

mean_abs_shap = pd.Series(
    abs(shap_values_summary).mean(axis=0), index=X_test.columns
).sort_values(ascending=False)

print("\nTop 10 most important features overall:")
print(mean_abs_shap.head(10))

plt.figure(figsize=(8, 6))
mean_abs_shap.head(10)[::-1].plot(kind="barh")
plt.xlabel("Average impact on fraud score (mean |SHAP value|)")
plt.title("Overall Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_global_importance.png")
plt.close()

