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
