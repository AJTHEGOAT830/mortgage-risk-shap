import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import shap
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

print("Step 1: Downloading real-world credit risk dataset from UCI Repository...")
# Fetch dataset (Taiwanese Bankruptcy Prediction Data)
# 1 = Financial Distress/Risk, 0 = Financially Stable
bankruptcy_dataset = fetch_ucirepo(id=572)

# Extract features and targets into Pandas DataFrames
X = bankruptcy_dataset.data.features
y = bankruptcy_dataset.data.targets.iloc[:, 0]  # Convert to a 1D Series

# Clean up column names (remove leading/trailing spaces for LightGBM compliance)
X.columns = X.columns.str.strip().str.replace(' ', '_').str.replace(':', '')

print(f"Dataset successfully loaded. Shape: {X.shape[0]} applicants, {X.shape[1]} financial metrics.")

# Step 2: Split into Train and Test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nStep 3: Training Industry-Standard LightGBM Credit Scoring Model...")
# Scale weight for imbalanced data (more stable companies than bankrupt ones)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Evaluate model performance
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
print("\n--- Model Performance Evaluation ---")
print(f"Model Accuracy: {accuracy_score(y_test, preds):.2%}")
print(f"ROC-AUC Credit Risk Score: {roc_auc_score(y_test, probs):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, preds, target_names=["Financially Stable", "High Risk / Default"]))

# =========================================================
# THE AI FLEX: SYSTEM AUDIT & INDIVIDUAL SHAP EXPLANATIONS
# =========================================================
print("\nStep 4: Computing SHAP values (Explainable AI Layer)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 1. Global Feature Importance (Saves a Summary Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, max_display=10, show=False)
plt.title("Top 10 Global Drivers of Credit & Bankruptcy Risk", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('global_credit_risk_drivers.png')
print("-> Global risk summary saved as 'global_credit_risk_drivers.png'")

# 2. Local Audit (Isolate a specific applicant flagged as "High Risk")
high_risk_applicants = X_test[y_test == 1]
sample_idx = high_risk_applicants.index[0]
sample_applicant_data = X_test.loc[[sample_idx]]

print(f"\n--- Auditing High-Risk Application Case ID: {sample_idx} ---")
# Pick out a few critical indicators to print cleanly to terminal
key_indicators = ['Debt_ratio_%', 'Net_Value_Growth_Rate', 'Borrowing_dependency', 'Net_Income_to_Total_Assets']
print(sample_applicant_data[key_indicators].to_string(index=False))

# Generate the personalized waterfall explanation graph for this specific case
applicant_shap = explainer(sample_applicant_data)

plt.figure(figsize=(11, 5))
shap.plots.waterfall(applicant_shap[0], max_display=10, show=False)
plt.title(f"Credit Audit Decision Trail (Case ID: {sample_idx})", fontsize=14, pad=25)
plt.tight_layout()
plt.savefig('individual_audit_explanation.png')
print("\n-> Individual waterfall explanation saved as 'individual_audit_explanation.png'")
print("\nExecution complete! Your project files and visuals are ready for GitHub.")