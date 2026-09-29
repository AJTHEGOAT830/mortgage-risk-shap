# Explainable AI (XAI) Credit Risk & Bankruptcy Scoring Engine

An end-to-end Machine Learning pipeline utilizing **LightGBM** and **SHAP (Shapley Additive exPlanations)** to predict high-risk financial defaults while providing complete decision transparency. 

In highly regulated environments like banking and mortgage lending, black-box AI models are unusable due to compliance frameworks (such as the EU AI Act and UK credit risk guidelines). This project demonstrates how to achieve state-of-the-art predictive accuracy while generating auditable, clear decision paths for risk officers and underwriters.

## Key Features
* **Real-World Tabular Classification:** Uses the UCI Taiwanese Bankruptcy dataset (6,819 companies, 95 financial metrics) to predict risk.
* **Imbalanced Data Optimization:** Implements customized objective weights (`scale_pos_weight`) to accurately capture rare but critical default events.
* **Global Interpretability:** Maps out the macro-level financial drivers that the model uses to determine creditworthiness across the entire portfolio.
* **Local Automated Audits:** Generates a granular, localized "decision trail" graph for individual applicants to explain precisely why a credit request was flagged or denied.

---

## Model & Explainability Visualizations

### 1. Global Risk Drivers
The model automatically ranks financial indicators by their structural impact on credit decisions. Look at your generated `global_credit_risk_drivers.png` file to see the top macroscopic risk patterns.

### 2. Individual Application Audit Trail
For any specific borrower flagged as "High Risk," the engine generates a localized waterfall plot showing the exact micro-contributions of their specific financial indicators (e.g., Debt Ratio, Net Income to Total Assets). Look at your generated `individual_audit_explanation.png` to inspect this audit path.

---

##  Tech Stack & Architecture
* **Modeling Engine:** LightGBM (Gradient Boosted Decision Trees)
* **Explainability Layer:** SHAP (TreeExplainer)
* **Data Processing:** Pandas, NumPy
* **Evaluation Framework:** Scikit-Learn (ROC-AUC focused optimization)
* **Data Source:** UCI Machine Learning Repository

---

## Quick Start Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/mortgage-risk-shap.git](https://github.com/YOUR_GITHUB_USERNAME/mortgage-risk-shap.git)
   cd mortgage-risk-shap
