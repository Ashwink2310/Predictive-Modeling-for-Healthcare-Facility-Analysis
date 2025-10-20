# Predictive Modeling for Healthcare Facility Analysis

## TLDR
Developed a **machine learning model to predict audit risks in hospitals** using California’s Annual Hospital Financial Data.  
By combining **Logistic Regression, Random Forest, and XGBoost** into a **Voting Classifier**, the model achieved **ROC-AUC 0.871**, identifying facilities with high audit probability and key risk factors such as bed utilization and managed care practices.  
This project demonstrates how **predictive analytics can enhance compliance, financial stability, and resource allocation in healthcare systems**.

---

## Objective
To build a **data-driven audit risk prediction framework** that helps healthcare administrators:
- Anticipate which facilities are most likely to face audits  
- Identify **key risk drivers** (e.g., utilization rates, managed care contracts)  
- Improve compliance and operational efficiency through proactive intervention  

---

## Dataset
**Source:** [California Annual Hospital Financial Data — Data.gov](https://catalog.data.gov/dataset/hospital-annual-financial-data-selected-data-pivot-tables-92074)  

| Category | Example Variables |
|-----------|------------------|
| Facility Info | FAC_NO, FAC_NAME, CITY, COUNTY |
| Operational | BED_AVL, BED_LIC, TYPE_CARE |
| Financial | Capitation Premium Revenue, Managed Care Contract Utilization |
| Target | AUDIT_IND (Audit Flag Indicator) |

- 400+ healthcare facilities across California  
- Multi-dimensional data with geographic, operational, and financial metrics  
- Missing values imputed, categorical variables consolidated, continuous variables normalized  

---

## Methodology

### **1. Preprocessing**
- Imputed missing data in key columns (e.g., `BED_AVL`, `BED_LIC`)  
- Engineered **Bed Utilization Rate** = `BED_AVL / BED_LIC`  
- Encoded and standardized categorical variables (`TYPE_CARE`)  
- Applied stratified train-test split to preserve class balance  

### **2. Algorithms Implemented**
| Model | Description | ROC-AUC | Log Loss |
|--------|--------------|----------|-----------|
| Logistic Regression | Baseline, interpretable benchmark | 0.668 | 0.626 |
| Random Forest | Non-linear model capturing feature interactions | 0.802 | 0.505 |
| XGBoost | High-regularization gradient boosting model | 0.840 | 0.477 |
| **Voting Classifier (Final)** | Ensemble (soft voting of all three) | **0.871** | **0.483** |

---

## Key Insights
- Most hospitals have an **audit probability between 0.1–0.4**, with the mode near 0.2  
- **Porterville State Hospital** and **Kaiser Santa Rosa** showed the highest audit probabilities (>85%)  
- **Stanford University Hospital** had the lowest risk (~4%)  
- **Managed Care Contract Utilization** and **Licensed Beds by Type of Care** were top predictors of audit likelihood  
- High-risk facilities are concentrated in the **San Francisco Bay Area**, reflecting regional healthcare density and managed care complexity  

---

## Evaluation Metrics
- **ROC-AUC:** Captures model discrimination performance  
- **Log Loss:** Penalizes overconfident misclassifications  
- Regularization and cross-validation were used to prevent overfitting and improve generalization  

---

## Discussion
The ensemble model significantly improved predictive accuracy compared to individual algorithms, demonstrating the strength of **bias-variance balancing through soft voting**.  
Findings also revealed regional concentration of high audit risk, with patterns strongly influenced by managed care intensity and resource utilization.

This model allows hospitals to:
- **Proactively mitigate compliance risks**
- **Prioritize internal audits or operational reviews**
- **Enhance decision-making in regulatory and financial planning**

---

## Future Work
- Expand dataset beyond California for national generalizability  
- Integrate **real-time operational data** for continuous audit risk monitoring  
- Explore **SHAP-based model interpretation** to visualize feature impacts  
- Deploy model in a **dashboard for hospital administrators**  

---


## 📁 Repository Structure
