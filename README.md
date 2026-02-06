# Predictive Modeling for Healthcare Facility Analysis

Advanced machine learning system achieving **88.6% ROC-AUC** for predicting healthcare facility audit risks using comprehensive feature engineering, ensemble methods, and state-of-the-art gradient boosting algorithms.

## Project Overview

This project analyzes California hospital financial data to predict which facilities are at highest risk for audit flags. By leveraging 228 financial variables, 22 engineered features, and advanced ML techniques, the system identifies compliance risks before audits occur, enabling proactive interventions.

## Key Results

| Model | Test ROC-AUC | Test F1 | CV ROC-AUC | Improvement vs Baseline |
|-------|--------------|---------|------------|-------------------------|
| **Gradient Boosting** | **0.886** | **0.873** | **0.944 ± 0.014** | **+32%** |
| LightGBM | 0.882 | 0.868 | 0.944 ± 0.016 | +32% |
| Stacking Ensemble | 0.881 | 0.877 | 0.945 ± 0.020 | +32% |
| XGBoost | 0.871 | 0.885 | 0.941 ± 0.018 | +30% |
| CatBoost | 0.867 | 0.864 | 0.946 ± 0.019 | +30% |
| Random Forest | 0.865 | 0.859 | 0.919 ± 0.017 | +29% |

*Baseline ROC-AUC: 0.668 (Logistic Regression with minimal features)*

## Features

### Advanced Data Processing
- **Long-to-Wide Transformation**: Converts 228 financial variables from long format to wide format
- **Missing Value Handling**: Median imputation for numeric features
- **Categorical Encoding**: One-hot encoding (low cardinality) + Target encoding (high cardinality with smoothing)
- **Feature Scaling**: StandardScaler normalization

### Comprehensive Feature Engineering (22 new features)

#### Financial Health Indicators
- **Deduction Rate**: `1 - (NET_PT_REV / GR_PT_REV)` - Revenue write-off proportion
- **Profit Margin**: `NET_INCOME / NET_PT_REV` - Overall profitability
- **Operating Margin**: `NET_FRM_OP / NET_PT_REV` - Core operational profitability
- **Cost-to-Revenue Ratio**: `TOT_OP_EXP / NET_PT_REV` - Expense efficiency

#### Utilization Metrics
- **Bed Utilization**: `BED_AVL / BED_LIC` - Capacity usage
- **Occupancy Rate**: `DAY_TOT / (BED_LIC × 365)` - Annual bed occupancy
- **Average Length of Stay**: `DAY_TOT / DIS_TOT` - Patient stay duration

#### Financial Efficiency
- **Revenue per Discharge**: `NET_PT_REV / DIS_TOT` - Revenue generation per patient
- **Cost per Discharge**: `TOT_OP_EXP / DIS_TOT` - Cost per patient
- **Cost per Patient Day**: `TOT_OP_EXP / DAY_TOT` - Daily care cost

#### Payer Mix Analysis
- **Medicare Total & Percentage**: Combined Medicare exposure and revenue share
- **Medicaid Total & Percentage**: Combined Medicaid exposure and revenue share

#### Liquidity & Solvency
- **Current Ratio**: `CUR_ASST / CUR_LIAB` - Short-term liquidity
- **Cash Ratio**: `CASH / CUR_LIAB` - Immediate liquidity
- **Debt-to-Equity**: `NET_LTDEBT / EQUITY` - Financial leverage
- **Debt Ratio**: `NET_LTDEBT / TOT_ASST` - Asset financing structure

#### Operational Metrics
- **Asset Turnover**: `NET_PT_REV / TOT_ASST` - Asset efficiency
- **FTE per Bed**: `HOSP_FTE / BED_LIC` - Staffing intensity
- **Labor Cost Percentage**: `EXP_SAL / TOT_OP_EXP` - Labor expense share
- **ER Visit Percentage**: `VIS_ER / VIS_TOT` - Emergency service dependence

### Rigorous Feature Selection
1. **Low Variance Removal**: Eliminates features with variance < 0.01 (7 features removed)
2. **Correlation Filtering**: Removes highly correlated features >0.95 (72 features removed)
3. **Mutual Information**: Selects top 100 most predictive features
4. **Final Feature Set**: 100 critical features from original 268

### Advanced Models
- **LightGBM**: Fast gradient boosting with categorical support
- **CatBoost**: Ordered boosting preventing prediction shift
- **XGBoost**: Regularized gradient boosting with tree pruning
- **Gradient Boosting**: Scikit-learn implementation (best performer)
- **Random Forest**: Bagging ensemble for stability
- **Stacking Ensemble**: Meta-learner combining all 5 base models

### Class Imbalance Handling
- **SMOTE**: Synthetic Minority Oversampling (121→256 samples)
- **Class Weights**: Balanced weighting in model configurations
- Original distribution: 67.9% audit-included, 32.1% audit-excluded

### Training the Model

The notebook executes:
1. Data loading and exploration (472 facilities, 228 variables)
2. Long-to-wide format transformation
3. Feature engineering (22 derived features)
4. Categorical encoding (one-hot + target encoding)
5. Feature selection (268 → 100 features)
6. SMOTE resampling for class balance
7. Training 6 advanced models with 5-fold CV
8. Comprehensive evaluation and visualization
9. Model persistence for deployment

## Dataset

**Source**: California Annual Hospital Financial Data  
**URL**: https://catalog.data.gov/dataset/hospital-annual-financial-data-selected-data-pivot-tables-92074

**Structure**:
- **Facilities**: 472 California hospitals
- **Records**: 578,350 (long format - multiple rows per facility)
- **Variables**: 228 financial and operational metrics
- **Features After Processing**: 268 (228 financial + 22 engineered + 18 categorical dummies)
- **Final Selected**: 100 features (via mutual information)

**Key Variables**:
- **Revenue**: Gross patient revenue, net revenue, deductions, capitation
- **Expenses**: Operating expenses by category (salaries, supplies, etc.)
- **Utilization**: Patient days, discharges, visits by payer type
- **Balance Sheet**: Assets, liabilities, equity, cash, debt
- **Labor**: FTEs, productive hours, staffing mix
- **Payer Mix**: Medicare, Medicaid, managed care breakdowns

## Methodology

### 1. Data Transformation
```
Original Format (Long):
FAC_NO | Variable        | Amount
123    | GR_PT_REV       | 5000000
123    | NET_PT_REV      | 3000000
123    | TOT_OP_EXP      | 2800000

Transformed Format (Wide):
FAC_NO | FIN_GR_PT_REV | FIN_NET_PT_REV | FIN_TOT_OP_EXP
123    | 5000000        | 3000000         | 2800000
```

### 2. Feature Engineering Pipeline
```python
# Financial health
Profit_Margin = NET_INCOME / NET_PT_REV
Operating_Margin = NET_FRM_OP / NET_PT_REV
Deduction_Rate = 1 - (NET_PT_REV / GR_PT_REV)

# Utilization
Occupancy_Rate = DAY_TOT / (BED_LIC × 365)
Bed_Utilization = BED_AVL / BED_LIC

# Efficiency
Revenue_per_Discharge = NET_PT_REV / DIS_TOT
Cost_per_Patient_Day = TOT_OP_EXP / DAY_TOT

# Payer mix
Medicare_Pct = Medicare_Total / NET_PT_REV
Medicaid_Pct = Medicaid_Total / NET_PT_REV

# Liquidity
Current_Ratio = CUR_ASST / CUR_LIAB
Debt_to_Equity = NET_LTDEBT / EQUITY
```

### 3. Categorical Encoding Strategy

**Low Cardinality (≤10 unique) → One-Hot Encoding:**
- TYPE_CNTRL (5 types): District, Non-Profit, Investor, State, Government
- TYPE_CARE (4 types): General, Children's, Psychiatric, Specialty
- TYPE_HOSP (6 types): Comparable, Kaiser, State, PHF, LTC Emphasis, Other

**High Cardinality (>10 unique) → Target Encoding with Smoothing:**
- COUNTY (58 counties): Encoded as mean(AUDIT_IND | COUNTY) with smoothing
- CITY (200+ cities): Encoded as mean(AUDIT_IND | CITY) with smoothing

Target encoding prevents sparse high-dimensional data while capturing geographic patterns.

### 4. Feature Selection Process

```
Initial: 268 features (228 financial + 22 engineered + 18 categorical)
    ↓
Remove Low Variance (<0.01): 261 features (-7)
    ↓
Remove High Correlation (>0.95): 189 features (-72)
    ↓
Select Top 100 by Mutual Information: 100 features
    ↓
Final Feature Set
```

### 5. Model Architecture

```
Training Data (377 facilities)
         ↓
   Feature Engineering (22 features)
         ↓
   Categorical Encoding (one-hot + target)
         ↓
   Feature Selection (100 features)
         ↓
   StandardScaler Normalization
         ↓
   SMOTE Resampling (256:256)
         ↓
    ┌────┴────┬────┬────┬────┐
    ↓         ↓    ↓    ↓    ↓
LightGBM  CatBoost XGB  RF   GB
    ↓         ↓    ↓    ↓    ↓
    └────┬────┴────┴────┴────┘
         ↓
  Stacking Meta-Learner
  (Logistic Regression)
         ↓
   Final Predictions
```

## Key Insights

### Top Predictive Features (by Mutual Information)
1. **FIN_PRD_HR_CLR** (0.151) - Clerical productive hours
2. **FIN_EQUIPMENT** (0.148) - Equipment value
3. **FIN_EXP_OTHPRO** (0.145) - Other professional fees
4. **FIN_GR_PT_REV** (0.145) - Gross patient revenue
5. **Asset_Turnover** (0.145) - Revenue generation efficiency
6. **FIN_GR_OP_OTH** (0.140) - Gross outpatient other revenue
7. **FIN_EXP_BEN** (0.136) - Employee benefits
8. **FIN_INC_INVEST** (0.136) - Investment income
9. **Current_Ratio** (0.128) - Short-term liquidity
10. **Labor_Cost_Pct** (0.127) - Labor expense percentage

### Model Performance Analysis

**Cross-Validation Stability**:
- Gradient Boosting: 0.944 ± 0.014 (lowest variance)
- LightGBM: 0.944 ± 0.016
- Stacking: 0.945 ± 0.020
- CatBoost: 0.946 ± 0.019

**Overfitting Control**:
- All models achieve perfect 1.000 training ROC-AUC
- Test ROC-AUC: 0.865-0.886 (controlled via regularization)
- Overfitting gap: 0.114-0.133 (acceptable for ensemble methods)

**Business Metrics**:
- Precision: 0.859-0.887 (low false positives)
- Recall: 0.844-0.906 (catches most high-risk facilities)
- F1: 0.859-0.885 (balanced performance)

## Improvements Over Baseline

| Aspect | Baseline | Advanced System | Improvement |
|--------|----------|-----------------|-------------|
| Features Used | 10 basic | 100 selected from 250+ | 10x more informative |
| Feature Engineering | Bed utilization only | 22 financial/operational metrics | Domain expertise |
| Categorical Handling | Ignored | One-hot + Target encoding | Proper encoding |
| Models | Logistic Regression | 6 advanced + stacking | State-of-the-art |
| Class Imbalance | Ignored | SMOTE + class weights | Balanced learning |
| Feature Selection | Manual | MI + variance + correlation | Data-driven |
| Test ROC-AUC | 0.668 | 0.886 | **+32% improvement** |
| Test F1 | 0.XX | 0.873 | Balanced precision/recall |

## Limitations

1. **Geographic Scope**: Model trained on California facilities only; generalization to other states uncertain
2. **Temporal Stability**: Regulatory changes may alter audit criteria over time
3. **Class Imbalance**: Original 68:32 ratio may not reflect true audit rates
4. **Feature Dependencies**: Some engineered features highly correlated with base financial metrics
5. **Missing External Factors**: No patient satisfaction, quality ratings, or staffing ratios

## Future Enhancements

### Short-term (1-3 months)
- Hyperparameter optimization via Bayesian search
- SHAP analysis for all models
- Threshold optimization for cost-sensitive decisions
- Feature interaction exploration

### Medium-term (3-6 months)
- Multi-state dataset expansion
- Temporal features (3-year trends)
- External data (CMS Star Ratings, patient outcomes)
- Interactive dashboard (Streamlit/Plotly Dash)

### Long-term (6-12 months)
- Deep learning with entity embeddings
- Causal inference for intervention recommendations
- Real-time prediction API
- Automated retraining pipeline with MLflow
- Fairness audits across demographic groups
