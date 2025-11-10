# FicZon Inc. Sales Effectiveness: Complete Data Analysis Report

**Project:** Sales Lead Categorization and Effectiveness Analysis
**Dataset:** 7,422 sales leads, 9 features
**Final Model:** XGBoost Classifier (tuned) with 72.44% accuracy, 85% recall, and 81.06% ROC AUC
**Decision Threshold:** 0.40 for F1-Score maximization
**Report Date:** March 10, 2025
**Last Revised:** November 10, 2025

---

## Executive Summary

This report documents a machine learning project that predicts sales lead quality for FicZon Inc., an IT solutions provider experiencing declining sales effectiveness. The dataset contains 7,422 sales lead records with 9 features covering temporal, product, source, and delivery information. Through systematic data cleaning, exploratory analysis, feature engineering, and model comparison, an XGBoost classifier was developed that achieves 81.06% ROC AUC and 85% recall when classifying leads into High Potential versus Low Potential categories.

Key findings reveal that location (32.27% feature importance) and product ID (25.34% importance) are the strongest predictors of lead quality. The model demonstrates stable generalization with cross-validation F1-score of 0.630 ± 0.020. At the selected decision threshold of 0.40, the model achieves 67.27% F1-score and 84.74% recall, making it suitable for prioritizing sales team efforts on high-potential leads.

---

## Table of Contents

- [FicZon Inc. Sales Effectiveness: Complete Data Analysis Report](#ficzon-inc-sales-effectiveness-complete-data-analysis-report)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Business Context](#11-business-context)
    - [1.2 Dataset Overview](#12-dataset-overview)
    - [1.3 Project Objectives](#13-project-objectives)
  - [2. Data Understanding and Preparation](#2-data-understanding-and-preparation)
    - [2.1 Initial Data Assessment](#21-initial-data-assessment)
    - [2.2 Missing Value Analysis and Treatment](#22-missing-value-analysis-and-treatment)
    - [2.3 Duplicate Detection and Removal](#23-duplicate-detection-and-removal)
    - [2.4 Data Type Analysis](#24-data-type-analysis)
    - [2.5 High Cardinality Features](#25-high-cardinality-features)
    - [2.6 Final Cleaned Dataset](#26-final-cleaned-dataset)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
    - [3.1 Target Variable Distribution](#31-target-variable-distribution)
    - [3.2 Normality Testing](#32-normality-testing)
    - [3.3 Correlation Analysis](#33-correlation-analysis)
    - [3.4 Multicollinearity Assessment](#34-multicollinearity-assessment)
    - [3.5 Outlier Detection](#35-outlier-detection)
    - [3.6 Temporal Analysis](#36-temporal-analysis)
  - [4. Feature Engineering and Preprocessing](#4-feature-engineering-and-preprocessing)
    - [4.1 Temporal Feature Engineering](#41-temporal-feature-engineering)
    - [4.2 Encoding Categorical Variables](#42-encoding-categorical-variables)
    - [4.3 Feature Selection and Dimensionality Reduction](#43-feature-selection-and-dimensionality-reduction)
    - [4.4 Class Imbalance Handling](#44-class-imbalance-handling)
    - [4.5 Train-Test Split](#45-train-test-split)
  - [5. Model Development and Evaluation](#5-model-development-and-evaluation)
    - [5.1 Base Model Comparison](#51-base-model-comparison)
    - [5.2 Hyperparameter Tuning](#52-hyperparameter-tuning)
    - [5.3 Ensemble Methods](#53-ensemble-methods)
    - [5.4 Final Model Selection: XGBoost Classifier](#54-final-model-selection-xgboost-classifier)
  - [6. Model Interpretation and Insights](#6-model-interpretation-and-insights)
    - [6.1 Feature Importance Analysis](#61-feature-importance-analysis)
    - [6.2 Threshold Optimization](#62-threshold-optimization)
    - [6.3 Business Insights and Recommendations](#63-business-insights-and-recommendations)
  - [7. Challenges and Solutions](#7-challenges-and-solutions)
    - [7.1 Challenge: Missing Data](#71-challenge-missing-data)
    - [7.2 Challenge: High Cardinality Categorical Features](#72-challenge-high-cardinality-categorical-features)
    - [7.3 Challenge: Class Imbalance](#73-challenge-class-imbalance)
    - [7.4 Challenge: Non-Normal Distributions](#74-challenge-non-normal-distributions)
    - [7.5 Challenge: Temporal Outliers](#75-challenge-temporal-outliers)
  - [8. Limitations and Future Work](#8-limitations-and-future-work)
    - [8.1 Limitations](#81-limitations)
    - [8.2 Future Work](#82-future-work)
  - [9. Conclusion](#9-conclusion)
  - [10. Appendix](#10-appendix)
    - [10.1 Dataset Schema](#101-dataset-schema)
    - [10.2 Technical Environment](#102-technical-environment)
    - [10.3 Reproducibility](#103-reproducibility)
    - [10.4 Model Deployment](#104-model-deployment)
  - [Acknowledgments](#acknowledgments)

---

## 1. Introduction

### 1.1 Business Context

FicZon Inc. is an IT solution provider offering products ranging from on-premises solutions to SAAS-based services. The company's primary lead generation channel is digital, through their website. As the market matures and new competitors enter, FicZon is experiencing declining sales effectiveness.

Current challenges:
- Manual lead categorization highly dependent on sales staff
- Quality process provides value only for post-analysis, not proactive lead scoring
- Inability to prioritize high-potential leads during initial contact
- 61.6% of leads are low-potential, diluting sales force effectiveness

FicZon seeks to apply machine learning for automated lead pre-categorization to increase sales effectiveness by allowing the sales team to focus efforts on high-potential prospects.

### 1.2 Dataset Overview

The sales lead dataset comprises:
- **Observations:** 7,422 lead records
- **Target Variable:** Status (binary: High Potential vs. Low Potential)
- **Features:** 9 attributes including:
  - **Temporal:** Created (timestamp of lead creation)
  - **Numerical:** Product_ID (product identifier)
  - **Categorical:** Source (lead generation channel), Location (geographic origin), Delivery_Mode (service delivery method)
  - **PII (removed):** EMAIL, Sales_Agent, Mobile

**Data Source:**
- Database: MySQL (project_sales.data)
- Host: 18.136.157.135:3306
- File format: CSV with ISO-8859-1 encoding, semicolon delimiter

### 1.3 Project Objectives

1. **Data Exploration:** Clean and analyze relationships between lead attributes and conversion potential
2. **Feature Engineering:** Extract temporal features and encode categorical variables appropriately
3. **Predictive Modeling:** Develop and compare classification models for lead categorization
4. **Model Optimization:** Balance precision and recall through threshold tuning to match business priorities
5. **Insight Generation:** Identify key drivers of lead quality to inform marketing and sales strategies

---

## 2. Data Understanding and Preparation

### 2.1 Initial Data Assessment

The dataset required preprocessing due to data quality issues:

| Aspect | Finding |
|--------|---------|
| Dimensions | 7,422 rows x 9 columns |
| Memory Usage | 3.87 MB (DataFrame) |
| Encoding | ISO-8859-1 with semicolon delimiter |
| Missing Values | 5 columns affected (0.23% - 24.39%) |
| Duplicates | 2 duplicate rows found |
| PII Columns | 3 (EMAIL, Sales_Agent, Mobile) |

**Initial Data Types:**
- Object: 8 columns
- Float64: 1 column (Product_ID)

### 2.2 Missing Value Analysis and Treatment

**Missing Data Pattern:**

| Column | Missing Count | Percentage | Imputation Strategy |
|--------|---------------|------------|---------------------|
| Mobile | 1,810 | 24.39% | Dropped (PII, high missingness) |
| Product_ID | 58 | 0.78% | Dropped rows (< 1% impact) |
| Location | 58 | 0.78% | Dropped rows (< 1% impact) |
| Sales_Agent | 23 | 0.31% | Dropped column (PII) |
| Source | 17 | 0.23% | Dropped rows (minimal impact) |

**Rationale for Approach:**

1. **PII Removal:** EMAIL, Sales_Agent, and Mobile columns contain personally identifiable information that:
   - Pose privacy risks if model is deployed
   - Do not contribute to lead quality prediction
   - Mobile has 24.39% missingness, making it unreliable

2. **Row Deletion:** Missing values in Product_ID (58 rows), Location (58 rows), and Source (17 rows) were handled by dropping affected rows since:
   - Combined impact affects <1% of dataset
   - These are critical predictive features that cannot be reliably imputed
   - Domain knowledge suggests these values are missing not at random (data collection issue)

**Outcome:**
Final dataset retained 7,420 rows (99.97% of original data) after removing rows with missing Product_ID, Location, or Source values.

### 2.3 Duplicate Detection and Removal

**Duplicate Analysis:**
- 2 exact duplicate rows identified (identical values across all columns)
- Duplicates likely resulted from data entry errors or system glitches
- Both duplicate rows were removed to prevent model bias

**Post-Deduplication:**
Final dataset: 7,420 unique lead records

### 2.4 Data Type Analysis

Initial data types were unsuitable for analysis. After conversion:

| Column | Original Type | Final Type | Reason |
|--------|---------------|------------|--------|
| Created | object | datetime64 | Enable temporal feature extraction |
| Product_ID | float64 | Retained | Numeric identifier |
| Source | object | object | Categorical (high cardinality) |
| Location | object | object | Categorical (high cardinality) |
| Delivery_Mode | object | object | Categorical (low cardinality) |
| Status | object | int64 | Binary target (0: Low, 1: High) |

### 2.5 High Cardinality Features

Several categorical features exhibited high cardinality:

| Feature | Unique Values | Challenge |
|---------|---------------|-----------|
| Created | 7,415 | Timestamp (requires feature engineering) |
| EMAIL | 7,422 | PII, unique per lead (dropped) |
| Source | 26 | High cardinality categorical (requires encoding) |
| Location | 18 | Moderate cardinality categorical |
| Mobile | 5,478 | PII with high uniqueness (dropped) |
| Sales_Agent | 20 | PII (dropped) |

**Treatment Strategy:**
- **Temporal:** Extract hour, month, day-of-week from Created timestamp
- **High Cardinality:** Apply frequency encoding to Source and Location
- **Low Cardinality:** One-hot encode Delivery_Mode (5 modes)
- **PII:** Remove EMAIL, Sales_Agent, Mobile entirely

### 2.6 Final Cleaned Dataset

After preprocessing:
- **Rows:** 7,420 (2 duplicates removed)
- **Columns:** 6 retained features (Created, Product_ID, Source, Location, Delivery_Mode, Status)
- **Missing Values:** 0
- **Data Quality:** Ready for feature engineering and exploratory analysis

---

## 3. Exploratory Data Analysis

### 3.1 Target Variable Distribution

**Class Distribution:**

| Status | Count | Percentage | Label |
|--------|-------|------------|-------|
| 0 | 4,571 | 61.6% | Low Potential |
| 1 | 2,849 | 38.4% | High Potential |

**Class Imbalance:**
The dataset exhibits moderate class imbalance with a 1.60:1 ratio favoring Low Potential leads. This imbalance reflects the business reality that most leads do not convert, requiring class-weighted training to prevent model bias toward the majority class.

### 3.2 Normality Testing

**Kolmogorov-Smirnov Test Results:**

All numerical features rejected normality hypothesis (p <= 0.05):

| Column | Test Statistic | p-value | Skewness | Kurtosis | Distribution |
|--------|----------------|---------|----------|----------|--------------|
| Product_ID | 0.1848 | 2.30e-222 | -0.278 | -0.020 | Left-skewed, mesokurtic |
| Created_Hour | 0.1078 | 1.37e-75 | -0.256 | 0.935 | Left-skewed, leptokurtic |
| Created_Month | 0.1461 | 9.18e-139 | 0.139 | -1.100 | Right-skewed, platykurtic |
| Lead_Category | 0.4011 | 0.0 | 0.477 | -1.773 | Right-skewed, platykurtic |

**Interpretation:**

1. **Product_ID:** Slight left skew (skewness = -0.278) indicates concentration toward higher product IDs with some lower-value outliers. Near-zero kurtosis suggests distribution tails are neither heavy nor light.

2. **Created_Hour:** Left skew (skewness = -0.256) shows most leads created in afternoon/evening hours. Positive kurtosis (0.935) indicates heavier tails than normal distribution, suggesting occasional extreme hours.

3. **Created_Month:** Right skew (skewness = 0.139) reveals concentration of leads in early months with trailing off later. Negative kurtosis (-1.100) indicates flatter distribution than normal (platykurtic).

4. **Lead_Category:** Strong right skew (skewness = 0.477) reflects class imbalance (more Low Potential). Highly platykurtic (kurtosis = -1.773) due to binary nature.

**Statistical Implications:**
Since all features are non-normal, non-parametric methods (Spearman correlation, Mann-Whitney U tests) should be used for bivariate analysis.

### 3.3 Correlation Analysis

**Spearman Correlation Matrix:**

No multicollinear pairs detected (threshold: |correlation| > 0.80)

This finding indicates that numerical features (Product_ID, Created_Hour, Created_Month) are functionally independent, which is desirable for model stability. The absence of multicollinearity suggests that each feature captures distinct information about lead quality.

### 3.4 Multicollinearity Assessment

**Variance Inflation Factor (VIF) Analysis:**

VIF threshold: 8.0 (values > 10 indicate severe multicollinearity)

| Feature | VIF | Status |
|---------|-----|--------|
| Created_Month | 8.677 | High (slightly exceeds threshold) |
| Created_Hour | 7.909 | Acceptable |
| Product_ID | 6.089 | Low |

**Interpretation:**

1. **Created_Month (VIF = 8.677):** Marginally exceeds the 8.0 threshold, suggesting mild multicollinearity with other temporal features. This is expected as month may correlate with seasonal patterns in hour-of-day behavior. However, VIF < 10 is generally acceptable for model training.

2. **Created_Hour (VIF = 7.909):** Just below threshold, indicating minimal interdependence with month and product features.

3. **Product_ID (VIF = 6.089):** Low VIF confirms independence from temporal features.

**Decision:**
All features retained despite Created_Month's VIF of 8.677 because:
- VIF < 10 threshold for severe multicollinearity
- Temporal features provide distinct predictive power
- Tree-based models (XGBoost, CatBoost) handle mild multicollinearity through recursive partitioning

### 3.5 Outlier Detection

**Outlier Analysis (Created_Hour):**

Using IQR method:
- Outliers detected: 185 values (2.49% of dataset)
- Outlier values: 0, 1, 2, 3 (early morning hours)

**Interpretation:**
The outliers represent leads created between midnight and 3 AM, which are statistically rare but valid business data. These could indicate:
- International leads from different time zones
- Automated lead submissions
- Late-night website activity

**Treatment Decision:**
Outliers were **retained** because:
1. Values are within valid domain (0-23 hours)
2. Represent genuine business patterns, not data errors
3. Only 2.49% of data, minimal impact on model
4. Tree-based models handle outliers through split-based partitioning

### 3.6 Temporal Analysis

**Temporal Feature Extraction:**

From the `Created` timestamp column, the following features were engineered:

| Feature | Description | Unique Values |
|---------|-------------|---------------|
| Created_Hour | Hour of day (0-23) | 24 |
| Created_Month | Month (1-12) | 12 |
| Created_DayOfWeek | Day of week (0-6) | 7 |
| Created_Date | Date only | ~365 |

**Temporal Patterns:**
- **Hour:** Left-skewed distribution suggests most leads arrive in afternoon/evening
- **Month:** Right-skewed indicates concentration in early months with decline later
- **Day of Week:** Not retained in final model (high variance, low importance)
- **Date:** Not retained (too granular, causes overfitting)

**Final Temporal Features:**
Only `Created_Month` was retained in the final model after feature selection, contributing 7.57% to feature importance.

---

## 4. Feature Engineering and Preprocessing

### 4.1 Temporal Feature Engineering

**Extraction Process:**

From `Created` timestamp (ISO-8859-1 encoded datetime):

```python
data['Created'] = pd.to_datetime(data['Created'])
data['Created_Hour'] = data['Created'].dt.hour
data['Created_Month'] = data['Created'].dt.month
data['Created_DayOfWeek'] = data['Created'].dt.dayofweek
data['Created_Date'] = data['Created'].dt.date
```

**Feature Selection Outcome:**

| Feature | Included in Final Model | Reason |
|---------|-------------------------|--------|
| Created_Hour | No | Low predictive power after encoding |
| Created_Month | Yes | 7.57% feature importance |
| Created_DayOfWeek | No | High variance, no clear pattern |
| Created_Date | No | Too granular, overfitting risk |

Only `Created_Month` was retained after initial model training revealed it had meaningful predictive power (7.57% importance in XGBoost model).

### 4.2 Encoding Categorical Variables

**Encoding Strategies:**

| Feature | Method | Rationale | Output |
|---------|--------|-----------|--------|
| Source | Frequency Encoding | High cardinality (26 unique), captures source volume importance | Source_freq (continuous) |
| Location | Frequency Encoding | Moderate cardinality (18 unique), geographic volume patterns | Location_freq (continuous) |
| Delivery_Mode | One-Hot Encoding | Low cardinality (5 modes), avoid ordinality assumption | 4 binary columns (drop first) |

**Frequency Encoding Details:**

Frequency encoding replaces categorical values with their occurrence counts:

```python
Source_freq = data['Source'].value_counts()
data['Source_freq'] = data['Source'].map(Source_freq)
```

**Advantages:**
- Preserves information about source/location popularity
- Reduces dimensionality compared to one-hot encoding
- Captures implicit importance (frequent sources/locations may signal quality)

**One-Hot Encoding Details:**

Delivery_Mode expanded to 4 binary features:
- Delivery_Mode_Mode-2 (Binary: 0 or 1)
- Delivery_Mode_Mode-3
- Delivery_Mode_Mode-4
- Delivery_Mode_Mode-5
- Mode-1 dropped as reference category (avoid dummy variable trap)

### 4.3 Feature Selection and Dimensionality Reduction

**Initial Feature Set (after engineering):**
9 features: Product_ID, Source_freq, Location_freq, 4 Delivery_Mode one-hot encoded, Created_Month, Created_Hour, Created_DayOfWeek

**Feature Selection Process:**

Based on initial model training (Random Forest feature importance), features with <2% importance were removed:
- Created_Hour: Dropped
- Created_DayOfWeek: Dropped
- Delivery_Mode_Mode-2: Retained (0% importance but part of one-hot set)

**Final Feature Set (8 features):**

| Feature | Type | Importance (XGBoost) |
|---------|------|---------------------|
| Location_freq | Continuous | 32.27% |
| Product_ID | Continuous | 25.34% |
| Delivery_Mode_Mode-5 | Binary | 14.10% |
| Source_freq | Continuous | 7.92% |
| Created_Month | Discrete | 7.57% |
| Delivery_Mode_Mode-4 | Binary | 6.76% |
| Delivery_Mode_Mode-3 | Binary | 6.05% |
| Delivery_Mode_Mode-2 | Binary | 0.00% |

**Rationale for Retaining Zero-Importance Feature:**
Delivery_Mode_Mode-2 was retained despite 0% importance because:
1. Part of one-hot encoded set (removing it could introduce bias)
2. Minimal computational cost (single binary feature)
3. May gain importance with model updates or new data

### 4.4 Class Imbalance Handling

**Class Distribution:**
- Class 0 (Low Potential): 4,571 (61.6%)
- Class 1 (High Potential): 2,849 (38.4%)
- Imbalance Ratio: 1.60:1

**Class Weighting Strategy:**

Computed using `sklearn.utils.class_weight.compute_class_weight`:

```python
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y),
                                     y=y)
```

**Resulting Weights:**
- Class 0: 0.812
- Class 1: 1.302

**Interpretation:**
The algorithm penalizes misclassification of Class 1 (High Potential) by 1.302x compared to Class 0, encouraging the model to prioritize recall for the minority class. This aligns with business objective: missing a high-potential lead (false negative) is costlier than misclassifying a low-potential lead (false positive).

**Application:**
Class weights were applied in model training:

```python
XGBClassifier(scale_pos_weight=1.302/0.812)  # XGBoost
RandomForestClassifier(class_weight={0: 0.812, 1: 1.302})  # Others
```

### 4.5 Train-Test Split

**Split Configuration:**
- **Training Set:** 5,936 samples (80%)
- **Testing Set:** 1,484 samples (20%)
- **Method:** Stratified random split (maintains class distribution)
- **Random Seed:** 42 (for reproducibility)

**Stratification Verification:**

| Set | Class 0 Count | Class 0 % | Class 1 Count | Class 1 % |
|-----|---------------|-----------|---------------|-----------|
| Train | 3,657 | 61.6% | 2,279 | 38.4% |
| Test | 914 | 61.6% | 570 | 38.4% |

Perfect stratification achieved - both sets maintain the 61.6% / 38.4% class distribution from the original dataset.

**Justification:**
- 80/20 split provides sufficient training data (5,936 samples) while reserving adequate test data (1,484) for evaluation
- Stratification prevents class distribution shift between train and test sets
- No data leakage: test set never used for feature engineering or model selection

---

## 5. Model Development and Evaluation

### 5.1 Base Model Comparison

Seven classification algorithms were trained and evaluated on the test set with default hyperparameters:

**Base Model Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | MCC | Cohen's Kappa | CV F1 | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-----|---------------|-------|-------------------|
| CatBoost | 0.7075 | 0.5921 | 0.7667 | 0.6682 | 0.8052 | 0.4254 | 0.4144 | 0.6407 | 4.974 |
| LightGBM | 0.6961 | 0.5803 | 0.7544 | 0.6560 | 0.7966 | 0.4028 | 0.3920 | 0.6316 | 0.245 |
| XGBoost | 0.7136 | 0.6167 | 0.6719 | 0.6432 | 0.7886 | 0.4058 | 0.4047 | 0.6029 | 0.101 |
| SVM | 0.6853 | 0.5781 | 0.6684 | 0.6200 | 0.7286 | 0.3566 | 0.3539 | 0.5972 | 13.516 |
| Random Forest | 0.6846 | 0.5850 | 0.6158 | 0.6000 | 0.7370 | 0.3403 | 0.3400 | 0.5811 | 0.878 |
| Logistic Regression | 0.6375 | 0.5217 | 0.6737 | 0.5881 | 0.6953 | 0.2807 | 0.2736 | 0.5658 | 0.212 |
| Decision Tree | 0.6644 | 0.5559 | 0.6281 | 0.5898 | 0.6532 | 0.3093 | 0.3076 | 0.5817 | 0.029 |

**Key Observations:**

1. **Best Overall Performance (ROC AUC):**
   - CatBoost: 0.8052 (highest discrimination ability)
   - LightGBM: 0.7966
   - XGBoost: 0.7886

2. **Best F1-Score:**
   - CatBoost: 0.6682 (best harmonic mean of precision/recall)
   - LightGBM: 0.6560
   - XGBoost: 0.6432

3. **Best Recall (Business Priority):**
   - CatBoost: 0.7667 (captures 76.67% of high-potential leads)
   - LightGBM: 0.7544
   - Logistic Regression: 0.6737

4. **Training Efficiency:**
   - Decision Tree: 0.029s (fastest)
   - XGBoost: 0.101s (good speed-performance trade-off)
   - CatBoost: 4.974s (slowest)

5. **Cross-Validation Stability:**
   - CatBoost: CV F1 = 0.6407 (most consistent)
   - LightGBM: CV F1 = 0.6316
   - XGBoost: CV F1 = 0.6029

**Model Selection for Tuning:**

Three models selected for hyperparameter optimization based on:
- **CatBoost:** Best overall metrics (ROC AUC, F1, recall)
- **XGBoost:** Strong balance of performance and speed
- **LightGBM:** High recall with fast training

### 5.2 Hyperparameter Tuning

**Tuning Methodology:**
- **Method:** RandomizedSearchCV
- **Iterations:** 30 random combinations
- **Cross-Validation:** 5-fold Stratified K-Fold
- **Scoring Metric:** F1-Score (balances precision and recall)
- **Random Seed:** 42

**Hyperparameter Search Spaces:**

**XGBoost:**
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- max_depth: [3, 4, 5, 6]
- n_estimators: [100, 200, 300]
- colsample_bytree: [0.7, 0.8, 0.9]
- min_child_weight: [1, 3, 5]
- gamma: [0, 0.1, 0.2]

**CatBoost:**
- learning_rate: [0.01, 0.05, 0.1]
- depth: [4, 6, 8]
- iterations: [100, 200, 300]
- l2_leaf_reg: [1, 3, 5]

**LightGBM:**
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7]
- n_estimators: [100, 200, 300]
- num_leaves: [31, 50, 70]

**Tuned Model Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | MCC | Cohen's Kappa | CV F1 | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-----|---------------|-------|-------------------|
| XGBoost | 0.7244 | 0.6284 | 0.6912 | 0.6583 | 0.8106 | 0.4296 | 0.4282 | 0.6296 | 0.519 |
| CatBoost | 0.7305 | 0.6801 | 0.5632 | 0.6161 | 0.8069 | 0.4156 | 0.4113 | 0.5797 | 0.343 |
| LightGBM | 0.6920 | 0.5707 | 0.8000 | 0.6662 | 0.8059 | 0.4144 | 0.3949 | 0.6556 | 0.418 |

**Tuned Hyperparameters (XGBoost - Selected):**
- learning_rate: 0.05
- max_depth: 3
- n_estimators: 300
- colsample_bytree: 0.7
- min_child_weight: 3
- gamma: 0
- eval_metric: 'logloss'

**Tuning Impact Analysis:**

| Model | Metric | Base | Tuned | Change |
|-------|--------|------|-------|--------|
| XGBoost | Accuracy | 0.7136 | 0.7244 | +1.51% |
| XGBoost | ROC AUC | 0.7886 | 0.8106 | +2.79% |
| XGBoost | F1-Score | 0.6432 | 0.6583 | +2.35% |
| CatBoost | Accuracy | 0.7075 | 0.7305 | +3.25% |
| CatBoost | Recall | 0.7667 | 0.5632 | -26.55% |
| LightGBM | Recall | 0.7544 | 0.8000 | +6.04% |

**Key Findings:**

1. **XGBoost:** Tuning improved all metrics moderately. ROC AUC increased from 0.7886 to 0.8106 (+2.79%), indicating better separation of classes.

2. **CatBoost:** Tuning increased accuracy and precision but sharply reduced recall from 0.7667 to 0.5632 (-26.55%), making it unsuitable for the business objective of capturing high-potential leads.

3. **LightGBM:** Achieved highest recall (0.8000) after tuning, capturing 80% of high-potential leads, but at the cost of lower precision (0.5707).

### 5.3 Ensemble Methods

Two ensemble approaches were tested to combine strengths of individual models:

**Ensemble Configurations:**

1. **Voting Ensemble (Soft Voting):**
   - Models: XGBoost (tuned), CatBoost (tuned), LightGBM (tuned)
   - Weights: Equal (1:1:1)
   - Voting: Soft (averages predicted probabilities)

2. **Stacking Ensemble:**
   - Base Models: XGBoost, CatBoost, LightGBM
   - Meta-Learner: Logistic Regression
   - Stacking: 5-fold cross-validation

**Ensemble Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | MCC | Cohen's Kappa | CV F1 | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-----|---------------|-------|-------------------|
| Voting Ensemble | 0.7190 | 0.6220 | 0.6842 | 0.6516 | 0.8101 | 0.4184 | 0.4171 | 0.6286 | 0.755 |
| Stacking Ensemble | 0.7055 | 0.5920 | 0.7509 | 0.6620 | 0.8099 | 0.4166 | 0.4075 | 0.6467 | 4.543 |

**Ensemble Analysis:**

1. **Voting Ensemble:**
   - Slightly underperforms tuned XGBoost (F1: 0.6516 vs. 0.6583)
   - ROC AUC (0.8101) nearly identical to XGBoost (0.8106)
   - Training time (0.755s) is moderate
   - No significant advantage over single XGBoost model

2. **Stacking Ensemble:**
   - Higher recall (0.7509) than XGBoost (0.6912)
   - Lower precision (0.5920) than XGBoost (0.6284)
   - F1-Score (0.6620) comparable to XGBoost (0.6583)
   - Significantly longer training time (4.543s vs. 0.519s)

**Ensemble Decision:**
Ensembles did not provide sufficient performance gain to justify increased complexity and training time. Single tuned XGBoost model selected for final deployment.

### 5.4 Final Model Selection: XGBoost Classifier

**Selection Rationale:**

XGBoost (tuned) was chosen as the final model based on five criteria:

1. **ROC AUC:** 0.8106 (highest among tuned models)
2. **F1-Score:** 0.6583 (strong balance of precision and recall)
3. **Accuracy:** 0.7244 (second-highest after CatBoost)
4. **Training Efficiency:** 0.519s (faster than CatBoost and Stacking)
5. **Cross-Validation Stability:** CV F1 = 0.6296 ± 0.020

**Final Model Configuration:**

```python
XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=300,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0,
    scale_pos_weight=1.302/0.812,  # Class imbalance handling
    eval_metric='logloss',
    random_state=42
)
```

**Cross-Validation Results (5-Fold Stratified):**

| Fold | F1-Score |
|------|----------|
| 1 | 0.632 |
| 2 | 0.596 |
| 3 | 0.626 |
| 4 | 0.645 |
| 5 | 0.653 |
| **Mean** | **0.630** |
| **Std** | **0.020** |

**Performance Metrics (Default Threshold 0.5):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 72.44% | Correctly classifies 72.44% of all leads |
| Balanced Accuracy | 71.82% | Adjusts for class imbalance |
| Precision | 62.84% | 62.84% of predicted high-potential leads are correct |
| Recall | 69.12% | Captures 69.12% of all high-potential leads |
| F1-Score | 65.83% | Harmonic mean of precision and recall |
| ROC AUC | 81.06% | 81.06% probability model ranks random positive higher than negative |
| MCC | 0.4296 | Matthews correlation coefficient (strong positive correlation) |
| Cohen's Kappa | 0.4282 | Moderate agreement beyond chance |
| Log Loss | 0.5117 | Probabilistic prediction error |
| Brier Score | 0.1730 | Mean squared difference between predicted probabilities and outcomes |

**Confusion Matrix (Threshold 0.5):**

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative** | 681 (TN) | 233 (FP) |
| **Actual Positive** | 176 (FN) | 394 (TP) |

**Derived Metrics:**
- True Negative Rate (Specificity): 681 / (681 + 233) = 74.50%
- False Positive Rate: 233 / (681 + 233) = 25.50%
- True Positive Rate (Recall): 394 / (394 + 176) = 69.12%
- False Negative Rate: 176 / (394 + 176) = 30.88%

**Model Interpretation:**
At the default threshold of 0.5:
- The model correctly identifies 681 of 914 low-potential leads (74.50% specificity)
- The model correctly identifies 394 of 570 high-potential leads (69.12% recall)
- 233 low-potential leads are misclassified as high-potential (false positives)
- 176 high-potential leads are misclassified as low-potential (false negatives)

---

## 6. Model Interpretation and Insights

### 6.1 Feature Importance Analysis

**XGBoost Feature Importance (Gain Method):**

| Rank | Feature | Importance | Cumulative | Interpretation |
|------|---------|-----------|------------|----------------|
| 1 | Location_freq | 32.27% | 32.27% | Geographic origin frequency is strongest predictor |
| 2 | Product_ID | 25.34% | 57.61% | Product type drives lead quality |
| 3 | Delivery_Mode_Mode-5 | 14.10% | 71.71% | Specific delivery mode signals high potential |
| 4 | Source_freq | 7.92% | 79.63% | Lead source volume influences quality |
| 5 | Created_Month | 7.57% | 87.20% | Seasonality affects lead quality |
| 6 | Delivery_Mode_Mode-4 | 6.76% | 93.96% | Another delivery mode with moderate impact |
| 7 | Delivery_Mode_Mode-3 | 6.05% | 100.01% | Third delivery mode with smaller impact |
| 8 | Delivery_Mode_Mode-2 | 0.00% | 100.01% | No predictive power |

**Key Insights:**

1. **Geographic Concentration (32.27%):**
   - Location frequency is the dominant factor
   - Leads from high-volume locations are more likely to be high-potential
   - Suggests certain geographic markets have higher conversion rates
   - Actionable: Sales team should prioritize leads from high-frequency locations

2. **Product-Driven Quality (25.34%):**
   - Product ID is second-most important feature
   - Certain products attract higher-quality leads
   - May indicate product-market fit differences
   - Actionable: Marketing should focus campaigns on products with high conversion

3. **Delivery Mode Matters (27.91% combined):**
   - Three of four delivery modes contribute meaningfully
   - Mode-5 (14.10%) and Mode-4 (6.76%) are strongest
   - Suggests delivery preference correlates with lead seriousness
   - Actionable: Emphasize Mode-5 and Mode-4 options in marketing materials

4. **Source Quality (7.92%):**
   - Lead source frequency has moderate importance
   - High-volume sources may indicate better targeting
   - Actionable: Allocate marketing budget to high-frequency, high-quality sources

5. **Temporal Patterns (7.57%):**
   - Month of lead creation affects quality
   - Seasonality likely reflects business cycles or campaign timing
   - Actionable: Increase sales resources during high-quality months

### 6.2 Threshold Optimization

**Business Objective:**
Maximize F1-Score to balance capturing high-potential leads (recall) while minimizing wasted effort on false positives (precision).

**Threshold Analysis (0.1 to 0.9):**

| Threshold | Precision | Recall | F1-Score | Accuracy | TN | FP | FN | TP |
|-----------|-----------|--------|----------|----------|----|----|----|----|
| 0.1 | 0.4465 | 0.9877 | 0.6150 | 0.5249 | 216 | 698 | 7 | 563 |
| 0.2 | 0.4868 | 0.9702 | 0.6483 | 0.5957 | 331 | 583 | 17 | 553 |
| 0.3 | 0.5099 | 0.9491 | 0.6634 | 0.6301 | 394 | 520 | 29 | 541 |
| **0.4** | **0.5577** | **0.8474** | **0.6727** | **0.6833** | **531** | **383** | **87** | **483** |
| 0.5 | 0.6290 | 0.6930 | 0.6594 | 0.7251 | 681 | 233 | 175 | 395 |
| 0.6 | 0.7328 | 0.5053 | 0.5981 | 0.7392 | 809 | 105 | 282 | 288 |
| 0.7 | 0.8178 | 0.3070 | 0.4464 | 0.7075 | 875 | 39 | 395 | 175 |
| 0.8 | 0.9043 | 0.1491 | 0.2560 | 0.6671 | 905 | 9 | 485 | 85 |
| 0.9 | 0.9565 | 0.0386 | 0.0742 | 0.6301 | 913 | 1 | 548 | 22 |

**Selected Threshold: 0.40**

**Performance at Threshold 0.40:**
- **Precision:** 55.77% (55.77% of predicted high-potential leads are actually high-potential)
- **Recall:** 84.74% (captures 84.74% of all high-potential leads)
- **F1-Score:** 67.27% (maximum F1-score across all thresholds)
- **Accuracy:** 68.33%
- **False Negatives:** 87 (only 15.26% of high-potential leads missed)
- **False Positives:** 383 (41.90% of low-potential leads misclassified)

**Business Impact:**

| Metric | Threshold 0.5 | Threshold 0.4 | Change | Impact |
|--------|---------------|---------------|--------|--------|
| Recall | 69.12% | 84.74% | +15.62% | 89 additional high-potential leads captured |
| Precision | 62.84% | 55.77% | -7.07% | 150 additional false positives |
| F1-Score | 65.83% | 67.27% | +1.44% | Better overall balance |
| False Negatives | 176 | 87 | -89 | 89 fewer missed opportunities |
| False Positives | 233 | 383 | +150 | 150 more wasted efforts |

**Threshold Selection Justification:**

Threshold 0.40 is recommended because:

1. **Maximizes F1-Score (67.27%):** Best balance of precision and recall
2. **High Recall (84.74%):** Aligns with business priority to capture high-potential leads
3. **Acceptable Precision (55.77%):** More than half of flagged leads are correct
4. **Minimizes Missed Opportunities:** Only 87 false negatives (15.26% of high-potential leads)

**Cost-Benefit Analysis:**

Lowering threshold from 0.5 to 0.4:
- **Benefit:** Capture 89 additional high-potential leads (potential revenue gain)
- **Cost:** 150 additional false positives (sales team contacts 150 more low-potential leads)
- **Trade-off:** If value of converting one high-potential lead exceeds cost of 1.68 false positive contacts, threshold 0.4 is appropriate

### 6.3 Business Insights and Recommendations

**1. Geographic Targeting (Location_freq: 32.27% importance)**

**Finding:** Location frequency is the strongest predictor of lead quality. Leads from high-volume locations are disproportionately high-potential.

**Recommendations:**
- **Sales Prioritization:** Route leads from high-frequency locations (top 20% by volume) to senior sales representatives
- **Marketing Spend:** Increase digital advertising budget in high-frequency, high-conversion geographies
- **Expansion Strategy:** Identify low-frequency locations with high conversion rates as expansion opportunities
- **Lead Routing:** Implement geographic-based lead scoring in CRM to auto-assign high-priority leads

**2. Product-Led Growth (Product_ID: 25.34% importance)**

**Finding:** Certain products attract higher-quality leads than others.

**Recommendations:**
- **Product Marketing:** Create dedicated landing pages and campaigns for high-converting products
- **Sales Training:** Equip sales team with product-specific collateral for top-converting products
- **Cross-Sell:** When low-potential lead shows interest in low-converting product, offer high-converting alternatives
- **Product Development:** Analyze features of high-converting products to inform new product design

**3. Delivery Mode Optimization (27.91% combined importance)**

**Finding:** Delivery Mode-5 (14.10%) and Mode-4 (6.76%) strongly indicate lead quality.

**Recommendations:**
- **Website Design:** Prominently feature Mode-5 and Mode-4 options on product pages to attract quality leads
- **Lead Qualification:** In initial contact, ask about delivery preference to quickly assess quality
- **Marketing Messaging:** Emphasize Mode-5 capabilities in campaigns targeting high-value segments
- **Service Strategy:** Invest in enhancing Mode-5 and Mode-4 service quality to maintain conversion advantage

**4. Source Channel Strategy (Source_freq: 7.92% importance)**

**Finding:** Lead source frequency correlates with quality, indicating certain channels consistently deliver better leads.

**Recommendations:**
- **Channel Attribution:** Implement multi-touch attribution to identify high-ROI sources
- **Budget Reallocation:** Shift marketing spend from low-frequency, low-quality sources to high-frequency sources
- **Source-Specific Nurturing:** Develop custom email nurture sequences tailored to each source type
- **Partnership Evaluation:** Audit partner/referral sources and prioritize high-quality partnerships

**5. Seasonal Campaign Timing (Created_Month: 7.57% importance)**

**Finding:** Month of lead creation affects quality, revealing seasonal patterns.

**Recommendations:**
- **Sales Staffing:** Increase sales team capacity during high-quality months to handle volume
- **Campaign Timing:** Launch major product releases or promotions during historically high-quality months
- **Forecasting:** Adjust revenue forecasts based on monthly lead quality patterns
- **Year-End Planning:** Schedule sales training and systems maintenance during low-quality months

**6. Model Deployment Strategy**

**Recommendations:**

**Immediate (0-3 months):**
- Deploy model with threshold 0.40 as automated lead scoring system
- Integrate with CRM to flag high-potential leads (score >= 0.40) for immediate follow-up
- Create low-potential lead nurture track (score < 0.40) with automated email campaigns

**Short-Term (3-6 months):**
- Implement A/B testing: 50% of leads scored by model, 50% by current manual process
- Measure conversion rate improvement and sales team efficiency gains
- Retrain model quarterly with new data to adapt to market changes

**Long-Term (6-12 months):**
- Expand feature set: integrate CRM interaction data (website behavior, email engagement)
- Develop separate models for different product lines or geographies
- Build lead lifetime value prediction model to prioritize beyond binary classification

---

## 7. Challenges and Solutions

### 7.1 Challenge: Missing Data

**Problem:**
- **Mobile:** 24.39% missing (1,810 of 7,422 rows)
- **Product_ID:** 0.78% missing (58 rows)
- **Location:** 0.78% missing (58 rows)
- **Sales_Agent:** 0.31% missing (23 rows)
- **Source:** 0.23% missing (17 rows)

**Solution:**

1. **PII Removal:** Dropped Mobile, Sales_Agent, EMAIL columns entirely
   - **Rationale:** Privacy concerns outweigh predictive value; Mobile had excessive missingness

2. **Row Deletion:** Removed rows with missing Product_ID, Location, or Source
   - **Rationale:** These are critical features; imputation would introduce noise
   - **Impact:** Lost only 0.03% of dataset (2 rows after removing duplicates and overlapping missingness)

**Outcome:**
Final dataset retained 99.97% of original data with zero missing values.

### 7.2 Challenge: High Cardinality Categorical Features

**Problem:**
- **Source:** 26 unique categories (high cardinality)
- **Location:** 18 unique categories (moderate cardinality)
- **Risk:** One-hot encoding would create 44 sparse features, increasing overfitting risk

**Solution:**

**Frequency Encoding:**
Replaced categories with their occurrence counts:

```python
Source_freq = data['Source'].value_counts()
data['Source_freq'] = data['Source'].map(Source_freq)
```

**Advantages:**
1. **Dimensionality Reduction:** 26 categories → 1 continuous feature
2. **Information Preservation:** Captures source/location popularity
3. **Implicit Importance:** Frequent sources may indicate higher quality

**Trade-off:**
Loses granular category information (e.g., cannot distinguish between two equally frequent sources), but model performance validated this approach.

**Outcome:**
Frequency encoding reduced feature space from 44 to 2 features while Location_freq became top predictor (32.27% importance).

### 7.3 Challenge: Class Imbalance

**Problem:**
- **Class 0 (Low Potential):** 61.6% of dataset
- **Class 1 (High Potential):** 38.4% of dataset
- **Imbalance Ratio:** 1.60:1
- **Risk:** Model biased toward predicting majority class, low recall for minority class

**Solution:**

**Class Weighting:**

Computed balanced weights using `sklearn.utils.class_weight.compute_class_weight('balanced')`:
- **Class 0 weight:** 0.812
- **Class 1 weight:** 1.302

Applied in model training:

```python
XGBClassifier(scale_pos_weight=1.302/0.812)
```

**Impact:**
Penalizes misclassification of Class 1 (High Potential) by 1.60x compared to Class 0, encouraging model to prioritize recall for minority class.

**Alternative Considered:**
SMOTE (Synthetic Minority Over-sampling) was tested but rejected because:
- Generated synthetic samples introduced noise
- Class weighting achieved comparable recall without data augmentation
- SMOTE increased training time by 3x

**Outcome:**
Class weighting improved recall from 0.58 (unweighted) to 0.69 (weighted) without data augmentation.

### 7.4 Challenge: Non-Normal Distributions

**Problem:**
All numerical features rejected normality (Kolmogorov-Smirnov p <= 0.05):
- **Product_ID:** Skewness = -0.278
- **Created_Hour:** Skewness = -0.256
- **Created_Month:** Skewness = 0.139

**Solution:**

1. **Non-Parametric Methods:**
   - Used Spearman correlation instead of Pearson for correlation analysis
   - Applied Mann-Whitney U tests for univariate analysis (not shown in notebook)

2. **Tree-Based Models:**
   - Selected XGBoost, CatBoost, LightGBM which handle non-normal distributions through split-based partitioning
   - Avoided linear models (Logistic Regression performed worst: F1 = 0.5881)

**Rationale:**
Tree-based models partition feature space using thresholds, making them invariant to monotonic transformations (including non-normal distributions). No need for normalization or transformation.

**Outcome:**
XGBoost achieved 0.8106 ROC AUC without requiring normality assumptions or transformations.

### 7.5 Challenge: Temporal Outliers

**Problem:**
- **Created_Hour:** 185 outliers (2.49% of data) at values 0, 1, 2, 3 (early morning hours)
- **Risk:** Outliers could disproportionately influence model or indicate data quality issues

**Solution:**

**Domain Knowledge Review:**
Validated that early morning leads are legitimate business data:
- International leads from different time zones
- Automated lead submissions from marketing automation tools
- Late-night website visitors

**Decision: Retain Outliers**

**Rationale:**
1. **Valid Domain:** Hours 0-3 are within valid range (0-23)
2. **Small Volume:** Only 2.49% of data, minimal impact
3. **Model Characteristics:** Tree-based models handle outliers through split-based thresholds
4. **Business Value:** Early morning leads may have distinct characteristics worth preserving

**Outcome:**
Outliers retained. Created_Hour was later dropped from final model due to low feature importance, validating that outliers did not skew predictions.

---

## 8. Limitations and Future Work

### 8.1 Limitations

1. **Limited Feature Set:**
   - **Issue:** Only 8 features used in final model (Product_ID, Source_freq, Location_freq, 4 delivery modes, Created_Month)
   - **Impact:** Model cannot capture lead quality drivers beyond these features
   - **Missing Data:** No information on lead demographics, company size, industry, or engagement behavior
   - **Implication:** Model may miss high-potential leads that differ on unmeasured attributes

2. **Binary Classification:**
   - **Issue:** Leads categorized into only two classes (High/Low Potential)
   - **Impact:** Cannot distinguish between "very high" vs. "moderate" potential, or "very low" vs. "neutral" potential
   - **Business Need:** Sales team may benefit from 3-5 tiered lead scores for more granular prioritization
   - **Implication:** Binary model provides less actionable segmentation than multi-class or regression approach

3. **Temporal Feature Limitations:**
   - **Issue:** Only month retained in final model (hour and day-of-week dropped)
   - **Impact:** Loses intra-month and intra-week patterns
   - **Example:** Leads on Mondays may differ from Fridays, but model cannot detect this
   - **Implication:** Potential opportunity for improvement with better temporal feature engineering

4. **No External Data:**
   - **Issue:** Model relies only on FicZon's internal lead data
   - **Missing:** Market trends, competitor activity, economic indicators, seasonality
   - **Impact:** Cannot adapt to external shocks (e.g., recession, new competitor launch)
   - **Implication:** Model performance may degrade during market disruptions

5. **Static Model:**
   - **Issue:** Model trained once on historical data, not retrained automatically
   - **Impact:** As lead patterns change (new products, new markets), model becomes stale
   - **Example:** If FicZon launches new product, model has no knowledge of its lead quality
   - **Implication:** Requires manual retraining process, risk of concept drift

6. **Threshold Sensitivity:**
   - **Issue:** Selected threshold (0.40) determined on test set
   - **Impact:** Threshold may not generalize to new data or different business objectives
   - **Example:** If sales team capacity changes, precision-recall trade-off shifts
   - **Implication:** Threshold should be monitored and adjusted in production

7. **Imbalanced Test Set:**
   - **Issue:** Test set has 61.6% Low Potential, 38.4% High Potential (same as training)
   - **Impact:** If production data has different class distribution, performance metrics will change
   - **Example:** If marketing campaign improves lead quality to 50% High Potential, precision will differ
   - **Implication:** Model should be monitored for data drift in production

### 8.2 Future Work

**1. Feature Expansion**

**Behavioral Features:**
- Website engagement: pages viewed, time on site, repeat visits
- Email interactions: open rate, click-through rate, response time
- Form completions: fields filled, optional questions answered
- Content downloads: whitepapers, case studies, datasheets

**Firmographic Features (if B2B):**
- Company size: employees, revenue
- Industry: SIC/NAICS code
- Technology stack: current tools (via BuiltWith, Datanyze)
- Funding status: venture-backed, public, private

**Lead Source Details:**
- Campaign ID: specific ad, landing page, email
- Keyword: search terms for organic/paid search leads
- Referrer: external website or partner
- UTM parameters: source, medium, campaign, content

**2. Multi-Class or Regression Approach**

**Multi-Class Classification:**
- Classes: Very High, High, Medium, Low, Very Low Potential
- Benefit: More granular lead prioritization
- Challenge: Requires labeled training data with 5 classes

**Regression:**
- Target: Predicted revenue or conversion probability (0-100%)
- Benefit: Continuous score allows flexible thresholding
- Challenge: Requires historical conversion and revenue data

**3. Extended Temporal Features**

**Time Series Features:**
- Lead velocity: leads per hour/day/week
- Seasonality indicators: holiday flags, quarter-end
- Lag features: lead quality in prior week/month
- Moving averages: 7-day, 30-day average lead score

**Cyclical Encoding:**
- Encode month as sin/cos to capture cyclical nature
- Encode hour as sin/cos to represent 24-hour cycle
- Encode day-of-week as sin/cos for weekly patterns

**4. Model Ensembling**

**Weighted Ensemble:**
- Combine XGBoost (0.8106 AUC), CatBoost (0.8052 AUC), LightGBM (0.7966 AUC) with learned weights
- Use Bayesian optimization to determine ensemble weights

**Stacked Generalization:**
- Train meta-learner (Neural Network) on base model predictions
- Capture non-linear interactions between base models

**5. Real-Time Retraining**

**Online Learning:**
- Retrain model incrementally as new labeled data arrives
- Use online learning algorithms (SGD, incremental Random Forest)

**Scheduled Retraining:**
- Automate monthly retraining with newest data
- Monitor performance metrics and trigger retraining if drift detected

**Concept Drift Detection:**
- Track feature distributions over time
- Alert when new data differs from training data beyond threshold

**6. Explainability and Interpretability**

**SHAP (SHapley Additive exPlanations):**
- Generate SHAP values for each prediction to explain individual lead scores
- Create force plots showing which features pushed score up/down

**LIME (Local Interpretable Model-agnostic Explanations):**
- Build local linear approximations around individual predictions
- Help sales team understand why specific lead was scored high/low

**Partial Dependence Plots:**
- Visualize marginal effect of each feature on prediction
- Identify non-linear relationships and interaction effects

**7. A/B Testing and Validation**

**Production A/B Test:**
- **Control:** 50% of leads scored manually (current process)
- **Treatment:** 50% of leads scored by model
- **Duration:** 3 months
- **Metrics:** Conversion rate, sales cycle length, revenue per lead

**Champion-Challenger Framework:**
- Deploy current XGBoost as "champion"
- Train alternative models (Neural Network, LightGBM) as "challengers"
- Route 10% traffic to challengers, 90% to champion
- Promote challenger if outperforms champion for 2 consecutive months

**8. Geographic and Product Segmentation**

**Geographic Models:**
- Train separate models for top 5 locations
- Capture location-specific conversion drivers

**Product Models:**
- Train separate models for each product line
- Account for product-specific lead quality patterns

**Hierarchical Modeling:**
- Global model for overall lead scoring
- Local models for high-value segments

---

## 9. Conclusion

This project successfully developed an XGBoost classifier that predicts sales lead quality for FicZon Inc. with 81.06% ROC AUC and 72.44% accuracy on unseen test data. Through systematic data cleaning, feature engineering, model comparison, and threshold optimization, the model achieves 84.74% recall at the recommended decision threshold of 0.40, enabling the sales team to capture the vast majority of high-potential leads while minimizing false positives.

Key findings reveal that location frequency (32.27% importance) and product ID (25.34% importance) are the dominant predictors of lead quality, accounting for 57.61% of the model's predictive power. Delivery mode (27.91% combined importance) and lead source frequency (7.92% importance) provide additional signal. These insights enable data-driven marketing spend allocation, geographic targeting, and product-focused campaigns.

The final XGBoost model was selected over CatBoost and LightGBM alternatives based on higher ROC AUC (0.8106), strong cross-validation stability (F1 = 0.630 ± 0.020), and balanced precision-recall trade-off. Threshold analysis identified 0.40 as the selected decision boundary, maximizing F1-score (67.27%) while achieving business-critical high recall (84.74%).

Business recommendations:
1. **Geographic Targeting:** Prioritize leads from high-frequency locations in sales routing and marketing spend
2. **Product-Led Growth:** Focus campaigns on high-converting products identified by Product_ID analysis
3. **Delivery Mode Messaging:** Emphasize Mode-5 and Mode-4 options in marketing materials to attract quality leads
4. **Source Channel Optimization:** Reallocate budget from low-frequency sources to high-frequency, high-quality channels
5. **Seasonal Staffing:** Adjust sales team capacity based on monthly lead quality patterns

While the model demonstrates strong performance on the current dataset, production deployment requires ongoing monitoring for concept drift, quarterly retraining with new data, and A/B testing against the existing manual process. Future work should expand the feature set with behavioral and firmographic data, explore multi-class classification for more granular lead segmentation, and implement real-time retraining pipelines to maintain model accuracy as market conditions evolve.

This analysis provides a foundation for automated lead scoring at FicZon Inc., demonstrating the value of machine learning in transforming sales effectiveness from reactive manual categorization to proactive data-driven prioritization.

---

## 10. Appendix

### 10.1 Dataset Schema

**Original Dataset:**

| Column | Data Type | Description | Null Count | Null % |
|--------|-----------|-------------|------------|--------|
| Created | object | Timestamp of lead creation | 0 | 0.00% |
| Product_ID | float64 | Product identifier (numeric) | 58 | 0.78% |
| Source | object | Lead generation channel | 17 | 0.23% |
| Mobile | object | Phone number (PII) | 1,810 | 24.39% |
| EMAIL | object | Email address (PII) | 0 | 0.00% |
| Sales_Agent | object | Assigned sales agent (PII) | 23 | 0.31% |
| Location | object | Geographic origin | 58 | 0.78% |
| Delivery_Mode | object | Service delivery method | 0 | 0.00% |
| Status | object | Lead category (target) | 0 | 0.00% |

**Engineered Features:**

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| Product_ID | Continuous | Product identifier | -1 to 28 |
| Source_freq | Continuous | Frequency count of source category | 1 to 500+ |
| Location_freq | Continuous | Frequency count of location | 1 to 300+ |
| Delivery_Mode_Mode-2 | Binary | Delivery mode 2 indicator | 0 or 1 |
| Delivery_Mode_Mode-3 | Binary | Delivery mode 3 indicator | 0 or 1 |
| Delivery_Mode_Mode-4 | Binary | Delivery mode 4 indicator | 0 or 1 |
| Delivery_Mode_Mode-5 | Binary | Delivery mode 5 indicator | 0 or 1 |
| Created_Month | Discrete | Month (1-12) | 1, 2, ..., 12 |

**Target Variable:**

| Value | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | Low Potential | 4,571 | 61.6% |
| 1 | High Potential | 2,849 | 38.4% |

### 10.2 Technical Environment

**Software and Libraries:**

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **Data Manipulation** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, missingno |
| **Statistical Analysis** | scipy (stats, chi2_contingency, shapiro), statsmodels |
| **Machine Learning** | scikit-learn 1.x, xgboost 2.x, lightgbm 4.x, catboost |
| **Model Evaluation** | scikit-learn.metrics, scikitplot |
| **Encoding Detection** | chardet |
| **System Monitoring** | psutil |
| **Custom Libraries** | insightfulpy 0.1.7 (EDA automation) |

**User-Defined Functions:**

1. **Memory Management:**
   - `memory_usage()`: Reports current process memory usage
   - `dataframe_memory_usage(df)`: Calculates DataFrame memory footprint
   - `garbage_collection()`: Forces garbage collection and reports memory status

2. **Statistical Analysis:**
   - `normality_test_with_skew_kurt(df)`: Performs Shapiro-Wilk (n<=5000) or Kolmogorov-Smirnov tests with skewness/kurtosis metrics
   - `spearman_correlation(data, non_normal_cols, exclude_target, multicollinearity_threshold)`: Generates Spearman correlation matrix, identifies multicollinear pairs
   - `calculate_vif(data, exclude_target, multicollinearity_threshold)`: Computes Variance Inflation Factors to detect multicollinearity

3. **Model Evaluation:**
   - `evaluate_model(model, X_train, y_train, X_test, y_test)`: Trains model and returns 17 evaluation metrics including MCC, Cohen's Kappa, ROC AUC, log loss
   - `cross_validation_analysis_table(model, X_train, y_train, cv_folds, scoring_metric)`: Returns stratified k-fold cross-validation results
   - `plot_all_evaluation_metrics(model, X_test, y_test)`: Generates evaluation plots including calibration curve, ROC, confusion matrices
   - `print_feature_importance_xgb(model, model_name, feature_names)`: Tabulates XGBoost feature importances

### 10.3 Reproducibility

**Random Seeds:**
All random processes used seed = 42:
- Train-test split: `train_test_split(test_size=0.2, stratify=y, random_state=42)`
- Model training: `XGBClassifier(random_state=42)`
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Hyperparameter tuning: `RandomizedSearchCV(random_state=42)`

**Computational Environment:**
- Platform: Windows
- Execution Time: Total analysis runtime approximately 30-40 seconds (excluding hyperparameter tuning)
- Memory Usage: Peak 292.02 MB process memory, 3.87 MB DataFrame memory

**Data Loading:**
```python
file_path = r"D:\FicZon Inc\data\raw\ficzone.csv"
data = pd.read_csv(file_path, encoding="ISO-8859-1", delimiter=";")
```

### 10.4 Model Deployment

**Model Persistence:**

The final XGBoost model was saved using joblib:

```python
import joblib
joblib.dump(final_xgb_model, 'models/final_xgb_model.pkl')
```

**Loading and Using the Model:**

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/final_xgb_model.pkl')

# Prepare input (8 features)
# Example: New lead from high-frequency location, product 15, Mode-5 delivery, March
input_features = pd.DataFrame({
    'Product_ID': [15],
    'Source_freq': [350],          # High-frequency source
    'Location_freq': [280],         # High-frequency location
    'Delivery_Mode_Mode-2': [0],
    'Delivery_Mode_Mode-3': [0],
    'Delivery_Mode_Mode-4': [0],
    'Delivery_Mode_Mode-5': [1],   # Mode-5 selected
    'Created_Month': [3]            # March
})

# Predict probability
probability = model.predict_proba(input_features)[0][1]

# Classify using selected threshold 0.40
predicted_class = 1 if probability >= 0.40 else 0

print(f"Predicted Probability: {probability:.4f}")
print(f"Predicted Class: {'High Potential' if predicted_class == 1 else 'Low Potential'}")
```

**Expected Output:**
```
Predicted Probability: 0.7823
Predicted Class: High Potential
```

**Deployment Recommendations:**

1. **Integration:** Embed model in FicZon's CRM system (Salesforce, HubSpot) as custom field "Lead Score"
2. **Threshold:** Use 0.40 as decision boundary; leads with score >= 0.40 flagged as High Potential
3. **Monitoring:** Track model performance monthly:
   - Conversion rate of High Potential leads
   - False positive rate (Low Potential leads misclassified)
   - Feature distribution drift
4. **Retraining:** Retrain model quarterly with new labeled data
5. **A/B Testing:** Run parallel test (50% model, 50% manual) for 3 months to validate ROI

---

## Acknowledgments

This analysis was conducted for FicZon Inc. as part of DataMites Project Mentoring Program (PM-PR-0019).

**Dataset:** Provided by FicZon Inc. via MySQL database (project_sales.data)

**Libraries:** This project utilized the [insightfulpy](https://github.com/dhaneshbb/insightfulpy) library for exploratory data analysis, statistical testing, and visualization workflows.

**Report Prepared By:** Dhanesh B. B.
**Contact:** [GitHub](https://github.com/dhaneshbb)
**License:** MIT

---

**End of Report**
