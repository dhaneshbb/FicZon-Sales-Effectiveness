# FicZon Inc. Sales Effectiveness: Exploratory Data Analysis Report

**Project:** Sales Lead Quality Prediction
**Dataset:** 7,420 sales leads
**Analysis Period:** April to November
**Report Date:** March 10, 2025
**Last Revised:** November 10, 2025

---

## Executive Summary

This exploratory data analysis examines 7,420 sales leads from FicZon Inc. to understand patterns in lead quality and identify key predictors for automated lead categorization. The analysis reveals strong class imbalance (61.6% Low Potential, 38.4% High Potential), high cardinality in categorical features (26 sources, 18 locations), and significant missing data in the Mobile field (24.39%).

Key findings show that geographic location, lead source, and delivery mode exhibit statistically significant relationships with lead quality (all p < 0.001). Bangalore contributes 44.5% of all high-potential leads despite representing only 28% of total volume. Customer referrals and existing client channels demonstrate conversion rates above 89%, while website sources show poor performance (1.5% high potential). All numerical features reject normality, requiring non-parametric statistical approaches.

---

## Table of Contents

- [1. Data Collection and Structure](#1-data-collection-and-structure)
  - [1.1 Dataset Dimensions](#11-dataset-dimensions)
  - [1.2 Feature Schema](#12-feature-schema)
  - [1.3 Engineered Temporal Features](#13-engineered-temporal-features)
- [2. Data Quality Assessment](#2-data-quality-assessment)
  - [2.1 Missing Values](#21-missing-values)
  - [2.2 Duplicate Records](#22-duplicate-records)
  - [2.3 Data Type Consistency](#23-data-type-consistency)
- [3. Target Variable Analysis](#3-target-variable-analysis)
  - [3.1 Lead_Category Distribution](#31-lead_category-distribution)
  - [3.2 Original Status Mapping](#32-original-status-mapping)
  - [3.3 Target Variable Statistics](#33-target-variable-statistics)
- [4. Categorical Features Analysis](#4-categorical-features-analysis)
  - [4.1 Source (Lead Generation Channel)](#41-source-lead-generation-channel)
  - [4.2 Location (Geographic Origin)](#42-location-geographic-origin)
  - [4.3 Delivery_Mode (Service Delivery Method)](#43-delivery_mode-service-delivery-method)
  - [4.4 Created_DayOfWeek (Temporal)](#44-created_dayofweek-temporal)
  - [4.5 Created_Month (Temporal)](#45-created_month-temporal)
- [5. Numerical Features Analysis](#5-numerical-features-analysis)
  - [5.1 Product_ID](#51-product_id)
  - [5.2 Created_Hour](#52-created_hour)
  - [5.3 Created_Month](#53-created_month)
- [6. Temporal Pattern Analysis](#6-temporal-pattern-analysis)
  - [6.1 Hourly Lead Distribution](#61-hourly-lead-distribution)
  - [6.2 Weekly Lead Distribution](#62-weekly-lead-distribution)
  - [6.3 Monthly Lead Distribution](#63-monthly-lead-distribution)
- [7. Bivariate Analysis](#7-bivariate-analysis)
  - [7.1 Product_ID vs Lead_Category](#71-product_id-vs-lead_category)
  - [7.2 Created_Hour vs Lead_Category](#72-created_hour-vs-lead_category)
  - [7.3 Source vs Lead_Category](#73-source-vs-lead_category)
  - [7.4 Location vs Lead_Category](#74-location-vs-lead_category)
  - [7.5 Delivery_Mode vs Lead_Category](#75-delivery_mode-vs-lead_category)
  - [7.6 Created_DayOfWeek vs Lead_Category](#76-created_dayofweek-vs-lead_category)
  - [7.7 Created_Month vs Lead_Category](#77-created_month-vs-lead_category)
- [8. Multicollinearity Assessment](#8-multicollinearity-assessment)
  - [8.1 Variance Inflation Factor (VIF) Analysis](#81-variance-inflation-factor-vif-analysis)
  - [8.2 Spearman Correlation Analysis](#82-spearman-correlation-analysis)
- [9. Key Findings and Implications](#9-key-findings-and-implications)
  - [9.1 Data Quality Findings](#91-data-quality-findings)
  - [9.2 Class Imbalance Findings](#92-class-imbalance-findings)
  - [9.3 Feature Predictive Power Findings](#93-feature-predictive-power-findings)
  - [9.4 Distribution Findings](#94-distribution-findings)
  - [9.5 High Cardinality Findings](#95-high-cardinality-findings)
  - [9.6 Temporal Findings](#96-temporal-findings)
- [10. Recommendations for Modeling](#10-recommendations-for-modeling)
  - [10.1 Feature Engineering Recommendations](#101-feature-engineering-recommendations)
  - [10.2 Modeling Approach Recommendations](#102-modeling-approach-recommendations)
  - [10.3 Feature Selection Recommendations](#103-feature-selection-recommendations)
  - [10.4 Model Evaluation Recommendations](#104-model-evaluation-recommendations)
  - [10.5 Deployment Recommendations](#105-deployment-recommendations)
- [Appendix: Statistical Tests Summary](#appendix-statistical-tests-summary)

---

## 1. Data Collection and Structure

### 1.1 Dataset Dimensions

**Initial Dataset:**
- Rows: 7,422 sales leads
- Columns: 9 attributes
- Data Source: MySQL database (project_sales.data, host: 18.136.157.135:3306)
- File Format: CSV with ISO-8859-1 encoding, semicolon delimiter

**After Data Cleaning:**
- Rows: 7,420 (2 duplicates removed)
- Columns: 6 retained features (Created, Product_ID, Source, Location, Delivery_Mode, Status)
- Missing Values: 0 (after treatment)

### 1.2 Feature Schema

| Column Name | Data Type | Description | Cardinality |
|-------------|-----------|-------------|-------------|
| Created | datetime64 | Lead creation timestamp | 6,752 unique |
| Product_ID | float64 | Product identifier (-1 to 28) | 30 unique |
| Source | object | Lead generation channel | 26 unique |
| Mobile | object | Phone number (PII) | 486 unique |
| EMAIL | object | Email address (PII) | 883 unique |
| Sales_Agent | object | Assigned sales agent (PII) | 12 unique |
| Location | object | Geographic origin | 18 unique |
| Delivery_Mode | object | Service delivery method | 5 unique |
| Status | object | Lead outcome category | 11 unique |

**Memory Footprint:**
- Total DataFrame Memory: 3.87 MB
- Process Memory Usage: 292.02 MB

### 1.3 Engineered Temporal Features

From the `Created` timestamp, the following features were extracted:

| Feature | Data Type | Description | Unique Values | Range |
|---------|-----------|-------------|---------------|-------|
| Created_Date | object | Date only (YYYY-MM-DD) | 201 | April to November |
| Created_Hour | int32 | Hour of day (0-23) | 24 | 0 to 23 |
| Created_DayOfWeek | object | Day name | 7 | Monday to Sunday |
| Created_Month | int32 | Month number | 8 | 4 to 11 |
| Lead_Category | int64 | Binary target (0: Low, 1: High) | 2 | 0 to 1 |

---

## 2. Data Quality Assessment

### 2.1 Missing Values

**Before Cleaning:**

| Column | Missing Count | Missing Percentage | Treatment |
|--------|---------------|-------------------|-----------|
| Mobile | 1,810 | 24.39% | Dropped (PII, high missingness) |
| Product_ID | 58 | 0.78% | Filled with -1 |
| Location | 58 | 0.78% | Filled with "Unknown" |
| Sales_Agent | 23 | 0.31% | Dropped (PII) |
| Source | 17 | 0.23% | Filled with "Unknown" |
| Created | 0 | 0.00% | No action |
| EMAIL | 0 | 0.00% | Dropped (PII) |
| Delivery_Mode | 0 | 0.00% | No action |
| Status | 0 | 0.00% | No action |

**After Cleaning:**
- Missing Values: 0 across all retained features
- Infinite Values: 0

**Treatment Justification:**

1. **Mobile (24.39% missing):** Dropped entirely due to:
   - Privacy concerns (PII)
   - Excessive missingness creates imputation unreliability
   - Missingness likely not random (users may refuse to provide phone numbers)

2. **Product_ID (0.78% missing):** Filled with -1 to:
   - Preserve all 7,420 rows
   - Create explicit "unknown product" category
   - Avoid losing rows with valuable information in other columns

3. **Location (0.78% missing):** Filled with "Unknown" to:
   - Maintain dataset size
   - Create distinct category for missing geographic data
   - Allow model to learn patterns of leads without location data

4. **Source (0.23% missing):** Filled with "Unknown" to:
   - Retain 17 rows with minimal impact
   - Explicit handling vs. dropping

5. **PII Columns (EMAIL, Sales_Agent, Mobile):** Dropped to:
   - Comply with data privacy requirements
   - Prevent data leakage (email domain may correlate with company size)
   - Remove features with no legitimate predictive value

### 2.2 Duplicate Records

**Duplicates Detected:** 2 rows (0.03% of dataset)

**Action:** Removed both duplicate rows

**Verification Method:** Checked for exact matches across all 9 columns

**Final Dataset:** 7,420 unique lead records

### 2.3 Data Type Consistency

**Data Type Conversions:**

| Column | Original Type | Converted Type | Reason |
|--------|---------------|----------------|--------|
| Created | object | datetime64[ns] | Enable temporal operations |
| Product_ID | float64 | Retained as float64 | Contains -1 for missing values |
| Status | object | int64 (Lead_Category) | Binary encoding for modeling |
| Created_Hour | N/A | int32 | Engineered feature |
| Created_Month | N/A | int32 | Engineered feature |

**Verification:**
- No infinite values detected after type conversion
- All datetime conversions successful
- Numerical ranges validated (Product_ID: -1 to 28, Created_Hour: 0 to 23)

---

## 3. Target Variable Analysis

### 3.1 Lead_Category Distribution

**Class Breakdown:**

| Lead_Category | Label | Count | Percentage | Proportion |
|---------------|-------|-------|------------|------------|
| 0 | Low Potential | 4,571 | 61.6% | Majority |
| 1 | High Potential | 2,849 | 38.4% | Minority |

**Class Imbalance Ratio:** 1.60:1 (Low:High)

**Visualization:** Distribution is moderately imbalanced, requiring class weighting or resampling techniques during modeling.

### 3.2 Original Status Mapping

**High Potential Categories (Lead_Category = 1):**

| Original Status | Count | % of High Potential | Business Interpretation |
|----------------|-------|---------------------|------------------------|
| CONVERTED | 834 | 29.3% | Closed deals |
| Potential | 708 | 24.9% | Strong prospects |
| Long Term | 646 | 22.7% | Future opportunities |
| In Progress Positive | 643 | 22.6% | Active positive engagements |
| converted | 18 | 0.6% | Closed deals (duplicate label) |

**Total High Potential:** 2,849 leads

**Low Potential Categories (Lead_Category = 0):**

| Original Status | Count | % of Low Potential | Business Interpretation |
|----------------|-------|-------------------|------------------------|
| Junk Lead | 1,536 | 33.6% | Unqualified/spam |
| Not Responding | 1,129 | 24.7% | No engagement |
| Just Enquiry | 760 | 16.6% | Informational only |
| In Progress Negative | 626 | 13.7% | Declining interest |
| LOST | 440 | 9.6% | Failed to convert |
| Open | 80 | 1.8% | Unclassified |

**Total Low Potential:** 4,571 leads

### 3.3 Target Variable Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 0.384 | 38.4% of leads are high potential |
| Standard Deviation | 0.486 | High variance (binary variable) |
| Minimum | 0.0 | Low Potential class |
| 25th Percentile | 0.0 | 75% are not high potential |
| Median | 0.0 | Median is Low Potential |
| 75th Percentile | 1.0 | Top 25% includes High Potential |
| Maximum | 1.0 | High Potential class |
| Mode | 0.0 | Most common class is Low Potential |
| Skewness | 0.477 | Right-skewed (more Low Potential) |
| Kurtosis | -1.773 | Platykurtic (flatter than normal) |

**Normality Test:**
- Kolmogorov-Smirnov Statistic: 0.401
- p-value: 0.00
- Conclusion: Rejects normality (expected for binary variable)

---

## 4. Categorical Features Analysis

### 4.1 Source (Lead Generation Channel)

**Cardinality:** 26 unique sources

**Top 10 Sources by Volume:**

| Rank | Source | Count | Percentage | Cumulative % |
|------|--------|-------|------------|--------------|
| 1 | Call | 2,547 | 34.33% | 34.33% |
| 2 | Live Chat-Direct | 1,834 | 24.72% | 59.05% |
| 3 | Website | 1,592 | 21.46% | 80.51% |
| 4 | Live Chat-Google Organic | 274 | 3.69% | 84.20% |
| 5 | Live Chat-PPC | 249 | 3.36% | 87.56% |
| 6 | Live Chat-Blog | 237 | 3.19% | 90.75% |
| 7 | Customer Referral | 180 | 2.43% | 93.18% |
| 8 | US Website | 137 | 1.85% | 95.03% |
| 9 | Just Dial | 56 | 0.75% | 95.78% |
| 10 | Existing Client | 51 | 0.69% | 96.47% |

**Long Tail Sources (16 sources):** Remaining 3.53% of volume distributed across:
- Personal Contact (50, 0.67%)
- Live Chat-CPC (50, 0.67%)
- By Recommendation (32, 0.43%)
- Existing Customer (22, 0.30%)
- Live Chat-Google Ads (21, 0.28%)
- Campaign (19, 0.26%)
- Unknown (17, 0.23%)
- E-mail Campaign (12, 0.16%)
- CRM form (23, 0.31%)
- 7 sources with <10 leads each

**Source Concentration:**
- Top 3 sources account for 80.51% of all leads
- Top 7 sources account for 93.18% of all leads
- 16 sources have <1% contribution each

**High Cardinality Implications:**
- One-hot encoding would create 26 sparse columns
- Frequency encoding recommended to reduce dimensionality
- Long-tail sources may lack statistical power for model training

### 4.2 Location (Geographic Origin)

**Cardinality:** 18 unique locations

**Distribution by Volume:**

| Rank | Location | Count | Percentage | Cumulative % |
|------|----------|-------|------------|--------------|
| 1 | Other Locations | 2,500 | 33.69% | 33.69% |
| 2 | Bangalore | 2,084 | 28.09% | 61.78% |
| 3 | Chennai | 909 | 12.25% | 74.03% |
| 4 | Hyderabad | 528 | 7.12% | 81.15% |
| 5 | Delhi | 471 | 6.35% | 87.50% |
| 6 | Mumbai | 402 | 5.42% | 92.92% |
| 7 | Pune | 142 | 1.91% | 94.83% |
| 8 | UAE | 79 | 1.06% | 95.89% |
| 9 | Trivandrum | 58 | 0.78% | 96.67% |
| 10 | Unknown | 56 | 0.75% | 97.42% |
| 11-18 | Others (8 cities) | 191 | 2.58% | 100.00% |

**Geographic Concentration:**
- Top 2 locations (Other Locations, Bangalore) account for 61.78% of leads
- Top 6 locations account for 92.92% of leads
- Indian cities dominate (91.43%), international locations contribute 8.57%

**International Locations:**
- UAE: 79 leads (1.06%)
- USA: 45 leads (0.61%)
- UK: 41 leads (0.55%)
- AUSTRALIA: 25 leads (0.34%)
- Singapore: 17 leads (0.23%)
- Malaysia: 4 leads (0.05%)
- EUROPE: 3 leads (0.04%)

**Data Quality Note:**
- "Other Locations" is a catch-all category representing 33.69% of data
- "Unknown" represents 0.75% with missing location data (filled during cleaning)

### 4.3 Delivery_Mode (Service Delivery Method)

**Cardinality:** 5 unique modes

**Distribution:**

| Delivery_Mode | Count | Percentage |
|---------------|-------|------------|
| Mode-5 | 2,973 | 40.07% |
| Mode-1 | 2,627 | 35.40% |
| Mode-3 | 1,688 | 22.75% |
| Mode-4 | 121 | 1.63% |
| Mode-2 | 11 | 0.15% |

**Observations:**
- Two dominant modes (Mode-5, Mode-1) account for 75.47% of leads
- Mode-3 represents moderate volume (22.75%)
- Mode-4 and Mode-2 are rare (1.78% combined)
- Mode-2 has only 11 occurrences, may lack statistical power

**Encoding Strategy:**
- Low cardinality (5 categories) suitable for one-hot encoding
- Mode-1 will be dropped as reference category (avoid dummy variable trap)
- 4 binary features created: Mode-2, Mode-3, Mode-4, Mode-5

### 4.4 Created_DayOfWeek (Temporal)

**Cardinality:** 7 unique days

**Distribution:**

| Day | Count | Percentage | Rank |
|-----|-------|------------|------|
| Monday | 1,513 | 20.39% | 1 |
| Tuesday | 1,319 | 17.78% | 2 |
| Wednesday | 1,272 | 17.14% | 3 |
| Thursday | 1,096 | 14.77% | 4 |
| Friday | 1,066 | 14.37% | 5 |
| Saturday | 756 | 10.19% | 6 |
| Sunday | 398 | 5.36% | 7 |

**Weekly Pattern:**
- Weekdays (Monday-Friday): 6,266 leads (84.45%)
- Weekends (Saturday-Sunday): 1,154 leads (15.55%)
- Monday has highest volume (20.39%), Sunday lowest (5.36%)
- Linear decline from Monday to Sunday (weekdays), sharp drop on weekends

**Business Implications:**
- Sales team most active Monday-Friday
- Weekend leads may be web-driven (no sales team contact)
- Monday spike may reflect weekend lead accumulation

### 4.5 Created_Month (Temporal)

**Cardinality:** 8 unique months (April to November)

**Distribution:**

| Month | Count | Percentage | Rank |
|-------|-------|------------|------|
| June | 1,285 | 17.32% | 1 |
| May | 1,260 | 16.98% | 2 |
| July | 1,157 | 15.59% | 3 |
| August | 1,119 | 15.08% | 4 |
| September | 1,103 | 14.87% | 5 |
| October | 1,000 | 13.48% | 6 |
| November | 381 | 5.13% | 7 |
| April | 115 | 1.55% | 8 |

**Seasonal Pattern:**
- Peak months: May-June (34.30% of all leads)
- Decline from June to November
- April is incomplete month (only 115 leads, likely partial data)
- November is incomplete month (381 leads, dataset ends mid-November)

**Quarterly Breakdown (excluding partial months):**
- Q2 (May-June): 2,545 leads (34.30%)
- Q3 (July-September): 3,379 leads (45.54%)
- Q4 (October): 1,000 leads (13.48%)

**Temporal Coverage:**
Dataset spans 7 complete months (May-October) plus partial April and November data.

---

## 5. Numerical Features Analysis

### 5.1 Product_ID

**Descriptive Statistics:**

| Statistic | Value |
|-----------|-------|
| Count | 7,420 |
| Unique Values | 30 |
| Mean | 15.82 |
| Standard Deviation | 6.23 |
| Minimum | -1.0 |
| 25th Percentile | 10.0 |
| Median | 18.0 |
| 75th Percentile | 19.0 |
| Maximum | 28.0 |
| Mode | 18.0 |
| Range | 29.0 |
| IQR | 9.0 |
| Variance | 38.75 |
| Skewness | -0.278 |
| Kurtosis | -0.020 |

**Distribution Characteristics:**
- **Left-skewed (skewness = -0.278):** Concentration toward higher product IDs (18-19) with tail extending to lower values
- **Mesokurtic (kurtosis ≈ 0):** Normal-like tail behavior, no extreme outliers
- **Missing Value Encoding:** -1 represents 56 leads (0.75%) with unknown product
- **Median > Mean:** Indicates left skew (median 18.0 > mean 15.82)

**Product ID Frequency:**
- Most common product: ID 18 (mode)
- 50% of leads involve products 18-19 (IQR: 10-19)
- Product range: 0 to 28 (29 products) plus -1 (unknown)

**Normality Test:**
- Kolmogorov-Smirnov Statistic: 0.185
- p-value: 2.30e-222
- Conclusion: Strongly rejects normality (p << 0.001)

**Outliers:**
- IQR Method: 0 outliers detected
- No values fall outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

### 5.2 Created_Hour

**Descriptive Statistics:**

| Statistic | Value |
|-----------|-------|
| Count | 7,420 |
| Unique Values | 24 |
| Mean | 13.33 |
| Standard Deviation | 4.05 |
| Minimum | 0 |
| 25th Percentile | 11 |
| Median | 13 |
| 75th Percentile | 16 |
| Maximum | 23 |
| Mode | 11 |
| Range | 23 |
| IQR | 5 |
| Variance | 16.44 |
| Skewness | -0.256 |
| Kurtosis | 0.936 |

**Distribution Characteristics:**
- **Left-skewed (skewness = -0.256):** Slight concentration in afternoon hours (11-16) with tail toward early morning
- **Leptokurtic (kurtosis = 0.936):** Heavier tails than normal distribution, indicating occasional extreme hours
- **Central Tendency:** Mean (13.33) and median (13) are close, both in early afternoon
- **Peak Hour:** 11 AM (mode)

**Hourly Pattern:**
- 50% of leads created between 11 AM and 4 PM (IQR: 11-16)
- Business hours (9 AM - 5 PM) likely account for majority
- Early morning hours (0-3) are outliers (185 leads, 2.49%)

**Normality Test:**
- Kolmogorov-Smirnov Statistic: 0.108
- p-value: 1.37e-75
- Conclusion: Rejects normality (p << 0.001)

**Outliers:**
- IQR Method: 185 outliers (2.49%)
- Outlier Values: Hours 0, 1, 2, 3 (early morning)
- Lower Bound: Q1 - 1.5×IQR = 11 - 7.5 = 3.5 (values ≤ 3 are outliers)
- Upper Bound: Q3 + 1.5×IQR = 16 + 7.5 = 23.5 (no outliers on upper end)

**Business Interpretation:**
- Peak lead activity in late morning to early afternoon
- Early morning leads (0-3 AM) may represent:
  - International time zones
  - Automated submissions
  - Website form fills during off-hours

### 5.3 Created_Month

**Descriptive Statistics:**

| Statistic | Value |
|-----------|-------|
| Count | 7,420 |
| Unique Values | 8 |
| Mean | 7.16 |
| Minimum | 4 |
| Maximum | 11 |
| Range | 7 |
| Skewness | 0.139 |
| Kurtosis | -1.100 |

**Distribution Characteristics:**
- **Right-skewed (skewness = 0.139):** Slight concentration in earlier months (May-July) with decline toward November
- **Platykurtic (kurtosis = -1.100):** Flatter distribution than normal, more uniform spread across months
- **Coverage:** April (4) to November (11)

**Normality Test:**
- Kolmogorov-Smirnov Statistic: 0.146
- p-value: 9.18e-139
- Conclusion: Rejects normality (p << 0.001)

**Monthly Distribution Pattern:**
Refer to Section 4.5 for detailed monthly volume breakdown.

---

## 6. Temporal Pattern Analysis

### 6.1 Hourly Lead Distribution

**Peak Hours (Top 5):**

| Hour | Count | Percentage | Cumulative % |
|------|-------|------------|--------------|
| 11 | 632 | 8.52% | 8.52% |
| 13 | 594 | 8.01% | 16.53% |
| 14 | 582 | 7.84% | 24.37% |
| 12 | 576 | 7.76% | 32.13% |
| 15 | 560 | 7.55% | 39.68% |

**Off-Peak Hours (Bottom 5):**

| Hour | Count | Percentage |
|------|-------|------------|
| 0 | 62 | 0.84% |
| 1 | 50 | 0.67% |
| 23 | 76 | 1.02% |
| 2 | 39 | 0.53% |
| 3 | 34 | 0.46% |

**Business Hours Analysis (9 AM - 6 PM):**
- Leads during business hours: Approximately 5,500+ (74%+)
- Leads outside business hours: Approximately 1,900+ (26%+)

**Hourly Pattern:**
- Gradual increase from 6 AM to 11 AM (peak)
- Sustained high volume from 11 AM to 4 PM
- Decline after 4 PM through midnight
- Minimal activity 12 AM to 6 AM

### 6.2 Weekly Lead Distribution

**Weekday Performance:**

| Day Category | Days | Total Count | Percentage | Avg per Day |
|--------------|------|-------------|------------|-------------|
| Weekdays | Mon-Fri | 6,266 | 84.45% | 1,253 |
| Weekends | Sat-Sun | 1,154 | 15.55% | 577 |

**Daily Pattern:**
- Monday has highest volume (1,513 leads)
- Progressive decline Tuesday through Friday
- Sharp drop on Saturday (756 leads, 50% of Monday)
- Further drop on Sunday (398 leads, 26% of Monday)

**Week-over-Week Stability:**
Consistent weekly pattern suggests stable lead generation processes (no data provided on week-to-week variance).

### 6.3 Monthly Lead Distribution

**Seasonal Trends:**

| Season | Months | Total Count | Percentage | Avg per Month |
|--------|--------|-------------|------------|---------------|
| Q2 (Spring) | May-Jun | 2,545 | 34.30% | 1,273 |
| Q3 (Summer) | Jul-Sep | 3,379 | 45.54% | 1,126 |
| Q4 (Fall) | Oct | 1,000 | 13.48% | 1,000 |

**Monthly Decline:**
- Peak: June (1,285 leads)
- Decline: -9.8% from June to July
- Further decline: -3.3% from July to August
- Continued decline through November

**Partial Months:**
- April: Only 115 leads (incomplete data, likely starts mid-month)
- November: Only 381 leads (incomplete data, likely ends mid-month)

**Business Cycle Hypothesis:**
- Higher lead volume in Q2-Q3 may reflect:
  - Fiscal year planning cycles
  - Seasonal IT project initiation
  - Marketing campaign timing

---

## 7. Bivariate Analysis

### 7.1 Product_ID vs Lead_Category

**Statistical Test:** Mann-Whitney U test (non-parametric, due to non-normality)

**Results:**

| Lead_Category | Count | Mean Product_ID | Std Dev |
|---------------|-------|-----------------|---------|
| 0 (Low Potential) | 4,571 | 15.3 | 6.2 |
| 1 (High Potential) | 2,849 | 16.6 | 6.2 |

**Findings:**
- High Potential leads have higher average Product_ID (16.6 vs. 15.3)
- Difference: +1.3 product IDs
- Standard deviations are identical (6.2), indicating similar spread
- p-value: < 0.001 (statistically significant)

**Interpretation:**
Higher-numbered products (IDs 16-28) are more likely to attract high-potential leads, suggesting premium or newer products may have better lead quality.

### 7.2 Created_Hour vs Lead_Category

**Statistical Test:** Mann-Whitney U test

**Results:**

| Lead_Category | Count | Mean Hour | Std Dev |
|---------------|-------|-----------|---------|
| 0 (Low Potential) | 4,571 | 13.2 | 4.2 |
| 1 (High Potential) | 2,849 | 13.6 | 3.8 |

**Findings:**
- High Potential leads created slightly later in day (13.6 vs. 13.2 hours)
- Difference: +0.4 hours (24 minutes)
- High Potential leads have lower variance (SD 3.8 vs. 4.2)
- p-value: < 0.001 (statistically significant)

**Interpretation:**
High-potential leads are more concentrated around midday (1:30 PM) with less spread, while low-potential leads have wider temporal distribution.

### 7.3 Source vs Lead_Category

**Statistical Test:** Chi-square test of independence

**p-value:** < 0.001 (highly significant)

**High Conversion Sources:**

| Source | Total Leads | High Potential | Conversion Rate |
|--------|-------------|----------------|-----------------|
| Existing Client | 51 | 48 | 94.1% |
| Existing Customer | 22 | 20 | 90.9% |
| Customer Referral | 180 | 161 | 89.4% |
| By Recommendation | 32 | 28 | 87.5% |
| Personal Contact | 50 | 35 | 70.0% |

**Low Conversion Sources:**

| Source | Total Leads | High Potential | Conversion Rate |
|--------|-------------|----------------|-----------------|
| US Website | 137 | 2 | 1.5% |
| Unknown | 17 | 1 | 5.9% |
| Website | 1,592 | 412 | 25.9% |
| Call | 2,547 | 851 | 33.4% |

**Key Insights:**
- **Referral channels** (Existing Client, Customer Referral, Recommendation) have 87-94% conversion
- **Cold channels** (US Website, Website) have 1.5-26% conversion
- **Volume vs. Quality Trade-off:** Website (1,592 leads, 25.9%) generates volume but lower quality than referrals (180 leads, 89.4%)

**Source Contribution to High Potential Leads:**

| Source | High Potential Count | % of Total High Potential |
|--------|---------------------|---------------------------|
| Call | 851 | 29.9% |
| Live Chat-Direct | 682 | 23.9% |
| Website | 412 | 14.5% |
| Customer Referral | 161 | 5.7% |
| Live Chat-Google Organic | 126 | 4.4% |

Despite low conversion rate, high-volume sources (Call, Live Chat-Direct) contribute most high-potential leads in absolute terms.

### 7.4 Location vs Lead_Category

**Statistical Test:** Chi-square test of independence

**p-value:** < 0.001 (highly significant)

**High Conversion Locations:**

| Location | Total Leads | High Potential | Conversion Rate |
|----------|-------------|----------------|-----------------|
| Bangalore | 2,084 | 1,269 | 60.9% |
| Chennai | 909 | 460 | 50.6% |
| Hyderabad | 528 | 246 | 46.6% |
| Pune | 142 | 68 | 47.9% |
| Delhi | 471 | 204 | 43.3% |

**Low Conversion Locations:**

| Location | Total Leads | High Potential | Conversion Rate |
|----------|-------------|----------------|-----------------|
| Unknown | 56 | 1 | 1.8% |
| Other Locations | 2,500 | 429 | 17.2% |
| US Website | 137 | 2 | 1.5% |

**Geographic Concentration of High Potential Leads:**

| Location | High Potential Count | % of Total High Potential |
|----------|---------------------|---------------------------|
| Bangalore | 1,269 | 44.5% |
| Chennai | 460 | 16.1% |
| Hyderabad | 246 | 8.6% |
| Delhi | 204 | 7.2% |
| Mumbai | 161 | 5.7% |
| **Top 5 Total** | **2,340** | **82.1%** |

**Key Insights:**
- **Bangalore dominance:** Nearly half (44.5%) of all high-potential leads originate from Bangalore
- **Top 5 cities:** Contribute 82.1% of high-potential leads despite representing only 58.71% of total volume
- **Other Locations:** 2,500 leads (33.69% of total) but only 429 high-potential (17.2% conversion), suggesting poor geographic targeting or data quality issue
- **International locations:** Low volume and low conversion (UAE, USA, UK combined: 165 leads, unknown conversion)

### 7.5 Delivery_Mode vs Lead_Category

**Statistical Test:** Chi-square test of independence

**p-value:** < 0.001 (highly significant)

**Delivery Mode Performance:**

| Delivery_Mode | Total Leads | High Potential | Conversion Rate |
|---------------|-------------|----------------|-----------------|
| Mode-4 | 121 | 81 | 66.9% |
| Mode-1 | 2,627 | 1,331 | 50.7% |
| Mode-3 | 1,688 | 704 | 41.7% |
| Mode-5 | 2,973 | 733 | 24.7% |
| Mode-2 | 11 | 0 | 0.0% |

**Contribution to High Potential Leads:**

| Delivery_Mode | High Potential Count | % of Total High Potential |
|---------------|---------------------|---------------------------|
| Mode-1 | 1,331 | 46.7% |
| Mode-5 | 733 | 25.7% |
| Mode-3 | 704 | 24.7% |
| Mode-4 | 81 | 2.8% |
| Mode-2 | 0 | 0.0% |

**Key Insights:**
- **Mode-4:** Highest conversion rate (66.9%) but low volume (121 leads), premium service indicator
- **Mode-1:** Best balance of volume (2,627 leads) and quality (50.7% conversion), contributes 46.7% of all high-potential leads
- **Mode-5:** Highest volume (2,973 leads) but lowest conversion (24.7%), may be entry-level or commodity service
- **Mode-2:** Only 11 leads, all low-potential, insufficient data for reliable conclusions

**Volume-Quality Trade-off:**
- Mode-1: High volume + high conversion (best balance)
- Mode-4: Low volume + high conversion (premium niche)
- Mode-5: High volume + low conversion (commodity)

### 7.6 Created_DayOfWeek vs Lead_Category

**Statistical Test:** Chi-square test of independence

**p-value:** 0.462 (NOT significant)

**Findings:**
- No statistically significant difference in lead quality across days of week
- Conversion rates are relatively uniform Monday through Sunday
- Day of week does NOT predict lead quality

**Implication for Modeling:**
Created_DayOfWeek feature should be excluded from predictive model due to lack of predictive power.

### 7.7 Created_Month vs Lead_Category

**Statistical Test:** Chi-square test of independence

**p-value:** < 0.001 (highly significant)

**Monthly Conversion Rates:**

| Month | Total Leads | High Potential | Conversion Rate | Rank |
|-------|-------------|----------------|-----------------|------|
| November | 381 | 177 | 46.5% | 1 |
| October | 1,000 | 453 | 45.3% | 2 |
| September | 1,103 | 489 | 44.3% | 3 |
| August | 1,119 | 442 | 39.5% | 4 |
| July | 1,157 | 437 | 37.8% | 5 |
| June | 1,285 | 450 | 35.0% | 6 |
| May | 1,260 | 382 | 30.3% | 7 |
| April | 115 | 19 | 16.5% | 8 |

**Key Insights:**
- **Increasing quality over time:** Conversion rate improves from May (30.3%) to November (46.5%)
- **Volume-Quality Trade-off:** Early months (May-June) have high volume but lower quality; later months (Oct-Nov) have lower volume but higher quality
- **April anomaly:** Very low conversion (16.5%) but only 115 leads, likely incomplete month data

**Seasonal Pattern:**
- Q2 (May-Jun): High volume (2,545), moderate quality (32.6% avg conversion)
- Q3 (Jul-Sep): Moderate volume (3,379), increasing quality (40.5% avg conversion)
- Q4 (Oct-Nov): Lower volume (1,381), highest quality (45.8% avg conversion)

**Business Hypothesis:**
- Early months: Broad marketing campaigns generate volume but lower quality
- Later months: Refined targeting or seasonal business cycles improve quality
- November: Fiscal year-end purchasing may explain 46.5% conversion spike

---

## 8. Multicollinearity Assessment

### 8.1 Variance Inflation Factor (VIF) Analysis

**Method:** VIF calculated for all numerical features

**Threshold:** 8.0 (values > 8.0 indicate problematic multicollinearity)

**Results:**

| Feature | VIF | Status | Severity |
|---------|-----|--------|----------|
| Created_Month | 8.68 | High | Marginally exceeds threshold |
| Created_Hour | 7.91 | Acceptable | Below threshold |
| Product_ID | 6.09 | Low | Well below threshold |

**Interpretation:**

1. **Created_Month (VIF = 8.68):**
   - Marginally exceeds 8.0 threshold
   - Indicates mild multicollinearity with other temporal features
   - Expected due to correlation between month and hour-of-day patterns (seasonal working hours)
   - VIF < 10 is generally acceptable for tree-based models
   - **Decision:** Retain for business insights despite VIF = 8.68

2. **Created_Hour (VIF = 7.91):**
   - Just below 8.0 threshold, acceptable
   - Minimal interdependence with month and product features
   - **Decision:** Retain

3. **Product_ID (VIF = 6.09):**
   - Low VIF confirms independence from temporal features
   - No multicollinearity concerns
   - **Decision:** Retain

**Overall Assessment:**
Multicollinearity is not a severe concern. All VIF values are below 10, indicating that features capture distinct information. The slight correlation between Created_Month and Created_Hour is expected and does not require remediation for tree-based modeling approaches.

### 8.2 Spearman Correlation Analysis

**Method:** Spearman rank correlation (non-parametric, suitable for non-normal distributions)

**Threshold:** |ρ| > 0.80 for multicollinearity concern

**Findings:**
- No feature pairs exhibited Spearman correlation exceeding 0.80
- All pairwise correlations were weak to moderate (|ρ| < 0.80)
- Conclusion: No multicollinear pairs detected

**Implication:**
Numerical features (Product_ID, Created_Hour, Created_Month) are functionally independent, providing non-redundant information to predictive models.

---

## 9. Key Findings and Implications

### 9.1 Data Quality Findings

**Finding 1: High Missing Data in Mobile**
- Mobile field has 24.39% missing values
- **Implication:** Feature unreliable for modeling; dropped entirely

**Finding 2: Low Missing Data in Critical Features**
- Product_ID (0.78%), Location (0.78%), Source (0.23%) have minimal missingness
- **Implication:** Imputation strategy (fill with sentinel values) preserves 99%+ of dataset

**Finding 3: Minimal Duplication**
- Only 2 duplicate records (0.03%)
- **Implication:** High data quality, minimal cleaning required

**Finding 4: PII Contamination**
- EMAIL, Sales_Agent, Mobile contain personally identifiable information
- **Implication:** Dropped for privacy compliance and data leakage prevention

### 9.2 Class Imbalance Findings

**Finding 5: Moderate Class Imbalance**
- 61.6% Low Potential, 38.4% High Potential (1.60:1 ratio)
- **Implication:** Requires class weighting or SMOTE during modeling to prevent majority class bias

**Finding 6: Binary Classification is Appropriate**
- Original 11-status categories cleanly map to 2 lead quality categories
- **Implication:** Binary classification (High/Low Potential) is sufficient for business needs

### 9.3 Feature Predictive Power Findings

**Finding 7: Location is Strongest Categorical Predictor**
- Chi-square p < 0.001, Bangalore alone contributes 44.5% of high-potential leads
- **Implication:** Geographic targeting is critical; frequency encoding recommended to capture volume patterns

**Finding 8: Source Quality Varies Substantially**
- Referral channels: 87-94% conversion
- Website channels: 1.5-26% conversion
- **Implication:** Marketing spend should shift from low-converting sources (US Website) to referral programs

**Finding 9: Delivery Mode Signals Customer Commitment**
- Mode-4: 66.9% conversion (premium service)
- Mode-5: 24.7% conversion (commodity service)
- **Implication:** Delivery mode is strong lead quality indicator; one-hot encoding preserves information

**Finding 10: Product ID Predicts Lead Quality**
- Higher product IDs (16-28) have better conversion
- Mean Product_ID for high potential: 16.6 vs. 15.3 for low potential (p < 0.001)
- **Implication:** Product segmentation strategy; certain products attract better leads

**Finding 11: Day of Week Has No Predictive Power**
- Chi-square p = 0.462 (not significant)
- **Implication:** Exclude Created_DayOfWeek from modeling to reduce dimensionality

**Finding 12: Month Shows Increasing Lead Quality**
- Conversion improves from May (30.3%) to November (46.5%)
- Chi-square p < 0.001
- **Implication:** Seasonality affects lead quality; include Created_Month in model

### 9.4 Distribution Findings

**Finding 13: All Features Reject Normality**
- Product_ID, Created_Hour, Created_Month, Lead_Category all p < 0.001 (Kolmogorov-Smirnov)
- **Implication:** Use non-parametric tests (Mann-Whitney U, Spearman) and tree-based models (XGBoost, Random Forest)

**Finding 14: Early Morning Leads are Outliers**
- Hours 0-3 represent 185 leads (2.49%), flagged as outliers by IQR method
- **Implication:** Retain outliers (valid domain values), tree-based models handle outliers through split-based partitioning

### 9.5 High Cardinality Findings

**Finding 15: Source Has 26 Categories**
- Top 3 sources account for 80.51% of volume, 16 sources have <1% each
- **Implication:** Frequency encoding recommended to avoid 26 sparse one-hot columns

**Finding 16: Location Has 18 Categories**
- "Other Locations" represents 33.69%, a catch-all with poor data quality
- **Implication:** Frequency encoding recommended; consider data quality improvement for "Other Locations"

### 9.6 Temporal Findings

**Finding 17: Business Hours Dominate**
- 74%+ of leads created 9 AM - 6 PM
- Peak hour: 11 AM
- **Implication:** Sales team capacity planning; after-hours leads may be web-driven (different quality profile)

**Finding 18: Monday Peak, Weekend Drop**
- Monday: 20.39% of leads, Sunday: 5.36%
- Weekdays: 84.45%, Weekends: 15.55%
- **Implication:** Staffing adjustments; weekend leads may have different conversion patterns

**Finding 19: Seasonal Volume Decline**
- June peak (1,285 leads) declining to October (1,000 leads)
- **Implication:** Marketing campaign timing; investigate causes of Q4 decline

---

## 10. Recommendations for Modeling

### 10.1 Feature Engineering Recommendations

**1. Apply Frequency Encoding to High Cardinality Features**
- **Features:** Source (26 categories), Location (18 categories)
- **Rationale:** Reduces dimensionality while preserving volume information
- **Implementation:** Map each category to its occurrence count

**2. Apply One-Hot Encoding to Low Cardinality Features**
- **Features:** Delivery_Mode (5 categories)
- **Rationale:** Low cardinality makes one-hot encoding feasible without excessive sparsity
- **Implementation:** Create 4 binary features (drop Mode-1 as reference)

**3. Extract and Retain Temporal Features Selectively**
- **Retain:** Created_Month (p < 0.001, VIF = 8.68 acceptable)
- **Drop:** Created_DayOfWeek (p = 0.462, no predictive power)
- **Consider:** Created_Hour (p < 0.001, VIF = 7.91, but may have low importance in tree models)

**4. Handle Missing Values with Sentinel Encoding**
- **Product_ID:** Use -1 for missing values (already implemented)
- **Location:** Use "Unknown" for missing values (already implemented)
- **Source:** Use "Unknown" for missing values (already implemented)
- **Rationale:** Preserves all rows, creates explicit "unknown" category for model to learn

**5. Do Not Transform or Normalize Features**
- **Rationale:** Tree-based models (XGBoost, Random Forest, CatBoost) are invariant to monotonic transformations
- **Exception:** If using logistic regression or neural networks, apply StandardScaler

### 10.2 Modeling Approach Recommendations

**1. Use Tree-Based Models as Primary Approach**
- **Recommended:** XGBoost, LightGBM, CatBoost, Random Forest
- **Rationale:**
  - Handle non-normal distributions through split-based partitioning (all features reject normality)
  - Handle multicollinearity well (Created_Month VIF = 8.68)
  - Handle outliers through split thresholds (early morning hours)
  - Capture non-linear relationships (Product_ID, hour, month effects)

**2. Implement Class Weighting**
- **Method:** Compute balanced class weights: `class_weight = n_samples / (n_classes * np.bincount(y))`
- **Weights:** Class 0: 0.812, Class 1: 1.302
- **Rationale:** Addresses 1.60:1 class imbalance, prioritizes recall for minority class (High Potential)
- **Alternative:** SMOTE (Synthetic Minority Over-sampling Technique) if class weighting insufficient

**3. Use Stratified K-Fold Cross-Validation**
- **Folds:** 5-fold stratified
- **Rationale:** Maintains 61.6% / 38.4% class distribution in each fold, provides stable performance estimates
- **Metric:** F1-Score (balances precision and recall for imbalanced data)

**4. Prioritize Recall Over Precision (if business requires)**
- **Business Objective:** Capture as many High Potential leads as possible (minimize false negatives)
- **Trade-off:** Accept some false positives (wasted sales effort on Low Potential leads)
- **Implementation:** Tune decision threshold below 0.5 to increase recall
- **Alternative Metric:** ROC AUC (threshold-independent) for model comparison

**5. Perform Threshold Analysis Post-Training**
- **Method:** Plot precision-recall curve, identify appropriate threshold for F1-Score or custom business metric
- **Business Metric Example:** `Revenue_Gain(TP) - Cost_of_Contact(FP)`
- **Expected Outcome:** Threshold 0.3-0.4 may maximize business value

### 10.3 Feature Selection Recommendations

**1. Drop Low-Predictive Features**
- **Drop:** Created_DayOfWeek (p = 0.462)
- **Consider Dropping:** Created_Hour (may have low feature importance in tree models, evaluate post-training)

**2. Monitor Feature Importance Post-Training**
- **Method:** Extract feature importance from trained XGBoost/Random Forest
- **Threshold:** Drop features with <2% importance
- **Re-train:** Iterate with reduced feature set if accuracy is maintained

**3. Test Interaction Features (Extended Analysis)**
- **Candidates:**
  - Source × Location (referrals may perform differently by geography)
  - Product_ID × Delivery_Mode (premium products may correlate with premium delivery)
- **Caution:** Increases model complexity, risk of overfitting

### 10.4 Model Evaluation Recommendations

**1. Use Multiple Metrics for Imbalanced Data**
- **Primary:** F1-Score, ROC AUC, Precision-Recall AUC
- **Secondary:** Precision, Recall, Accuracy, Matthews Correlation Coefficient (MCC)
- **Avoid:** Accuracy alone (misleading with 61.6% / 38.4% imbalance)

**2. Analyze Confusion Matrix**
- **Focus:** False Negative Rate (missed High Potential leads)
- **Business Cost:** Assign cost to FN, FP, TP, TN; optimize for minimum total cost

**3. Perform Calibration Analysis**
- **Method:** Plot calibration curve (predicted probability vs. observed frequency)
- **Goal:** Ensure predicted probabilities are well-calibrated (P(High Potential | score=0.7) ≈ 70%)
- **Tool:** Platt scaling or isotonic regression if calibration is poor

**4. Validate on Holdout Test Set**
- **Split:** 80% train, 20% test, stratified
- **Random Seed:** Fix seed for reproducibility
- **Ensure:** No data leakage (test set never used for feature engineering or hyperparameter tuning)

### 10.5 Deployment Recommendations

**1. Integrate Model with CRM**
- **Output:** Lead score (0-100) and binary flag (High/Low Potential)
- **Threshold:** Use selected threshold (likely 0.3-0.4) for binary classification
- **Refresh:** Score new leads in real-time or batch (nightly)

**2. Monitor Model Drift**
- **Features:** Track distribution of Product_ID, Source, Location, Delivery_Mode over time
- **Performance:** Monitor monthly precision, recall, F1-Score on labeled data
- **Trigger:** Retrain if F1-Score drops >5% or feature distributions shift beyond threshold (e.g., >10% change)

**3. Implement A/B Testing**
- **Control:** 50% of leads scored manually (current process)
- **Treatment:** 50% of leads scored by model
- **Duration:** 3 months
- **Metrics:** Conversion rate, sales cycle length, revenue per lead

**4. Establish Feedback Loop**
- **Collect:** Actual lead outcomes (converted, lost, in-progress)
- **Retrain:** Quarterly retraining with new labeled data
- **Versioning:** Track model version, performance metrics, and deployment date

---

**End of Exploratory Data Analysis Report**

---

## Appendix: Statistical Tests Summary

| Analysis | Test Used | p-value | Significance | Conclusion |
|----------|-----------|---------|--------------|------------|
| Product_ID normality | Kolmogorov-Smirnov | 2.30e-222 | p < 0.001 | Reject normality |
| Created_Hour normality | Kolmogorov-Smirnov | 1.37e-75 | p < 0.001 | Reject normality |
| Created_Month normality | Kolmogorov-Smirnov | 9.18e-139 | p < 0.001 | Reject normality |
| Lead_Category normality | Kolmogorov-Smirnov | 0.00 | p < 0.001 | Reject normality |
| Product_ID vs Lead_Category | Mann-Whitney U | <0.001 | p < 0.001 | Significant difference |
| Created_Hour vs Lead_Category | Mann-Whitney U | <0.001 | p < 0.001 | Significant difference |
| Source vs Lead_Category | Chi-square | <0.001 | p < 0.001 | Significant association |
| Location vs Lead_Category | Chi-square | <0.001 | p < 0.001 | Significant association |
| Delivery_Mode vs Lead_Category | Chi-square | <0.001 | p < 0.001 | Significant association |
| Created_DayOfWeek vs Lead_Category | Chi-square | 0.462 | p > 0.05 | No significant association |
| Created_Month vs Lead_Category | Chi-square | <0.001 | p < 0.001 | Significant association |
| Spearman Correlation (all pairs) | Spearman | N/A | |ρ| < 0.80 | No multicollinearity |

---

**Report Prepared By:** Dhanesh B.B.
**Analysis Date:** Based on April-November lead data
**Report Date:** March 10, 2025
**Last Revised:** November 10, 2025
**Report Version:** 1.0
