# Table of Contents

- [Data Analysis](#data-analysis)
    - [Import data](#import-data)
    - [Imports & functions](#imports-functions)
  - [Data understanding and cleaning](#data-understanding-and-cleaning)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Univariate Analysis](#univariate-analysis)
      - [num_analysis](#num-analysis)
      - [cat_analysis](#cat-analysis)
    - [Bivariate & Multivariate Analysis](#bivariate-multivariate-analysis)
- [Predictive Model](#predictive-model)
  - [Preprocessing](#preprocessing)
      - [outliers](#outliers)
      - [relation](#relation)
      - [Association](#association)
      - [Encoding](#encoding)
      - [Splitting](#splitting)
  - [Model Development](#model-development)
    - [Model Training & Evaluation](#model-training-evaluation)
    - [Model comparision & Interpretation](#model-comparision-interpretation)
    - [Best Model](#best-model)
    - [Saving the Model](#saving-the-model)
    - [Loading the Model Further use](#loading-the-model-further-use)
- [Table of Contents](#table-of-contents)
- [Acknowledgment](#acknowledgment)
- [Report](#report)
- [Author Information](#author-information)
- [References](#references)
- [Appendix](#appendix)
  - [About data](#about-data)
  - [Source Code and Dependencies](#source-code-and-dependencies)

---

# Acknowledgment  

I appreciate the opportunity to work on the PRCL-0019: Sales Effectiveness & Lead Qualification project. This project helped me apply data science and machine learning to improve lead categorization and sales efficiency.

I acknowledge the dataset and business case provided, which guided the approach for data processing, feature engineering, and model development.

This project has been a valuable learning experience in sales analytics and predictive modeling.

---

# Report

**Final Data Analysis Report**

Final Data Analysis Report for FicZon Inc. Lead Categorization  
Project Ref: PM-PR-0019  

 1. Executive Summary  
FicZon Inc. aims to enhance sales effectiveness by automating lead categorization into High Potential (38.4% of leads) and Low Potential (61.6% of leads) using machine learning. This report details the data preparation, exploratory analysis, and predictive modeling steps to achieve this goal. Key findings include:  
- Temporal trends: Peak lead activity occurs on Mondays (20.4% of leads) and in June/May (17% monthly contribution).  
- Critical features: Status, Location, and Product_ID are most influential in predicting lead quality.  
- Class imbalance: Addressed through stratified sampling and class weighting (0.81 for Low Potential, 1.30 for High Potential).  



 2. Data Overview  
- Dataset Details  
    - Size: 7,422 entries, 9 initial features.  
    - Key Features:  
      - Temporal: Created (datetime), Created_Hour, Created_DayOfWeek, Created_Month.  
      - Categorical: Source, Location, Delivery_Mode, Status.  
      - Numerical: Product_ID.  

- Initial Data Quality Issues  
    - Missing Values: Mobile (24.4%), Product_ID (0.78%), Location (0.78%).  
    - High Cardinality: Created, Source, Mobile, EMAIL (removed to prevent leakage).  
    - Duplicates: 2 rows removed.  



 3. Data Cleaning & Preprocessing  
- Steps Taken:  
    1. Handling Missing Values:  
       - Product_ID and Location: Filled with -1 and Unknown, respectively.  
       - Mobile, Sales_Agent, EMAIL: Dropped (high cardinality/identifiers).  
    2. Feature Engineering:  
       - Extracted Created_Hour, Created_DayOfWeek, Created_Month from Created.  
       - Target Variable: Lead_Category derived from Status:  
         - High Potential: CONVERTED, Potential, In Progress Positive, etc.  
         - Low Potential: Junk Lead, Not Responding, LOST, etc.  
    3. Outliers:  
       - Created_Hour had 185 outliers (2.49%) in early hours (0–3 AM), retained for completeness.  



 4. Exploratory Data Analysis (EDA)  
- Key Insights  
    1. Temporal Patterns:  
       - Hourly: Peak activity at 11 AM; minimal activity post 8 PM.  
       - Monthly: Highest lead volume in June (17.3%), but highest conversion rates in November (46%).  
       - Weekly: 20.4% of leads generated on Mondays; weekends had the lowest activity.  
    
    2. Feature Distributions:  
       - Sources: 34.3% from calls, 24.7% from live chat.  
       - Locations: 33.7% from Other Locations, 28.1% from Bangalore.  
       - Delivery Modes: Mode-5 (40.1%) and Mode-1 (35.4%) dominated.  
    
    3. Correlations:  
       - Spearman Correlation: Weak relationships between features (all |r| < 0.8).  
       - VIF Analysis: Created_Month showed multicollinearity (VIF=8.68), retained for model context.  


 5. Feature Engineering & Modeling Preparation  
- Encoding & Transformations:  
    - Frequency Encoding: Applied to Source and Location (replaced with occurrence counts).  
    - One-Hot Encoding: Delivery_Mode converted to binary columns.  
    - Dropped Redundant Columns: Created, Status, Created_Date, Created_DayOfWeek.  

- Train-Test Split:  
    - Stratified Sampling: 80% training (5,936 samples), 20% testing (1,484 samples).  
    - Class Weights: Adjusted to mitigate imbalance (0:0.81, 1:1.30).  



 6. Predictive Modeling  
- Feature Importance  
    - Random Forest Classifier identified Status (60.1% importance) as the top predictor.  
    - Secondary drivers: Location (9.1%), Product_ID (8.1%).   


- Next Steps for Model Development:  
    1. Algorithm Selection: Test ensemble methods (e.g., XGBoost, LightGBM).  
    2. Hyperparameter Tuning: Optimize using GridSearchCV or RandomizedSearchCV.  
    3. Evaluation Metrics: Focus on precision (minimize false High Potential labels) and recall (capture true High Potential leads).  



 7. Business Recommendations  
    1. Target High-Value Channels: Prioritize leads from Customer Referral (5.7% conversion rate) and Existing Client (1.7%).  
    2. Geographic Focus: Allocate resources to Bangalore (44.5% High Potential leads).  
    3. Temporal Optimization: Intensify follow-ups during peak hours (10 AM–2 PM) and high-conversion months (October–November).  



 8. Conclusion  
This analysis provides a robust foundation for automating lead categorization at FicZon Inc. By leveraging temporal, geographic, and behavioral patterns, the model can prioritize high-potential leads, enabling the sales team to focus on high-impact opportunities. Further model refinement and real-time deployment will drive measurable improvements in sales efficiency.  

---

**Data Exploration Insights: Understanding Sales Effectiveness**  

 1. Lead Source Effectiveness  
- Key Findings:  
  - Calls (34.3%), Live Chat-Direct (24.7%), and Website (21.5%) dominate lead generation.  
  - Customer Referrals (5.7% of high-potential leads) and Existing Clients (1.7%) show higher conversion rates, indicating quality leads.  
  - US Website and Unknown sources contribute mostly to low-potential leads (3% and 0.4%, respectively).  
- Actionable Insight: Prioritize high-conversion channels like referrals and existing clients. Optimize low-performing channels (e.g., website forms) to reduce junk leads.  



 2. Geographical Performance  
- Key Findings:  
  - Bangalore (44.5% high-potential) and Chennai (16.1%) are top-performing locations.  
  - Other Locations (33.7% of total leads) contribute primarily to low-potential leads (45.3%).  
  - International regions (e.g., UAE, USA) show moderate conversion rates but low volume.  
- Actionable Insight: Focus sales efforts on high-performing regions like Bangalore. Investigate Other Locations to improve lead quality through targeted campaigns.  



 3. Temporal Trends  
- Hourly: Peak lead activity occurs between 10 AM – 2 PM, with conversion rates stable across hours.  
- Weekly: Mondays (20.4%) and Tuesdays (17.8%) have the highest lead volume, while weekends see minimal activity.  
- Monthly: June (17.3%) and May (17.0%) drive the most leads, but November (6.2% conversion rate) has the highest quality despite lower volume.  
- Actionable Insight: Align sales team availability with peak hours/weekdays. Leverage high-conversion months (e.g., November) for targeted outreach.  



 4. Product and Delivery Mode Impact  
- Product ID: Higher IDs (avg. 16.6 for high-potential vs. 15.3 for low) correlate with better conversion rates.  
- Delivery Modes:  
  - Mode-1 (46.7% high-potential) and Mode-3 (24.5%) are effective.  
  - Mode-5 (40.1% of total leads) is heavily associated with low-potential leads (49%).  
- Actionable Insight: Promote products with higher IDs and streamline delivery processes using Mode-1/Mode-3. Investigate why Mode-5 underperforms.  



 5. Lead Status and Conversion Rates  
- High-Potential Statuses: CONVERTED (29.3%), In Progress Positive (22.6%), and Potential (24.9%).  
- Low-Potential Statuses: Junk Lead (33.6%), Not Responding (24.7%), and Just Enquiry (16.6%) dominate.  
- Actionable Insight: Implement automated follow-ups for In Progress Positive leads. Improve lead qualification to reduce junk entries.  



 Recommendations for Sales Effectiveness  
1. Channel Optimization: Focus on high-conversion sources (referrals, calls) and reduce reliance on low-quality channels (e.g., US Website).  
2. Regional Targeting: Allocate resources to Bangalore and Chennai; improve data collection for Other Locations.  
3. Temporal Alignment: Schedule sales activities during peak hours (10 AM–2 PM) and high-volume months (May–July).  
4. Product & Delivery Strategy: Promote high-ID products and prioritize Mode-1/Mode-3 delivery.  
5. Lead Management: Automate follow-ups for promising leads and refine lead qualification criteria to minimize junk entries.  

By leveraging these insights, FicZon Inc. can enhance lead prioritization, reduce manual effort, and drive higher conversion rates.

----


**Final Model Comparison and Report**

  Model Comparison and Final Report  



 Model Performance Comparison  
The following table summarizes key performance metrics across base models, tuned models, and ensembles. Metrics include accuracy, balanced accuracy, precision, recall, F1-score, and overfitting margin:  

| Model Type       | Model           | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | Overfitting (Train - Test) |  
|----------------------|---------------------|--------------|-----------------------|---------------|------------|--------------|--------------------------------|  
| Base Models      | CatBoost            | 70.75%       | 71.87%                | 59.21%        | 76.67%     | 66.82%       | +3.18%                         |  
|                      | XGBoost             | 71.36%       | 70.58%                | 61.67%        | 67.19%     | 64.32%       | +6.89%                         |  
|                      | LightGBM            | 69.61%       | 70.71%                | 58.03%        | 75.44%     | 65.60%       | +5.27%                         |  
| Tuned Models     | CatBoost            | 73.05%       | 69.90%                | 68.01%        | 56.32%     | 61.61%       | +0.84%                         |  
|                      | XGBoost             | 72.44%       | 71.82%                | 62.84%        | 69.12%     | 65.83%       | -0.64%                         |  
|                      | LightGBM            | 69.20%       | 71.24%                | 57.07%        | 80.00% | 66.62%       | -0.42%                         |  
| Ensembles        | Voting              | 71.90%       | 71.24%                | 62.20%        | 68.42%     | 65.16%       | +0.42%                         |  
|                      | Stacking            | 70.55%       | 71.41%                | 59.20%        | 75.09%     | 66.20%       | +0.40%                         |  

Key Observations:  
1. Tuned XGBoost achieved the best balance between accuracy (72.44%), balanced accuracy (71.82%), and recall (69.12%) while minimizing overfitting.  
2. LightGBM excelled in recall (80.00%), making it ideal for minimizing false negatives.  
3. CatBoost prioritized precision (68.01%) but had lower recall.  
4. Ensembles (Voting/Stacking) showed stable generalization but did not outperform the best-tuned individual models.  



 Feature Importance Analysis  
For the final XGBoost model, the most influential features were:  
1. Location Frequency (32.27%): Geographic distribution of leads significantly impacts classification.  
2. Product_ID (25.33%): Product variations drive lead conversion behavior.  
3. Delivery Mode (26.92% combined): Modes 3, 4, and 5 are critical predictors.  
4. Source Frequency (7.92%) and Created Month (7.57%): Lead source and timing influence outcomes.  



 Final Model Selection: XGBoost  
Rationale:  
- Performance: Highest accuracy (72.44%) and balanced accuracy (71.82%) among tuned models.  
- Generalization: Minimal overfitting (-0.64%) and strong cross-validation F1-score (63.04% ±1.96%).  
- Threshold Optimization: At 0.40 threshold, it balances precision (56%) and recall (85%), prioritizing lead capture.  
- Interpretability: Clear feature importance aligns with business logic (e.g., location and product factors).  

Confusion Matrix at Threshold 0.40:  
- True Positives (TP): 483 | False Negatives (FN): 87  
- True Negatives (TN): 531 | False Positives (FP): 383  

Key Metrics:  
- F1-Score: 67.27%  
- ROC AUC: 81.06%  
- Brier Score: 0.173 (Well-calibrated probabilities).  


 Conclusion  
The XGBoost model is recommended for deployment due to its robust performance, interpretability, and balance between precision and recall. By prioritizing recall (85% at threshold 0.40), it effectively identifies high-potential leads while maintaining reasonable precision. Geographic and product-related features are critical drivers, providing actionable insights for marketing strategies.  

Next Steps:  
1. Deploy the model with a 0.40 decision threshold to prioritize lead capture.  
2. Monitor performance quarterly to address potential data drift.  
3. Optimize delivery modes and geographic targeting based on feature importance.  

---

# Author Information

- Dhanesh B. B.  

- Contact Information:  
    - [Email](dhaneshbb5@gmail.com) 
    - [LinkedIn](https://www.linkedin.com/in/dhanesh-b-b-2a8971225/) 
    - [GitHub](https://github.com/dhaneshbb)

---

# References

1. DataMites™ Solutions Pvt Ltd, "PRCL-0019: Sales Effectiveness & Lead Qualification," Internal Documentation, 2018.
2. FicZon Inc., "Sales Effectiveness Dataset," Internal Data, 2025.

----

# Appendix

## About data
 About the Data  

The dataset for the PRCL-0019: Sales Effectiveness & Lead Qualification project consists of 7422 records and 9 features, capturing key aspects of lead generation and sales conversion. The data is sourced from FicZon Inc., an IT solutions provider, and is used to improve lead categorization through Machine Learning.

 Data Overview  
- Total Entries: 7422  
- Total Features: 9  
- Primary Objective: Predict lead category (High Potential vs. Low Potential) to enhance sales effectiveness.

 Column Description
| Column Name   | Description  | Type |
|------------------|----------------|---------|
| Created | Timestamp of lead creation | Date/Time |
| Product_ID | ID of the product associated with the lead | Numerical |
| Source | Lead source (e.g., Website, Referral) | Categorical |
| Mobile | Contact number of the lead | Text |
| EMAIL | Email address of the lead | Text |
| Sales_Agent | Sales agent handling the lead | Categorical |
| Location | Geographical location of the lead | Categorical |
| Delivery_Mode | Mode of product/service delivery | Categorical |
| Status | Current status of the lead (Open, Closed, etc.) | Categorical |


## Source Code and Dependencies

In the development of the predictive models for this project, I extensively utilized several functions from my custom library "insightfulpy." This library, available on both GitHub and PyPI, provided crucial functionalities that enhanced the data analysis and modeling process. For those interested in exploring the library or using it in their own projects, you can inspect the source code and documentation available. The functions from "insightfulpy" helped streamline data preprocessing, feature engineering, and model evaluation, making the analytic processes more efficient and reproducible.

You can find the source and additional resources on GitHub here: [insightfulpy on GitHub](https://github.com/dhaneshbb/insightfulpy), and for installation or further documentation, visit [insightfulpy on PyPI](https://pypi.org/project/insightfulpy/). These resources provide a comprehensive overview of the functions available and instructions on how to integrate them into your data science workflows.

---

Below is an overview of each major tool (packages, user-defined functions, and imported functions) that appears in this project.

<pre>
Imported packages:
1: builtins
2: builtins
3: chardet
4: csv
5: pandas
6: warnings
7: researchpy
8: matplotlib.pyplot
9: missingno
10: seaborn
11: numpy
12: scipy.stats
13: textwrap
14: logging
15: time
16: joblib
17: statsmodels.api
18: sklearn.utils.class_weight
19: scikitplot
20: xgboost
21: lightgbm
22: catboost
23: psutil
24: os
25: gc
26: types
27: inspect

User-defined functions:
1: memory_usage
2: dataframe_memory_usage
3: garbage_collection
4: normality_test_with_skew_kurt
5: spearman_correlation
6: calculate_vif
7: evaluate_model
8: threshold_analysis
9: cross_validation_analysis_table
10: plot_all_evaluation_metrics
11: print_feature_importance_xgb

Imported functions:
1: open
2: tabulate
3: display
4: is_datetime64_any_dtype
5: skew
6: kurtosis
7: shapiro
8: kstest
9: compare_df_columns
10: linked_key
11: display_key_columns
12: interconnected_outliers
13: grouped_summary
14: calc_stats
15: iqr_trimmed_mean
16: mad
17: comp_cat_analysis
18: comp_num_analysis
19: detect_mixed_data_types
20: missing_inf_values
21: columns_info
22: cat_high_cardinality
23: analyze_data
24: num_summary
25: cat_summary
26: calculate_skewness_kurtosis
27: detect_outliers
28: show_missing
29: plot_boxplots
30: kde_batches
31: box_plot_batches
32: qq_plot_batches
33: num_vs_num_scatterplot_pair_batch
34: cat_vs_cat_pair_batch
35: num_vs_cat_box_violin_pair_batch
36: cat_bar_batches
37: cat_pie_chart_batches
38: num_analysis_and_plot
39: cat_analyze_and_plot
40: chi2_contingency
41: fisher_exact
42: pearsonr
43: spearmanr
44: ttest_ind
45: mannwhitneyu
46: linkage
47: dendrogram
48: leaves_list
49: variance_inflation_factor
50: train_test_split
51: cross_val_score
52: learning_curve
53: resample
54: compute_class_weight
55: mutual_info_classif
56: accuracy_score
57: precision_score
58: recall_score
59: f1_score
60: roc_auc_score
61: confusion_matrix
62: balanced_accuracy_score
63: matthews_corrcoef
64: log_loss
65: brier_score_loss
66: cohen_kappa_score
67: precision_recall_curve
68: roc_curve
69: auc
70: classification_report
71: calibration_curve
</pre>